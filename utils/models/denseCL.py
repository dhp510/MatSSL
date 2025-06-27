# This example requires the following dependencies to be installed:
# pip install lightly

# Note: The model and training settings do not follow the reference settings
# from the paper. The settings are chosen such that the example can easily be
# run on a small dataset with a single GPU.

import copy

import torch
import torchvision
from torch import nn

from lightly.loss import NTXentLoss
from lightly.models import utils
from lightly.models.modules import DenseCLProjectionHead
from lightly.transforms import DenseCLTransform
from lightly.utils.scheduler import cosine_schedule

# Modify the ResNet without using Sequential
class ModifiedResNet(nn.Module):
    def __init__(self):
        super(ModifiedResNet, self).__init__()
        self.backbone = torchvision.models.resnet50(weights=True)
        self.backbone.fc = nn.Identity()
        self.backbone.avgpool = nn.Identity()  # This prevents the pooling operation

    def forward(self, x):
        # Pass the input through each layer of the original resnet50
        # Process all layers up to the penultimate layer (before avgpool)
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        # Do not perform avgpool or flattening, just return the feature map
        return x

class DenseCL(nn.Module):
    def __init__(self, momentum=0.999):
        super().__init__()
        self.backbone = ModifiedResNet()
        # Fix: ResNet50 outputs 2048-dimensional features, not 512
        self.projection_head_global = DenseCLProjectionHead(2048, 512, 128)
        self.projection_head_local = DenseCLProjectionHead(2048, 512, 128)

        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_global_momentum = copy.deepcopy(
            self.projection_head_global
        )
        self.projection_head_local_momentum = copy.deepcopy(self.projection_head_local)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.momentum = momentum

        utils.deactivate_requires_grad(self.backbone_momentum)
        utils.deactivate_requires_grad(self.projection_head_global_momentum)
        utils.deactivate_requires_grad(self.projection_head_local_momentum)

    def forward(self, x):
        query_features = self.backbone(x)
        query_global = self.pool(query_features).flatten(start_dim=1)
        query_global = self.projection_head_global(query_global)
        query_features = query_features.flatten(start_dim=2).permute(0, 2, 1)
        query_local = self.projection_head_local(query_features)
        # Shapes: (B, H*W, C), (B, D), (B, H*W, D)
        return query_features, query_global, query_local

    @torch.no_grad()
    def forward_momentum(self, x):
        key_features = self.backbone_momentum(x)
        key_global = self.pool(key_features).flatten(start_dim=1)
        key_global = self.projection_head_global_momentum(key_global)
        key_features = key_features.flatten(start_dim=2).permute(0, 2, 1)
        key_local = self.projection_head_local_momentum(key_features)
        return key_features, key_global, key_local

    def update_momentum(self, momentum=None):
        """Update momentum parameters"""
        if momentum is None:
            momentum = self.momentum

        utils.update_momentum(self.backbone, self.backbone_momentum, momentum)
        utils.update_momentum(self.projection_head_global, self.projection_head_global_momentum, momentum)
        utils.update_momentum(self.projection_head_local, self.projection_head_local_momentum, momentum)


class DenseCLLoss(nn.Module):
    def __init__(self, memory_bank_size=(4096, 128)):
        super().__init__()
        self.criterion_global = NTXentLoss(memory_bank_size=memory_bank_size)
        self.criterion_local = NTXentLoss(memory_bank_size=memory_bank_size)

    def forward(self, query_global, key_global, query_local, key_local):
        """
        Compute DenseCL loss combining global and local contrastive losses.

        Args:
            query_global: Global query features (B, D)
            key_global: Global key features (B, D)
            query_local: Local query features (B, H*W, D)
            key_local: Local key features (B, H*W, D)

        Returns:
            Dictionary containing total loss and individual loss components
        """
        # Compute global contrastive loss
        loss_global = self.criterion_global(query_global, key_global)

        # Reshape local features for contrastive loss computation
        # From (B, H*W, D) to (B*H*W, D)
        batch_size, num_patches, dim = query_local.shape
        query_local_flat = query_local.reshape(-1, dim)
        key_local_flat = key_local.reshape(-1, dim)

        # Compute local contrastive loss
        loss_local = self.criterion_local(query_local_flat, key_local_flat)

        # Combine losses
        total_loss = loss_global + loss_local

        return {
            "total_loss": total_loss,
            "global_loss": loss_global,
            "local_loss": loss_local,
        }
