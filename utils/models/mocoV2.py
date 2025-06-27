import copy
from typing import List, Tuple

import torch
from torch import nn, Tensor
import torchvision

from lightly.loss import NTXentLoss
from lightly.models.modules import MoCoProjectionHead
from lightly.models import utils
from lightly.transforms import MoCoV2Transform


class MoCoV2(nn.Module):
    def __init__(self, momentum=0.999):
        super().__init__()
        
        # Backbone network
        resnet = torchvision.models.resnet50(weights=True)
        resnet.fc = nn.Identity()  # Remove classification head
        self.backbone = resnet
        
        # Projection heads
        self.projection_head = MoCoProjectionHead()
        
        # Momentum encoder
        self.key_backbone = copy.deepcopy(self.backbone)
        self.key_projection_head = copy.deepcopy(self.projection_head)
        
        self.momentum = momentum
        
        # Deactivate gradients for momentum encoder
        utils.deactivate_requires_grad(self.key_backbone)
        utils.deactivate_requires_grad(self.key_projection_head)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for query encoder"""
        features = self.backbone(x).flatten(start_dim=1)
        projections = self.projection_head(features)
        return projections

    @torch.no_grad()
    def forward_momentum(self, x: Tensor) -> Tensor:
        """Forward pass for key encoder (momentum)"""
        features = self.key_backbone(x).flatten(start_dim=1)
        projections = self.key_projection_head(features)
        return projections

    def update_momentum(self, momentum=None):
        """Update momentum parameters"""
        if momentum is None:
            momentum = self.momentum
            
        utils.update_momentum(self.backbone, self.key_backbone, momentum)
        utils.update_momentum(self.projection_head, self.key_projection_head, momentum)


class MoCoV2Loss(nn.Module):
    def __init__(self, temperature=0.2, memory_bank_size=(65536, 128)):
        super().__init__()
        self.criterion = NTXentLoss(
            temperature=temperature,
            memory_bank_size=memory_bank_size,
        )

    def forward(self, query_projections, key_projections):
        """Compute MoCoV2 contrastive loss"""
        loss = self.criterion(query_projections, key_projections)
        return loss


transform = MoCoV2Transform()