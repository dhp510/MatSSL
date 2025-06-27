import torch
import torch.nn as nn
from lightly.models.modules import SimCLRProjectionHead
import torchvision

class GatedFeatureFusion(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.gates = nn.ParameterList([nn.Parameter(torch.ones(1, d)) for d in dims])

    def forward(self, features):
        gated = [torch.sigmoid(g) * f for g, f in zip(self.gates, features)]
        return torch.cat(gated, dim=1)


class MatSSL(nn.Module):
    def __init__(self):
        super(MatSSL, self).__init__()
        # Initialize ResNet50 backbone
        backbone = torchvision.models.resnet50(pretrained=True)

        backbone.fc = nn.Identity() 

        projection_head = SimCLRProjectionHead(
            input_dim=3840,
            hidden_dim=2048,
            output_dim=128,
        )
        
        self.backbone = backbone
        self.projection_head = projection_head

        # Extract intermediate layers from ResNet50
        self.layer1 = nn.Sequential(
            self.backbone.conv1,
            self.backbone.bn1,
            self.backbone.relu,
            self.backbone.maxpool,
            self.backbone.layer1,
        )
        self.layer2 = self.backbone.layer2
        self.layer3 = self.backbone.layer3
        self.layer4 = self.backbone.layer4  # Last block before fc

        # Feature dimensions from each layer (ResNet50: 256, 512, 1024, 2048)
        feature_dims = [256, 512, 1024, 2048]
        self.attn_module = GatedFeatureFusion(feature_dims)

    def forward(self, x):
        x1 = self.layer1(x)  # Low-level features
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)  # High-level features

        # Global average pooling
        pooled_x1 = torch.mean(x1, dim=[2, 3])
        pooled_x2 = torch.mean(x2, dim=[2, 3])
        pooled_x3 = torch.mean(x3, dim=[2, 3])
        pooled_x4 = torch.mean(x4, dim=[2, 3])

        # Weighted combination using attention
        z_attn = self.attn_module([pooled_x1, pooled_x2, pooled_x3, pooled_x4])

        # Pass through projection head
        z = self.projection_head(z_attn)
        return z