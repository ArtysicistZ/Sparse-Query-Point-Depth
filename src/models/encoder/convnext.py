from typing import Dict
import timm
import torch
import torch.nn as nn

class ConvNeXtV2Encoder(nn.Module):

    def __init__(self, pretrained: bool = True):
        super().__init__()
        encoder = timm.create_model(
            'convnextv2_tiny.fcmae_ft_in22k_in1k', 
            pretrained=pretrained
        )
        self.stem = encoder.stem
        self.stages = encoder.stages


    def forward_l(self, x) -> dict[str, torch.Tensor]:
        """Forward pass through the first 3 stages"""
        x = self.stem(x)
        L1 = self.stages[0](x)
        L2 = self.stages[1](L1)
        L3 = self.stages[2](L2)
        return L1, L2, L3
    

    def forward_l4(self, L3_enhanced) -> torch.Tensor:
        """Forward pass through the 4th stage, with optional self-attention on L3 features."""
        L4 = self.stages[3](L3_enhanced)
        return L4