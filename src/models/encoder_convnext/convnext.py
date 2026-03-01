from typing import Dict
import timm
import torch
import torch.nn as nn


class ConvNeXtV2Encoder(nn.Module):

    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.encoder = timm.create_model(
            'convnextv2_tiny.fcmae_ft_in22k_in1k',
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.encoder(x)
        return {
            "L1": features[0],
            "L2": features[1],
            "L3": features[2],
            "L4": features[3],
        }