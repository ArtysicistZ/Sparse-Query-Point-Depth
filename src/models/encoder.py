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
            features_only=True
        )

    def forward(self, x) -> Dict[str, torch.Tensor]:
        features = self.encoder(x)
        return {
            'L1': features[0],  
            'L2': features[1],  
            'L3': features[2],  
            'L4': features[3],  
        }

if __name__ == "__main__":
    model = ConvNeXtV2Encoder(pretrained=True)
    x = torch.randn(1, 3, 480, 640)
    features = model(x)
    for name, feat in features.items():
        print(f"{name}: {feat.shape}")