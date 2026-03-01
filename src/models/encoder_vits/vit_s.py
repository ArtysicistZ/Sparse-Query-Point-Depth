import torch
import torch.nn as nn
import warnings

warnings.filterwarnings("ignore", message="xFormers is not available")

class ViTSEncoder(nn.Module):

    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.encoder = torch.hub.load(
            'facebookresearch/dinov2', 
            'dinov2_vits14', 
            pretrained=pretrained
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.encoder.get_intermediate_layers(
            x, [2, 5, 8, 11],
            return_class_token=False
        )
        h, w = x.shape[2] // 14, x.shape[3] // 14
        out = []
        for i, feat in enumerate(features):
            out.append(feat.reshape(x.shape[0], h, w, -1).permute(0, 3, 1, 2))
        return {
            "L1": out[0],
            "L2": out[1],
            "L3": out[2],
            "L4": out[3],
        }