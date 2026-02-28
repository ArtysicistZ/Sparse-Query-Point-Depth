import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoder.convnext import ConvNeXtV2Encoder

class ProjectionNeck(nn.Module):

    def __init__(self, 
                 enc_channels: list[int] = [96, 192, 384, 768], 
                 d_model: int = 64):
        
        super().__init__()
        dec_channels = [d_model] * len(enc_channels)
        self.projections = nn.ModuleList([
            nn.Conv2d(in_ch, out_ch, kernel_size=1) 
            for in_ch, out_ch in zip(enc_channels, dec_channels)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(out_ch) for out_ch in dec_channels
        ])


    def _proj(self, x: torch.Tensor, conv, ln) -> torch.Tensor:
        B, C, H, W = x.shape
        x = conv(x)                # [B, out_ch, H, W]
        x = x.permute(0, 2, 3, 1)  # [B, H, W, out_ch]
        x = ln(x)
        x = x.permute(0, 3, 1, 2)  # [B, out_ch, H, W]
        return x
    
    
    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        keys = list(features.keys())
        projects = {}
        for k, conv, ln in zip(keys, self.projections, self.norms):
            projects[k] = self._proj(features[k], conv, ln)
        return projects
    
