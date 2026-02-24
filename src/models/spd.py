import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoder import ConvNeXtV2Encoder
from models.pyramid_neck import ProjectionNeck
from models.precompute import PreCompute
from models.query_encoder import TokenConstructor
from models.local_cross_attn import LocalCrossAttn
from models.global_cross_attn import GlobalCrossAttn

class SPD(nn.Module):

    def __init__(self, pretrained=True):
        super().__init__()
        self.encoder = ConvNeXtV2Encoder(pretrained=pretrained)
        self.neck = ProjectionNeck()
        self.precompute = PreCompute()
        self.b1 = TokenConstructor()
        self.b2 = LocalCrossAttn()
        self.b3 = GlobalCrossAttn()
        self.depth_head = nn.Sequential(
            nn.Linear(192, 384),
            nn.GELU(),
            nn.Linear(384, 1)
        )

    def forward(self, images: torch.Tensor, query_coords: torch.Tensor) -> torch.Tensor:
        features = self.encoder(images)
        features = self.neck(features)
        features = self.precompute(features)

        query_tokens, seed = self.b1(features, query_coords)
        h = self.b2(seed, query_tokens)
        h, top_indices = self.b3(h, features)

        r_q = self.depth_head(h).squeeze(-1)  # [B, K]
        s = F.softplus(features['s'])
        b = features['b']
        inv_depth = F.softplus(r_q * s + b) + 1e-6  # [B, K]
        depth = 1.0 / inv_depth
        return depth, inv_depth
    
if __name__ == "__main__":
    model = SPD(pretrained=False)
    images = torch.randn(1, 3, 480, 640)
    coords = torch.randint(0, 480, (1, 16, 2)).float()
    depth = model(images, coords)
    print(f"depth: {depth.shape}")  # expect [1, 16]
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")