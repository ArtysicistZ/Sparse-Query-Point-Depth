import torch
import torch.nn as nn

from models.encoder import ConvNeXtV2Encoder
from models.pyramid_neck import ProjectionNeck
from models.precompute import PreCompute
from models.query_encoder import TokenConstructor
from models.local_cross_attn import LocalCrossAttn
from models.global_cross_attn import GlobalCrossAttn
from models.deformable_read import DeformableRead
from models.fused_decoder import FusedDecoder

class SPD(nn.Module):

    def __init__(self, pretrained=True):
        super().__init__()
        self.encoder = ConvNeXtV2Encoder(pretrained=pretrained)
        self.neck = ProjectionNeck()
        self.precompute = PreCompute()
        self.b1 = TokenConstructor()
        self.b2 = LocalCrossAttn()
        self.b3 = GlobalCrossAttn()
        self.b4 = DeformableRead()
        self.b5 = FusedDecoder()

    def forward(self, images: torch.Tensor, query_coords: torch.Tensor) -> torch.Tensor:
        features = self.encoder(images)
        features = self.neck(features)
        features = self.precompute(features)

        query_tokens, seed = self.b1(features, query_coords)
        h = self.b2(seed, query_tokens)
        h, top_indices = self.b3(h, features)

        deform_tokens = self.b4(h, top_indices, query_coords, features)
        fused_tokens = torch.cat([query_tokens, deform_tokens], dim=2)

        log_depth = self.b5(h, fused_tokens)  # [B, K]
        depth = torch.exp(log_depth)           # metric depth
        return depth

if __name__ == "__main__":
    model = SPD(pretrained=False)
    images = torch.randn(1, 3, 256, 320)
    coords = torch.randint(0, 256, (1, 16, 2)).float()
    depth = model(images, coords)
    print(f"depth: {depth.shape}")  # expect [1, 16]
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
