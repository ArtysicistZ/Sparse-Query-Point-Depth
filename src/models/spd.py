import torch
import torch.nn as nn

from models.encoder.convnext import ConvNeXtV2Encoder
from models.encoder.pyramid_neck import ProjectionNeck
from models.encoder.precompute import PreCompute
from models.decoder.query_encoder import TokenConstructor
from models.decoder.local_cross_attn import LocalCrossAttn
from models.decoder.global_cross_attn import GlobalCrossAttn, GlobalCrossAttnNoRouting
# from models.decoder.deformable_read import DeformableRead
from models.decoder.fused_decoder import FusedDecoder

class SPD(nn.Module):

    def __init__(self, pretrained=True):
        super().__init__()
        self.encoder = ConvNeXtV2Encoder(pretrained=pretrained)
        self.neck = ProjectionNeck()
        self.precompute = PreCompute()
        self.b1 = TokenConstructor()
        self.b2 = LocalCrossAttn()
        self.b3a = GlobalCrossAttnNoRouting()
        self.b3b = GlobalCrossAttn()
        # self.b4 = DeformableRead()
        self.b5 = FusedDecoder()

    def forward(self, images: torch.Tensor, query_coords: torch.Tensor) -> torch.Tensor:
        features = self.encoder(images)
        features = self.neck(features)
        features = self.precompute(features)

        query_tokens, center_tokens, seed = self.b1(features, query_coords)
        h = self.b2(seed, query_tokens)
        h, top_indices_l3 = self.b3b(h, features, lev=3, k=20)
        h, top_indices_l4 = self.b3b(h, features, lev=4, k=10)  # B3b on L4 for better global feature fusion

        log_depth = self.b5(h, center_tokens, top_indices_l4, top_indices_l3)
        depth = torch.exp(log_depth)           # metric depth
        return depth

if __name__ == "__main__":
    model = SPD(pretrained=False)
    images = torch.randn(1, 3, 256, 320)
    coords = torch.randint(0, 256, (1, 16, 2)).float()
    depth = model(images, coords)
    print(f"depth: {depth.shape}")  # expect [1, 16]
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
