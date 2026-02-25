import torch
import torch.nn as nn

from models.encoder.convnext import ConvNeXtV2Encoder
from models.encoder.pyramid_neck import ProjectionNeck, SelfAttention
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
        self.l3_self_attn = SelfAttention(d_model=384, num_layers=2)
        self.neck = ProjectionNeck()
        self.precompute = PreCompute()
        self.b1 = TokenConstructor()
        self.b2 = LocalCrossAttn()
        self.b3a = GlobalCrossAttn()
        self.b3b = GlobalCrossAttn()
        # self.b4 = DeformableRead()
        self.b5 = FusedDecoder()

    def forward(self, images: torch.Tensor, query_coords: torch.Tensor,
                return_debug: bool = False) -> torch.Tensor:
        L1, L2, L3 = self.encoder.forward_l(images)
        L3 = self.l3_self_attn(L3)
        L4 = self.encoder.forward_l4(L3)
        features = {'L1': L1, 'L2': L2, 'L3': L3, 'L4': L4}
        features = self.neck(features)
        features = self.precompute(features)

        query_tokens, center_tokens, seed = self.b1(features, query_coords)
        h = self.b2(seed, query_tokens)
        h, top_indices_l3 = self.b3a(h, features, lev=3, queries=20)
        h, top_indices_l4 = self.b3b(h, features, lev=4, queries=10)

        log_depth = self.b5(h, center_tokens, top_indices_l4, top_indices_l3, features=features)  # [B, K]
        depth = torch.exp(log_depth)           # metric depth

        if return_debug:
            return depth, {
                'log_depth': log_depth,
                'top_indices_l3': top_indices_l3,
                'top_indices_l4': top_indices_l4,
            }
        return depth

if __name__ == "__main__":
    model = SPD(pretrained=False)
    images = torch.randn(1, 3, 256, 320)
    coords = torch.randint(0, 256, (1, 16, 2)).float()
    depth = model(images, coords)
    print(f"depth: {depth.shape}")  # expect [1, 16]
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
