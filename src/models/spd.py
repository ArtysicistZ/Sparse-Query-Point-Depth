import torch
import torch.nn as nn

from models.encoder.convnext import ConvNeXtV2Encoder
from models.encoder.pyramid_neck import ProjectionNeck
from models.encoder.precompute import PreCompute
from models.decoder.query_seed import SeedConstructor
from models.decoder.msda_decoder import MSDADecoder
from models.decoder.global_cross_attn import GlobalCrossAttn
from models.decoder.depth_head import DepthHead

class SPD(nn.Module):

    def __init__(self, pretrained=True):
        super().__init__()
        self.encoder = ConvNeXtV2Encoder(pretrained=pretrained)
        self.neck = ProjectionNeck()
        self.precompute = PreCompute()
        self.seed_constructor = SeedConstructor()
        self.msda_decoder = MSDADecoder()
        self.global_cross_attn = GlobalCrossAttn()
        self.depth_head = DepthHead()


    def forward(self, images: torch.Tensor, query_coords: torch.Tensor,
                return_debug: bool = False) -> torch.Tensor:

        features = self.encoder(images)
        features = self.neck(features)
        features = self.precompute(features)

        canvas_L2 = features['L2'].clone()  # [B, d_model, H_l2, W_l2]
        canvas_L3 = features['L3'].clone()  # [B, d_model, H_l3, W_l3]

        h, pos_q, center_grid = self.seed_constructor(features, query_coords)

        h, canvas_L2, canvas_L3 = self.msda_decoder(h, pos_q, features, canvas_L2, canvas_L3, center_grid, query_coords)

        h, canvas_L2, canvas_L3 = self.global_cross_attn(h, pos_q, canvas_L2, canvas_L3, features, lev=4, center_grid=center_grid, coords=query_coords)

        log_depth = self.depth_head(h)

        depth = torch.exp(log_depth)           # metric depth

        return depth


if __name__ == "__main__":
    model = SPD(pretrained=False)
    images = torch.randn(1, 3, 256, 320)
    coords = torch.randint(0, 256, (1, 16, 2)).float()
    depth = model(images, coords)
    print(f"depth: {depth.shape}")  # expect [1, 16]
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
