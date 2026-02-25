import torch
import torch.nn as nn

from models.encoder.convnext import ConvNeXtV2Encoder
from models.encoder.pyramid_neck import ProjectionNeck
from models.encoder.precompute import PreCompute
from models.decoder.query_seed import SeedConstructor
from models.decoder.global_cross_attn import GlobalCrossAttn

class SPD(nn.Module):

    def __init__(self, pretrained=True):
        super().__init__()
        self.encoder = ConvNeXtV2Encoder(pretrained=pretrained)
        self.neck = ProjectionNeck()
        self.precompute = PreCompute()
        self.b1 = SeedConstructor()
        self.b3 = GlobalCrossAttn()


    def forward(self, images: torch.Tensor, query_coords: torch.Tensor,
                return_debug: bool = False) -> torch.Tensor:

        features = self.encoder(images)
        features = self.neck(features)
        features = self.precompute(features)

        seed, pos_q, center_grid = self.b1(features, query_coords)
        
        h = self.b3(seed, features)

        depth = torch.exp(log_depth)           # metric depth

        return depth

if __name__ == "__main__":
    model = SPD(pretrained=False)
    images = torch.randn(1, 3, 256, 320)
    coords = torch.randint(0, 256, (1, 16, 2)).float()
    depth = model(images, coords)
    print(f"depth: {depth.shape}")  # expect [1, 16]
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
