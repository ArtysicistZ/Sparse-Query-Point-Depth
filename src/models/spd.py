import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoder.convnext import ConvNeXtV2Encoder
from models.encoder.pyramid_neck import ProjectionNeck
from models.decoder.global_dpt import GlobalDPT
from models.decoder.local_dpt import LocalDPT

class SPD(nn.Module):

    def __init__(self, pretrained=True):
        super().__init__()
        self.encoder = ConvNeXtV2Encoder(pretrained=pretrained)
        self.neck = ProjectionNeck()
        self.global_dpt = GlobalDPT()
        self.local_dpt = LocalDPT()


    def forward(self, images: torch.Tensor, query_coords: torch.Tensor = None) -> torch.Tensor:

        features = self.encoder(images)
        features = self.neck(features)

        r3 = self.global_dpt(features)

        if self.training:
            depth = self.local_dpt.forward_train(r3, features)

        else:
            depth = self.local_dpt.forward_infer(r3, features, query_coords)

        return depth


if __name__ == "__main__":
    model = SPD(pretrained=False)
    images = torch.randn(1, 3, 352, 480)
    coords = torch.randint(0, 256, (1, 16, 2)).float()

    model.train()
    depth_train = model(images, coords)
    print(f"depth: {depth_train.shape}") # [1, 1, 352, 480]

    model.eval()
    with torch.no_grad():
        depth_infer = model(images, coords)
    print(f"depth: {depth_infer.shape}") # [1, 16]