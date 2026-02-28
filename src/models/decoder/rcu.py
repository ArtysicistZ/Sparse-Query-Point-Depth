import torch
import torch.nn as nn
import torch.nn.functional as F

class RCU(nn.Module):

    def __init__(self, d_model: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(d_model, d_model, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(d_model, d_model, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(F.gelu(self.conv1(F.gelu(x))))
    
    def forward_infer(self, x: torch.Tensor) -> torch.Tensor:
        out = F.gelu(x)
        out = F.conv2d(out, self.conv1.weight, self.conv1.bias, padding=0)
        out = F.gelu(out)
        out = F.conv2d(out, self.conv2.weight, self.conv2.bias, padding=0)
        size = (x.shape[2] - out.shape[2]) // 2
        return x[:, :, size:size + out.shape[2], size:size + out.shape[3]] + out