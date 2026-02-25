import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthHead(nn.Module):

    def __init__(self, d_model: int = 192):
        super().__init__()

        self.ffn = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, 1)
        )
        self.s = nn.Parameter(torch.tensor(0.0))  # Learnable scaling factor


    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        h: [B, K, d_model]
        """
        B, K, D = h.shape

        h = self.ffn(h)  # [B, K, 1]
        log_depth = torch.exp(self.s) * h.squeeze(-1)  # [B, K]

        return log_depth


