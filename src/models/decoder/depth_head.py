import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthHead(nn.Module):

    def __init__(self, d_model: int = 192):
        super().__init__()

        self.W_final = nn.Linear(2 * d_model, d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, 1)
        )
        self.s = nn.Parameter(torch.tensor(1.0))  # Learnable scaling factor


    def forward(self, h: torch.Tensor, canvas_L2: torch.Tensor, canvas_L3: torch.Tensor, center_grid: torch.Tensor) -> torch.Tensor:
        """
        h: [B, K, d_model]
        """
        B, K, D = h.shape

        read_L2 = F.grid_sample(canvas_L2, center_grid, align_corners=True).squeeze(-1).transpose(1, 2)  # [B, K, d_model]
        read_L3 = F.grid_sample(canvas_L3, center_grid, align_corners=True).squeeze(-1).transpose(1, 2)  # [B, K, d_model]

        h = h + self.W_final(torch.cat([read_L2, read_L3], dim=-1))  # [B, K, d_model]

        h = self.ffn(h)  # [B, K, 1]
        log_depth = torch.log(self.s) * h.squeeze(-1)  # [B, K]

        return log_depth


