import torch
import torch.nn as nn
import torch.nn.functional as F

from config import H_IMG as H_img, W_IMG as W_img

class CanvasSmooth(nn.Module):

    def __init__(self, d_model: int = 192):
        super().__init__()

        self.smooth_L2 = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=5, padding=2, groups=d_model), 
            nn.GELU(), 
            nn.Conv2d(d_model, d_model, kernel_size=5, padding=2, groups=d_model)
        )

        self.smooth_L3 = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, groups=d_model), 
            nn.GELU(), 
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
        )


    def forward(self, canvas_L2: torch.Tensor, canvas_L3: torch.Tensor) -> torch.Tensor:
        
        # Smooth canvas
        canvas_L2 = canvas_L2 + self.smooth_L2(canvas_L2)  # [B, d_model, H_l2, W_l2]
        canvas_L3 = canvas_L3 + self.smooth_L3(canvas_L3)  # [B, d_model, H_l3, W_l3]

        return canvas_L2, canvas_L3  # [B, d_model, H_l2, W_l2], [B, d_model, H_l3, W_l3]



class CanvasLayer(nn.Module):

    def __init__(self, d_model: int = 192):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.W_read = nn.Linear(2 * d_model, d_model)
        self.W_gate = nn.Linear(d_model, 1)
        
        self.W_write_L2 = nn.Linear(d_model, d_model)
        self.W_write_L3 = nn.Linear(d_model, d_model)


    
    def write(self, w: torch.Tensor, canvas: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """
        w: [B, d_model, K]
        """
        B, D, K = w.shape

        stride = H_img // canvas.shape[2]
        H_canvas = H_img // stride
        W_canvas = W_img // stride

        col = (coords[:, :, 0] // stride).long()
        row = (coords[:, :, 1] // stride).long()

        flat_idx = row * W_canvas + col
        canvas_flat = canvas.view(B, D, -1)  # [B, d_model, H_canvas*W_canvas]
        idx = flat_idx.unsqueeze(1).expand(-1, D, -1)  # [B, d_model, K]

        canvas_flat.scatter_add_(2, idx, w)  # [B, d_model, H_canvas*W_canvas]
        canvas = canvas_flat.view(B, D, H_canvas, W_canvas)  # [B, d_model, H_canvas, W_canvas]

        return canvas
    
    
    def forward(self, h: torch.Tensor, canvas_L2: torch.Tensor, canvas_L3: torch.Tensor, center_grid: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:

        # read from canvas
        B, K, D = h.shape
        h_norm = self.ln(h)

        read_L2 = F.grid_sample(canvas_L2, center_grid, align_corners=True).squeeze(-1).transpose(1, 2)  # [B, K, d_model]
        read_L3 = F.grid_sample(canvas_L3, center_grid, align_corners=True).squeeze(-1).transpose(1, 2)  # [B, K, d_model]

        gate = torch.sigmoid(self.W_gate(h_norm))  # [B, K, 1]
        h = h + gate * self.W_read(torch.cat([read_L2, read_L3], dim=-1))  # [B, K, d_model]

        w_2 = self.W_write_L2(h).transpose(1, 2)  # [B, d_model, K]
        canvas_L2 = self.write(w_2, canvas_L2, coords)  # [B, d_model, H_l2, W_l2]

        w_3 = self.W_write_L3(h).transpose(1, 2)  # [B, d_model, K]
        canvas_L3 = self.write(w_3, canvas_L3, coords)  # [B, d_model, H_l3, W_l3]

        return h  # [B, K, d_model]