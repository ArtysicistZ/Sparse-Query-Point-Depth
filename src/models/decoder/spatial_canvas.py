import torch
import torch.nn as nn
import torch.nn.functional as F

from config import H_IMG as H_img, W_IMG as W_img


class CanvasLayer(nn.Module):

    def __init__(self, d_model: int = 192):
        super().__init__()
        
        self.W_write_L2 = nn.Linear(d_model, d_model)
        self.W_write_L3 = nn.Linear(d_model, d_model)
        self.W_write_L4 = nn.Linear(d_model, d_model)

        self.gate_43_fine = nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
        self.gate_43_coarse = nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)

        self.gate_32_fine = nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
        self.gate_32_coarse = nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)

        self.smooth_L2 = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=5, padding=2, groups=d_model), 
            nn.GELU(), 
            nn.Conv2d(d_model, d_model, kernel_size=5, padding=2, groups=d_model)
        )

        self.smooth_L3 = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=5, padding=2, groups=d_model), 
            nn.GELU(), 
            nn.Conv2d(d_model, d_model, kernel_size=5, padding=2, groups=d_model)
        )

        self.smooth_L4 = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, groups=d_model), 
            nn.GELU(), 
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
        )

        self.ln = nn.LayerNorm(d_model)
        self.W_read = nn.Linear(3 * d_model, d_model)
        self.W_gate = nn.Linear(d_model, 1)


    
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

        update = torch.zeros_like(canvas_flat)
        update.scatter_add_(2, idx, w.to(update.dtype))  # [B, d_model, H_canvas*W_canvas]

        canvas = (canvas_flat + update).view(B, D, H_canvas, W_canvas)  # [B, d_model, H_canvas, W_canvas]

        return canvas


    def forward(self, h: torch.Tensor, canvas: dict[str, torch.Tensor], center_grid: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:

        # read from canvas
        B, K, D = h.shape

        w_2 = self.W_write_L2(h).transpose(1, 2)  # [B, d_model, K]
        canvas['L2'] = self.write(w_2, canvas['L2'], coords)  # [B, d_model, H_l2, W_l2]

        w_3 = self.W_write_L3(h).transpose(1, 2)  # [B, d_model, K]
        canvas['L3'] = self.write(w_3, canvas['L3'], coords)  # [B, d_model, H_l3, W_l3]

        w_4 = self.W_write_L4(h).transpose(1, 2)  # [B, d_model, K]
        canvas['L4'] = self.write(w_4, canvas['L4'], coords)  # [B, d_model, H_l4, W_l4]

        # cross-level gating
        l4_up = F.interpolate(canvas['L4'], size=canvas['L3'].shape[2:], mode='bilinear', align_corners=True)
        gate_43 = torch.sigmoid(self.gate_43_fine(canvas['L3']) + self.gate_43_coarse(l4_up))  # [B, d_model, H_l3, W_l3]
        canvas['L3'] = canvas['L3'] + gate_43 * l4_up

        l3_up = F.interpolate(canvas['L3'], size=canvas['L2'].shape[2:], mode='bilinear', align_corners=True)
        gate_32 = torch.sigmoid(self.gate_32_fine(canvas['L2']) + self.gate_32_coarse(l3_up))  # [B, d_model, H_l2, W_l2]
        canvas['L2'] = canvas['L2'] + gate_32 * l3_up

        # Smooth canvas
        canvas['L2'] = canvas['L2'] + self.smooth_L2(canvas['L2'])  # [B, d_model, H_l2, W_l2]
        canvas['L3'] = canvas['L3'] + self.smooth_L3(canvas['L3'])  # [B, d_model, H_l3, W_l3]
        canvas['L4'] = canvas['L4'] + self.smooth_L4(canvas['L4'])  # [B, d_model, H_l4, W_l4]

        h_norm = self.ln(h)

        read_L2 = F.grid_sample(canvas['L2'], center_grid, align_corners=True).squeeze(-1).transpose(1, 2)  # [B, K, d_model]
        read_L3 = F.grid_sample(canvas['L3'], center_grid, align_corners=True).squeeze(-1).transpose(1, 2)  # [B, K, d_model]
        read_L4 = F.grid_sample(canvas['L4'], center_grid, align_corners=True).squeeze(-1).transpose(1, 2)  # [B, K, d_model]

        gate = torch.sigmoid(self.W_gate(h_norm))  # [B, K, 1]
        h = h + gate * self.W_read(torch.cat([read_L2, read_L3, read_L4], dim=-1))  # [B, K, d_model]

        return h, canvas  # [B, K, d_model]