import torch
import torch.nn as nn
import torch.nn.functional as F

from config import W_IMG as W_img, H_IMG as H_img

class SeedConstructor(nn.Module):

    def __init__(self, d_model: int = 192):
        super().__init__()

        self.W_seed = nn.Linear(d_model * 4 + 32, d_model)
        self.ln = nn.LayerNorm(d_model)
        self.W_pos = nn.Linear(32, d_model)


    def _fourier_pe(self, coords, n_freq):

        freqs = 2 ** torch.arange(n_freq, device=coords.device, dtype=torch.float32).float()  # [n_freq]
        x = coords[..., 0:1] * freqs * 2 * torch.pi  # [B, K, n_freq]
        y = coords[..., 1:2] * freqs * 2 * torch.pi  # [B, K, n_freq]
        return torch.cat([x.sin(), x.cos(), y.sin(), y.cos()], dim=-1)  # [B, K, n_freq*4]


    def forward(self, features, coords) -> dict[str, torch.Tensor]:
        """
        features: dict from PreCompute (L1, L2, L3, L4, d_model = 192)
        coords: [B, K, 2] query pixel coordinates
        Returns: seed: [B, K, d_model]
        """
        B, K, _ = coords.shape
 
        # L1 tokens with deformable sampling
        feat_coords = coords.float()
        center_norm = torch.stack([
            2 * feat_coords[..., 0] / (W_img - 1) - 1,
            2 * feat_coords[..., 1] / (H_img - 1) - 1
        ], dim=-1)
        center_grid = center_norm.view(B, K, 1, 2)  # [B, K, 1, 2]

        coords_norm = coords.float() / torch.tensor([W_img, H_img], device=coords.device)  # [B, K, 2]
        pe_q = self._fourier_pe(coords_norm, n_freq=8)  # [B, K, 32]        

        f_1 = F.grid_sample(features['L1'], center_grid, align_corners=True).squeeze(-1).transpose(1, 2)  
        f_2 = F.grid_sample(features['L2'], center_grid, align_corners=True).squeeze(-1).transpose(1, 2)  
        f_3 = F.grid_sample(features['L3'], center_grid, align_corners=True).squeeze(-1).transpose(1, 2)  
        f_4 = F.grid_sample(features['L4'], center_grid, align_corners=True).squeeze(-1).transpose(1, 2)  

        seed = self.W_seed(torch.cat([f_1, f_2, f_3, f_4, pe_q], dim=-1))  # [B, K, d_model]
        seed = self.ln(seed)
        pos = self.W_pos(pe_q)  # [B, K, d_model]

        return seed, pos, center_grid  # [B, K, d_model], [B, K, 2], [B, K, 1, 2]

