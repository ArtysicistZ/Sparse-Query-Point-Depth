import torch
import torch.nn as nn
import torch.nn.functional as F

from config import W_IMG as W_img, H_IMG as H_img

class TokenConstructor(nn.Module):

    def __init__(self, d_model: int = 192,
                 l1_channels: int = 64, 
                 l2_channels: int = 128,
                 l4_channels: int = 384):
        super().__init__()
        self.d_model = d_model

        self.proj_l2 = nn.Linear(l2_channels, d_model)
        self.proj_l4 = nn.Linear(l4_channels, d_model)

        self.e_L1 = nn.Parameter(torch.zeros(1, 1, 1, d_model))
        self.e_L2 = nn.Parameter(torch.zeros(1, 1, 1, d_model))
        self.e_L3 = nn.Parameter(torch.zeros(1, 1, 1, d_model))
        self.e_L4 = nn.Parameter(torch.zeros(1, 1, 1, d_model))

        self.w_off = nn.Linear(l1_channels, 8 * 2)
        self.w_loc = nn.Linear(l1_channels + 16, d_model)
        self.w_q = nn.Linear(l1_channels + 32, d_model)

        self.rpe = nn.Parameter(torch.randn(25, d_model) * 0.02)  # Learnable RPE for 5x5 local grid

        self.register_buffer('l4_rpe_idx', torch.tensor([6,7,8,11,12,13,16,17,18]))  # 3x3 subset for L4


    def _fourier_pe(self, coords, n_freq):

        B, K, _ = coords.shape
        freqs = 2 ** torch.arange(n_freq, device=coords.device, dtype=torch.float32).float()  # [n_freq]
        x = coords[..., 0:1] * freqs * 2 * torch.pi  # [B, K, n_freq]
        y = coords[..., 1:2] * freqs * 2 * torch.pi  # [B, K, n_freq]
        return torch.cat([x.sin(), x.cos(), y.sin(), y.cos()], dim=-1)  # [B, K, n_freq*4]


    def _make_local_grid(self, coords, patch_size, stride):
        
        B, K, _ = coords.shape
        P = patch_size ** 2
        H_feat, W_feat = H_img // stride, W_img // stride

        feat_coords = coords.float() / stride 

        r = patch_size // 2

        off = torch.arange(-r, r + 1, device=coords.device, dtype=torch.float32)  # [-r, ..., 0, ..., r]
        dy, dx = torch.meshgrid(off, off, indexing='ij')  # [patch_size, patch_size]
        offsets = torch.stack([dx.flatten(), dy.flatten()], dim=-1)  # [P, 2]

        grid = feat_coords.unsqueeze(2) + offsets[None, None]  # Broadcast: [B, K, P, 2]
        grid[..., 0] = 2 * grid[..., 0] / (W_feat - 1) - 1
        grid[..., 1] = 2 * grid[..., 1] / (H_feat - 1) - 1

        return grid.view(B, K * P, 1, 2) 
    
    
    def _sample_and_reshape(self, feat, grid, B, K, P):
        sampled = F.grid_sample(
            feat,
            grid,
            align_corners=True,
            mode='bilinear',
        )
        C = sampled.shape[1]
        return sampled.squeeze(-1).permute(0, 2, 1).view(B, K, P, C)


    def forward(self, features, coords) -> dict[str, torch.Tensor]:
        """
        features: dict from PreCompute (L1, L2, L3 etc.)
        coords: [B, K, 2] query pixel coordinates
        Returns: [B, K, num_tokens, d_model]
        """
        B, K, _ = coords.shape
 
        grid_L2 = self._make_local_grid(coords, patch_size=5, stride=8) 
        grid_L3 = self._make_local_grid(coords, patch_size=5, stride=16) 
        grid_L4 = self._make_local_grid(coords, patch_size=3, stride=32)

        l2_raw = self._sample_and_reshape(features['L2'], grid_L2, B, K, 25)
        l3_raw = self._sample_and_reshape(features['L3'], grid_L3, B, K, 25)
        l4_raw = self._sample_and_reshape(features['L4'], grid_L4, B, K, 9)

        l2_token = self.proj_l2(l2_raw) + self.e_L2 + self.rpe[None, None]
        l3_token = l3_raw + self.e_L3 + self.rpe[None, None]
        l4_token = self.proj_l4(l4_raw) + self.e_L4 + self.rpe[self.l4_rpe_idx][None, None]

        
        stride = 4
        H_feat, W_feat = H_img // stride, W_img // stride
        feat_coords = coords.float() / stride
        center_norm = torch.stack([
            2 * feat_coords[..., 0] / (W_feat - 1) - 1,
            2 * feat_coords[..., 1] / (H_feat - 1) - 1
        ], dim=-1)
        center_grid = center_norm.view(B, K, 1, 2)  # [B, K, 1, 2]
        f_center = self._sample_and_reshape(features['L1'], center_grid, B, K, 1).squeeze(2)  # [B, K, C]

        coords_norm = coords.float() / torch.tensor([W_img, H_img], device=coords.device)  # [B, K, 2]
        pe_q = self._fourier_pe(coords_norm, n_freq=8)  # [B, K, 32]
        seed = self.w_q(torch.cat([f_center, pe_q], dim=-1))  # [B, K, d_model]

        offsets_learned = self.w_off(f_center).view(B, K, 8, 2)
        offsets_learned = 6.0 * offsets_learned.tanh()  # Limit to [-6, 6] pixels in feature space
        
        r = 2
        off = torch.arange(-r, r + 1, device=coords.device, dtype=torch.float32)  # [-2, -1, 0, 1, 2]
        dy, dx = torch.meshgrid(off, off, indexing='ij')  # [5, 5]
        all_grid = torch.stack([dx.flatten(), dy.flatten()], dim=-1)  # [25, 2]
        mask = (all_grid[:, 0] != 0) | (all_grid[:, 1] != 0)  # Exclude center
        fixed_offsets = all_grid[mask]  # [24, 2]

        fixed_offsets = fixed_offsets[None, None].expand(B, K, -1, -1)  # [B, K, 24, 2]
        all_l1_offsets = torch.cat([fixed_offsets, offsets_learned], dim=2)  # [B, K, 32, 2]

        grid_pts = feat_coords.unsqueeze(2) + all_l1_offsets  # [B, K, 32, 2]
        grid_pts_norm = torch.stack([
            2 * grid_pts[..., 0] / (W_feat - 1) - 1,
            2 * grid_pts[..., 1] / (H_feat - 1) - 1
        ], dim=-1)  # [B, K, 32, 2]

        l1_grid = grid_pts_norm.view(B, K * 32, 1, 2)  # [B, K*32, 1, 2]
        l1_raw = self._sample_and_reshape(features['L1'], l1_grid, B, K, 32)  # [B, K, 32, C]

        offsets_flat = all_l1_offsets.view(B, K * 32, 2)  # [B, K*32, 2]
        phi_delta = self._fourier_pe(offsets_flat, n_freq=4).view(B, K, 32, 16)  # [B, K, 32, 16]

        l1_input = torch.cat([l1_raw, phi_delta], dim=-1)  # [B, K, 32, C+16]
        l1_token = F.gelu(self.w_loc(l1_input)) + self.e_L1  # [B, K, 32, d_model]

        final_tokens = torch.cat([l1_token, l2_token, l3_token, l4_token], dim=2)  # [B, K, 91, d_model]

        return final_tokens, seed


if __name__ == "__main__":
    features = {
        'L1': torch.randn(2, 64, 120, 160),
        'L2': torch.randn(2, 128, 60, 80),
        'L3': torch.randn(2, 192, 30, 40),
        'L4': torch.randn(2, 384, 15, 20),
    }
    coords = torch.randint(0, 480, (2, 16, 2)).float()  # B=2, K=16 query points

    model = TokenConstructor()
    tokens, seed = model(features, coords)
    print(f"tokens: {tokens.shape}")  # expect [2, 16, 91, 192]
    print(f"seed:   {seed.shape}")    # expect [2, 16, 192]