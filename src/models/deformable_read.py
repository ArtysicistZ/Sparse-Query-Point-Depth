import torch
import torch.nn as nn
import torch.nn.functional as F

from config import W_IMG, H_IMG


class DeformableRead(nn.Module):
    """B4: Deformable Multi-Scale Read (Section 6.4).

    For each of 32 anchors from B3 routing, predict offsets and importance
    weights, sample from multi-scale feature pyramid, and produce one
    d-dim token per anchor.

    Budget: 32 anchors × 72 samples = 2,304 deformable lookups.
    Params: ~159K.
    """

    def __init__(self, d_model: int = 192, n_head: int = 6,
                 n_levels: int = 3, n_points: int = 4):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.n_levels = n_levels
        self.n_points = n_points
        self.d_head = d_model // n_head

        # Conditioning: [h'; g_r; phi(delta_p)] -> u_r
        # Input dim: d + d + 32(fourier) = 416
        self.w_u = nn.Linear(d_model + d_model + 32, d_model)
        self.ln_u = nn.LayerNorm(d_model)

        # Offset prediction: u_r -> H*L*M*2 = 6*3*4*2 = 144
        self.w_delta = nn.Linear(d_model, n_head * n_levels * n_points * 2)

        # Importance weights: u_r -> H*L*M = 6*3*4 = 72
        self.w_a = nn.Linear(d_model, n_head * n_levels * n_points)

        # Output projection: concat heads -> d
        self.w_o = nn.Linear(d_model, d_model)

        # Type embedding for deformable tokens
        self.e_deform = nn.Parameter(torch.zeros(1, 1, 1, d_model))

        # Offset bounds per level (in feature-space pixels)
        # L2(stride=8): ±4 feat px = ±32 img px
        # L3(stride=16): ±2 feat px = ±32 img px
        # L4(stride=32): ±1 feat px = ±32 img px
        self.register_buffer('sigma', torch.tensor([4.0, 2.0, 1.0]))
        self.register_buffer('strides', torch.tensor([8.0, 16.0, 32.0]))


    def _fourier_pe(self, coords, n_freq=8):
        """Fourier positional encoding for arbitrary [..., 2] input."""
        freqs = 2 ** torch.arange(n_freq, device=coords.device, dtype=torch.float32)
        x = coords[..., 0:1] * freqs * 2 * torch.pi
        y = coords[..., 1:2] * freqs * 2 * torch.pi
        return torch.cat([x.sin(), x.cos(), y.sin(), y.cos()], dim=-1)


    def forward(self, h: torch.Tensor, top_indices: torch.Tensor,
                query_coords: torch.Tensor,
                features: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            h: [B, K, d] — B3 output (globally-aware query representation)
            top_indices: [B, K, 32] — L4 token indices from B3 routing
            query_coords: [B, K, 2] — pixel coordinates (x, y)
            features: dict with 'g', 'L2_proj', 'L3_proj', 'L4_proj'

        Returns:
            deform_tokens: [B, K, 32, d] — one token per anchor
        """
        B, K, _ = h.shape
        R = top_indices.shape[2]       # 32 anchors
        d, H, d_h = self.d_model, self.n_head, self.d_head
        L, M = self.n_levels, self.n_points

        # ── 1. Anchor pixel coordinates from L4 indices ──
        _, _, H4, W4 = features['L4_proj'].shape
        anchor_x = (top_indices % W4).float()     # [B, K, R]
        anchor_y = (top_indices // W4).float()     # [B, K, R]
        stride_4 = self.strides[2]                 # 32
        anchor_px = torch.stack([
            anchor_x * stride_4 + stride_4 / 2,   # center of L4 cell
            anchor_y * stride_4 + stride_4 / 2,
        ], dim=-1)                                 # [B, K, R, 2]

        # ── 2. Gather pre-projected anchor features g_r ──
        g = features['g']                          # [B, N_L4, d]
        idx = top_indices.view(B, -1)              # [B, K*R]
        g_r = torch.gather(g, 1, idx.unsqueeze(-1).expand(-1, -1, d))
        g_r = g_r.view(B, K, R, d)                # [B, K, R, d]

        # ── 3. Conditioning ──
        delta_p = anchor_px - query_coords.unsqueeze(2)          # [B, K, R, 2]
        delta_p_norm = delta_p / torch.tensor(
            [W_IMG, H_IMG], device=h.device, dtype=h.dtype)      # normalize to ~[-1,1]
        phi = self._fourier_pe(delta_p_norm)                     # [B, K, R, 32]

        h_exp = h.unsqueeze(2).expand(-1, -1, R, -1)            # [B, K, R, d]
        u_input = torch.cat([h_exp, g_r, phi], dim=-1)          # [B, K, R, 416]
        u_r = self.ln_u(F.gelu(self.w_u(u_input)))              # [B, K, R, d]

        # ── 4. Predict offsets and importance weights ──
        raw_offsets = self.w_delta(u_r).view(B, K, R, H, L, M, 2)
        weights = self.w_a(u_r).view(B, K, R, H, L * M)
        weights = F.softmax(weights, dim=-1)                     # over all L*M=12 samples
        weights = weights.view(B, K, R, H, L, M)

        # ── 5. Multi-scale deformable sampling ──
        proj_maps = [features['L2_proj'], features['L3_proj'], features['L4_proj']]

        h_r = torch.zeros(B, K, R, H, d_h, device=h.device, dtype=h.dtype)

        for l_idx in range(L):
            feat_map = proj_maps[l_idx]             # [B, d, H_l, W_l]
            _, _, H_l, W_l = feat_map.shape
            stride = self.strides[l_idx]
            sigma = self.sigma[l_idx]

            # Anchor position in this level's feature space
            anchor_feat = anchor_px / stride         # [B, K, R, 2]

            # Bounded offsets: tanh * sigma
            level_off = raw_offsets[:, :, :, :, l_idx, :, :].tanh() * sigma  # [B,K,R,H,M,2]

            for h_idx in range(H):
                head_off = level_off[:, :, :, h_idx]               # [B, K, R, M, 2]
                pts = anchor_feat.unsqueeze(3) + head_off           # [B, K, R, M, 2]

                # Normalize to [-1, 1] for grid_sample
                grid = torch.stack([
                    2 * pts[..., 0] / (W_l - 1) - 1,
                    2 * pts[..., 1] / (H_l - 1) - 1,
                ], dim=-1)                                          # [B, K, R, M, 2]
                grid = grid.view(B, K * R * M, 1, 2)

                # Sample this head's d_h channel slice
                feat_head = feat_map[:, h_idx * d_h:(h_idx + 1) * d_h]  # [B, d_h, H_l, W_l]
                sampled = F.grid_sample(feat_head, grid, align_corners=True, mode='bilinear')
                sampled = sampled.squeeze(-1).permute(0, 2, 1)     # [B, K*R*M, d_h]
                sampled = sampled.view(B, K, R, M, d_h)

                # Weighted accumulation over M points at this level
                w = weights[:, :, :, h_idx, l_idx]                 # [B, K, R, M]
                h_r[:, :, :, h_idx] += (sampled * w.unsqueeze(-1)).sum(dim=3)

        # ── 6. Concat heads + output projection + type embedding ──
        h_r = h_r.view(B, K, R, d)                # concat H heads of d_h each
        deform_tokens = self.w_o(h_r) + self.e_deform   # [B, K, R, d]

        return deform_tokens


if __name__ == "__main__":
    features = {
        'L2_proj': torch.randn(2, 192, 32, 40),
        'L3_proj': torch.randn(2, 192, 16, 20),
        'L4_proj': torch.randn(2, 192, 8, 10),
        'g': torch.randn(2, 80, 192),
    }
    h = torch.randn(2, 16, 192)
    top_indices = torch.randint(0, 80, (2, 16, 32))
    query_coords = torch.randint(0, 256, (2, 16, 2)).float()

    model = DeformableRead()
    tokens = model(h, top_indices, query_coords, features)
    print(f"deform_tokens: {tokens.shape}")    # expect [2, 16, 32, 192]
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e3:.1f}K")
