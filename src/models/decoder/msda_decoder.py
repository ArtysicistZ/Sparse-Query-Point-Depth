import torch
import torch.nn as nn
import torch.nn.functional as F

from models.decoder.spatial_canvas import CanvasLayer

class MSDADecoder(nn.Module):
    
    def __init__(self, d_model: int = 192, n_layers: int = 3):
        super().__init__()
        self.layers = nn.ModuleList([
            MSDADecoderLayer(d_model=d_model) for _ in range(n_layers)
        ])


    def forward(self, h: torch.Tensor, pos_q: torch.Tensor, features: dict[str, torch.Tensor], canvas_L2: torch.Tensor, canvas_L3: torch.Tensor, center_grid: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            h, canvas_L2, canvas_L3 = layer(h, pos_q, features, canvas_L2, canvas_L3, center_grid, coords)
        return h, canvas_L2, canvas_L3  # [B, K, d_model], [B, d_model, H_l2, W_l2], [B, d_model, H_l3, W_l3]



class MSDADecoderLayer(nn.Module):

    def __init__(self, d_model: int = 192):
        super().__init__()
        self.msda = MSDABlock(d_model=d_model)
        self.canvas_layer = CanvasLayer(d_model=d_model)
        self.q2q = Q2QBlock(d_model=d_model)

    def forward(self, h: torch.Tensor, pos_q: torch.Tensor, features: dict[str, torch.Tensor], canvas_L2: torch.Tensor, canvas_L3: torch.Tensor, center_grid: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:

        h = self.msda(h, features, center_grid)
        h, canvas_L2, canvas_L3 = self.canvas_layer(h, canvas_L2, canvas_L3, center_grid, coords)
        h = self.q2q(h, pos_q)
        return h, canvas_L2, canvas_L3


class MSDABlock(nn.Module):

    def __init__(self, d_model: int = 192, n_head: int = 6, n_levels: int = 4, n_points: int = 4):
        super().__init__()
        
        self.n_head = n_head
        self.n_levels = n_levels
        self.n_points = n_points

        self.ln = nn.LayerNorm(d_model)
        self.W_off = nn.Linear(d_model, n_head * n_levels * n_points * 2)
        self.W_attn = nn.Linear(d_model, n_head * n_levels * n_points)
        self.W_o = nn.Linear(d_model, d_model)


    def forward(self, h: torch.Tensor, features: dict[str, torch.Tensor], center_grid: torch.Tensor) -> torch.Tensor:
        """
        h: [B, K, d_model]
        features: dict from PreCompute (L1, L2, L3, L4, d_model = 192)
        """
        B, K, D = h.shape
        h_norm = self.ln(h)

        offsets = self.W_off(h_norm).view(B, K, self.n_head, self.n_levels, self.n_points, 2)  # [B, K, 6, 4, 4, 2]

        attn_weights = self.W_attn(h_norm).view(B, K, self.n_head, self.n_levels * self.n_points)  # [B, K, 6, 4 * 4]

        attn_weights = F.softmax(attn_weights, dim=-1).view(B, K, self.n_head, self.n_levels, self.n_points)  # [B, K, 6, 4, 4]

        # Deformable attention sampling and aggregation
        sampled_feats = []
        for level in range(self.n_levels):

            feat = features[f'L{level+1}']  # [B, C, H_l, W_l]

            # Compute sampling locations
            sampling_pos = center_grid.unsqueeze(2) + offsets[:, :, :, level, :, :].view(B, K, self.n_head, self.n_points, 2)  # [B, K, 6, 4, 2]
            sampling_pos = sampling_pos.view(B, K * self.n_head * self.n_points, 2).unsqueeze(2)  # [B, K*n_head*n_points, 1, 2]

            # Sample features using grid_sample
            feat_sampled = F.grid_sample(feat, sampling_pos, align_corners=True)  # [B, C, 1, K*n_head*n_points]
            feat_sampled = feat_sampled.squeeze(2).transpose(1, 2)  # [B, K*n_head*n_points, C]

            feat_sampled = feat_sampled.view(B, K, self.n_head, self.n_points, D)  # [B, K, n_head, n_points, D]

            d_head = D // self.n_head
            head_idx = torch.arange(D, device=h.device).view(self.n_head, d_head)  # [4, 32]

            idx = head_idx.view(1, 1, self.n_head, 1, d_head).expand(B, K, -1, self.n_points, -1)  # [B, K, 6, 4, 32]

            feat_sampled = feat_sampled.gather(-1, idx)  # [B, K, n_head, n_points, d_head]

            w = attn_weights[:, :, :, level, :]

            weighted = feat_sampled * w.unsqueeze(-1)  # [B, K, n_head, n_points, d_head]
            weighted_sum = weighted.sum(dim=3)  # [B, K, n_head, d_head]
            sampled_feats.append(weighted_sum)

        output = torch.stack(sampled_feats, dim=0).sum(dim=0)  # [B, K, n_head, d_head]
        output = output.reshape(B, K, D)  # [B, K, d_model]
        h = h + self.W_o(output)  # [B, K, d_model]
        return h
    


class Q2QBlock(nn.Module):

    def __init__(self, d_model: int = 192, n_head: int = 6, ffn_ratio: int = 4):
        super().__init__()

        self.ln1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model, n_head, 
            batch_first=True,
            dropout=0.1
        )

        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_ratio * d_model),
            nn.GELU(),
            nn.Linear(ffn_ratio * d_model, d_model)
        )


    def forward(self, h: torch.Tensor, pos_q: torch.Tensor) -> torch.Tensor:
        h_norm = self.ln1(h)
        q = h_norm + pos_q
        k = h_norm + pos_q
        v = h_norm
        attn_out, _ = self.self_attn(q, k, v)  # [B, K, d_model]
        h = h + attn_out

        h_norm = self.ln2(h)
        ffn_out = self.ffn(h_norm)  # [B, K, d_model]
        return h + ffn_out
    



