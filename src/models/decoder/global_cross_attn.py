import torch
import torch.nn as nn
import torch.nn.functional as F

from spatial_canvas import CanvasLayer, CanvasSmooth

class CrossAttnLayer(nn.Module):

    def __init__(self, d_model: int = 192, n_head: int = 6):
        super().__init__()
        self.n_head = n_head
        self.d_head = d_model // n_head

        self.ln_q = nn.LayerNorm(d_model)

        self.wq = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)

        self.ln_ff = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )

        self.canvas_layer = CanvasLayer(d_model=d_model)
        self.canvas_smooth = CanvasSmooth(d_model=d_model)


    def forward(self, h: torch.Tensor, canvas_L2: torch.Tensor, canvas_L3: torch.Tensor, K, V, center_grid: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:

        B, n_q, D = h.shape
        N = K.shape[1]

        residual = h
        h = self.ln_q(h)
        q = self.wq(h).unsqueeze(2).view(B, n_q, 1, self.n_head, self.d_head).transpose(2, 3)
        k = K.unsqueeze(1).view(B, 1, N, self.n_head, self.d_head).transpose(2, 3)
        v = V.unsqueeze(1).view(B, 1, N, self.n_head, self.d_head).transpose(2, 3)

        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.squeeze(3).view(B, n_q, D)
        attn_out = self.wo(attn_out)

        h = residual + attn_out  # Residual connection
        h = h + self.ffn(self.ln_ff(h))  # FFN with residual

        # canvas
        h, canvas_L2, canvas_L3 = self.canvas_layer(h, canvas_L2, canvas_L3, center_grid, coords)
        canvas_L2, canvas_L3 = self.canvas_smooth(canvas_L2, canvas_L3)
        return h, canvas_L2, canvas_L3

    

class GlobalCrossAttn(nn.Module):

    def __init__(self, d_model: int = 192, n_head: int = 6, num_layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList([
            CrossAttnLayer(d_model, n_head) for _ in range(num_layers)
        ])

    def forward(self, h: torch.Tensor, canvas_L2: torch.Tensor, canvas_L3: torch.Tensor, precomputed: dict, lev: int, center_grid: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        
        for i, layer in enumerate(self.layers):
            h, canvas_L2, canvas_L3 = layer(h, canvas_L2, canvas_L3, precomputed[f'K_{i}_l{lev}'], precomputed[f'V_{i}_l{lev}'], center_grid, coords)
        return h, canvas_L2, canvas_L3
