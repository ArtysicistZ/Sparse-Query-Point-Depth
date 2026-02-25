import torch
import torch.nn as nn
import torch.nn.functional as F


class FusedCrossAttnLayer(nn.Module):

    def __init__(self, d_model: int = 192, n_head: int = 6):
        super().__init__()
        self.n_head = n_head
        self.d_head = d_model // n_head

        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)

        self.ln_ff = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )

    def forward(self, h: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        B, K, D = h.shape
        N = tokens.shape[2]

        residual = h
        q = self.wq(self.ln_q(h)).unsqueeze(2).view(B, K, 1, self.n_head, self.d_head).transpose(2, 3)
        kv_in = self.ln_kv(tokens)
        k = self.wk(kv_in).view(B, K, N, self.n_head, self.d_head).transpose(2, 3)
        v = self.wv(kv_in).view(B, K, N, self.n_head, self.d_head).transpose(2, 3)

        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.squeeze(3).view(B, K, D)
        h = residual + self.wo(attn_out)
        h = h + self.ffn(self.ln_ff(h))
        return h


class FusedDecoder(nn.Module):
    """B5: 3-layer fused cross-attention over 123 tokens + depth head.

    Params: ~1,430K.
    """

    def __init__(self, d_model: int = 192, n_head: int = 6, num_layers: int = 3):
        super().__init__()
        self.e_g3 = nn.Parameter(torch.zeros(1, 1, 1, d_model))
        self.e_g4 = nn.Parameter(torch.zeros(1, 1, 1, d_model))
        self.layers = nn.ModuleList([
            FusedCrossAttnLayer(d_model, n_head) for _ in range(num_layers)
        ])
        self.depth_head = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, 1)
        )
        nn.init.constant_(self.depth_head[-1].bias, 0.916) 


    def forward(self, h: torch.Tensor, center_tokens: torch.Tensor, top_indices_l4: torch.Tensor, top_indices_l3: torch.Tensor, features: dict) -> torch.Tensor:
        """
        h: [B, K, d] — B3 output
        fused_tokens: [B, K, 123, d] — 91 local + 32 deformable
        Returns: r_q [B, K] — raw depth code (before calibration)
        """
        
        _, K, d = h.shape

        g_l4 = features['L4_g'].unsqueeze(1).expand(-1, K, -1, -1) # [B, K, H*W, d]
        g_l3 = features['L3_g'].unsqueeze(1).expand(-1, K, -1, -1) # [B, K, H*W, d]

        tokens_l4 = g_l4.gather(dim=2, index=top_indices_l4.unsqueeze(-1).expand(-1, -1, -1, d)) + self.e_g4  # [B, K, 10, d]
        tokens_l3 = g_l3.gather(dim=2, index=top_indices_l3.unsqueeze(-1).expand(-1, -1, -1, d)) + self.e_g3  # [B, K, 20, d]

        fused_tokens = torch.cat([center_tokens, tokens_l3, tokens_l4], dim=2)  # [B, K, 33, d]

        for layer in self.layers:
            h = layer(h, fused_tokens)
        return self.depth_head(h).squeeze(-1)  # [B, K]
