import torch
import torch.nn as nn
import torch.nn.functional as F

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


    def forward(self, h: torch.Tensor, K, V, ret_attn: bool) -> torch.Tensor:

        B, n_q, D = h.shape
        N = K.shape[1]

        residual = h
        h = self.ln_q(h)
        q = self.wq(h).unsqueeze(2).view(B, n_q, 1, self.n_head, self.d_head).transpose(2, 3)
        k = K.unsqueeze(1).view(B, 1, N, self.n_head, self.d_head).transpose(2, 3)
        v = V.unsqueeze(1).view(B, 1, N, self.n_head, self.d_head).transpose(2, 3)

        if ret_attn:
            scale = self.d_head ** -0.5
            attn_weights = F.softmax(q @ k.transpose(-2, -1) * scale, dim=-1)
            attn_out = attn_weights @ v
            attn_out = attn_out.squeeze(3).view(B, n_q, D)
            attn_out = self.wo(attn_out)

            h = residual + attn_out  # Residual connection
            h = h + self.ffn(self.ln_ff(h))  # FFN with residual
            return h, attn_weights
        
        else:
            attn_out = F.scaled_dot_product_attention(q, k, v)
            attn_out = attn_out.squeeze(3).view(B, n_q, D)
            attn_out = self.wo(attn_out)

            h = residual + attn_out  # Residual connection
            h = h + self.ffn(self.ln_ff(h))  # FFN with residual
            return h, None

        
    
    
class GlobalCrossAttn(nn.Module):

    def __init__(self, d_model: int = 192, n_head: int = 6, num_layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList([
            CrossAttnLayer(d_model, n_head) for _ in range(num_layers)
        ])

    def forward(self, h: torch.Tensor, precomputed: dict) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            if i == len(self.layers) - 1:  # Last layer, return attn weights for routing
                h, attn_weight = layer(h, precomputed[f'K_{i}'], precomputed[f'V_{i}'], ret_attn=True)
            else:
                h, _ = layer(h, precomputed[f'K_{i}'], precomputed[f'V_{i}'], ret_attn=False)

        avg_attn = attn_weight.squeeze(3).mean(dim=2) 
        _, top_indices = avg_attn.topk(32, dim=-1)
        return h, top_indices
    

if __name__ == "__main__":
    precomputed = {
        'K_0': torch.randn(2, 300, 192),
        'V_0': torch.randn(2, 300, 192),
        'K_1': torch.randn(2, 300, 192),
        'V_1': torch.randn(2, 300, 192),
    }
    h = torch.randn(2, 16, 192)  # B=2, K=16

    model = GlobalCrossAttn()
    h_out, top_idx = model(h, precomputed)
    print(f"h: {h_out.shape}")          # expect [2, 16, 192]
    print(f"top_idx: {top_idx.shape}")  # expect [2, 16, 32]