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
        h = self.ln_q(h)
        q = self.wq(h).unsqueeze(2).view(B, K, 1, self.n_head, self.d_head).transpose(2, 3)
        k = self.wk(tokens).view(B, K, N, self.n_head, self.d_head).transpose(2, 3)
        v = self.wv(tokens).view(B, K, N, self.n_head, self.d_head).transpose(2, 3)

        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.squeeze(3).view(B, K, D)
        attn_out = self.wo(attn_out)

        h = residual + attn_out  # Residual connection
        h = h + self.ffn(self.ln_ff(h))  # FFN with residual
        return h


class LocalCrossAttn(nn.Module):

    def __init__(self, d_model: int = 192, n_head: int = 6, num_layers: int = 2):
        super().__init__()
        self.ln_kv = nn.LayerNorm(d_model)
        self.layers = nn.ModuleList([
            CrossAttnLayer(d_model, n_head) for _ in range(num_layers)
        ])

    def forward(self, h: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        tokens = self.ln_kv(tokens)
        for layer in self.layers:
            h = layer(h, tokens)
        return h
    

if __name__ == "__main__":
    tokens = torch.randn(2, 16, 91, 192)  # B=2, K=16, 91 tokens
    seed = torch.randn(2, 16, 192)         # query seed

    model = LocalCrossAttn()
    h = model(seed, tokens)
    print(f"h: {h.shape}")  # expect [2, 16, 192]