import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoder.convnext import ConvNeXtV2Encoder

class ProjectionNeck(nn.Module):

    def __init__(self, 
                 enc_channels: list[int] = [96, 192, 384, 768], 
                 dec_channels: list[int] = [192, 192, 192, 192]):
        
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Conv2d(in_ch, out_ch, kernel_size=1) 
            for in_ch, out_ch in zip(enc_channels, dec_channels)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(out_ch) for out_ch in dec_channels
        ])
        self.l4_self_attn = SelfAttention(d_model=dec_channels[3], num_layers=2)


    def _proj(self, x: torch.Tensor, conv, ln) -> torch.Tensor:
        B, C, H, W = x.shape
        x = conv(x)                # [B, out_ch, H, W]
        x = x.permute(0, 2, 3, 1)  # [B, H, W, out_ch]
        x = ln(x)
        x = x.permute(0, 3, 1, 2)  # [B, out_ch, H, W]
        return x
    
    
    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        keys = list(features.keys())
        projects = {}
        for k, conv, ln in zip(keys, self.projections, self.norms):
            projects[k] = self._proj(features[k], conv, ln)
        projects['L4'] = self.l4_self_attn(projects['L4'])
        return projects
    

class SelfAttentionLayer(nn.Module):
    """One layer of self-attention for the L4 feature map."""

    def __init__(self, d_model: int = 192, n_head: int = 6, ffn_ratio: int = 4):
        super().__init__()
        self.n_head = n_head
        self.d_head = d_model // n_head

        self.norm1 = nn.LayerNorm(d_model)

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)

        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_ratio * d_model),
            nn.GELU(),
            nn.Linear(ffn_ratio * d_model, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape

        h = self.norm1(x)
        q = self.wq(h)
        k = self.wk(h)
        v = self.wv(h) 
        
        # [B, n_head, N, d_head]
        q = q.view(B, N, self.n_head, self.d_head).transpose(1, 2)  
        k = k.view(B, N, self.n_head, self.d_head).transpose(1, 2)  
        v = v.view(B, N, self.n_head, self.d_head).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, 
            dropout_p=0.1 if self.training else 0.0
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, N, D)

        x = x + self.wo(attn_out)  # Residual connection
        x = x + self.ffn(self.norm2(x))  # FFN with residual

        return x
    

class SelfAttention(nn.Module):

    def __init__(self, d_model: int = 192, num_layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList([
            SelfAttentionLayer(d_model) for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 192, H, W] → [B, 192, H, W]"""
        B, C, H, W = x.shape
        x = x.reshape(B, C, -1).transpose(1, 2)

        for layer in self.layers:
            x = layer(x)
        x = x.transpose(1, 2).reshape(B, C, H, W) 
        return x
    

    
if __name__ == "__main__":
    from models.encoder.convnext import ConvNeXtV2Encoder
    
    encoder = ConvNeXtV2Encoder(pretrained=True)
    neck = ProjectionNeck()
    
    x = torch.randn(1, 3, 480, 640)
    enc_out = encoder(x)
    dec_out = neck(enc_out)
    
    for name, feat in dec_out.items():
        print(f"{name}: {feat.shape}")
