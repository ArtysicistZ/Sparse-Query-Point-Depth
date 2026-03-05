import torch
import torch.nn as nn
import torch.nn.functional as F

from models.decoder.rcu import RCU
from config import H_IMG, W_IMG

class LocalAttn(nn.Module):
    
    def __init__(self, d_model: int = 64, mid_channels: int = 128, layers: int = 2):
        super().__init__()

        self.d_model = d_model

        self.s8_rcu = RCU(d_model=d_model)
        self.l2_rcu = RCU(d_model=d_model)
        self.l1_rcu = RCU(d_model=d_model)
        
        self.emb = nn.Embedding(3, d_model)
        self.pos = nn.Parameter(torch.randn(25, d_model))
        
        self.s8_attn = CrossAttn(d_model=d_model)
        self.l2_attn = CrossAttn(d_model=d_model)
        self.l1_attn = CrossAttn(d_model=d_model)

        self.output = nn.Sequential(
            nn.Linear(d_model, mid_channels),
            nn.ReLU(),
            nn.Linear(mid_channels, 1),
            nn.Softplus()
        )


    def forward_train(self, s8: torch.Tensor, features: dict[str, torch.Tensor]) -> torch.Tensor:
        
        s8 = self.s8_rcu(s8)
        l2 = self.l2_rcu(features['L2'])
        l1 = self.l1_rcu(features['L1'])

        B, D, H8, W8 = s8.shape
        _, _, H4, W4 = l1.shape

        x = s8.permute(0, 2, 3, 1).reshape(B*H8*W8, -1, D)

        kv_s8 = F.unfold(s8, kernel_size = 5, padding = 2)
        kv_s8 = kv_s8.view(B, D, 25, -1)
        kv_s8 = kv_s8.permute(0, 3, 2, 1)
        kv_s8 = kv_s8.reshape(B*H8*W8, -1, D)
        kv_s8 = kv_s8 + self.pos.unsqueeze(0) + self.emb(torch.tensor(0, device=kv_s8.device)).reshape(1, 1, D)

        kv_l2 = F.unfold(l2, kernel_size = 5, padding = 2)
        kv_l2 = kv_l2.view(B, D, 25, -1)
        kv_l2 = kv_l2.permute(0, 3, 2, 1)
        kv_l2 = kv_l2.reshape(B*H8*W8, -1, D)
        kv_l2 = kv_l2 + self.pos.unsqueeze(0) + self.emb(torch.tensor(1, device=kv_l2.device)).reshape(1, 1, D)

        x = self.s8_attn(x, kv_s8)
        x = self.l2_attn(x, kv_l2)

        x = x.squeeze(1).reshape(B, H8, W8, D).permute(0, 3, 1, 2)
        x = F.interpolate(x, scale_factor = 2, mode = 'bilinear', align_corners = True) # [B, D, H4, W4]
        x = x.permute(0, 2, 3, 1).reshape(B*H4*W4, -1, D)

        kv_l1 = F.unfold(l1, kernel_size = 5, padding = 2)
        kv_l1 = kv_l1.view(B, D, 25, -1)
        kv_l1 = kv_l1.permute(0, 3, 2, 1)
        kv_l1 = kv_l1.reshape(B*H4*W4, -1, D)
        kv_l1 = kv_l1 + self.pos.unsqueeze(0) + self.emb(torch.tensor(2, device=kv_l1.device)).reshape(1, 1, D)

        x = self.l1_attn(x, kv_l1)

        x = self.output(x).reshape(B, 1, H4, W4)
        x = F.interpolate(x, size=(H_IMG, W_IMG), mode='bilinear', align_corners=True)
        return x        
    

    def forward_infer(self, s8: torch.Tensor, features: dict[str, torch.Tensor], coords: torch.Tensor) -> torch.Tensor:

        B, K, _ = coords.shape

        stride_8 = W_IMG / s8.shape[3]  
        stride_4 = W_IMG / features['L1'].shape[3]
        
        cx8 = coords[..., 0] / stride_8
        cy8 = coords[..., 1] / stride_8

        cx4 = coords[..., 0] / stride_4
        cy4 = coords[..., 1] / stride_4

        H8, W8 = s8.shape[2], s8.shape[3]
        H4, W4 = features['L1'].shape[2], features['L1'].shape[3]

        N = 9
        half_N = N // 2

        # Stride 8 sampling
        dx_s8 = torch.arange(-half_N, half_N + 1, device=s8.device, dtype=torch.float32)
        dy_s8 = torch.arange(-half_N, half_N + 1, device=s8.device, dtype=torch.float32)

        dx_s4 = torch.arange(-half_N, half_N + 1, device=features['L1'].device, dtype=torch.float32)
        dy_s4 = torch.arange(-half_N, half_N + 1, device=features['L1'].device, dtype=torch.float32)

        gy_s8, gx_s8 = torch.meshgrid(dy_s8, dx_s8, indexing='ij')
        gy_s4, gx_s4 = torch.meshgrid(dy_s4, dx_s4, indexing='ij')

        sx_8 = gx_s8 + cx8[:, :, None, None]
        sy_8 = gy_s8 + cy8[:, :, None, None]

        sx_4 = gx_s4 + cx4[:, :, None, None]
        sy_4 = gy_s4 + cy4[:, :, None, None]

        grid_8 = torch.stack([
            2.0 * sx_8 / (W8 - 1) - 1.0,
            2.0 * sy_8 / (H8 - 1) - 1.0
        ], dim=-1).reshape(B, K * N, N, 2)

        grid_4 = torch.stack([
            2.0 * sx_4 / (W4 - 1) - 1.0,
            2.0 * sy_4 / (H4 - 1) - 1.0
        ], dim=-1).reshape(B, K * N, N, 2)

        sampled_s8 = F.grid_sample(s8, grid_8, mode='bilinear', padding_mode='zeros', align_corners=True)  # [B, C, K*N_s8, N_s8]
        sampled_s8 = sampled_s8.view(B, self.d_model, K, N, N).permute(0, 2, 1, 3, 4).reshape(B*K, self.d_model, N, N)  # [B*K, 64, 9, 9]

        sampled_l2 = F.grid_sample(features['L2'], grid_8, mode='bilinear', padding_mode='zeros', align_corners=True)
        sampled_l2 = sampled_l2.view(B, self.d_model, K, N, N).permute(0, 2, 1, 3, 4).reshape(B*K, self.d_model, N, N)  # [B*K, 64, 9, 9]

        sampled_l1 = F.grid_sample(features['L1'], grid_4, mode='bilinear', padding_mode='zeros', align_corners=True)
        sampled_l1 = sampled_l1.view(B, self.d_model, K, N, N).permute(0, 2, 1, 3, 4).reshape(B*K, self.d_model, N, N)  # [B*K, 64, 9, 9]

        sampled_s8 = self.s8_rcu.forward_infer(sampled_s8)
        sampled_l2 = self.l2_rcu.forward_infer(sampled_l2)
        sampled_l1 = self.l1_rcu.forward_infer(sampled_l1)

        x = sampled_s8[:, :, 2, 2].unsqueeze(1)  # [B*K, 1, D]

        kv_s8 = sampled_s8.flatten(2).permute(0, 2, 1)
        kv_l2 = sampled_l2.flatten(2).permute(0, 2, 1)
        kv_l1 = sampled_l1.flatten(2).permute(0, 2, 1)

        kv_s8 = kv_s8 + self.pos.unsqueeze(0) + self.emb(torch.tensor(0, device=kv_s8.device)).reshape(1, 1, self.d_model)
        kv_l2 = kv_l2 + self.pos.unsqueeze(0) + self.emb(torch.tensor(1, device=kv_l2.device)).reshape(1, 1, self.d_model)
        kv_l1 = kv_l1 + self.pos.unsqueeze(0) + self.emb(torch.tensor(2, device=kv_l1.device)).reshape(1, 1, self.d_model)

        x = self.s8_attn(x, kv_s8)
        x = self.l2_attn(x, kv_l2)
        x = self.l1_attn(x, kv_l1)

        x = self.output(x).reshape(B, K)
        return x

        

class CrossAttn(nn.Module):

    def __init__(self, d_model: int = 64, n_head: int = 4, multiplier: int = 4):
        super().__init__()

        self.cross_attn = nn.MultiheadAttention(
            embed_dim = d_model,
            num_heads = n_head, 
            dropout = 0.1,
            batch_first = True
        )

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * multiplier),
            nn.GELU(),
            nn.Linear(d_model * multiplier, d_model)
        )

    def forward(self, x: torch.Tensor, kv: torch.Tensor):
        '''
        x: query point, [B*K, 1, D]
        kv: cross attention vectors, [B*K, 5*5, D]
        return: x after MHCrossAttn and FFN
        '''
        z = self.ln1(x)
        out, _ = self.cross_attn(z, kv, kv)
        x = x + out
        x = self.ln2(x)
        x = x + self.ffn(x)
        return x

