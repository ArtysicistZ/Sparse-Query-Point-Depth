import torch
import torch.nn as nn

class PreCompute(nn.Module):

    def __init__(self, d_model: int = 192, l4_channels: int = 384,
                 l3_channels: int = 192, l2_channels: int = 128, 
                 n_layers: int = 2):
        super().__init__()

        self.n_layers = n_layers
        self.wk = nn.ModuleList([
            nn.Linear(l4_channels, d_model) for _ in range(n_layers)
        ])

        self.wv = nn.ModuleList([
            nn.Linear(l4_channels, d_model) for _ in range(n_layers)
        ])

        self.proj_l4 = nn.Conv2d(l4_channels, d_model, kernel_size=1)
        self.proj_l3 = nn.Conv2d(l3_channels, d_model, kernel_size=1)
        self.proj_l2 = nn.Conv2d(l2_channels, d_model, kernel_size=1) 

        self.wg = nn.Linear(l4_channels, d_model)

        '''
        self.calib_l4 = nn.Linear(l4_channels, 1)
        self.calib_bias = nn.Linear(l4_channels, 1)
        '''
        

    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out = features.copy()
        B, C, H, W = out['L4'].shape

        L4_flat = out['L4'].flatten(2).transpose(1, 2)  # [B, H*W, C]

        for i in range(self.n_layers):
            out[f'K_{i}'] = self.wk[i](L4_flat)
            out[f'V_{i}'] = self.wv[i](L4_flat)

        out['L2_proj'] = self.proj_l2(out['L2'])
        out['L3_proj'] = self.proj_l3(out['L3'])
        out['L4_proj'] = self.proj_l4(out['L4'])

        out['g'] = self.wg(L4_flat)  # [B, H*W, d_model]

        '''
        L4_pooled = out['L4'].mean(dim=[2, 3])  # [B, C]
        out['s'] = self.calib_l4(L4_pooled)  # [B, 1]
        out['b'] = self.calib_bias(L4_pooled)  # [B, 1]
        '''

        return out
    


if __name__ == "__main__":
    features = {
        'L1': torch.randn(1, 64, 120, 160),
        'L2': torch.randn(1, 128, 60, 80),
        'L3': torch.randn(1, 192, 30, 40),
        'L4': torch.randn(1, 384, 15, 20),
    }
    model = PreCompute()
    out = model(features)
    for name, feat in out.items():
        print(f"{name}: {feat.shape}")