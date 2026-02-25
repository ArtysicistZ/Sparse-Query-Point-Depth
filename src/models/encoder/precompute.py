import torch
import torch.nn as nn

class PreCompute(nn.Module):

    def __init__(self, d_model: int = 192, l4_channels: int = 192,
                 l3_channels: int = 192, l2_channels: int = 192, 
                 n_layers: int = 2):
        super().__init__()

        self.n_layers = n_layers
        self.wk_l4 = nn.ModuleList([
            nn.Linear(l4_channels, d_model) for _ in range(n_layers)
        ])

        self.wv_l4 = nn.ModuleList([
            nn.Linear(l4_channels, d_model) for _ in range(n_layers)
        ])

        '''
        self.calib_l4 = nn.Linear(l4_channels, 1)
        self.calib_bias = nn.Linear(l4_channels, 1)
        '''
        

    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        B, C, H, W = features['L4'].shape

        L4_flat = features['L4'].flatten(2).transpose(1, 2)  # [B, H*W, C]

        for i in range(self.n_layers):
            features[f'K_{i}_l4'] = self.wk_l4[i](L4_flat)
            features[f'V_{i}_l4'] = self.wv_l4[i](L4_flat)

        return features



if __name__ == "__main__":
    features = {
        'L1': torch.randn(1, 192, 120, 160),
        'L2': torch.randn(1, 192, 60, 80),
        'L3': torch.randn(1, 192, 30, 40),
        'L4': torch.randn(1, 192, 15, 20),
    }
    model = PreCompute()
    out = model(features)
    for name, feat in out.items():
        print(f"{name}: {feat.shape}")