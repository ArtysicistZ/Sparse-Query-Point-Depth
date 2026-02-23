import torch
import torch.nn as nn

class PreCompute(nn.Module):

    def __init__(self, d_model: int = 192, l4_channels: int = 384,
                 l3_channels: int = 192, l2_channels: int = 128, 
                 n_b3_layers: int = 2):
        super().__init__()

        self.wk = nn.ModuleList([
            nn.Linear(d_model, l4_channels) for _ in range(n_b3_layers)
        ])

        self.wv = nn.ModuleList([
            nn.Linear(d_model, l4_channels) for _ in range(n_b3_layers)
        ])