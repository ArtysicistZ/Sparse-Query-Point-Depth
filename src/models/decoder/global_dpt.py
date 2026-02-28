import torch
import torch.nn as nn
import torch.nn.functional as F

from models.decoder.rcu import RCU

class GlobalDPT(nn.Module):
    
    def __init__(self, d_model: int = 64):
        super().__init__()
        self.rcu_L4 = RCU(d_model=d_model)
        self.rcu_L4_out = nn.Conv2d(d_model, d_model, kernel_size=1)
        self.rcu_L3 = RCU(d_model=d_model)
        self.rcu_L3_merge = RCU(d_model=d_model)
        self.rcu_L3_out = nn.Conv2d(d_model, d_model, kernel_size=1)

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:

        r4 = self.rcu_L4(features['L4'])
        r4 = F.interpolate(r4, size=features['L3'].shape[2:], mode='bilinear', align_corners=True)
        r4 = self.rcu_L4_out(r4)

        r3 = self.rcu_L3(features['L3'])
        r3 = r3 + r4
        r3 = self.rcu_L3_merge(r3)
        r3 = F.interpolate(r3, size=features['L2'].shape[2:], mode='bilinear', align_corners=True)
        r3 = self.rcu_L3_out(r3)

        return r3