import torch
import torch.nn as nn
import torch.nn.functional as F

from models.decoder.rcu import RCU

class LocalDPT(nn.Module):
    
    def __init__(self, d_model: int = 64, mid_channels: int = 32):
        super().__init__()

        self.d_model = d_model

        self.s8_rcu = RCU(d_model=d_model)
        self.l2_rcu = RCU(d_model=d_model)
        self.rn2_out = nn.Conv2d(d_model, d_model, kernel_size=1)

        self.s4_rcu = RCU(d_model=d_model)
        self.l1_rcu = RCU(d_model=d_model)
        self.rn1_out = nn.Conv2d(d_model, d_model, kernel_size=1)

        self.output_conv1 = nn.Conv2d(d_model, mid_channels, kernel_size=3, padding=1)
        self.output_conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 1, kernel_size=1)
        )


    def forward_train(self, s8: torch.Tensor, features: dict[str, torch.Tensor]) -> torch.Tensor:

        s8 = self.s8_rcu(s8)
        l2 = self.l2_rcu(features['L2'])
        rn2 = s8 + l2
        rn2 = F.interpolate(rn2, scale_factor=2, mode='bilinear', align_corners=True)
        rn2 = self.rn2_out(rn2)

        s4 = self.s4_rcu(rn2)
        l1 = self.l1_rcu(features['L1'])
        rn1 = s4 + l1
        rn1 = F.interpolate(rn1, scale_factor=2, mode='bilinear', align_corners=True)
        rn1 = self.rn1_out(rn1)

        out = self.output_conv1(rn1)
        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)
        out = self.output_conv2(out)
        depth = torch.exp(out)

        return depth
    

    def recenter_crop(self, x: torch.Tensor, offset_x: int, offset_y: int, crop: int) -> torch.Tensor:
        half = crop // 2
        cy = x.shape[2] // 2 - 1 - half
        cx = x.shape[3] // 2 - 1 - half
        crop_y0 = x[:, :, cy:cy + crop, :]
        crop_y1 = x[:, :, cy + 1:cy + 1 + crop, :]
        out = torch.where(offset_y.view(-1, 1, 1, 1).bool(), crop_y1, crop_y0)
        crop_x0 = out[:, :, :, cx:cx + crop]
        crop_x1 = out[:, :, :, cx + 1:cx + 1 + crop]
        out = torch.where(offset_x.view(-1, 1, 1, 1).bool(), crop_x1, crop_x0)
        return out
    

    def forward_infer(self, s8: torch.Tensor, features: dict[str, torch.Tensor], coords: torch.Tensor) -> torch.Tensor:

        B, K, _ = coords.shape
        
        cx8 = coords[..., 0] / 8.0
        cy8 = coords[..., 1] / 8.0

        H8, W8 = s8.shape[2], s8.shape[3]
        H4, W4 = features['L1'].shape[2], features['L1'].shape[3]

        cx4 = coords[..., 0] / 4.0
        cy4 = coords[..., 1] / 4.0

        N_s8 = 9
        N_s4 = 7

        half_s8 = N_s8 // 2
        half_s4 = N_s4 // 2

        # Stride 8 sampling
        dx_s8 = torch.arange(-half_s8, half_s8 + 1, device=s8.device, dtype=torch.float32)
        dy_s8 = torch.arange(-half_s8, half_s8 + 1, device=s8.device, dtype=torch.float32)

        gy_s8, gx_s8 = torch.meshgrid(dy_s8, dx_s8, indexing='ij')

        sx_8 = gx_s8 + cx8[:, :, None, None]
        sy_8 = gy_s8 + cy8[:, :, None, None]

        grid_s8 = torch.stack([
            2.0 * sx_8 / (W8 - 1) - 1.0,
            2.0 * sy_8 / (H8 - 1) - 1.0
        ], dim=-1).reshape(B, K * N_s8, N_s8, 2)

        sampled_s8 = F.grid_sample(s8, grid_s8, mode='bilinear', padding_mode='zeros', align_corners=True)  # [B, C, K*N_s8, N_s8]

        sampled_s8 = sampled_s8.view(B, self.d_model, K, N_s8, N_s8).permute(0, 2, 1, 3, 4).reshape(B*K, self.d_model, N_s8, N_s8)  # [B*K, 64, 9, 9]

        sampled_l2 = F.grid_sample(features['L2'], grid_s8, mode='bilinear', padding_mode='zeros', align_corners=True)
        sampled_l2 = sampled_l2.view(B, self.d_model, K, N_s8, N_s8).permute(0, 2, 1, 3, 4).reshape(B*K, self.d_model, N_s8, N_s8)  # [B*K, 64, 9, 9]

        sampled_s8 = self.s8_rcu.forward_infer(sampled_s8)
        sampled_l2 = self.l2_rcu.forward_infer(sampled_l2)
        s4 = sampled_s8 + sampled_l2
        s4 = F.interpolate(s4, scale_factor=2, mode='bilinear', align_corners=True) # 10*10

        off_x_s8 = ((coords[..., 0] % 8) >= 4).long()
        off_y_s8 = ((coords[..., 1] % 8) >= 4).long()
        s4 = self.recenter_crop(s4, offset_x=off_x_s8, offset_y=off_y_s8, crop=N_s4)
        s4 = self.rn2_out(s4)


        # Stride 4 sampling
        dx_s4 = torch.arange(-half_s4, half_s4 + 1, device=s8.device, dtype=torch.float32)
        dy_s4 = torch.arange(-half_s4, half_s4 + 1, device=s8.device, dtype=torch.float32)

        gy_s4, gx_s4 = torch.meshgrid(dy_s4, dx_s4, indexing='ij')

        sx_4 = gx_s4 + cx4[:, :, None, None]
        sy_4 = gy_s4 + cy4[:, :, None, None]

        grid_s4 = torch.stack([
            2.0 * sx_4 / (W4 - 1) - 1.0,
            2.0 * sy_4 / (H4 - 1) - 1.0
        ], dim=-1).reshape(B, K * N_s4, N_s4, 2)

        sampled_l1 = F.grid_sample(features['L1'], grid_s4, mode='bilinear', padding_mode='zeros', align_corners=True)
        sampled_l1 = sampled_l1.view(B, self.d_model, K, N_s4, N_s4).permute(0, 2, 1, 3, 4).reshape(B*K, self.d_model, N_s4, N_s4)  # [B*K, 64, 7, 7]

        sampled_s4 = self.s4_rcu.forward_infer(s4)
        sampled_l1 = self.l1_rcu.forward_infer(sampled_l1)
        s2 = sampled_s4 + sampled_l1
        s2 = F.interpolate(s2, scale_factor=2, mode='bilinear', align_corners=True)

        off_x_s4 = ((coords[..., 0] % 4) >= 2).long()
        off_y_s4 = ((coords[..., 1] % 4) >= 2).long()
        s2 = self.recenter_crop(s2, offset_x=off_x_s4, offset_y=off_y_s4, crop=5)
        s2 = self.rn1_out(s2)

        s2 = F.conv2d(s2, self.output_conv1.weight, self.output_conv1.bias, padding=0)
        s1 = F.interpolate(s2, scale_factor=2, mode='bilinear', align_corners=True)

        off_x_s2 = ((coords[..., 0] % 2) >= 1).long()
        off_y_s2 = ((coords[..., 1] % 2) >= 1).long()
        s1 = self.recenter_crop(s1, offset_x=off_x_s2, offset_y=off_y_s2, crop=3)

        out = F.conv2d(s1, self.output_conv2[0].weight, self.output_conv2[0].bias, padding=0)
        out = F.relu(out)
        out = F.conv2d(out, self.output_conv2[2].weight, self.output_conv2[2].bias, padding=0)

        depth = torch.exp(out.reshape(B, K))

        return depth
