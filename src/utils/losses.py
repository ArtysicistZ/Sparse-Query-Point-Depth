import torch

eps = 1e-8

def l_silog(pred_depth, gt_depth, lam_var=0.50):
    d = torch.log(pred_depth + eps) - torch.log(gt_depth + eps)
    per_img = torch.mean(d ** 2, dim=1) - lam_var * (torch.mean(d, dim=1) ** 2) 
    loss = torch.sqrt(per_img.mean() + eps)
    return loss


def l_dense_silog(pred_depth, gt_depth, lam_var=0.50):
    mask = (gt_depth > 0).squeeze(1)  # [B, H, W]
    d = torch.log(pred_depth.squeeze(1) + eps) - torch.log(gt_depth.squeeze(1) + eps)  # [B, 1, H, W]
    d = d * mask  # Zero out invalid pixels

    n_valid = mask.sum(dim=[1, 2]).clamp(min=1)  # [B]
    mean_d2 = (d ** 2).sum(dim=[1, 2]) / n_valid  # [B]
    mean_d = d.sum(dim=[1, 2]) / n_valid  # [B]

    per_img = mean_d2 - lam_var * (mean_d ** 2)  # [B]

    return torch.sqrt(per_img.mean() + eps)