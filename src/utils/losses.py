import torch
import torch.nn.functional as F

def l_point(inv_depth, gt_depth):
    inv_gt = 1.0 / (gt_depth + 1e-8)
    return F.smooth_l1_loss(inv_depth, inv_gt, reduction='mean')

def l_silog(pred_depth, gt_depth, lam_var=0.15):
    eps = 1e-8
    d = torch.log(pred_depth + eps) - torch.log(gt_depth + eps)
    per_img = torch.mean(d ** 2, dim=1) - lam_var * (torch.mean(d, dim=1) ** 2) 
    loss = torch.sqrt(per_img.mean() + eps)
    return loss