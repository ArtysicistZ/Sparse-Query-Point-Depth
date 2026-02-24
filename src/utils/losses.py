import torch
import torch.nn.functional as F

def l_point(inv_depth, gt_depth):
    inv_gt = 1.0 / (gt_depth + 1e-8)
    return F.smooth_l1_loss(inv_depth, inv_gt, reduction='mean')

def l_silog(pred_depth, gt_depth, lam_var=0.5):
    d = torch.log(pred_depth + 1e-8) - torch.log(gt_depth + 1e-8)
    return torch.sqrt(torch.mean(d ** 2) - lam_var * (torch.mean(d) ** 2) + 1e-8)