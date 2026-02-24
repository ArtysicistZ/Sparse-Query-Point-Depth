import torch.nn.functional as F

def l_point(inv_depth, gt_depth):
    inv_gt = 1.0 / (gt_depth + 1e-8)
    return F.smooth_l1_loss(inv_depth, inv_gt, reduction='mean')