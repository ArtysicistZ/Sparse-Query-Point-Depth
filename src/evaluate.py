import torch
import numpy as np

from torch.utils.data import DataLoader
from data.nyu_dataset import NYUDataset
eps = 1e-8


def _compute_metrics(p, g):
    """Compute core depth metrics from flattened pred/gt tensors."""
    abs_rel = (torch.abs(p - g) / g).mean().item()
    sq_rel = ((p - g) ** 2 / g).mean().item()
    rmse = torch.sqrt(((p - g) ** 2).mean()).item()

    log_diff = torch.log(p) - torch.log(g)
    rmse_log = torch.sqrt((log_diff ** 2).mean()).item()
    silog = torch.sqrt((log_diff ** 2).mean() - 0.5 * (log_diff.mean() ** 2) + eps).item()

    thresh = torch.max(p / g, g / p)
    d1 = (thresh < 1.25).float().mean().item() * 100
    d2 = (thresh < 1.25 ** 2).float().mean().item() * 100
    d3 = (thresh < 1.25 ** 3).float().mean().item() * 100

    optimal_scale = (g / p).median().item()
    p_scaled = p * optimal_scale
    abs_rel_scaled = (torch.abs(p_scaled - g) / g).mean().item()

    return {
        'abs_rel': abs_rel, 'sq_rel': sq_rel, 'rmse': rmse,
        'rmse_log': rmse_log, 'silog': silog,
        'd1': d1, 'd2': d2, 'd3': d3,
        'optimal_scale': optimal_scale, 'abs_rel_scaled': abs_rel_scaled,
    }


def _eval_sparse(model, val_loader, device, verbose):
    """Sparse evaluation with K=128 query points per image."""
    model.eval()
    all_pred, all_gt = [], []

    with torch.no_grad():
        for cnt, (image, coords, gt_depth, _) in enumerate(val_loader):
            image, coords, gt_depth = image.to(device), coords.to(device), gt_depth.to(device)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                depth = model(image, coords)  # [B, K]
            all_pred.append(depth.float().cpu())
            all_gt.append(gt_depth.float().cpu())
            if verbose and cnt % 20 == 0:
                print(f"  sparse eval {cnt}/{len(val_loader)}")

    p = torch.cat(all_pred, dim=0).reshape(-1)
    g = torch.cat(all_gt, dim=0).reshape(-1)
    p = p.clamp(min=1e-3)
    return _compute_metrics(p, g), p, g


def _eval_dense(model, val_loader, device, verbose):
    """Dense evaluation using forward_train on full images."""
    all_pred, all_gt = [], []

    with torch.no_grad():
        for cnt, (image, _, _, depth_map) in enumerate(val_loader):
            image, depth_map = image.to(device), depth_map.to(device)

            # Use forward_train (dense) path
            model.train()
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                pred = model(image)  # [B, 1, H, W]
            model.eval()

            pred = pred.float().cpu().squeeze(1)   # [B, H, W]
            gt = depth_map.float().cpu().squeeze(1) # [B, H, W]
            all_pred.append(pred)
            all_gt.append(gt)
            if verbose and cnt % 20 == 0:
                print(f"  dense eval {cnt}/{len(val_loader)}")

    pred = torch.cat(all_pred, dim=0)  # [N, H, W]
    gt = torch.cat(all_gt, dim=0)

    # Mask valid pixels (gt > 0)
    mask = gt > 0
    p = pred[mask].clamp(min=1e-3)
    g = gt[mask]
    return _compute_metrics(p, g), p, g


def evaluate(model, device, verbose=True):
    """Run both dense and sparse evaluation, print comparison."""
    val_ds = NYUDataset(split="validation", K=128)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

    model.eval()
    dense_metrics, _, _ = _eval_dense(model, val_loader, device, verbose)
    sparse_metrics, p_sparse, g_sparse = _eval_sparse(model, val_loader, device, verbose)

    # Extended diagnostics (on sparse, as before)
    p, g = p_sparse, g_sparse
    pred_mean, pred_std = p.mean().item(), p.std().item()
    pred_med = p.median().item()
    gt_mean, gt_std = g.mean().item(), g.std().item()
    gt_med = g.median().item()
    ratio = p / g
    ratio_mean, ratio_med, ratio_std = ratio.mean().item(), ratio.median().item(), ratio.std().item()
    pred_cv = pred_std / (pred_mean + eps)
    gt_cv = gt_std / (gt_mean + eps)
    within_2x = ((ratio > 0.5) & (ratio < 2.0)).float().mean().item() * 100

    log_d = torch.log(p)
    ld_mean, ld_std = log_d.mean().item(), log_d.std().item()
    ld_min, ld_max = log_d.min().item(), log_d.max().item()

    # Depth-range breakdown (sparse)
    bins = [(0, 2), (2, 5), (5, 10)]
    bin_results = {}
    for lo, hi in bins:
        mask = (g >= lo) & (g < hi)
        n = mask.sum().item()
        if n > 0:
            bin_rel = (torch.abs(p[mask] - g[mask]) / g[mask]).mean().item()
            bin_d1 = (torch.max(p[mask] / g[mask], g[mask] / p[mask]) < 1.25).float().mean().item() * 100
        else:
            bin_rel, bin_d1 = float('nan'), float('nan')
        bin_results[f'{lo}-{hi}m'] = (bin_rel, bin_d1, n)

    if verbose:
        dm, sm = dense_metrics, sparse_metrics
        print("\n" + "=" * 65)
        print("  EVALUATION REPORT")
        print("=" * 65)

        # Dense vs Sparse comparison table
        print(f"\n  [Dense vs Sparse Comparison]")
        print(f"    {'Metric':>12s}  {'Dense':>10s}  {'Sparse':>10s}  {'Gap':>10s}")
        print(f"    {'-' * 46}")
        for key, label in [('abs_rel', 'AbsRel'), ('d1', 'd<1.25'),
                           ('rmse', 'RMSE'), ('silog', 'SILog'),
                           ('optimal_scale', 'Scale s*'),
                           ('abs_rel_scaled', 'Scaled AR')]:
            dv, sv = dm[key], sm[key]
            if key == 'd1':
                gap = f"{dv - sv:+.1f}%"
                print(f"    {label:>12s}  {dv:>9.1f}%  {sv:>9.1f}%  {gap:>10s}")
            else:
                gap = f"{dv - sv:+.4f}"
                print(f"    {label:>12s}  {dv:>10.4f}  {sv:>10.4f}  {gap:>10s}")

        # Full sparse metrics (existing report)
        print(f"\n  [Standard Metrics (Sparse)]")
        print(f"    AbsRel : {sm['abs_rel']:.4f}      SqRel  : {sm['sq_rel']:.4f}")
        print(f"    RMSE   : {sm['rmse']:.4f}      RMSE_log: {sm['rmse_log']:.4f}")
        print(f"    SILog  : {sm['silog']:.4f}")
        print(f"    d<1.25 : {sm['d1']:.1f}%      d<1.25^2: {sm['d2']:.1f}%      d<1.25^3: {sm['d3']:.1f}%")

        print(f"\n  [Prediction Distribution (Sparse)]")
        print(f"    Pred : mean={pred_mean:.3f}  std={pred_std:.3f}  "
              f"med={pred_med:.3f}  [{p.min().item():.3f}, {p.max().item():.3f}]")
        print(f"    GT   : mean={gt_mean:.3f}  std={gt_std:.3f}  "
              f"med={gt_med:.3f}  [{g.min().item():.3f}, {g.max().item():.3f}]")
        print(f"    Ratio: mean={ratio_mean:.3f}  med={ratio_med:.3f}  std={ratio_std:.3f}")
        print(f"    Pred CV={pred_cv:.3f}  GT CV={gt_cv:.3f}  "
              f"{'!! COLLAPSE (pred CV << GT CV)' if pred_cv < gt_cv * 0.3 else 'OK'}")
        print(f"    Within 2x of GT: {within_2x:.1f}%")

        print(f"\n  [Log-Depth Head]")
        print(f"    mean={ld_mean:.4f}  std={ld_std:.4f}  [{ld_min:.4f}, {ld_max:.4f}]  (init=0)")
        if ld_std < 0.1:
            print(f"    !! HEAD NOT LEARNING (std < 0.1, predictions ~constant)")
        elif ld_std < 0.3:
            print(f"    .. Head barely learning (std < 0.3)")
        else:
            print(f"    OK")

        print(f"\n  [Depth-Range Breakdown (Sparse)]")
        print(f"    {'Range':>8s}  {'AbsRel':>8s}  {'d<1.25':>8s}  {'Count':>8s}")
        print(f"    {'-' * 40}")
        for rng, (rel, bd1, n) in bin_results.items():
            if np.isnan(rel):
                print(f"    {rng:>8s}  {'N/A':>8s}  {'N/A':>8s}  {n:>8d}")
            else:
                print(f"    {rng:>8s}  {rel:>8.4f}  {bd1:>7.1f}%  {n:>8d}")

        print(f"\n  [Scale Analysis (Sparse)]")
        s = sm['optimal_scale']
        print(f"    Optimal scale s* = {s:.4f}  (1.0 = perfect)")
        imp = (sm['abs_rel'] - sm['abs_rel_scaled']) / (sm['abs_rel'] + eps) * 100
        print(f"    AbsRel raw={sm['abs_rel']:.4f}  ->  scaled={sm['abs_rel_scaled']:.4f}  "
              f"({imp:.1f}% improvement)")
        if imp > 20:
            print(f"    !! SCALE BIAS: model learned structure but not scale")
        else:
            print(f"    OK")

        print("=" * 65)

    # Return sparse metrics (backward compatible) + dense metrics prefixed
    metrics = {**sparse_metrics}
    metrics['pred_mean'] = pred_mean
    metrics['pred_std'] = pred_std
    metrics['pred_med'] = pred_med
    metrics['gt_mean'] = gt_mean
    metrics['gt_std'] = gt_std
    metrics['ratio_mean'] = ratio_mean
    metrics['ratio_med'] = ratio_med
    metrics['pred_cv'] = pred_cv
    metrics['within_2x'] = within_2x
    metrics['log_depth_mean'] = ld_mean
    metrics['log_depth_std'] = ld_std
    metrics['n_nan'] = 0
    metrics['n_inf'] = 0
    for rng, (rel, bd1, n) in bin_results.items():
        metrics[f'absrel_{rng}'] = rel
        metrics[f'd1_{rng}'] = bd1
    metrics['scale_improvement_pct'] = (sm['abs_rel'] - sm['abs_rel_scaled']) / (sm['abs_rel'] + eps) * 100

    # Dense metrics
    for k, v in dense_metrics.items():
        metrics[f'dense_{k}'] = v

    return metrics
