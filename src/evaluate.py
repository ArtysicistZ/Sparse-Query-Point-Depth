import torch
import numpy as np

from torch.utils.data import DataLoader
from data.nyu_dataset import NYUDataset
from config import H_IMG, W_IMG


def evaluate(model, device, verbose=True):
    """Comprehensive evaluation with diagnostic metrics.

    Returns AbsRel (float) for backward compatibility.
    Prints full diagnostic report when verbose=True.
    """
    model.eval()
    val_ds = NYUDataset(split="validation", K=256)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

    all_pred = []
    all_gt = []
    all_log_depth = []
    all_top_l3 = []
    all_top_l4 = []

    with torch.no_grad():
        for cnt, (image, coords, gt_depth) in enumerate(val_loader):
            image = image.to(device)
            coords = coords.to(device)
            gt_depth = gt_depth.to(device)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                depth, debug = model(image, coords, return_debug=True)

            all_pred.append(depth.float().cpu())
            all_gt.append(gt_depth.float().cpu())
            all_log_depth.append(debug['log_depth'].float().cpu())
            all_top_l3.append(debug['top_indices_l3'].cpu())
            all_top_l4.append(debug['top_indices_l4'].cpu())

            if verbose and cnt % 20 == 0:
                print(f"  eval {cnt}/{len(val_loader)}")

    # [N_images, K]
    pred = torch.cat(all_pred, dim=0)
    gt = torch.cat(all_gt, dim=0)
    log_d = torch.cat(all_log_depth, dim=0)
    top_l3 = torch.cat(all_top_l3, dim=0)  # [N, K, 20]
    top_l4 = torch.cat(all_top_l4, dim=0)  # [N, K, 10]

    # Flatten for global metrics
    p = pred.reshape(-1)
    g = gt.reshape(-1)

    # --- NaN / Inf check ---
    n_nan = torch.isnan(p).sum().item()
    n_inf = torch.isinf(p).sum().item()
    p = p.clamp(min=1e-3)  # safety clamp

    # ============================
    #  1. Standard Depth Metrics
    # ============================
    abs_rel = (torch.abs(p - g) / g).mean().item()
    sq_rel = ((p - g) ** 2 / g).mean().item()
    rmse = torch.sqrt(((p - g) ** 2).mean()).item()

    log_diff = torch.log(p) - torch.log(g)
    rmse_log = torch.sqrt((log_diff ** 2).mean()).item()
    silog = torch.sqrt((log_diff ** 2).mean() - 0.5 * (log_diff.mean() ** 2) + 1e-8).item()

    thresh = torch.max(p / g, g / p)
    d1 = (thresh < 1.25).float().mean().item() * 100
    d2 = (thresh < 1.25 ** 2).float().mean().item() * 100
    d3 = (thresh < 1.25 ** 3).float().mean().item() * 100

    # ============================
    #  2. Prediction Distribution
    # ============================
    pred_mean, pred_std = p.mean().item(), p.std().item()
    pred_med = p.median().item()
    gt_mean, gt_std = g.mean().item(), g.std().item()
    gt_med = g.median().item()

    ratio = p / g
    ratio_mean = ratio.mean().item()
    ratio_med = ratio.median().item()
    ratio_std = ratio.std().item()

    pred_cv = pred_std / (pred_mean + 1e-8)
    gt_cv = gt_std / (gt_mean + 1e-8)

    # ============================
    #  3. Log-Depth Head Analysis
    # ============================
    ld = log_d.reshape(-1)
    ld_mean, ld_std = ld.mean().item(), ld.std().item()
    ld_min, ld_max = ld.min().item(), ld.max().item()

    # ============================
    #  4. Per-Image Prediction Diversity
    # ============================
    per_img_std = pred.std(dim=1)  # [N_images]
    intra_std_mean = per_img_std.mean().item()
    intra_std_min = per_img_std.min().item()

    # What fraction of queries predict within 2x of GT?
    within_2x = ((ratio > 0.5) & (ratio < 2.0)).float().mean().item() * 100

    # ============================
    #  5. Depth-Range Breakdown
    # ============================
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

    mask = g >= 10
    n = mask.sum().item()
    if n > 0:
        bin_rel = (torch.abs(p[mask] - g[mask]) / g[mask]).mean().item()
        bin_d1 = (torch.max(p[mask] / g[mask], g[mask] / p[mask]) < 1.25).float().mean().item() * 100
    else:
        bin_rel, bin_d1 = float('nan'), float('nan')
    bin_results['10m+'] = (bin_rel, bin_d1, n)

    # ============================
    #  6. Scale Analysis
    # ============================
    optimal_scale = (g / p).median().item()
    p_scaled = p * optimal_scale
    abs_rel_scaled = (torch.abs(p_scaled - g) / g).mean().item()
    improvement_pct = (abs_rel - abs_rel_scaled) / (abs_rel + 1e-8) * 100

    # ============================
    #  7. Routing Diversity
    # ============================
    n_l3 = (H_IMG // 16) * (W_IMG // 16)
    n_l4 = (H_IMG // 32) * (W_IMG // 32)
    N_img = top_l3.shape[0]

    l3_uniq = [top_l3[i].reshape(-1).unique().numel() for i in range(N_img)]
    l4_uniq = [top_l4[i].reshape(-1).unique().numel() for i in range(N_img)]
    l3_uniq_mean, l3_uniq_std = np.mean(l3_uniq), np.std(l3_uniq)
    l4_uniq_mean, l4_uniq_std = np.mean(l4_uniq), np.std(l4_uniq)

    # ============================
    #  Print Report
    # ============================
    if verbose:
        print("\n" + "=" * 65)
        print("  EVALUATION REPORT")
        print("=" * 65)

        if n_nan > 0 or n_inf > 0:
            print(f"\n  !! {n_nan} NaN, {n_inf} Inf in predictions !!")

        print(f"\n  [Standard Metrics]")
        print(f"    AbsRel : {abs_rel:.4f}      SqRel  : {sq_rel:.4f}")
        print(f"    RMSE   : {rmse:.4f}      RMSE_log: {rmse_log:.4f}")
        print(f"    SILog  : {silog:.4f}")
        print(f"    d<1.25 : {d1:.1f}%      d<1.25^2: {d2:.1f}%      d<1.25^3: {d3:.1f}%")

        print(f"\n  [Prediction Distribution]")
        print(f"    Pred : mean={pred_mean:.3f}  std={pred_std:.3f}  "
              f"med={pred_med:.3f}  [{p.min().item():.3f}, {p.max().item():.3f}]")
        print(f"    GT   : mean={gt_mean:.3f}  std={gt_std:.3f}  "
              f"med={gt_med:.3f}  [{g.min().item():.3f}, {g.max().item():.3f}]")
        print(f"    Ratio: mean={ratio_mean:.3f}  med={ratio_med:.3f}  std={ratio_std:.3f}")
        print(f"    Pred CV={pred_cv:.3f}  GT CV={gt_cv:.3f}  "
              f"{'!! COLLAPSE (pred CV << GT CV)' if pred_cv < gt_cv * 0.3 else 'OK'}")
        print(f"    Within 2x of GT: {within_2x:.1f}%")

        print(f"\n  [Log-Depth Head]")
        print(f"    mean={ld_mean:.4f}  std={ld_std:.4f}  [{ld_min:.4f}, {ld_max:.4f}]  (init=0.916)")
        if ld_std < 0.1:
            print(f"    !! HEAD NOT LEARNING (std < 0.1, predictions ~constant)")
        elif ld_std < 0.3:
            print(f"    .. Head barely learning (std < 0.3)")
        else:
            print(f"    OK")

        print(f"\n  [Per-Image Diversity]")
        print(f"    Intra-image pred std: mean={intra_std_mean:.4f}  min={intra_std_min:.4f}")
        if intra_std_min < 0.01:
            print(f"    !! SPATIAL COLLAPSE (some images: all queries -> same depth)")
        else:
            print(f"    OK")

        print(f"\n  [Depth-Range Breakdown]")
        print(f"    {'Range':>8s}  {'AbsRel':>8s}  {'d<1.25':>8s}  {'Count':>8s}")
        print(f"    {'-' * 40}")
        for rng, (rel, bd1, n) in bin_results.items():
            if np.isnan(rel):
                print(f"    {rng:>8s}  {'N/A':>8s}  {'N/A':>8s}  {n:>8d}")
            else:
                print(f"    {rng:>8s}  {rel:>8.4f}  {bd1:>7.1f}%  {n:>8d}")

        print(f"\n  [Scale Analysis]")
        print(f"    Optimal scale s* = {optimal_scale:.4f}  (1.0 = perfect)")
        print(f"    AbsRel raw={abs_rel:.4f}  ->  scaled={abs_rel_scaled:.4f}  "
              f"({improvement_pct:.1f}% improvement)")
        if improvement_pct > 20:
            print(f"    !! SCALE BIAS: model learned structure but not scale")
        else:
            print(f"    OK")

        print(f"\n  [Routing Diversity]")
        print(f"    L3 ({n_l3:>3d} total): {l3_uniq_mean:.1f} +/- {l3_uniq_std:.1f} unique/img "
              f"({l3_uniq_mean / n_l3 * 100:.1f}%)")
        print(f"    L4 ({n_l4:>3d} total): {l4_uniq_mean:.1f} +/- {l4_uniq_std:.1f} unique/img "
              f"({l4_uniq_mean / n_l4 * 100:.1f}%)")
        # Collapse: unique count barely exceeds the top-k value itself
        if l3_uniq_mean < 30:
            print(f"    !! L3 ROUTING COLLAPSE (all queries pick ~same 20 positions)")
        if l4_uniq_mean < 15:
            print(f"    !! L4 ROUTING COLLAPSE (all queries pick ~same 10 positions)")
        if l3_uniq_mean >= 30 and l4_uniq_mean >= 15:
            print(f"    OK")

        print("=" * 65)

    return abs_rel
