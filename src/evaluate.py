import torch

from torch.utils.data import DataLoader
from data.nyu_dataset import NYUDataset

def evaluate(model, device):
    model.eval()
    val_ds = NYUDataset(split="validation", K=256)
    val_loader = DataLoader(val_ds, batch_size=12, shuffle=False, num_workers=4, pin_memory=True)
    total_abs_rel = 0.0
    num_samples = 0

    with torch.no_grad():
        for cnt, (image, coords, gt_depth) in enumerate(val_loader):
            image = image.to(device)
            coords = coords.to(device)
            gt_depth = gt_depth.to(device)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                depth, _ = model(image, coords)

            # Per-sample AbsRel: mean over K query points, then accumulate per image
            per_sample = torch.mean(torch.abs(depth - gt_depth) / gt_depth, dim=1)  # [B]

            total_abs_rel += per_sample.sum().item()
            num_samples += per_sample.shape[0]

            if cnt % 10 == 0:
                print(f"Sample {num_samples}/{len(val_ds)}, AbsRel: {per_sample.mean().item():.4f}, "
                      f"pred [{depth.min().item():.2f}, {depth.max().item():.2f}], "
                      f"gt [{gt_depth.min().item():.2f}, {gt_depth.max().item():.2f}]")

    return total_abs_rel / num_samples if num_samples > 0 else float('inf')