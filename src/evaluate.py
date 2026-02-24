import torch

from torch.utils.data import DataLoader
from data.nyu_dataset import NYUDataset

def evaluate(model, device):
    model.eval()
    val_ds = NYUDataset(split="validation", K=256)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    total_abs_rel = 0.0
    num_samples = 0

    with torch.no_grad():
        for cnt, (image, coords, gt_depth) in enumerate(val_loader):
            image = image.to(device)
            coords = coords.to(device)
            gt_depth = gt_depth.to(device)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                depth, _ = model(image, coords)
            # Compute metrics (e.g., RMSE, MAE) using output and gt_depth
            abs_rel = torch.mean(torch.abs(depth - gt_depth) / gt_depth)

            total_abs_rel += abs_rel.item()
            num_samples += 1

            print(f"Sample {num_samples}, AbsRel: {abs_rel.item():.4f}") if cnt % 100 == 0 else None

    return total_abs_rel / num_samples if num_samples > 0 else float('inf')