import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from models.spd import SPD
from utils.losses import l_silog, l_dense_silog
from data.nyu_dataset import NYUDataset
from evaluate import evaluate
from config import EPOCHS, H_IMG, W_IMG


def build_optimizer(model):
    encoder_params = list(model.encoder.parameters())
    decoder_params = [p for n, p in model.named_parameters() if not n.startswith("encoder")]

    optimizer = torch.optim.AdamW([
        {'params': encoder_params, 'lr': 1e-5},
        {'params': decoder_params, 'lr': 1e-4}
    ], weight_decay=0.01)

    return optimizer


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SPD(pretrained=True).to(device)

    optimizer = build_optimizer(model)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    scaler = torch.amp.GradScaler()

    dataset = NYUDataset(split="train", K=256)
    train_loader = DataLoader(dataset, batch_size=6, shuffle=True, num_workers=4, pin_memory=True)

    for epoch in range(EPOCHS):
        model.train()
        for step, (images, coords, gt_depth, depth_map) in enumerate(train_loader):
            images = images.to(device)
            coords = coords.to(device)
            gt_depth = gt_depth.to(device)
            depth_map = depth_map.to(device)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                pred_depth, aux_l2 = model(images, coords)

                gt_l2 = F.interpolate(depth_map, scale_factor=0.25, mode='bilinear', align_corners=True)

                loss_main = l_silog(pred_depth, gt_depth)
                loss_aux_l2 = l_dense_silog(aux_l2, gt_l2)
                loss = loss_main + 0.5 * loss_aux_l2

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            if step % 10 == 0:
                with torch.no_grad():
                    p_mean = pred_depth.mean().item()
                    p_std = pred_depth.std().item()
                print(f"Epoch {epoch+1}, Step {step}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f}, "
                      f"pred: {p_mean:.2f}+/-{p_std:.2f} "
                      f"[{pred_depth.min().item():.2f}, {pred_depth.max().item():.2f}], "
                      f"gt: [{gt_depth.min().item():.2f}, {gt_depth.max().item():.2f}]")

        scheduler.step()
        absrel = evaluate(model, device)
        print(f"Epoch {epoch+1} Validation AbsRel: {absrel:.4f}")


if __name__ == "__main__":
    train()
