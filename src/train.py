import json
import time
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from models.spd import SPD
from utils.losses import l_dense_silog
from data.nyu_dataset import NYUDataset
from evaluate import evaluate
from config import EPOCHS, BATCH_SIZE, ENCODER_LR, DECODER_LR

import os

SAVE_DIR = "checkpoints/v15.1.1"
LOG_FILE = os.path.join(SAVE_DIR, "train_log.json")


def build_optimizer(model):
    encoder_params = list(model.encoder.parameters())
    decoder_params = [p for n, p in model.named_parameters() if not n.startswith("encoder")]

    optimizer = torch.optim.AdamW([
        {'params': encoder_params, 'lr': ENCODER_LR},
        {'params': decoder_params, 'lr': DECODER_LR}
    ], weight_decay=0.01)

    return optimizer


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SPD(pretrained=True).to(device)

    optimizer = build_optimizer(model)
    

    dataset = NYUDataset(split="train", K=256)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    total_steps = len(train_loader) * EPOCHS
    warmup_steps = 500
    warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_steps])

    log = {"config": {"epochs": EPOCHS, "batch_size": BATCH_SIZE, 
                      "encoder_lr": ENCODER_LR,
                      "decoder_lr": DECODER_LR, "warmup_steps": warmup_steps,
                      "total_steps": total_steps}, "epochs": []}

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0
        t0 = time.time()

        for step, (images, depth_map) in enumerate(train_loader):
            images = images.to(device)
            depth_map = depth_map.to(device)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                pred_depth = model(images)
                loss = l_dense_silog(pred_depth, depth_map)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            epoch_steps += 1

            if step % 100 == 0:
                with torch.no_grad():
                    pred = pred_depth.float()
                    gt = depth_map.float()
                    mask = gt > 0
                    p_valid = pred[mask].clamp(min=1e-3)
                    g_valid = gt[mask]
                    abs_rel = (torch.abs(p_valid - g_valid) / g_valid).mean().item()
                    s_star = (g_valid / p_valid).median().item()
                    abs_rel_s = (torch.abs(p_valid * s_star - g_valid) / g_valid).mean().item()
                    p_mean = p_valid.mean().item()
                    p_std = p_valid.std().item()
                    g_mean = g_valid.mean().item()
                    g_std = g_valid.std().item()
                lr_enc = optimizer.param_groups[0]['lr']
                lr_dec = optimizer.param_groups[1]['lr']
                print(f"E{epoch+1} S{step}/{len(train_loader)} "
                      f"loss={loss.item():.4f} "
                      f"AR={abs_rel:.3f} s*={s_star:.3f} sAR={abs_rel_s:.3f} "
                      f"pred={p_mean:.2f}±{p_std:.2f} "
                      f"gt={g_mean:.2f}±{g_std:.2f} "
                      f"lr={lr_enc:.1e}/{lr_dec:.1e}")

        epoch_time = time.time() - t0
        avg_loss = epoch_loss / epoch_steps

        metrics = evaluate(model, device)
        print(f"Epoch {epoch+1} Validation AbsRel: {metrics['abs_rel']:.4f}")

        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": round(avg_loss, 6),
            "epoch_time_s": round(epoch_time, 1),
            "lr_encoder": optimizer.param_groups[0]['lr'],
            "lr_decoder": optimizer.param_groups[1]['lr'],
            **{k: round(v, 6) if isinstance(v, float) else v for k, v in metrics.items()}
        }
        log["epochs"].append(epoch_record)

        with open(LOG_FILE, "w") as f:
            json.dump(log, f, indent=2)

        torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"checkpoint_epoch{epoch+1}.pt"))
        print(f"  Log saved to {LOG_FILE}  |  Checkpoint saved")


if __name__ == "__main__":
    train()
