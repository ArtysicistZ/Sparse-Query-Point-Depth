import torch
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from datasets import load_dataset

from config import H_IMG as H, W_IMG as W

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# NYU native resolution
NYU_H, NYU_W = 480, 640
MIN_DEPTH, MAX_DEPTH = 1e-3, 10.0


class NYUDataset(Dataset):
    """NYU Depth V2 dataset with BTS-style augmentation."""

    def __init__(self, split="train", K=256):
        self.split = split
        self.K = K
        self.is_train = split == "train"
        hf_split = "train" if split == "train" else "validation"
        self.nyu_data = load_dataset("sayakpaul/nyu_depth_v2", split=hf_split)

    def __len__(self):
        return len(self.nyu_data)

    def __getitem__(self, idx):
        item = self.nyu_data[idx]
        image = item["image"]    # PIL 480x640
        depth = item["depth_map"]  # PIL 480x640

        if self.is_train:
            # --- Random rotation +/-2.5 degrees ---
            if random.random() < 0.5:
                angle = random.uniform(-2.5, 2.5)
                image = image.rotate(angle, resample=Image.BILINEAR, fillcolor=0)
                depth = depth.rotate(angle, resample=Image.NEAREST, fillcolor=0)

            # --- Random crop 416x544 from 480x640 ---
            top = random.randint(0, NYU_H - H)
            left = random.randint(0, NYU_W - W)
            image = image.crop((left, top, left + W, top + H))
            depth = depth.crop((left, top, left + W, top + H))

            # --- Horizontal flip ---
            if random.random() < 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                depth = depth.transpose(Image.FLIP_LEFT_RIGHT)

            # --- Color augmentation (BTS-style, on float32 numpy) ---
            image = np.array(image).astype(np.float32) / 255.0

            # Gamma
            gamma = random.uniform(0.9, 1.1)
            image = np.power(image, gamma)

            # Brightness
            brightness = random.uniform(0.75, 1.25)
            image = image * brightness

            # Per-channel color
            colors = np.random.uniform(0.9, 1.1, size=3).astype(np.float32)
            image = image * colors[np.newaxis, np.newaxis, :]

            image = np.clip(image, 0.0, 1.0)

        else:
            # --- Validation: resize to target resolution ---
            image = image.resize((W, H), resample=Image.BILINEAR)
            depth = depth.resize((W, H), resample=Image.NEAREST)
            image = np.array(image).astype(np.float32) / 255.0

        # --- Normalize and convert ---
        image = (image - MEAN[np.newaxis, np.newaxis, :]) / STD[np.newaxis, np.newaxis, :]
        image = torch.from_numpy(image).permute(2, 0, 1).float()  # [C, H, W]

        depth = np.array(depth).astype(np.float32)
        depth = np.clip(depth, 0.0, MAX_DEPTH)

        # --- Sample query points from valid pixels ---
        valid_y, valid_x = np.where(depth >= MIN_DEPTH)
        indices = np.random.choice(len(valid_y), self.K, replace=True)

        qx = valid_x[indices]
        qy = valid_y[indices]

        depth_map = torch.from_numpy(depth).unsqueeze(0)  # [1, H, W]

        if self.is_train:
            return image, depth_map

        query_coords = torch.tensor(np.stack((qx, qy), axis=-1), dtype=torch.float32)
        gt_depth = torch.tensor(depth[qy, qx], dtype=torch.float32)

        return image, query_coords, gt_depth, depth_map


if __name__ == "__main__":
    ds = NYUDataset(split="train", K=16)
    image, coords, gt_depth, depth_map = ds[0]
    print(f"image: {image.shape}, range [{image.min():.2f}, {image.max():.2f}]")
    print(f"coords: {coords.shape}")
    print(f"gt_depth: {gt_depth.shape}, range [{gt_depth.min():.2f}, {gt_depth.max():.2f}]")
    print(f"depth_map: {depth_map.shape}, range [{depth_map.min():.2f}, {depth_map.max():.2f}]")

    ds_val = NYUDataset(split="validation", K=16)
    image, coords, gt_depth, depth_map = ds_val[0]
    print(f"\nVal image: {image.shape}, range [{image.min():.2f}, {image.max():.2f}]")
    print(f"Val depth_map: {depth_map.shape}, range [{depth_map.min():.2f}, {depth_map.max():.2f}]")
