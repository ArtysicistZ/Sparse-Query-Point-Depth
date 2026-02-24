import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from datasets import load_dataset

from config import H_IMG as H, W_IMG as W

class NYUDataset(Dataset):
    """NYU Depth V2 dataset for monocular depth estimation."""
    
    def __init__(self, split="train", K=256):
        self.split = split
        self.K = K
        hf_split = "train" if split == "train" else "validation"
        self.nyu_data = load_dataset("sayakpaul/nyu_depth_v2", split=hf_split, revision="refs/convert/parquet")

    def __len__(self):
        return len(self.nyu_data)

    def __getitem__(self, idx):
        item = self.nyu_data[idx]
        image = item["image"].resize((W, H))  # PIL resize to (width, height)
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)  # [C, H, W]

        depth = item["depth_map"].resize((W, H), resample=Image.NEAREST)
        depth = np.array(depth).astype(np.float32)

        valid_y, valid_x = np.where(depth > 0)
        indices = np.random.choice(len(valid_y), self.K, replace=True)

        qx = valid_x[indices]
        qy = valid_y[indices]

        query_coords = torch.tensor(np.stack((qx, qy), axis=-1), dtype=torch.float32)
        gt_depth = torch.tensor(depth[qy, qx], dtype=torch.float32)

        return image, query_coords, gt_depth
    

if __name__ == "__main__":
    ds = NYUDataset(split="train", K=16)
    image, coords, gt_depth = ds[0]
    print(f"image: {image.shape}, range [{image.min():.2f}, {image.max():.2f}]")
    print(f"coords: {coords.shape}")
    print(f"gt_depth: {gt_depth.shape}, range [{gt_depth.min():.2f}, {gt_depth.max():.2f}]")