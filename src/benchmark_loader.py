"""Benchmark script: training batch size sweep + inference speed."""

import time
import torch

from models.spd import SPD
from utils.losses import l_dense_silog
from config import H_IMG as H, W_IMG as W


WARMUP_STEPS = 3
MEASURE_STEPS = 50
N_DATASET = 47584


def run_step(model, images, depth_map):
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        pred = model(images)
        loss = l_dense_silog(pred, depth_map)
    loss.backward()
    model.zero_grad(set_to_none=True)


def test_batch_size(model, device):
    """Find max batch size and measure throughput for each."""
    results = []
    for bs in [1, 2, 3, 4, 6, 8, 10, 12]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        images = torch.randn(bs, 3, H, W, device=device)
        depth_map = torch.rand(bs, 1, H, W, device=device) * 9.0 + 1.0

        try:
            for _ in range(WARMUP_STEPS):
                run_step(model, images, depth_map)

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(MEASURE_STEPS):
                run_step(model, images, depth_map)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

            peak_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
            imgs_per_sec = (MEASURE_STEPS * bs) / elapsed
            results.append((bs, imgs_per_sec, peak_mb, elapsed / MEASURE_STEPS))
            print(f"  batch_size={bs:>2}  |  {imgs_per_sec:.1f} img/s  |  "
                  f"{peak_mb:.0f} MB peak  |  {elapsed/MEASURE_STEPS:.3f} s/step")

        except (RuntimeError, torch.AcceleratorError) as e:
            if "out of memory" in str(e).lower() or "illegal memory" in str(e).lower():
                torch.cuda.empty_cache()
                print(f"  batch_size={bs:>2}  |  OOM")
                break
            raise

        del images, depth_map

    return results


def test_inference(model, device):
    """Measure inference speed at different K values."""
    results = []
    image = torch.randn(1, 3, H, W, device=device)

    for K in [1, 32, 64, 128, 256, 512, 1024]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        coords = torch.randint(0, min(H, W), (1, K, 2), device=device).float()

        try:
            with torch.no_grad():
                for _ in range(WARMUP_STEPS):
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        model(image, coords)

                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(MEASURE_STEPS):
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        model(image, coords)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - t0

            peak_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
            ms_per_img = elapsed / MEASURE_STEPS * 1000
            queries_per_sec = K * MEASURE_STEPS / elapsed
            results.append((K, ms_per_img, peak_mb, queries_per_sec))
            print(f"  K={K:>5}  |  {ms_per_img:.1f} ms/img  |  "
                  f"{peak_mb:.0f} MB peak  |  {queries_per_sec:.0f} queries/s")

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                print(f"  K={K:>5}  |  OOM")
                break
            raise

        del coords

    # Also test dense inference (full forward_train path, no grads)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model.train()  # use dense path
    with torch.no_grad():
        for _ in range(WARMUP_STEPS):
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                model(image)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(MEASURE_STEPS):
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                model(image)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

    peak_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
    ms_per_img = elapsed / MEASURE_STEPS * 1000
    total_queries = H * W
    print(f"  dense   |  {ms_per_img:.1f} ms/img  |  "
          f"{peak_mb:.0f} MB peak  |  {total_queries} pixels ({total_queries/ms_per_img*1000:.0f} px/s)")
    model.eval()

    del image
    return results


if __name__ == "__main__":
    device = torch.device("cuda")
    model = SPD(pretrained=True).to(device)

    print(f"Resolution: {H}x{W}")
    print(f"GPU: {torch.cuda.get_device_name()}")
    free, total = torch.cuda.mem_get_info()
    print(f"VRAM: {total / 1024**2:.0f} MB ({free / 1024**2:.0f} MB free)")
    print()


    print("=== Phase 1: Training batch size sweep ===")
    model.train()
    bs_results = test_batch_size(model, device)

    if bs_results:
        best = max(bs_results, key=lambda x: x[1])
        print(f"\n  Best: batch_size={best[0]}  ({best[1]:.1f} img/s, {best[2]:.0f} MB)")
        print(f"  ~{N_DATASET // best[0]} steps/epoch  |  "
              f"~{(N_DATASET // best[0]) * best[3] / 60:.1f} min/epoch")
    

    print(f"\n=== Phase 2: Inference speed (eval mode) ===")
    model.eval()
    inf_results = test_inference(model, device)
