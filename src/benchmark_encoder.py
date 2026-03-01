"""Benchmark: Compare ConvNeXt V2-T vs DINOv2 ViT-S encoder speed and VRAM."""

import time
import torch

WARMUP_STEPS = 5
MEASURE_STEPS = 50

# ConvNeXt uses 352x480 (divisible by 32), ViT-S uses 350x476 (divisible by 14)
CONFIGS = {
    "ConvNeXt V2-T": {"res": (352, 480), "build": "convnext"},
    "DINOv2 ViT-S":  {"res": (350, 476), "build": "vits"},
}


def build_convnext(device):
    from models.encoder_convnext.convnext import ConvNeXtV2Encoder
    from models.encoder_convnext.pyramid_neck import ProjectionNeck
    encoder = ConvNeXtV2Encoder(pretrained=True).to(device)
    neck = ProjectionNeck(enc_channels=[96, 192, 384, 768], d_model=64).to(device)
    return encoder, neck


def build_vits(device):
    from models.encoder_vits.vit_s import ViTSEncoder
    from models.encoder_vits.pyramid_neck import ProjectionNeck
    encoder = ViTSEncoder(pretrained=True).to(device)
    neck = ProjectionNeck(enc_channels=[384, 384, 384, 384], d_model=64).to(device)
    return encoder, neck


def count_params(modules):
    seen = set()
    total = 0
    for m in modules:
        for p in m.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                total += p.numel()
    return total


def bench_encoder(name, encoder, neck, image, device):
    """Measure forward pass speed and VRAM for encoder + neck."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Warmup
    with torch.no_grad():
        for _ in range(WARMUP_STEPS):
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                feats = encoder(image)
                neck(feats)

    # Measure
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(MEASURE_STEPS):
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                feats = encoder(image)
                out = neck(feats)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    peak_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
    ms_per_img = elapsed / MEASURE_STEPS * 1000

    # Output shapes
    shapes = {k: tuple(v.shape) for k, v in out.items()}

    print(f"\n  {name}")
    print(f"    Forward:  {ms_per_img:.1f} ms/img  ({1000/ms_per_img:.1f} img/s)")
    print(f"    Peak VRAM: {peak_mb:.0f} MB")
    print(f"    Params:   {count_params([encoder, neck]):,}")
    print(f"    Output shapes: {shapes}")

    return ms_per_img, peak_mb


def bench_training_step(name, encoder, neck, H, W, device):
    """Measure a training forward+backward pass at bs=1."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    image = torch.randn(1, 3, H, W, device=device)

    # Warmup
    for _ in range(WARMUP_STEPS):
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            feats = encoder(image)
            out = neck(feats)
            loss = sum(v.mean() for v in out.values())
        loss.backward()
        encoder.zero_grad(set_to_none=True)
        neck.zero_grad(set_to_none=True)

    # Measure
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(MEASURE_STEPS):
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            feats = encoder(image)
            out = neck(feats)
            loss = sum(v.mean() for v in out.values())
        loss.backward()
        encoder.zero_grad(set_to_none=True)
        neck.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    peak_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
    ms_per_step = elapsed / MEASURE_STEPS * 1000

    print(f"    Train step: {ms_per_step:.1f} ms/step  |  {peak_mb:.0f} MB peak")
    del image
    return ms_per_step, peak_mb


if __name__ == "__main__":
    device = torch.device("cuda")

    print(f"GPU: {torch.cuda.get_device_name()}")
    free, total = torch.cuda.mem_get_info()
    print(f"VRAM: {total / 1024**2:.0f} MB ({free / 1024**2:.0f} MB free)")

    results = {}

    for name, cfg in CONFIGS.items():
        H, W = cfg["res"]
        print(f"\n{'='*50}")
        print(f"  {name}  (input: {H}x{W})")
        print(f"{'='*50}")

        if cfg["build"] == "convnext":
            encoder, neck = build_convnext(device)
        else:
            encoder, neck = build_vits(device)

        encoder.eval()
        neck.eval()

        image = torch.randn(1, 3, H, W, device=device)
        infer_ms, infer_mb = bench_encoder(name, encoder, neck, image, device)
        del image

        encoder.train()
        neck.train()
        train_ms, train_mb = bench_training_step(name, encoder, neck, H, W, device)

        results[name] = {
            "infer_ms": infer_ms, "infer_mb": infer_mb,
            "train_ms": train_ms, "train_mb": train_mb,
            "params": count_params([encoder, neck]),
        }

        del encoder, neck
        torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*50}")
    print("  SUMMARY")
    print(f"{'='*50}")
    print(f"  {'':20s} {'ConvNeXt V2-T':>15s} {'DINOv2 ViT-S':>15s}")
    print(f"  {'-'*50}")
    for metric, label in [
        ("params", "Parameters"),
        ("infer_ms", "Inference (ms)"),
        ("infer_mb", "Infer VRAM (MB)"),
        ("train_ms", "Train step (ms)"),
        ("train_mb", "Train VRAM (MB)"),
    ]:
        vals = []
        for name in CONFIGS:
            v = results[name][metric]
            if metric == "params":
                vals.append(f"{v:,}")
            elif "ms" in metric:
                vals.append(f"{v:.1f}")
            else:
                vals.append(f"{v:.0f}")
        print(f"  {label:20s} {vals[0]:>15s} {vals[1]:>15s}")
