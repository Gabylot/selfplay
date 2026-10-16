"""Benchmark GPU forward pass latency at various batch sizes.

Measures the fixed-overhead theory: if latency stays roughly flat across
batch sizes 1→256, then aggregating multiple workers' batches into one
mega-batch will dramatically increase throughput.

Usage:
    python benchmark_batch_latency.py
"""

import time
import numpy as np
import torch
import torch.nn.functional as F
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import get_config
from network import AlphaZeroNet, create_model_from_config, load_checkpoint
from encoding import NUM_PLANES, NUM_ACTIONS
from pathlib import Path


def benchmark_batch_latency():
    """Measure forward pass latency at various batch sizes on DirectML."""
    try:
        device = torch.device("cuda")
        print(f"[GPU-Server] ROCm device: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("[WARN] torch_directml not available, falling back to CPU")
        device = torch.device("cpu")

    config = get_config()
    net = create_model_from_config(config).to(device)
    net.eval()

    # Try to load latest checkpoint for realistic weights
    output_dir = Path(config.main.output_dir) / config.main.run_name
    ckpt_path = output_dir / "checkpoints" / "latest.pt"
    if ckpt_path.exists():
        print(f"[INFO] Loading checkpoint: {ckpt_path}")
        load_checkpoint(str(ckpt_path), net)
    else:
        print("[INFO] No checkpoint found — using random weights")

    batch_sizes = [1, 8, 16, 32, 64, 128, 256]
    num_warmup = 10
    num_iters = 30

    print(f"\n{'='*72}")
    print(f"  BATCH LATENCY BENCHMARK (warmup={num_warmup}, iters={num_iters})")
    print(f"  Network: {sum(p.numel() for p in net.parameters())} params, "
          f"NUM_PLANES={NUM_PLANES}, NUM_ACTIONS={NUM_ACTIONS}")
    print(f"{'='*72}\n")

    print(f"  {'Batch':>6}  {'Min(ms)':>10}  {'Median(ms)':>12}  {'Mean(ms)':>10}  "
          f"{'Throughput':>14}  {'Latency/32':>12}")
    print("  " + "-" * 70)

    results = {}
    baseline_latency_32 = None

    for bs in batch_sizes:
        dummy = torch.randn(bs, NUM_PLANES, 8, 8, device=device)

        # Warmup
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = net(dummy)

        # Measure
        times = []
        with torch.no_grad():
            for _ in range(num_iters):
                t0 = time.perf_counter()
                policy_logits, values = net(dummy)
                # Force synchronization — DirectML may be async
                _ = policy_logits.cpu()
                _ = values.cpu()
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000)

        min_ms = min(times)
        median_ms = float(np.median(times))
        mean_ms = float(np.mean(times))
        throughput = bs / (mean_ms / 1000)  # samples/sec
        latency_per_32 = mean_ms / (bs / 32) if bs > 0 else mean_ms

        if bs == 32:
            baseline_latency_32 = mean_ms

        ratio_str = f"{mean_ms / baseline_latency_32:.2f}x" if baseline_latency_32 else "—"
        print(f"  {bs:>6}  {min_ms:>10.2f}  {median_ms:>12.2f}  {mean_ms:>10.2f}  "
              f"{throughput:>10.0f}/s  {ratio_str:>12}")

        results[bs] = {
            'min_ms': min_ms,
            'median_ms': median_ms,
            'mean_ms': mean_ms,
            'throughput': throughput,
        }

    # Analysis
    print(f"\n{'='*72}")
    print("  ANALYSIS")
    print(f"{'='*72}")

    lat_32 = results[32]['mean_ms']
    lat_128 = results[128]['mean_ms']
    lat_256 = results[256]['mean_ms']

    print(f"\n  Latency at batch=32:  {lat_32:.2f} ms")
    print(f"  Latency at batch=128: {lat_128:.2f} ms  ({lat_128/lat_32:.2f}x of 32)")
    print(f"  Latency at batch=256: {lat_256:.2f} ms  ({lat_256/lat_32:.2f}x of 32)")

    # Throughput comparison
    tp_32 = results[32]['throughput']
    tp_128 = results[128]['throughput']
    tp_256 = results[256]['throughput']

    print(f"\n  Throughput at batch=32 (8 passes):  {tp_32:.0f} samples/s")
    print(f"  Throughput at batch=128 (2 passes): {tp_128:.0f} samples/s  "
          f"({tp_128/tp_32:.2f}x)")
    print(f"  Throughput at batch=256 (1 pass):   {tp_256:.0f} samples/s  "
          f"({tp_256/tp_32:.2f}x)")

    # Decision gate — use MEDIAN (robust to DirectML shader recompilation outliers)
    med_32 = results[32]['median_ms']
    med_128 = results[128]['median_ms']
    med_256 = results[256]['median_ms']

    print(f"\n  Median-based analysis (robust to outliers):")
    print(f"  Median latency at batch=32:  {med_32:.2f} ms")
    print(f"  Median latency at batch=128: {med_128:.2f} ms  ({med_128/med_32:.2f}x of 32)")
    print(f"  Median latency at batch=256: {med_256:.2f} ms  ({med_256/med_32:.2f}x of 32)")

    # Throughput comparison using median
    tp_med_32 = 32 / (med_32 / 1000)
    tp_med_128 = 128 / (med_128 / 1000)
    tp_med_256 = 256 / (med_256 / 1000)

    print(f"\n  Median throughput at batch=32 (8 passes):  {tp_med_32:.0f} samples/s")
    print(f"  Median throughput at batch=128 (2 passes): {tp_med_128:.0f} samples/s  "
          f"({tp_med_128/tp_med_32:.2f}x)")
    print(f"  Median throughput at batch=256 (1 pass):   {tp_med_256:.0f} samples/s  "
          f"({tp_med_256/tp_med_32:.2f}x)")

    print(f"\n  {'='*60}")
    if med_256 <= 2.5 * med_32:
        print("  [PROCEED] median latency(256) <= 2.5 x median latency(32)")
        print("    Aggregation will provide significant throughput improvement.")
        print(f"    Expected speedup: {8 * med_32 / med_256:.1f}x "
              f"(8x{med_32:.1f}ms -> 1x{med_256:.1f}ms)")
    elif med_256 <= 3.0 * med_32:
        print("  [MARGINAL] median latency(256) is 2.5-3.0 x median latency(32)")
        print("    Aggregation will help but benefit is moderate.")
    else:
        print("  [RECONSIDER] median latency(256) > 3.0 x median latency(32)")
        print("    Aggregation benefit is marginal -- not worth the complexity.")
    print(f"  {'='*60}")


if __name__ == "__main__":
    benchmark_batch_latency()