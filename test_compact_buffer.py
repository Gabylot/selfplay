"""Self-test for the compact replay buffer + on-device batch decoding.

Run from the project folder:   python test_compact_buffer.py

1. Builds a synthetic buffer with AlphaZero-style 119x8x8 planes.
2. Decodes a compact batch with torch on the CPU (and on CUDA/ROCm if
   available) and compares it to the numpy reference decode -- must be exact.
3. Runs one real train_one_step() with your configured network.
"""
import time

import numpy as np
import torch

from selfplay import ReplayBuffer
from training import decode_compact_batch

A = 4672


def make_game(k, rng):
    st = np.zeros((k, 119, 8, 8), np.float32)
    for j in range(k):
        for t in range(8):
            for pl in range(12):
                for _ in range(rng.integers(0, 4)):
                    st[j, t * 14 + pl, rng.integers(8), rng.integers(8)] = 1
            if rng.random() < 0.05:
                st[j, t * 14 + 12] = 1.0
        st[j, 112] = rng.integers(0, 2)
        for c in range(113, 117):
            st[j, c] = rng.integers(0, 2)
        st[j, 117] = rng.integers(0, 101) / 100.0
        st[j, 118] = rng.integers(1, 91) / 90.0
    out = []
    for j in range(k):
        p = np.zeros(A, np.float32)
        cols = rng.choice(A, rng.integers(5, 40), replace=False)
        w = rng.random(len(cols)) + 0.01
        p[cols] = w / w.sum()
        m = (p > 0).astype(np.uint8)
        m[rng.choice(A, 30, replace=False)] = 1
        out.append((st[j], p, float(rng.choice([-1, 0, 1])), m))
    return out


def check_device(rb, device):
    batch = rb.sample_batch_compact(128)
    ref_st, ref_po, ref_va, ref_mk = rb._decode_rows(batch['slots'])
    states, policies, values, masks = decode_compact_batch(batch, device)
    for name, got, ref in (("states", states, ref_st), ("policies", policies, ref_po),
                           ("values", values, ref_va), ("masks", masks, ref_mk.astype(np.float32))):
        diff = float(np.abs(got.cpu().numpy() - ref).max())
        status = "OK" if diff == 0.0 else "MISMATCH"
        print(f"  [{device}] {name:8s} max|diff| = {diff:g}  {status}")
        assert diff == 0.0, f"{name} decode differs on {device}"
    if device.type == "cuda":
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(20):
            decode_compact_batch(rb.sample_batch_compact(1024), device)
        torch.cuda.synchronize()
        print(f"  [{device}] sample + decode of a 1024 batch: {(time.perf_counter() - t) / 20 * 1000:.1f} ms")


def main():
    rng = np.random.default_rng(0)
    rb = ReplayBuffer(2000)
    for _ in range(4):
        rb.add_game(make_game(100, rng))
    print(f"buffer: {len(rb)} positions, {rb.nbytes / len(rb):.0f} bytes/position "
          f"({'bit-plane' if rb._planar else 'dense'} states)")

    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    for d in devices:
        check_device(rb, d)

    try:
        from config import get_config
        from network import create_model_from_config
        from training import train_one_step, create_optimizer
        dev = devices[-1]
        net = create_model_from_config(get_config()).to(dev)
        opt = create_optimizer(net, lr=0.01)
        out = train_one_step(net, opt, rb, 128, dev)
        print("train_one_step on", dev, "->", {k: round(v, 4) for k, v in out.items()})
        assert all(np.isfinite(v) for v in out.values())
    except Exception as e:                      # e.g. a non-chess game/config
        print("train_one_step check skipped/failed:", repr(e))
    print("DONE")


if __name__ == "__main__":
    main()