"""Test different batch sizes to find optimal IPC/throughput trade-off.

This runs a single MCTS search (no multiprocessing) for each batch size
and reports the resulting throughput.  The goal is to find the batch_size
that minimises time per simulation.

Results are saved to 'optimization_results.txt'.
"""

import time
import numpy as np
import chess

from encoding import board_to_tensor
from network import create_model_from_config
from config import get_config
from mcts import MCTS


def test_batch_sizes(batch_sizes, num_simulations=1024, num_runs=2):
    cfg = get_config()
    net = create_model_from_config(cfg)
    net.eval()
    board = chess.Board()

    print(f"Network: {cfg.network.num_residual_blocks} blocks, "
          f"{cfg.network.num_filters} filters")
    print(f"Simulations: {num_simulations}")
    print()
    print(f"{'Batch':>6} {'Total(s)':>10} {'Sims/s':>8} {'Calls':>6} "
          f"{'Net(s)':>8} {'Avg/call(ms)':>14} {'%Net':>6}")
    print("-" * 68)

    results = []
    for bs in batch_sizes:
        run_times = []
        run_net_times = []
        run_calls = []
        for _ in range(num_runs):
            mcts = MCTS(
                net,
                num_simulations=num_simulations,
                batch_size=bs,
            )
            mcts.profile = True
            root = mcts.get_root(board)
            t0 = time.perf_counter()
            _, _, stats = mcts.search(root)
            elapsed = time.perf_counter() - t0
            pf = stats.get('profiling', {})

            run_times.append(elapsed)
            run_net_times.append(pf.get('network_batch_predict', 0))
            run_calls.append(pf.get('network_batch_calls', 0))

        avg_t = np.mean(run_times)
        avg_net = np.mean(run_net_times)
        avg_calls = int(np.mean(run_calls))
        sims_s = num_simulations / avg_t
        net_pct = 100.0 * avg_net / avg_t
        avg_call_ms = (avg_net / avg_calls) * 1000 if avg_calls > 0 else 0

        print(f"{bs:>6} {avg_t:>10.4f} {sims_s:>8.0f} {avg_calls:>6} "
              f"{avg_net:>8.4f} {avg_call_ms:>13.3f} {net_pct:>5.1f}%")
        results.append((bs, avg_t, sims_s, avg_calls, avg_call_ms))

    # Find best
    best = max(results, key=lambda r: r[2])
    print(f"\n--> Best batch_size: {best[0]} ({best[2]:.0f} sims/s)")

    # Save to file
    with open("optimization_results.txt", "w") as f:
        f.write(f"Batch size optimization for {cfg.network.num_filters}f "
                f"{cfg.network.num_residual_blocks}b network\n")
        f.write(f"Simulations: {num_simulations}\n\n")
        f.write(f"{'Batch':>6} {'Sims/s':>8} {'Avg/call(ms)':>14}\n")
        for bs, _, sims_s, _, avg_ms in results:
            f.write(f"{bs:>6} {sims_s:>8.0f} {avg_ms:>13.3f}\n")
        f.write(f"\nBest: batch_size={best[0]} ({best[2]:.0f} sims/s)\n")
        f.write(f"Ideal (no IPC): ~{1000/1.3 * 32:.0f} sims/s at bs=32\n")

    return results


if __name__ == "__main__":
    # Test batch sizes from 1 to 512 in powers of 2, plus 48 for a middle ground
    batch_sizes = [1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 192, 256, 384, 512]
    test_batch_sizes(batch_sizes)