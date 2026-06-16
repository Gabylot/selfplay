"""Profiler for MCTS performance optimization.

Measures:
- Simulations per second
- Time breakdown: selection vs expansion vs backpropagation
- Network call latency (single vs batched) with statistics (min/max/stdev)
- Board.copy() and legal_moves generation cost
- CPU vs GPU inference comparison
- GPU shader pre-warming effect

Run modes:
    python profiler.py              # Default: baseline profile at 200 sims
    python profiler.py --batch      # Profile with batch sizes [1, 4, 8, 16, 32]
    python profiler.py --sweep      # Sweep num_simulations [200, 400, 600, 800]
    python profiler.py --gpu        # GPU profiling (requires torch_directml)
    python profiler.py --compare    # CPU vs GPU comparison across batch sizes
    python profiler.py --latency    # Detailed latency stats (min/max/stdev)
    python profiler.py --all        # Run all profiling modes
"""

import argparse
import time
import statistics
import numpy as np
import chess
from typing import Tuple, List, Optional
from dataclasses import dataclass, field

from encoding import board_to_tensor, get_legal_move_mask, move_to_policy_index, NUM_ACTIONS
from network import AlphaZeroNet, create_model_from_config
from config import Config, get_config
from mcts import MCTS, MCTSNode


@dataclass
class ProfileResult:
    """Results from a single profiling run."""
    label: str
    num_simulations: int
    batch_size: int
    sims_per_sec: float = 0.0
    total_time: float = 0.0
    network_time: float = 0.0
    legal_moves_time: float = 0.0
    board_copy_time: float = 0.0
    selection_time: float = 0.0
    backprop_time: float = 0.0
    expand_other_time: float = 0.0
    total_expand_time: float = 0.0
    num_expansions: int = 0
    num_network_calls: int = 0
    avg_depth: float = 0.0
    max_depth: int = 0
    visit_counts: List[int] = field(default_factory=list)


@dataclass
class LatencyResult:
    """Detailed latency statistics for a single configuration."""
    label: str
    device: str
    batch_size: int
    num_samples: int
    mean_ms: float
    median_ms: float
    stdev_ms: float
    min_ms: float
    max_ms: float
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    throughput_sps: float = 0.0  # samples per second


# ─────────────────────────────────────────────────────────────────────────────
# Board helpers
# ─────────────────────────────────────────────────────────────────────────────

def create_test_board() -> chess.Board:
    """Create a standard starting position for profiling."""
    return chess.Board()


def create_midgame_board() -> chess.Board:
    """Create a midgame position for more realistic profiling."""
    board = chess.Board()
    moves = ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6"]
    for m in moves:
        board.push(chess.Move.from_uci(m))
    return board


# ─────────────────────────────────────────────────────────────────────────────
# Device helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_cpu_device():
    import torch
    return torch.device("cpu")


def get_gpu_device():
    try:
        import torch_directml
        return torch_directml.device()
    except ImportError:
        print("[WARN] torch_directml not installed — GPU profiling unavailable")
        return None


def move_network_to_device(network, device):
    """Return a copy of the network on the given device."""
    import torch
    net = AlphaZeroNet(
        num_residual_blocks=network.num_residual_blocks,
        num_filters=network.num_filters,
    )
    net.load_state_dict(network.state_dict())
    net = net.to(device)
    net.eval()
    return net


# ─────────────────────────────────────────────────────────────────────────────
# Latency measurement
# ─────────────────────────────────────────────────────────────────────────────

def measure_latency(func, num_samples=200, warmup=20, label="", device="cpu",
                    batch_size=1) -> LatencyResult:
    """Measure latency of a callable with full statistics.

    Runs *warmup* iterations first (discarded), then *num_samples* timed
    iterations.  Reports mean, median, stdev, min, max, p95, p99.
    """
    import torch

    # Warmup (forces any lazy compilation / cache population)
    for _ in range(warmup):
        func()

    # Timed runs
    times_ms = []
    for _ in range(num_samples):
        t0 = time.perf_counter()
        func()
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    times_ms = np.array(times_ms)
    sorted_times = np.sort(times_ms)
    p95_idx = int(0.95 * len(sorted_times))
    p99_idx = int(0.99 * len(sorted_times))

    total_sec = times_ms.sum() / 1000.0
    throughput = (num_samples * batch_size) / total_sec if total_sec > 0 else 0

    return LatencyResult(
        label=label,
        device=device,
        batch_size=batch_size,
        num_samples=num_samples,
        mean_ms=float(np.mean(times_ms)),
        median_ms=float(np.median(times_ms)),
        stdev_ms=float(np.std(times_ms)),
        min_ms=float(np.min(times_ms)),
        max_ms=float(np.max(times_ms)),
        p95_ms=float(sorted_times[p95_idx]),
        p99_ms=float(sorted_times[p99_idx]),
        throughput_sps=throughput,
    )


def profile_network_latency_single(network, board, device_label="cpu",
                                   num_samples=200) -> LatencyResult:
    """Measure single-inference latency."""
    import torch
    device = next(network.parameters()).device
    state = board_to_tensor(board)

    def run():
        with torch.no_grad():
            x = torch.from_numpy(state).unsqueeze(0).float().to(device)
            policy_logits, value = network(x)

    return measure_latency(run, num_samples=num_samples, label=f"single_{device_label}",
                           device=device_label, batch_size=1)


def profile_network_latency_batch(network, board, batch_size, device_label="cpu",
                                  num_samples=200) -> LatencyResult:
    """Measure batched inference latency."""
    import torch
    device = next(network.parameters()).device
    states = np.stack([board_to_tensor(board)] * batch_size, axis=0)

    def run():
        with torch.no_grad():
            x = torch.from_numpy(states).float().to(device)
            policy_logits, value = network(x)

    return measure_latency(run, num_samples=num_samples,
                           label=f"batch{batch_size}_{device_label}",
                           device=device_label, batch_size=batch_size)


# ─────────────────────────────────────────────────────────────────────────────
# GPU warmup profiling
# ─────────────────────────────────────────────────────────────────────────────

def profile_gpu_warmup(network, board, batch_sizes=None, num_samples=100):
    """Compare GPU latency before and after shader pre-warming."""
    if batch_sizes is None:
        batch_sizes = [1, 8, 16, 32, 64, 128]

    device = get_gpu_device()
    if device is None:
        return []

    import torch
    from gpu_server import warmup_shaders

    net_gpu = move_network_to_device(network, device)
    results = []

    # Measure BEFORE pre-warming (cold — may trigger lazy compilation)
    print("  Measuring cold GPU latency (before shader pre-warming)...")
    cold_results = {}
    for bs in batch_sizes:
        if bs > 128:
            continue
        states = np.stack([board_to_tensor(board)] * bs, axis=0)
        def run_cold(bss=bs, sts=states):
            with torch.no_grad():
                x = torch.from_numpy(sts).float().to(device)
                _ = net_gpu(x)
        lr = measure_latency(run_cold, num_samples=10, warmup=0,
                             label=f"cold_gpu_b{bs}", device="gpu", batch_size=bs)
        cold_results[bs] = lr

    # Pre-warm
    print(f"  Pre-warming shaders for batch sizes: {batch_sizes}")
    t0 = time.perf_counter()
    warmup_shaders(net_gpu, device, batch_sizes)
    warmup_time = (time.perf_counter() - t0) * 1000
    print(f"  Pre-warming completed in {warmup_time:.0f} ms")

    # Measure AFTER pre-warming (warm — cached fast path)
    print("  Measuring warm GPU latency (after shader pre-warming)...")
    for bs in batch_sizes:
        if bs > 128:
            continue
        states = np.stack([board_to_tensor(board)] * bs, axis=0)
        def run_warm(bss=bs, sts=states):
            with torch.no_grad():
                x = torch.from_numpy(sts).float().to(device)
                _ = net_gpu(x)
        lr = measure_latency(run_warm, num_samples=num_samples, warmup=10,
                             label=f"warm_gpu_b{bs}", device="gpu", batch_size=bs)
        lr.cold_mean_ms = cold_results[bs].mean_ms
        lr.cold_max_ms = cold_results[bs].max_ms
        lr.warmup_time_ms = warmup_time
        results.append(lr)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# CPU vs GPU comparison
# ─────────────────────────────────────────────────────────────────────────────

def profile_cpu_vs_gpu(network, board, batch_sizes=None, num_samples=200):
    """Compare CPU and GPU latency across batch sizes."""
    if batch_sizes is None:
        batch_sizes = [1, 8, 16, 32, 64, 128]

    import torch
    from gpu_server import warmup_shaders

    results = []

    # CPU
    print("  Profiling CPU...")
    net_cpu = move_network_to_device(network, get_cpu_device())
    for bs in batch_sizes:
        lr = profile_network_latency_batch(net_cpu, board, bs, "cpu", num_samples)
        results.append(lr)

    # GPU
    device = get_gpu_device()
    if device is None:
        return results

    print("  Profiling GPU...")
    net_gpu = move_network_to_device(network, device)

    # Pre-warm shaders
    print(f"  Pre-warming shaders...")
    warmup_shaders(net_gpu, device, batch_sizes)
    print(f"  Pre-warming done")

    for bs in batch_sizes:
        lr = profile_network_latency_batch(net_gpu, board, bs, "gpu", num_samples)
        results.append(lr)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# MCTS end-to-end profiling
# ─────────────────────────────────────────────────────────────────────────────

def profile_mcts_overall(
    network: AlphaZeroNet,
    board: chess.Board,
    num_simulations: int,
    batch_size: int = 1,
    c_virtual_loss: float = 0.5,
    label: str = "",
    num_runs: int = 1
) -> List[ProfileResult]:
    """Run MCTS end-to-end and measure overall timing."""
    results = []

    for run in range(num_runs):
        mcts = MCTS(
            network=network,
            num_simulations=num_simulations,
            c_puct=1.5,
            dirichlet_alpha=0.3,
            dirichlet_epsilon=0.25,
            batch_size=batch_size,
            c_virtual_loss=c_virtual_loss,
        )

        root = mcts.get_root(board)

        t_start = time.perf_counter()
        visit_policy, best_move, stats = mcts.search(root)
        t_end = time.perf_counter()

        total_time = t_end - t_start

        visit_counts = []
        for action_idx, child in root.children.items():
            if child.N > 0:
                visit_counts.append(child.N)
        visit_counts.sort(reverse=True)

        run_label = f"{label}" if num_runs == 1 else f"{label}_run{run+1}"

        result = ProfileResult(
            label=run_label,
            num_simulations=num_simulations,
            batch_size=batch_size,
            sims_per_sec=num_simulations / total_time if total_time > 0 else 0,
            total_time=total_time,
            num_expansions=len(root.children),
            num_network_calls=0,
            avg_depth=stats.get('avg_depth', 0),
            max_depth=stats.get('max_depth', 0),
            visit_counts=visit_counts,
        )

        results.append(result)

    return results


def profile_baseline(network: AlphaZeroNet, num_simulations: int = 200,
                     num_runs: int = 3) -> List[ProfileResult]:
    """Profile baseline MCTS performance (batch_size=1)."""
    print(f"\n=== Baseline Profile (batch_size=1, {num_simulations} sims, {num_runs} runs) ===")

    board = create_test_board()
    results = []

    for run in range(num_runs):
        result_list = profile_mcts_overall(
            network, board, num_simulations, batch_size=1,
            label=f"baseline_run{run+1}", num_runs=1
        )
        results.extend(result_list)

    return results


def profile_batch_sweep(network: AlphaZeroNet, num_simulations: int = 200,
                        batch_sizes: List[int] = None) -> List[ProfileResult]:
    """Profile across different batch sizes."""
    if batch_sizes is None:
        batch_sizes = [1, 4, 8, 16, 32]

    print(f"\n=== Batch Size Sweep ({num_simulations} sims) ===")

    board = create_test_board()
    results = []

    for bs in batch_sizes:
        result_list = profile_mcts_overall(
            network, board, num_simulations, batch_size=bs,
            label=f"batch_{bs}"
        )
        results.extend(result_list)

    return results


def profile_sims_sweep(network: AlphaZeroNet,
                       sim_values: List[int] = None,
                       batch_size: int = 8) -> List[ProfileResult]:
    """Profile across different num_simulations values."""
    if sim_values is None:
        sim_values = [200, 400, 600, 800]

    print(f"\n=== Num Simulations Sweep (batch={batch_size}) ===")

    board = create_test_board()
    results = []

    for sims in sim_values:
        result_list = profile_mcts_overall(
            network, board, sims, batch_size=batch_size,
            label=f"sims_{sims}_batch{batch_size}"
        )
        results.extend(result_list)

    return results


def profile_separate_costs(network: AlphaZeroNet, board: chess.Board,
                           num_simulations: int = 200, batch_size: int = 1,
                           num_measurements: int = 100) -> dict:
    """Measure individual operation costs in isolation."""
    results = {}

    state = board_to_tensor(board)
    times = []
    for _ in range(num_measurements):
        t0 = time.perf_counter()
        policy, value = network.predict(state)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    results['single_network_call_us'] = 1e6 * np.mean(times)

    if batch_size > 1:
        states = np.stack([board_to_tensor(board)] * batch_size, axis=0)
        times = []
        for _ in range(num_measurements):
            t0 = time.perf_counter()
            policies, values = network.predict_batch(states)
            t1 = time.perf_counter()
            times.append(t1 - t0)
        results[f'batch{batch_size}_network_call_us'] = 1e6 * np.mean(times)
        results[f'batch{batch_size}_speedup_vs_single'] = (
            results['single_network_call_us'] * batch_size / results[f'batch{batch_size}_network_call_us']
        )

    times_stack_true = []
    times_stack_false = []
    for _ in range(num_measurements):
        b = board.copy()
        t0 = time.perf_counter()
        b2 = b.copy(stack=True)
        t1 = time.perf_counter()
        times_stack_true.append(t1 - t0)

        t0 = time.perf_counter()
        b3 = b.copy(stack=False)
        t1 = time.perf_counter()
        times_stack_false.append(t1 - t0)
    results['copy_stack_true_us'] = 1e6 * np.mean(times_stack_true)
    results['copy_stack_false_us'] = 1e6 * np.mean(times_stack_false)
    results['copy_speedup'] = (
        results['copy_stack_true_us'] / results['copy_stack_false_us']
    )

    times = []
    for _ in range(num_measurements):
        t0 = time.perf_counter()
        moves = list(board.legal_moves)
        t1 = time.perf_counter()
        _ = len(moves)
        times.append(t1 - t0)
    results['legal_moves_us'] = 1e6 * np.mean(times)
    results['num_legal_moves'] = len(list(board.legal_moves))

    times = []
    for _ in range(num_measurements // 10):
        t0 = time.perf_counter()
        mask = get_legal_move_mask(board)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    results['legal_mask_full_us'] = 1e6 * np.mean(times)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Print helpers
# ─────────────────────────────────────────────────────────────────────────────

def print_profile_table(results: List[ProfileResult]):
    """Print a formatted table of profile results."""
    print(f"\n{'Label':<30} {'Sims':>5} {'Batch':>5} {'Sims/s':>8} {'Total(s)':>8} "
          f"{'#Moves':>7} {'AvgD':>5} {'Top-1%':>7} {'Top-3%':>7}")
    print("-" * 90)
    for r in results:
        total_moves = len(r.visit_counts)
        if total_moves > 0:
            total_visits = sum(r.visit_counts)
            top1_pct = 100 * r.visit_counts[0] / total_visits if total_visits > 0 else 0
            top3_pct = 100 * sum(r.visit_counts[:3]) / total_visits if len(r.visit_counts) >= 3 else 100 * sum(r.visit_counts) / total_visits
        else:
            top1_pct = top3_pct = 0
        print(f"{r.label:<30} {r.num_simulations:>5} {r.batch_size:>5} {r.sims_per_sec:>8.1f} "
              f"{r.total_time:>8.3f} {total_moves:>7} {r.avg_depth:>5.1f} "
              f"{top1_pct:>6.1f}% {top3_pct:>6.1f}%")


def print_visit_summary(results: List[ProfileResult]):
    """Print visit distribution summary."""
    print(f"\n{'Label':<30} {'#Moves>5%':>12} {'#Moves>0':>10} {'#MovesTotal':>14} {'Top-1%':>8} "
          f"{'Top-3%':>8} {'Top-5%':>8}")
    print("-" * 100)
    for r in results:
        total_moves = len(r.visit_counts)
        if total_moves > 0:
            total_visits = sum(r.visit_counts)
            top1_pct = 100 * r.visit_counts[0] / total_visits if total_visits > 0 else 0
            top3_pct = 100 * sum(r.visit_counts[:3]) / total_visits if len(r.visit_counts) >= 3 else 100 * sum(r.visit_counts) / total_visits
            top5_pct = 100 * sum(r.visit_counts[:5]) / total_visits if len(r.visit_counts) >= 5 else 100 * sum(r.visit_counts) / total_visits
            moves_gt_5pct = sum(1 for c in r.visit_counts if 100 * c / total_visits > 5)
            moves_gt_0 = sum(1 for c in r.visit_counts if c > 0)
            print(f"{r.label:<30} {moves_gt_5pct:>12} {moves_gt_0:>10} {total_moves:>14} "
                  f"{top1_pct:>7.1f}% {top3_pct:>7.1f}% {top5_pct:>7.1f}%")


def print_separate_costs(costs: dict):
    """Print the isolated cost measurements."""
    print("\n--- Isolated Operation Costs ---")
    print(f"  Single network call:   {costs['single_network_call_us']:.1f} us")

    for key, val in costs.items():
        if key.startswith('batch') and 'speedup' in key:
            print(f"  {key}: {val:.1f}x")
        elif key.startswith('batch') and 'us' in key:
            print(f"  {key}: {val:.1f} us")

    print(f"  board.copy(stack=True):  {costs['copy_stack_true_us']:.1f} us")
    print(f"  board.copy(stack=False): {costs['copy_stack_false_us']:.1f} us")
    print(f"  copy speedup:            {costs['copy_speedup']:.1f}x")
    print(f"  legal_moves gen:         {costs['legal_moves_us']:.1f} us ({costs['num_legal_moves']} moves)")
    print(f"  get_legal_move_mask:      {costs['legal_mask_full_us']:.1f} us")


def print_latency_table(results: List[LatencyResult]):
    """Print a formatted table of latency results with full statistics."""
    print(f"\n{'Label':<28} {'Device':<6} {'BS':>4} {'Mean':>8} {'Median':>8} "
          f"{'Stdev':>8} {'Min':>8} {'Max':>8} {'P95':>8} {'P99':>8} {'Throughput':>12}")
    print("-" * 120)
    for r in results:
        print(f"{r.label:<28} {r.device:<6} {r.batch_size:>4} "
              f"{r.mean_ms:>7.2f}ms {r.median_ms:>7.2f}ms "
              f"{r.stdev_ms:>7.2f}ms {r.min_ms:>7.2f}ms {r.max_ms:>7.2f}ms "
              f"{r.p95_ms:>7.2f}ms {r.p99_ms:>7.2f}ms "
              f"{r.throughput_sps:>10.0f}/s")


def print_cpu_vs_gpu_comparison(results: List[LatencyResult]):
    """Print a side-by-side CPU vs GPU comparison table."""
    # Group by batch size
    by_bs = {}
    for r in results:
        by_bs.setdefault(r.batch_size, {})[r.device] = r

    print(f"\n{'='*100}")
    print(f"  CPU vs GPU Latency Comparison (pre-warmed)")
    print(f"{'='*100}")
    print(f"{'Batch':>6} {'CPU mean':>10} {'GPU mean':>10} {'Speedup':>8} "
          f"{'CPU max':>10} {'GPU max':>10} {'GPU stdev':>10} {'GPU P99':>10}")
    print("-" * 100)

    for bs in sorted(by_bs.keys()):
        cpu = by_bs[bs].get('cpu')
        gpu = by_bs[bs].get('gpu')
        if cpu and gpu:
            speedup = cpu.mean_ms / gpu.mean_ms if gpu.mean_ms > 0 else 0
            print(f"{bs:>6} "
                  f"{cpu.mean_ms:>8.2f}ms {gpu.mean_ms:>8.2f}ms {speedup:>7.2f}x "
                  f"{cpu.max_ms:>8.2f}ms {gpu.max_ms:>8.2f}ms "
                  f"{gpu.stdev_ms:>8.2f}ms {gpu.p99_ms:>8.2f}ms")
        elif cpu:
            print(f"{bs:>6} {cpu.mean_ms:>8.2f}ms {'N/A':>10} {'N/A':>8} "
                  f"{cpu.max_ms:>8.2f}ms {'N/A':>10} {'N/A':>10} {'N/A':>10}")
        elif gpu:
            print(f"{bs:>6} {'N/A':>10} {gpu.mean_ms:>8.2f}ms {'N/A':>8} "
                  f"{'N/A':>10} {gpu.max_ms:>8.2f}ms {gpu.stdev_ms:>8.2f}ms {gpu.p99_ms:>8.2f}ms")


def print_warmup_comparison(results: List[LatencyResult]):
    """Print GPU cold vs warm (pre-warmed) comparison."""
    if not results:
        print("\n[SKIP] No GPU warmup results (torch_directml not available)")
        return

    print(f"\n{'='*100}")
    print(f"  GPU Shader Pre-warming Effect")
    print(f"{'='*100}")
    print(f"{'Batch':>6} {'Cold mean':>10} {'Cold max':>10} {'Warm mean':>10} "
          f"{'Warm max':>10} {'Warm stdev':>10} {'Variance elim':>14}")
    print("-" * 100)

    for r in results:
        cold_mean = getattr(r, 'cold_mean_ms', 0)
        cold_max = getattr(r, 'cold_max_ms', 0)
        variance_ratio = cold_max / r.mean_ms if r.mean_ms > 0 else 0
        print(f"{r.batch_size:>6} "
              f"{cold_mean:>8.2f}ms {cold_max:>8.2f}ms "
              f"{r.mean_ms:>8.2f}ms {r.max_ms:>8.2f}ms "
              f"{r.stdev_ms:>8.2f}ms "
              f"{variance_ratio:>12.1f}x")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MCTS & Inference Profiler")
    parser.add_argument('--batch', action='store_true', help='Run batch size sweep')
    parser.add_argument('--sweep', action='store_true', help='Run num_simulations sweep')
    parser.add_argument('--gpu', action='store_true', help='GPU profiling (requires torch_directml)')
    parser.add_argument('--compare', action='store_true', help='CPU vs GPU comparison')
    parser.add_argument('--latency', action='store_true', help='Detailed latency stats')
    parser.add_argument('--warmup', action='store_true', help='GPU shader pre-warming comparison')
    parser.add_argument('--all', action='store_true', help='Run all profiling modes')
    parser.add_argument('--sims', type=int, default=200, help='Num simulations')
    parser.add_argument('--runs', type=int, default=1, help='Num runs per config')
    parser.add_argument('--midgame', action='store_true', help='Use midgame position')
    parser.add_argument('--costs', action='store_true', help='Measure individual operation costs')
    parser.add_argument('--detailed', action='store_true', help='Show visit summary')
    parser.add_argument('--samples', type=int, default=200, help='Latency measurement samples')
    parser.add_argument('--batch-sizes', type=str, default="1,8,16,32,64,128",
                        help='Comma-separated batch sizes for latency tests')
    args = parser.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    # Create a small network for profiling (faster)
    print("Creating network...")
    network = AlphaZeroNet(num_residual_blocks=4, num_filters=64)
    board = create_midgame_board() if args.midgame else create_test_board()
    if args.midgame:
        print(f"Using midgame position, FEN: {board.fen()}")

    # ── Isolated cost measurements ──
    if args.costs or args.all:
        costs = profile_separate_costs(network, board, args.sims, batch_size=8)
        print_separate_costs(costs)

    # ── MCTS batch sweep ──
    if args.all or args.batch:
        results = profile_batch_sweep(network, args.sims)
        print_profile_table(results)
        if args.detailed:
            print_visit_summary(results)

    # ── MCTS sims sweep ──
    if args.all or args.sweep:
        results = profile_sims_sweep(network)
        print_profile_table(results)
        if args.detailed:
            print_visit_summary(results)

    # ── Latency stats (CPU) ──
    if args.latency or args.all:
        print(f"\n=== Network Latency — CPU ({args.samples} samples) ===")
        lat_results = []
        lat_results.append(profile_network_latency_single(network, board, "cpu", args.samples))
        for bs in batch_sizes:
            lat_results.append(profile_network_latency_batch(network, board, bs, "cpu", args.samples))
        print_latency_table(lat_results)

    # ── GPU profiling ──
    if args.gpu or args.all:
        device = get_gpu_device()
        if device is None:
            print("[SKIP] GPU profiling — torch_directml not available")
        else:
            import torch
            from gpu_server import warmup_shaders

            print(f"\n=== GPU Profiling ===")
            net_gpu = move_network_to_device(network, device)

            # Pre-warm
            print(f"Pre-warming shaders for: {batch_sizes}")
            t0 = time.perf_counter()
            warmup_shaders(net_gpu, device, batch_sizes)
            warmup_ms = (time.perf_counter() - t0) * 1000
            print(f"Pre-warming done in {warmup_ms:.0f} ms\n")

            # Latency table
            lat_results = []
            lat_results.append(profile_network_latency_single(net_gpu, board, "gpu", args.samples))
            for bs in batch_sizes:
                lat_results.append(profile_network_latency_batch(net_gpu, board, bs, "gpu", args.samples))
            print_latency_table(lat_results)

    # ── CPU vs GPU comparison ──
    if args.compare or args.all:
        print(f"\n=== CPU vs GPU Comparison ({args.samples} samples) ===")
        comp_results = profile_cpu_vs_gpu(network, board, batch_sizes, args.samples)
        print_cpu_vs_gpu_comparison(comp_results)

    # ── GPU warmup effect ──
    if args.warmup or args.all:
        print(f"\n=== GPU Shader Pre-warming Effect ===")
        warmup_results = profile_gpu_warmup(network, board, batch_sizes, args.samples)
        print_warmup_comparison(warmup_results)

    # ── Default: baseline profile ──
    if not any([args.batch, args.sweep, args.gpu, args.compare,
                args.latency, args.warmup, args.all]):
        results = profile_baseline(network, args.sims, args.runs)
        print_profile_table(results)
        if args.detailed:
            print_visit_summary(results)


if __name__ == "__main__":
    main()