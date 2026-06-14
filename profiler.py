"""Profiler for MCTS performance optimization.

Measures:
- Simulations per second
- Time breakdown: selection vs expansion (sub-divided) vs backpropagation
- Network call latency (single vs batched)
- Board.copy() and legal_moves generation cost

Run modes:
    python profiler.py              # Default: baseline profile at 200 sims
    python profiler.py --batch     # Profile with batch sizes [1, 4, 8, 16, 32]
    python profiler.py --sweep     # Sweep num_simulations [200, 400, 600, 800]
    python profiler.py --all       # Run all profiling modes
"""

import argparse
import time
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


def create_test_board() -> chess.Board:
    """Create a standard starting position for profiling."""
    return chess.Board()


def create_midgame_board() -> chess.Board:
    """Create a midgame position for more realistic profiling."""
    board = chess.Board()
    # Play some standard opening moves (full UCI notation)
    moves = ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6"]
    for m in moves:
        board.push(chess.Move.from_uci(m))
    return board


def profile_separate_costs(network: AlphaZeroNet, board: chess.Board,
                           num_simulations: int = 200, batch_size: int = 1,
                           num_measurements: int = 100) -> dict:
    """Measure individual operation costs in isolation.
    
    This avoids the instrumentation overhead issues by measuring each
    operation independently.
    """
    results = {}
    
    # Measure single network call latency
    state = board_to_tensor(board)
    times = []
    for _ in range(num_measurements):
        t0 = time.perf_counter()
        policy, value = network.predict(state)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    results['single_network_call_us'] = 1e6 * np.mean(times)
    
    # Measure batch network call latency for various batch sizes
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
    
    # Measure board.copy() with and without stack
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
    
    # Measure legal_moves generation
    times = []
    for _ in range(num_measurements):
        t0 = time.perf_counter()
        moves = list(board.legal_moves)
        t1 = time.perf_counter()
        _ = len(moves)  # prevent optimization
        times.append(t1 - t0)
    results['legal_moves_us'] = 1e6 * np.mean(times)
    results['num_legal_moves'] = len(list(board.legal_moves))
    
    # Measure get_legal_move_mask (full version)
    times = []
    for _ in range(num_measurements // 10):  # fewer iterations, it's slower
        t0 = time.perf_counter()
        mask = get_legal_move_mask(board)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    results['legal_mask_full_us'] = 1e6 * np.mean(times)
    
    return results


def profile_mcts_overall(
    network: AlphaZeroNet,
    board: chess.Board,
    num_simulations: int,
    batch_size: int = 1,
    c_virtual_loss: float = 0.5,
    label: str = "",
    num_runs: int = 1
) -> List[ProfileResult]:
    """Run MCTS end-to-end and measure overall timing.
    
    Uses a simple timing wrapper instead of class instrumentation,
    which could interfere with the batched code path.
    """
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
        
        # Get visit distribution
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
            num_network_calls=0,  # Not measured at this granularity
            avg_depth=stats.get('avg_depth', 0),
            max_depth=stats.get('max_depth', 0),
            visit_counts=visit_counts,
        )
        
        results.append(result)
    
    return results


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


def main():
    parser = argparse.ArgumentParser(description="MCTS Profiler")
    parser.add_argument('--batch', action='store_true', help='Run batch size sweep')
    parser.add_argument('--sweep', action='store_true', help='Run num_simulations sweep')
    parser.add_argument('--all', action='store_true', help='Run all profiling modes')
    parser.add_argument('--sims', type=int, default=200, help='Num simulations')
    parser.add_argument('--runs', type=int, default=1, help='Num runs per config')
    parser.add_argument('--midgame', action='store_true', help='Use midgame position')
    parser.add_argument('--costs', action='store_true', help='Measure individual operation costs')
    parser.add_argument('--detailed', action='store_true', help='Show visit summary')
    args = parser.parse_args()
    
    # Create a small network for profiling (faster)
    print("Creating network...")
    network = AlphaZeroNet(num_residual_blocks=4, num_filters=64)
    
    board = create_midgame_board() if args.midgame else create_test_board()
    if args.midgame:
        print(f"Using midgame position, FEN: {board.fen()}")
    
    # Isolated cost measurements (useful for understanding bottlenecks)
    if args.costs or args.all:
        costs = profile_separate_costs(network, board, args.sims, batch_size=8)
        print_separate_costs(costs)
    
    if args.all or args.batch:
        results = profile_batch_sweep(network, args.sims)
        print_profile_table(results)
        if args.detailed:
            print_visit_summary(results)
    
    if args.all or args.sweep:
        results = profile_sims_sweep(network)
        print_profile_table(results)
        if args.detailed:
            print_visit_summary(results)
    
    if not args.batch and not args.sweep and not args.all:
        # Default: baseline profile
        results = profile_baseline(network, args.sims, args.runs)
        print_profile_table(results)
        if args.detailed:
            print_visit_summary(results)


if __name__ == "__main__":
    main()