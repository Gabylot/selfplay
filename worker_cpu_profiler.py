"""Worker CPU profiler for AlphaZero self-play.

Instruments the hot paths in MCTS and the worker process to show
exactly where CPU time is spent during real self-play games.

Two usage modes:
  A) Standalone (creates its own network, runs on CPU):
       python worker_cpu_profiler.py --games 1 --sims 50 --blocks 2 --filters 32

  B) Standalone with GPU inference (uses the real GPU server pipeline):
       python worker_cpu_profiler.py --gpu --games 1 --sims 200

  C) Worker-integrated (runs inside a real _worker_process with config.yaml):
       Main process calls ParallelSelfPlay.dispatch_profile(network, num_games=3)
"""

import time
import chess
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional as _Optional

from mcts import MCTS


# ── Timers ───────────────────────────────────────────────────────────────────

@dataclass
class TimerSnapshot:
    """Accumulates timing statistics for a single phase."""
    calls: int = 0
    total: float = 0.0
    min: float = float('inf')
    max: float = 0.0
    mean: float = 0.0

    def record(self, elapsed: float):
        self.calls += 1
        self.total += elapsed
        if elapsed < self.min:
            self.min = elapsed
        if elapsed > self.max:
            self.max = elapsed

    def finalize(self):
        if self.calls > 0:
            self.mean = self.total / self.calls
        if self.min == float('inf'):
            self.min = 0.0


# ── Profiler class ──────────────────────────────────────────────────────────

class WorkerProfiler:
    """Attaches timers to an MCTS instance and records time spent in each phase.

    Patches two categories of methods:

    1. **MCTS instance methods** -- monkey-patched on the specific MCTS object.
       The original is an unbound function from ``type(mcts).__dict__``, so we
       call it as ``original(self.mcts, *args)``.

    2. **Python-chess class-level methods** -- patched globally on ``chess.Board``
       and ``chess.LegalMoveGenerator`` so *every* call in the worker is timed.
       Originals are stored and restored in ``detach()``.

    Usage:
        profiler = WorkerProfiler(mcts_instance)
        profiler.attach()
        # ... run games ...
        profiler.detach()
        profiler.print_report()
    """

    PHASES = [
        'selection',
        'network_eval',
        'backpropagation',
        'dirichlet_noise',
        'visit_policy',
        'board_to_tensor',
        'chess_copy',
        'chess_legal_moves',
        'chess_push',
        'chess_game_over',
        'chess_result',
    ]

    def __init__(self, mcts: MCTS):
        self.mcts = mcts
        self.timers: Dict[str, TimerSnapshot] = {
            name: TimerSnapshot() for name in self.PHASES
        }
        self._mcts_originals: Dict[str, object] = {}
        self._board_originals: Dict[str, object] = {}
        self._module_originals: Dict[str, object] = {}

    def attach(self):
        self._patch_mcts_methods()
        self._patch_chess_methods()
        self._patch_board_to_tensor()

    def detach(self):
        self._detach_mcts()
        self._detach_chess()
        self._detach_board_to_tensor()

    # MCTS patching

    def _patch_mcts_methods(self):
        m = self.mcts
        patches = [
            ('_select_child',                     'selection'),
            ('_collect_batch',                    'selection'),
            ('_evaluate_batch',                   'network_eval'),
            ('_expand_node',                      'network_eval'),
            ('_backpropagate',                    'backpropagation'),
            ('_backpropagate_with_virtual_loss',   'backpropagation'),
            ('_add_dirichlet_noise',              'dirichlet_noise'),
            ('_get_visit_policy',                 'visit_policy'),
            ('select_move_with_temperature',      'visit_policy'),
        ]
        for attr_name, timer_name in patches:
            self._patch_mcts_method(m, attr_name, timer_name)

    def _patch_mcts_method(self, obj, attr_name: str, timer_name: str):
        original = type(obj).__dict__.get(attr_name)
        if original is None:
            original = getattr(obj, attr_name)
        timer = self.timers[timer_name]
        self._mcts_originals[attr_name] = original

        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            result = original(self.mcts, *args, **kwargs)
            timer.record(time.perf_counter() - t0)
            return result

        setattr(obj, attr_name, wrapper)

    def _detach_mcts(self):
        m = self.mcts
        for name, original in self._mcts_originals.items():
            setattr(m, name, original)
        self._mcts_originals.clear()

    # Python-chess patching (class-level)

    def _patch_chess_methods(self):
        Board = chess.Board
        self._patch_board_class(Board, 'copy', 'chess_copy')
        self._patch_board_class(Board, 'push', 'chess_push')
        self._patch_board_class(Board, 'is_game_over', 'chess_game_over')
        self._patch_board_class(Board, 'result', 'chess_result')
        self._patch_legal_moves_iter()

    def _patch_board_class(self, cls, attr_name: str, timer_name: str):
        original = cls.__dict__.get(attr_name)
        if original is None:
            raise RuntimeError(f"Attribute {attr_name} not found on {cls.__name__}")
        timer = self.timers[timer_name]
        self._board_originals[attr_name] = original

        def wrapper(self_or_board, *args, **kwargs):
            t0 = time.perf_counter()
            result = original(self_or_board, *args, **kwargs)
            timer.record(time.perf_counter() - t0)
            return result

        setattr(cls, attr_name, wrapper)

    def _patch_legal_moves_iter(self):
        LegalGen = chess.LegalMoveGenerator
        orig_iter = LegalGen.__iter__
        timer = self.timers['chess_legal_moves']
        self._board_originals['legal_moves_iter'] = orig_iter

        def timed_iter(self_gen):
            t0 = time.perf_counter()
            result = orig_iter(self_gen)
            timer.record(time.perf_counter() - t0)
            return result

        LegalGen.__iter__ = timed_iter

        if hasattr(LegalGen, '__len__'):
            orig_len = LegalGen.__len__
            self._board_originals['legal_moves_len'] = orig_len

            def timed_len(self_gen):
                t0 = time.perf_counter()
                result = orig_len(self_gen)
                timer.record(time.perf_counter() - t0)
                return result

            LegalGen.__len__ = timed_len

    def _detach_chess(self):
        Board = chess.Board
        for attr_name, original in self._board_originals.items():
            if attr_name == 'legal_moves_iter':
                chess.LegalMoveGenerator.__iter__ = original
            elif attr_name == 'legal_moves_len':
                chess.LegalMoveGenerator.__len__ = original
            else:
                setattr(Board, attr_name, original)
        self._board_originals.clear()

    # board_to_tensor patching (module-level)

    def _patch_board_to_tensor(self):
        import encoding as encoding_mod
        self._module_originals['board_to_tensor'] = encoding_mod.board_to_tensor
        bt_timer = self.timers['board_to_tensor']

        def _timed_board_to_tensor(board):
            t0 = time.perf_counter()
            result = self._module_originals['board_to_tensor'](board)
            bt_timer.record(time.perf_counter() - t0)
            return result

        encoding_mod.board_to_tensor = _timed_board_to_tensor

    def _detach_board_to_tensor(self):
        if 'board_to_tensor' in self._module_originals:
            import encoding as encoding_mod
            encoding_mod.board_to_tensor = self._module_originals['board_to_tensor']
        self._module_originals.clear()

    # Report

    def finalize(self):
        for ts in self.timers.values():
            ts.finalize()

    def print_report(self, num_games: int = 1, num_simulations: int = 200,
                     batch_size: int = 1, total_moves: int = 0,
                     avg_moves_per_game: float = 0.0):
        self.finalize()
        total_time = sum(ts.total for ts in self.timers.values())
        sep = "-" * 65

        print()
        print("=" * 75)
        print(f"  Worker CPU Profile -- {num_games} game(s), "
              f"batch={batch_size}, sims={num_simulations}")
        print("=" * 75)

        if total_time == 0:
            print("  (no timing data collected)")
            return

        print()
        header = (f"{'Phase':<25} {'Calls':>8} {'Total(s)':>10} "
                  f"{'Mean(ms)':>10} {'%Total':>8}")
        print(header)
        print(sep)

        sorted_phases = sorted(self.PHASES,
                               key=lambda p: self.timers[p].total,
                               reverse=True)

        for phase in sorted_phases:
            ts = self.timers[phase]
            if ts.calls == 0:
                continue
            pct = 100.0 * ts.total / total_time if total_time > 0 else 0
            hot = "  *** HOT" if pct > 30 else ""
            print(f"{phase:<25} {ts.calls:>8} {ts.total:>10.4f} "
                  f"{ts.mean * 1000:>9.2f} {pct:>7.1f}%{hot}")

        accounted = sum(ts.total for ts in self.timers.values() if ts.calls > 0)
        other_time = total_time - accounted
        if other_time > 0.001:
            opct = 100.0 * other_time / total_time
            print(f"{'other':<25} {'-':>8} {other_time:>10.4f} "
                  f"{'-':>9} {opct:>7.1f}%")

        print(sep)
        print(f"{'TOTAL':<25} {'-':>8} {total_time:>10.4f} "
              f"{'-':>9} {'100.0%':>8}")
        print()

        if total_moves > 0:
            sims_per_sec = ((num_simulations * total_moves) / total_time
                            if total_time > 0 else 0)
            print(f"  Sims/sec: {sims_per_sec:.1f}    "
                  f"Games: {num_games}    "
                  f"Avg moves/game: {avg_moves_per_game:.1f}    "
                  f"Sims/move: {num_simulations}")
        print()

    def get_summary_dict(self) -> dict:
        self.finalize()
        total = sum(ts.total for ts in self.timers.values())
        return {
            phase: {
                'calls': self.timers[phase].calls,
                'total_s': self.timers[phase].total,
                'mean_ms': self.timers[phase].mean * 1000,
                'pct': (100.0 * self.timers[phase].total / total
                        if total > 0 else 0),
            }
            for phase in self.PHASES
        }


# ── Run profiled game ────────────────────────────────────────────────────────

def run_profiled_game(mcts_engine: MCTS,
                      **game_kwargs) -> Tuple[list, dict, WorkerProfiler]:
    profiler = WorkerProfiler(mcts_engine)
    profiler.attach()
    try:
        from selfplay import play_one_game
        game_data, game_info = play_one_game(mcts_engine, **game_kwargs)
    finally:
        profiler.detach()
    return game_data, game_info, profiler


# ── Standalone CPU profiling ────────────────────────────────────────────────

def profile_standalone(num_games: int = 3,
                       num_simulations: int = 200,
                       batch_size: int = 1,
                       num_filters: int = 64,
                       num_blocks: int = 4,
                       max_game_length: int = 150,
                       verbosity: int = 0):
    from network import AlphaZeroNet
    network = AlphaZeroNet(
        num_residual_blocks=num_blocks,
        num_filters=num_filters,
        num_policy_channels=32,
        num_value_channels=16,
        value_fc_size=256,
    )
    network.eval()
    _run_profile_loop(network, None, num_games, num_simulations,
                      batch_size, max_game_length, verbosity)


# ── Standalone GPU profiling ────────────────────────────────────────────────

def profile_gpu(num_games: int = 3,
                num_simulations: int = 200,
                batch_size: int = 64,
                max_game_length: int = 150,
                verbosity: int = 0):
    """Profile with real GPU inference server + InferenceClient.

    Starts a GPUInferenceServer subprocess, creates an InferenceClient
    per game (or reuses one), and profiles the MCTS with GPU inference.
    """
    import multiprocessing as mp
    from config import Config
    from config import get_config
    from gpu_server import GPUInferenceServer
    from inference_client import InferenceClient
    import torch

    # Build a full config from yaml (needed for GPU server params)
    # Use the default config with use_gpu=true
    cfg = get_config('config.yaml')
    if not hasattr(cfg, 'inference') or not getattr(cfg.inference, 'use_gpu', False):
        # Manually ensure GPU inference is enabled
        from types import SimpleNamespace
        cfg.inference = SimpleNamespace(
            use_gpu=True,
            max_batch=batch_size,
            max_wait_ms=3.0,
            prewarm_batch_sizes=[1, batch_size],
        )

    piece_values = {'P': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9}
    game_kwargs = dict(
        max_game_length=max_game_length,
        adjudicate_material=True,
        piece_values=piece_values,
        temp_threshold=30,
        temp_high=1.0,
        temp_low=0.1,
        adjudicate_graded=True,
        adjudicate_scaling=9.0,
    )

    # Create queues
    request_q = mp.Queue()
    weight_q = mp.Queue()
    response_qs = {0: mp.Queue(maxsize=256)}
    gpu_ready = mp.Event()
    gpu_shutdown = mp.Event()

    # Create and start GPU server
    server = GPUInferenceServer(
        config=cfg,
        request_queue=request_q,
        response_queues=response_qs,
        weight_queue=weight_q,
        ready_event=gpu_ready,
        shutdown_event=gpu_shutdown,
    )
    server_process = mp.Process(target=server.run, daemon=True)
    server_process.start()

    print("[GPU] Waiting for server warm-up...")
    gpu_ready.wait()
    print("[GPU] Server ready. Starting profiled games...")

    try:
        # Create inference client
        client = InferenceClient(0, request_q, response_qs[0])

        all_profilers: List[WorkerProfiler] = []
        total_moves = 0

        for game_idx in range(num_games):
            mcts_engine = MCTS(
                network=client,
                num_simulations=num_simulations,
                c_puct=2.5,
                dirichlet_alpha=0.3,
                dirichlet_epsilon=0.35,
                batch_size=batch_size,
                c_virtual_loss=0.1,
            )

            if verbosity >= 1:
                print(f"\n--- GPU Game {game_idx + 1}/{num_games} ---")

            try:
                game_data, game_info, profiler = run_profiled_game(
                    mcts_engine, **game_kwargs)
            except Exception as e:
                print(f"[GPU ERROR] Game {game_idx + 1} failed: {e}")
                import traceback
                traceback.print_exc()
                continue

            all_profilers.append(profiler)
            total_moves += game_info['length']

            if verbosity >= 1:
                print(f"  Result: {game_info['result_str']}  "
                      f"Moves: {game_info['length']}  "
                      f"Termination: {game_info['termination']}")

        if not all_profilers:
            print("[GPU ERROR] No games completed successfully.")
            return

        merged: Dict[str, TimerSnapshot] = {}
        for profiler in all_profilers:
            profiler.finalize()
            for phase, ts in profiler.timers.items():
                if phase not in merged:
                    merged[phase] = TimerSnapshot()
                merged[phase].calls += ts.calls
                merged[phase].total += ts.total
                merged[phase].min = min(merged[phase].min, ts.min)
                merged[phase].max = max(merged[phase].max, ts.max)
        for ts in merged.values():
            ts.finalize()

        dummy = WorkerProfiler.__new__(WorkerProfiler)
        dummy.timers = merged
        dummy.PHASES = list(merged.keys())

        avg_moves_per_game = (total_moves / len(all_profilers)
                              if all_profilers else 0)
        dummy.print_report(
            num_games=len(all_profilers),
            num_simulations=num_simulations,
            batch_size=batch_size,
            total_moves=total_moves,
            avg_moves_per_game=avg_moves_per_game,
        )

    finally:
        gpu_shutdown.set()
        try:
            request_q.put_nowait(None)
        except:
            pass
        server_process.join(timeout=5)
        if server_process.is_alive():
            server_process.kill()


# ── Shared profile loop (CPU) ───────────────────────────────────────────────

def _run_profile_loop(network, inference_client, num_games, num_simulations,
                      batch_size, max_game_length, verbosity):
    piece_values = {'P': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9}
    game_kwargs = dict(
        max_game_length=max_game_length,
        adjudicate_material=True,
        piece_values=piece_values,
        temp_threshold=30,
        temp_high=1.0,
        temp_low=0.1,
        adjudicate_graded=True,
        adjudicate_scaling=9.0,
    )

    net_or_client = inference_client if inference_client is not None else network

    all_profilers: List[WorkerProfiler] = []
    total_moves = 0

    for game_idx in range(num_games):
        mcts_engine = MCTS(
            network=net_or_client,
            num_simulations=num_simulations,
            c_puct=1.5,
            dirichlet_alpha=0.3,
            dirichlet_epsilon=0.25,
            batch_size=batch_size,
            c_virtual_loss=0.5,
        )

        if verbosity >= 1:
            print(f"\n--- Game {game_idx + 1}/{num_games} ---")

        try:
            game_data, game_info, profiler = run_profiled_game(
                mcts_engine, **game_kwargs)
        except Exception as e:
            print(f"[ERROR] Game {game_idx + 1} failed: {e}")
            import traceback
            traceback.print_exc()
            continue

        all_profilers.append(profiler)
        total_moves += game_info['length']

        if verbosity >= 1:
            print(f"  Result: {game_info['result_str']}  "
                  f"Moves: {game_info['length']}  "
                  f"Termination: {game_info['termination']}")

    if not all_profilers:
        print("[ERROR] No games completed successfully.")
        return

    merged: Dict[str, TimerSnapshot] = {}
    for profiler in all_profilers:
        profiler.finalize()
        for phase, ts in profiler.timers.items():
            if phase not in merged:
                merged[phase] = TimerSnapshot()
            merged[phase].calls += ts.calls
            merged[phase].total += ts.total
            merged[phase].min = min(merged[phase].min, ts.min)
            merged[phase].max = max(merged[phase].max, ts.max)
    for ts in merged.values():
        ts.finalize()

    dummy = WorkerProfiler.__new__(WorkerProfiler)
    dummy.timers = merged
    dummy.PHASES = list(merged.keys())

    avg_moves_per_game = (total_moves / len(all_profilers)
                          if all_profilers else 0)
    dummy.print_report(
        num_games=len(all_profilers),
        num_simulations=num_simulations,
        batch_size=batch_size,
        total_moves=total_moves,
        avg_moves_per_game=avg_moves_per_game,
    )


# ── Worker-integrated profile entry point ─────────────────────────────────────

def profile_in_worker(mcts_engine: MCTS, config, piece_values: dict,
                      num_games: int = 3) -> dict:
    from selfplay import play_one_game
    game_kwargs = dict(
        max_game_length=config.selfplay.max_game_length,
        adjudicate_material=config.selfplay.adjudicate_material,
        piece_values=piece_values,
        temp_threshold=config.selfplay.temperature_threshold,
        temp_high=config.selfplay.temperature_high,
        temp_low=config.selfplay.temperature_low,
        adjudicate_graded=getattr(config.selfplay, 'adjudicate_graded', True),
        adjudicate_scaling=getattr(config.selfplay, 'adjudicate_scaling', 9.0),
    )

    all_profilers = []
    total_moves = 0
    print(f"\n[Worker Profiler] Starting {num_games} profiled game(s)...")

    for game_idx in range(num_games):
        profiler = WorkerProfiler(mcts_engine)
        profiler.attach()
        try:
            game_data, game_info = play_one_game(mcts_engine, **game_kwargs)
        except Exception as e:
            print(f"[Worker Profiler] Game {game_idx + 1} failed: {e}")
            import traceback
            traceback.print_exc()
            profiler.detach()
            continue
        profiler.detach()

        all_profilers.append(profiler)
        total_moves += game_info['length']
        print(f"[Worker Profiler] Game {game_idx + 1}/{num_games}: "
              f"{game_info['result_str']}, {game_info['length']} moves, "
              f"{game_info['termination']}")

    if not all_profilers:
        print("[Worker Profiler] No games completed.")
        return {}

    merged: Dict[str, TimerSnapshot] = {}
    for profiler in all_profilers:
        profiler.finalize()
        for phase, ts in profiler.timers.items():
            if phase not in merged:
                merged[phase] = TimerSnapshot()
            merged[phase].calls += ts.calls
            merged[phase].total += ts.total
            merged[phase].min = min(merged[phase].min, ts.min)
            merged[phase].max = max(merged[phase].max, ts.max)
    for ts in merged.values():
        ts.finalize()

    dummy = WorkerProfiler.__new__(WorkerProfiler)
    dummy.timers = merged
    dummy.PHASES = list(merged.keys())

    avg_moves_per_game = (total_moves / len(all_profilers)
                          if all_profilers else 0)
    dummy.print_report(
        num_games=len(all_profilers),
        num_simulations=config.mcts.num_simulations,
        batch_size=getattr(config.mcts, 'batch_size', 1),
        total_moves=total_moves,
        avg_moves_per_game=avg_moves_per_game,
    )

    return {
        'num_games': len(all_profilers),
        'total_moves': total_moves,
        'config': {
            'num_simulations': config.mcts.num_simulations,
            'batch_size': getattr(config.mcts, 'batch_size', 1),
        },
        'phases': {
            phase: {
                'calls': ts.calls,
                'total_s': ts.total,
                'mean_ms': ts.mean * 1000,
                'pct': (100.0 * ts.total / total_moves
                        if total_moves > 0 else 0),
            }
            for phase, ts in merged.items()
        },
    }


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Profile CPU time distribution in the self-play worker")
    parser.add_argument('--games', type=int, default=3,
                        help='Number of games to profile (default: 3)')
    parser.add_argument('--sims', type=int, default=200,
                        help='MCTS simulations per move (default: 200)')
    parser.add_argument('--batch', type=int, default=1,
                        help='MCTS batch size (default: 1, 1=sequential)')
    parser.add_argument('--filters', type=int, default=64,
                        help='Network filter count (CPU only, default: 64)')
    parser.add_argument('--blocks', type=int, default=4,
                        help='Network residual blocks (CPU only, default: 4)')
    parser.add_argument('--moves', type=int, default=150,
                        help='Max game length (default: 150)')
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU inference server (requires torch_directml)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Show per-game results')
    args = parser.parse_args()

    if args.gpu:
        profile_gpu(
            num_games=args.games,
            num_simulations=args.sims,
            batch_size=args.batch if args.batch > 1 else 64,
            max_game_length=args.moves,
            verbosity=1 if args.verbose else 0,
        )
    else:
        profile_standalone(
            num_games=args.games,
            num_simulations=args.sims,
            batch_size=args.batch,
            num_filters=args.filters,
            num_blocks=args.blocks,
            max_game_length=args.moves,
            verbosity=1 if args.verbose else 0,
        )


if __name__ == '__main__':
    main()