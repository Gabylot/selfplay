"""Real-pipeline profiler for AlphaZero self-play.

Each worker runs in its own OS process (mp.Process, spawn mode) — exactly
like the real training stack — so Task Manager shows N+1 processes and CPU
usage reflects true parallelism.

Workers return plain dicts of timing data via a Queue so no unpicklable
objects cross the process boundary.

Usage
-----
    python profile_real.py                          # 3 games, 1 worker, auto-detect checkpoint
    python profile_real.py --workers 8 --games 5   # mirrors training scale
    python profile_real.py --no-gpu --sims 100     # CPU-only, quick
    python profile_real.py --checkpoint output/run1/checkpoints/step_500.pt

Output sections
---------------
  PHASE BREAKDOWN   — every MCTS sub-phase as % of wall time + latency dist
  NETWORK LATENCY   — mean / p50 / p95 / p99 for inference calls
  BOARD OPS         — chess.Board.copy / push / legal_moves per-call cost
  ENCODING          — board_to_tensor and move-index encoding
  GAME STATS        — moves/game, sims/sec, throughput
  BOTTLENECK HINTS  — auto-analysis with concrete advice
"""

import argparse
import glob
import io
import math
import os
import sys
import time
import queue as _queue
import multiprocessing as mp
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── project root on sys.path (both parent process AND worker subprocess) ────
_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


# ════════════════════════════════════════════════════════════════════════════
# Serialisable timer  (must survive pickle to cross mp boundary)
# ════════════════════════════════════════════════════════════════════════════

class Timer:
    """Accumulates timing samples; fully picklable (plain lists/ints)."""

    __slots__ = ("calls", "total_s", "_samples")

    def __init__(self):
        self.calls    = 0
        self.total_s  = 0.0
        self._samples: List[float] = []

    def record(self, elapsed: float):
        self.calls   += 1
        self.total_s += elapsed
        self._samples.append(elapsed)

    # ── Stats ────────────────────────────────────────────────────────────

    @property
    def mean_ms(self) -> float:
        return (self.total_s / self.calls * 1000) if self.calls else 0.0

    def _pct(self, p: float) -> float:
        if not self._samples:
            return 0.0
        arr = sorted(self._samples)
        idx = max(min(int(math.ceil(p / 100 * len(arr))) - 1, len(arr) - 1), 0)
        return arr[idx] * 1000

    @property
    def p50_ms(self)  -> float: return self._pct(50)
    @property
    def p95_ms(self)  -> float: return self._pct(95)
    @property
    def p99_ms(self)  -> float: return self._pct(99)
    @property
    def min_ms(self)  -> float: return min(self._samples) * 1000 if self._samples else 0.0
    @property
    def max_ms(self)  -> float: return max(self._samples) * 1000 if self._samples else 0.0

    def merge(self, other: "Timer"):
        self.calls    += other.calls
        self.total_s  += other.total_s
        self._samples.extend(other._samples)

    # Make pickling explicit so nothing breaks on edge-case Python builds
    def __getstate__(self):  return (self.calls, self.total_s, self._samples)
    def __setstate__(self, s): self.calls, self.total_s, self._samples = s


# ════════════════════════════════════════════════════════════════════════════
# Instrumented MCTS  (runs inside worker subprocess)
# ════════════════════════════════════════════════════════════════════════════

# Import deferred to worker to avoid importing torch in the parent process
def _make_instrumented_mcts_class():
    """Build and return InstrumentedMCTS. Called inside the worker process."""
    import numpy as np
    import chess
    from mcts import MCTS, MCTSNode
    from encoding import (board_to_tensor, get_legal_move_mask_from_moves,
                          move_to_policy_index, policy_index_to_move, NUM_ACTIONS)

    class InstrumentedMCTS(MCTS):
        PHASES = [
            "selection",
            "expansion_cpu",
            "network_infer",
            "backprop",
            "board_to_tensor",
            "move_encoding",
            "legal_moves",
            "dirichlet_noise",
            "visit_policy",
        ]

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.timers: Dict[str, Timer] = {p: Timer() for p in self.PHASES}

        # ── override selection ──────────────────────────────────────────

        def _select_child(self, node):
            t0 = time.perf_counter()
            r = super()._select_child(node)
            self.timers["selection"].record(time.perf_counter() - t0)
            return r

        # ── override single-leaf expansion ─────────────────────────────

        def _expand_node(self, node):
            if node._game_over_cached is None:
                node._game_over_cached = (
                    node.board.is_game_over() or
                    self._tree_repetition_check(node, 3) or
                    node.board.is_fifty_moves() or
                    node.board.ply() >= self.max_game_length
                )
            if node._game_over_cached:
                return self._get_terminal_value(node)

            t0 = time.perf_counter()
            state = board_to_tensor(node.board)
            self.timers["board_to_tensor"].record(time.perf_counter() - t0)

            t0 = time.perf_counter()
            policy, value = self.network.predict(state)
            self.timers["network_infer"].record(time.perf_counter() - t0)

            t0 = time.perf_counter()
            result = self._expand_node_with_data(node, policy, value)
            self.timers["expansion_cpu"].record(time.perf_counter() - t0)
            return result

        # ── override batched evaluation ─────────────────────────────────

        def _evaluate_batch(self, leaf_nodes):
            terminal_values = {}
            expand_idx, expand_nodes = [], []

            for i, node in enumerate(leaf_nodes):
                if node._game_over_cached is None:
                    node._game_over_cached = (
                        node.board.is_game_over() or
                        self._tree_repetition_check(node, 3) or
                        node.board.is_fifty_moves() or
                        node.board.ply() >= self.max_game_length
                    )
                if node._game_over_cached:
                    terminal_values[i] = self._get_terminal_value(node)
                elif not node.is_expanded:
                    expand_idx.append(i); expand_nodes.append(node)
                else:
                    terminal_values[i] = self._get_terminal_value(node)

            if expand_nodes:
                t0 = time.perf_counter()
                states = np.stack([board_to_tensor(n.board) for n in expand_nodes])
                self.timers["board_to_tensor"].record(time.perf_counter() - t0)

                t0 = time.perf_counter()
                policies, values = self.network.predict_batch(states)
                self.timers["network_infer"].record(time.perf_counter() - t0)

                t0 = time.perf_counter()
                for k, node in enumerate(expand_nodes):
                    self._expand_node_with_data(node, policies[k], values[k])
                    terminal_values[expand_idx[k]] = float(values[k])
                self.timers["expansion_cpu"].record(time.perf_counter() - t0)

            return [terminal_values.get(i, 0.0) for i in range(len(leaf_nodes))]

        # ── override expansion inner work ───────────────────────────────

        def _expand_node_with_data(self, node, policy, value):
            t0 = time.perf_counter()
            legal_moves = list(node.board.legal_moves)
            self.timers["legal_moves"].record(time.perf_counter() - t0)
            node.legal_moves_cached = legal_moves

            if not legal_moves:
                node.is_expanded = True
                return 0.0

            mask = get_legal_move_mask_from_moves(legal_moves, node.board)
            lp = policy * mask
            s = lp.sum()
            lp = lp / s if s > 0 else mask / mask.sum()

            t0 = time.perf_counter()
            for move in legal_moves:
                try:
                    idx = move_to_policy_index(move, node.board)
                except ValueError:
                    continue
                child = MCTSNode(parent=node, move=move, prior=float(lp[idx]))
                node.children[idx] = child
            self.timers["move_encoding"].record(time.perf_counter() - t0)

            node.is_expanded = True
            return float(value)

        # ── override backprop ───────────────────────────────────────────

        def _backpropagate(self, node, value):
            t0 = time.perf_counter()
            super()._backpropagate(node, value)
            self.timers["backprop"].record(time.perf_counter() - t0)

        def _backpropagate_with_virtual_loss(self, node, value):
            t0 = time.perf_counter()
            super()._backpropagate_with_virtual_loss(node, value)
            self.timers["backprop"].record(time.perf_counter() - t0)

        # ── override noise + visit policy ───────────────────────────────

        def _add_dirichlet_noise(self, node):
            t0 = time.perf_counter()
            super()._add_dirichlet_noise(node)
            self.timers["dirichlet_noise"].record(time.perf_counter() - t0)

        def _get_visit_policy(self, root):
            t0 = time.perf_counter()
            r = super()._get_visit_policy(root)
            self.timers["visit_policy"].record(time.perf_counter() - t0)
            return r

        def select_move_with_temperature(self, root, temperature):
            t0 = time.perf_counter()
            r = super().select_move_with_temperature(root, temperature)
            self.timers["visit_policy"].record(time.perf_counter() - t0)
            return r

    return InstrumentedMCTS


# ════════════════════════════════════════════════════════════════════════════
# Board-op patcher  (class-level monkey-patch inside worker process)
# ════════════════════════════════════════════════════════════════════════════

class BoardOpPatcher:
    OPS = ["board_copy", "board_push", "board_game_over", "board_result"]

    def __init__(self):
        self.timers: Dict[str, Timer] = {k: Timer() for k in self.OPS}
        self._orig: dict = {}

    def attach(self):
        import chess
        Board = chess.Board
        for attr, key in [("copy",         "board_copy"),
                           ("push",         "board_push"),
                           ("is_game_over", "board_game_over"),
                           ("result",       "board_result")]:
            orig = Board.__dict__.get(attr)
            if orig is None:
                continue
            self._orig[attr] = orig
            timer = self.timers[key]

            def _wrap(fn, t):
                def wrapper(self_b, *a, **kw):
                    t0 = time.perf_counter()
                    r = fn(self_b, *a, **kw)
                    t.record(time.perf_counter() - t0)
                    return r
                return wrapper

            setattr(Board, attr, _wrap(orig, timer))

    def detach(self):
        import chess
        for attr, orig in self._orig.items():
            setattr(chess.Board, attr, orig)
        self._orig.clear()

    def dump(self) -> Dict[str, Timer]:
        return dict(self.timers)


# ════════════════════════════════════════════════════════════════════════════
# Worker subprocess entry point
# ════════════════════════════════════════════════════════════════════════════

def _worker_entry(worker_id: int,
                  config_dict: dict,
                  checkpoint_path: Optional[str],
                  num_games: int,
                  result_q: mp.Queue,
                  request_q: Optional[mp.Queue],
                  response_q: Optional[mp.Queue]):
    """Runs inside a real subprocess. Plays `num_games` games, sends
    results back through `result_q` as plain picklable dicts."""
    import sys, os
    _here = str(Path(__file__).resolve().parent)
    if _here not in sys.path:
        sys.path.insert(0, _here)

    import torch
    import numpy as np
    from config import Config
    from network import AlphaZeroNet
    from selfplay import play_one_game

    config = Config(config_dict)
    num_sims   = config.mcts.num_simulations
    batch_size = getattr(config.mcts, "batch_size", 1)

    # ── Build / load network ────────────────────────────────────────────
    def make_net():
        n = AlphaZeroNet(
            num_residual_blocks=config.network.num_residual_blocks,
            num_filters=config.network.num_filters,
            num_policy_channels=config.network.num_policy_channels,
            num_value_channels=config.network.num_value_channels,
            value_fc_size=config.network.value_fc_size,
        )
        n.eval()
        return n

    net = make_net()
    if checkpoint_path and Path(checkpoint_path).exists():
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "model_state_dict" in state:
            net.load_state_dict(state["model_state_dict"])
        else:
            net.load_state_dict(state)
        net.eval()

    # ── Pick inference backend ──────────────────────────────────────────
    use_client = (request_q is not None and response_q is not None)
    if use_client:
        from inference_client import InferenceClient
        network_or_client = InferenceClient(worker_id, request_q, response_q)
    else:
        network_or_client = net

    # ── Build instrumented MCTS class (deferred import) ─────────────────
    InstrumentedMCTS = _make_instrumented_mcts_class()

    piece_values = dict(config.selfplay.piece_values) \
        if hasattr(config.selfplay.piece_values, "items") \
        else config.selfplay.piece_values

    # ── Play games ───────────────────────────────────────────────────────
    for game_num in range(num_games):
        mcts_engine = InstrumentedMCTS(
            network=network_or_client,
            num_simulations=num_sims,
            c_puct=config.mcts.c_puct,
            dirichlet_alpha=config.mcts.dirichlet_alpha,
            dirichlet_epsilon=config.mcts.dirichlet_epsilon,
            batch_size=batch_size,
            c_virtual_loss=getattr(config.mcts, "c_virtual_loss", 0.5),
            max_game_length=config.selfplay.max_game_length,
            adjudicate_material=config.selfplay.adjudicate_material,
            piece_values=piece_values,
            adjudicate_graded=getattr(config.selfplay, "adjudicate_graded", True),
            adjudicate_scaling=getattr(config.selfplay, "adjudicate_scaling", 9.0),
        )

        board_patcher = BoardOpPatcher()
        board_patcher.attach()
        t_start = time.perf_counter()
        try:
            _, game_info = play_one_game(
                mcts_engine,
                max_game_length=config.selfplay.max_game_length,
                adjudicate_material=config.selfplay.adjudicate_material,
                piece_values=piece_values,
                temp_threshold=config.selfplay.temperature_threshold,
                temp_high=config.selfplay.temperature_high,
                temp_low=config.selfplay.temperature_low,
                adjudicate_graded=getattr(config.selfplay, "adjudicate_graded", True),
                adjudicate_scaling=getattr(config.selfplay, "adjudicate_scaling", 9.0),
            )
            wall_s = time.perf_counter() - t_start
            board_patcher.detach()

            # Merge board-op timers into mcts timers dict
            combined_timers = dict(mcts_engine.timers)
            for k, t in board_patcher.dump().items():
                combined_timers[k] = t

            game_info["wall_s"] = wall_s
            result_q.put({
                "status":    "ok",
                "worker_id": worker_id,
                "game_num":  game_num,
                "game_info": game_info,
                "timers":    combined_timers,   # Dict[str, Timer] — picklable
            })

        except Exception as exc:
            board_patcher.detach()
            import traceback
            result_q.put({
                "status":    "error",
                "worker_id": worker_id,
                "game_num":  game_num,
                "error":     str(exc),
                "traceback": traceback.format_exc(),
            })

    result_q.put({"status": "done", "worker_id": worker_id})


# ════════════════════════════════════════════════════════════════════════════
# GPU server context  (mirrors main.py setup exactly)
# ════════════════════════════════════════════════════════════════════════════

class GPUServerContext:
    def __init__(self, config, num_workers: int):
        self.config      = config
        self.num_workers = num_workers
        self.request_q   = mp.Queue()
        self.weight_q    = mp.Queue()
        self.response_qs = {i: mp.Queue(maxsize=256) for i in range(num_workers)}
        self._ready      = mp.Event()
        self._shutdown   = mp.Event()
        self._proc: Optional[mp.Process] = None

    def start(self):
        from gpu_server import GPUInferenceServer
        server = GPUInferenceServer(
            config=self.config,
            request_queue=self.request_q,
            response_queues=self.response_qs,
            weight_queue=self.weight_q,
            ready_event=self._ready,
            shutdown_event=self._shutdown,
        )
        self._proc = mp.Process(target=server.run, daemon=True)
        self._proc.start()
        print("[profiler] Waiting for GPU server …")
        self._ready.wait()
        print("[profiler] GPU server ready")

    def push_weights(self, network):
        import torch
        buf = io.BytesIO()
        torch.save(network.state_dict(), buf)
        self.weight_q.put(buf.getvalue())

    def stop(self):
        self._shutdown.set()
        try: self.request_q.put_nowait(None)
        except Exception: pass
        if self._proc and self._proc.is_alive():
            self._proc.join(timeout=8)
            if self._proc.is_alive():
                self._proc.kill()


# ════════════════════════════════════════════════════════════════════════════
# Checkpoint helpers
# ════════════════════════════════════════════════════════════════════════════

def find_latest_checkpoint(config) -> Optional[str]:
    out = Path(config.main.output_dir) / config.main.run_name / "checkpoints"
    for pat in [str(out / "latest.pt"), str(out / "step_*.pt"),
                "output/*/checkpoints/latest.pt", "checkpoints/latest.pt",
                "output/*/checkpoints/step_*.pt"]:
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[-1]
    return None


def load_network_for_report(config, checkpoint_path: Optional[str]):
    """Load into parent process just to push weights to GPU server."""
    import torch
    from network import AlphaZeroNet
    net = AlphaZeroNet(
        num_residual_blocks=config.network.num_residual_blocks,
        num_filters=config.network.num_filters,
        num_policy_channels=config.network.num_policy_channels,
        num_value_channels=config.network.num_value_channels,
        value_fc_size=config.network.value_fc_size,
    )
    if checkpoint_path and Path(checkpoint_path).exists():
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        sd = state.get("model_state_dict", state)
        net.load_state_dict(sd)
    net.eval()
    return net


# ════════════════════════════════════════════════════════════════════════════
# Reporting
# ════════════════════════════════════════════════════════════════════════════

SEP = "─" * 76

def _pct(v, total):
    return f"{100*v/total:5.1f}%" if total > 0 else "   — "

def merge_timers(all_dicts: List[Dict[str, Timer]]) -> Dict[str, Timer]:
    merged: Dict[str, Timer] = {}
    for d in all_dicts:
        for k, t in d.items():
            if k not in merged:
                merged[k] = Timer()
            merged[k].merge(t)
    return merged


def print_phase_table(merged: Dict[str, Timer], total_wall_s: float,
                      total_sims: int, total_moves: int):
    print(f"\n{'═'*76}")
    print(f"  PHASE BREAKDOWN  "
          f"(wall={total_wall_s:.2f}s | sims={total_sims:,} | moves={total_moves:,})")
    print(f"{'═'*76}")

    accounted = sum(t.total_s for t in merged.values())
    rows = sorted(merged.items(), key=lambda x: x[1].total_s, reverse=True)

    print(f"\n  {'Phase':<26} {'Calls':>8} {'Total(s)':>9} {'%Wall':>7} "
          f"{'Mean':>8} {'P50':>8} {'P95':>8} {'P99':>8}")
    print(f"  {SEP}")
    for name, t in rows:
        if t.calls == 0:
            continue
        hot = "  ◀" if t.total_s / total_wall_s > 0.25 else ""
        print(f"  {name:<26} {t.calls:>8,} {t.total_s:>9.3f} "
              f"{_pct(t.total_s, total_wall_s):>7} "
              f"{t.mean_ms:>7.2f}ms {t.p50_ms:>7.2f}ms "
              f"{t.p95_ms:>7.2f}ms {t.p99_ms:>7.2f}ms{hot}")

    other = max(0.0, total_wall_s - accounted)
    print(f"  {'(overhead / other)':<26} {'—':>8} {other:>9.3f} "
          f"{_pct(other, total_wall_s):>7}")
    print(f"  {SEP}")
    print(f"  {'TOTAL WALL':<26} {'—':>8} {total_wall_s:>9.3f} {'100.0%':>7}")


def print_network_section(t: Optional[Timer], batch_size: int, use_gpu: bool):
    print(f"\n{'═'*76}")
    print(f"  NETWORK INFERENCE  ({'GPU' if use_gpu else 'CPU'}, batch_size={batch_size})")
    print(f"{'═'*76}")
    if not t or t.calls == 0:
        print("  (no inference recorded)")
        return
    print(f"  Calls   : {t.calls:,}")
    print(f"  Total   : {t.total_s:.3f}s")
    print(f"  Mean    : {t.mean_ms:.2f}ms")
    print(f"  P50     : {t.p50_ms:.2f}ms")
    print(f"  P95     : {t.p95_ms:.2f}ms")
    print(f"  P99     : {t.p99_ms:.2f}ms")
    print(f"  Min/Max : {t.min_ms:.2f}ms / {t.max_ms:.2f}ms")


def print_board_ops(merged: Dict[str, Timer]):
    print(f"\n{'═'*76}")
    print(f"  BOARD OPERATIONS  (chess.Board.*)")
    print(f"{'═'*76}")
    print(f"  {'Op':<24} {'Calls':>8} {'Total(s)':>9} {'Mean':>9} {'P95':>9}")
    print(f"  {SEP}")
    for k in ("board_copy", "board_push", "board_game_over",
               "board_result", "legal_moves"):
        t = merged.get(k)
        if not t or t.calls == 0:
            continue
        print(f"  {k:<24} {t.calls:>8,} {t.total_s:>9.4f} "
              f"{t.mean_ms*1000:>8.1f}µs {t.p95_ms*1000:>8.1f}µs")


def print_encoding(merged: Dict[str, Timer]):
    print(f"\n{'═'*76}")
    print(f"  ENCODING")
    print(f"{'═'*76}")
    print(f"  {'Op':<28} {'Calls':>8} {'Total(s)':>9} {'Mean':>9} {'P95':>9}")
    print(f"  {SEP}")
    for k in ("board_to_tensor", "move_encoding", "policy_index_to_move"):
        t = merged.get(k)
        if not t or t.calls == 0:
            continue
        print(f"  {k:<28} {t.calls:>8,} {t.total_s:>9.4f} "
              f"{t.mean_ms*1000:>8.1f}µs {t.p95_ms*1000:>8.1f}µs")


def print_game_stats(game_infos: List[dict], total_wall_s: float,
                     num_sims: int, num_workers: int):
    print(f"\n{'═'*76}")
    print(f"  GAME STATS  ({len(game_infos)} games · {num_workers} worker process(es))")
    print(f"{'═'*76}")
    import numpy as np
    lengths = [g["length"]  for g in game_infos]
    walls   = [g["wall_s"]  for g in game_infos]
    terms   = defaultdict(int)
    for g in game_infos:
        terms[g["termination"]] += 1
    total_moves = sum(lengths)
    total_sims  = total_moves * num_sims
    print(f"  Avg moves/game  : {np.mean(lengths):.1f}  (min {min(lengths)} max {max(lengths)})")
    print(f"  Avg wall/game   : {np.mean(walls):.2f}s")
    print(f"  Total moves     : {total_moves:,}")
    print(f"  Total sims      : {total_sims:,}  ({num_sims}/move)")
    print(f"  Sims/sec        : {total_sims / total_wall_s:.1f}")
    print(f"  Moves/sec       : {total_moves / total_wall_s:.1f}")
    print(f"  Terminations    :")
    for term, cnt in sorted(terms.items(), key=lambda x: -x[1]):
        print(f"    {term:<30} {cnt}")


def print_hints(merged: Dict[str, Timer], total_wall_s: float,
                use_gpu: bool, batch_size: int):
    print(f"\n{'═'*76}")
    print(f"  BOTTLENECK HINTS")
    print(f"{'═'*76}")

    def pct(key):
        t = merged.get(key)
        return 100 * t.total_s / total_wall_s if (t and t.calls) else 0.0

    hints = []
    net_pct   = pct("network_infer")
    sel_pct   = pct("selection")
    brd_pct   = pct("board_to_tensor")
    enc_pct   = pct("move_encoding")
    copy_pct  = pct("board_copy")
    legal_pct = pct("legal_moves")
    bp_pct    = pct("backprop")

    if net_pct > 40:
        if not use_gpu:
            hints.append((f"Network = {net_pct:.0f}% (CPU)",
                          "Enable GPU: config.inference.use_gpu=true, or reduce\n"
                          "    num_residual_blocks / num_filters."))
        elif batch_size == 1:
            hints.append((f"Network = {net_pct:.0f}% at batch_size=1",
                          "Increase config.mcts.batch_size to 32–128.\n"
                          "    With GPU each forward pass is the same cost regardless\n"
                          "    of batch size up to a point — batching is nearly free."))
        else:
            hints.append((f"Network = {net_pct:.0f}% (GPU, batch={batch_size})",
                          "Network is genuinely the limit. Options:\n"
                          "    - Fewer blocks/filters\n"
                          "    - More GPU memory / larger batch ceiling\n"
                          "    - FP16 if not already in use (gpu_server.py uses .half())"))

    if sel_pct > 25:
        hints.append((f"PUCT selection = {sel_pct:.0f}%",
                      "Very deep trees or very high num_simulations.\n"
                      "    Try reducing num_simulations or raising c_puct (shallower)."))

    if brd_pct > 15:
        hints.append((f"board_to_tensor = {brd_pct:.0f}%",
                      "Encoding is hot. Pre-allocate a (20,8,8) array per worker\n"
                      "    and fill in-place rather than zeros + slice assignment."))

    if legal_pct > 10:
        hints.append((f"legal_moves = {legal_pct:.0f}%",
                      "python-chess move gen is unavoidable, but confirm\n"
                      "    get_legal_move_mask_from_moves is used (no second iteration)."))

    if enc_pct > 10:
        hints.append((f"move_encoding = {enc_pct:.0f}%",
                      "move_to_policy_index called for every legal move on expand.\n"
                      "    The LUT already eliminates the direction loop; further\n"
                      "    gains would need a C extension."))

    if copy_pct > 8:
        hints.append((f"board_copy = {copy_pct:.0f}%",
                      "Lazy-board MCTSNode should minimise copies to visited nodes.\n"
                      "    If this is high, check that stack=False is used (not stack=True)\n"
                      "    in the lazy property (mcts.py:MCTSNode.board getter)."))

    if not hints:
        hints.append(("No single dominant bottleneck",
                      "Time is spread across phases — healthy balance.\n"
                      "    To go faster: add more workers or raise num_simulations\n"
                      "    to improve move quality rather than raw throughput."))

    for title, advice in hints:
        print(f"\n  ⚠  {title}")
        for line in advice.split("\n"):
            print(f"     {line}")
    print()


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="AlphaZero real-pipeline profiler")
    parser.add_argument("--config",     type=str,  default=None)
    parser.add_argument("--checkpoint", type=str,  default=None,
                        help="Path to .pt file (default: auto-detect latest)")
    parser.add_argument("--games",      type=int,  default=3,
                        help="Games per worker (default: 3)")
    parser.add_argument("--workers",    type=int,  default=1,
                        help="Worker processes (default: 1; use 8 to match training)")
    parser.add_argument("--sims",       type=int,  default=None,
                        help="Override num_simulations")
    parser.add_argument("--batch",      type=int,  default=None,
                        help="Override mcts.batch_size")
    parser.add_argument("--no-gpu",     action="store_true",
                        help="Force CPU even if config enables GPU")
    parser.add_argument("--max-moves",  type=int,  default=None,
                        help="Override max_game_length (shorter = faster profiling)")
    args = parser.parse_args()

    # ── Config ───────────────────────────────────────────────────────────
    overrides = {}
    if args.sims:      overrides.setdefault("mcts", {})["num_simulations"] = args.sims
    if args.batch:     overrides.setdefault("mcts", {})["batch_size"]      = args.batch
    if args.max_moves: overrides.setdefault("selfplay", {})["max_game_length"] = args.max_moves
    if args.no_gpu:    overrides.setdefault("inference", {})["use_gpu"]    = False

    try:
        config = get_config(path=args.config, overrides=overrides or None)
    except Exception:
        from config import get_config as _gc
        config = _gc(overrides=overrides or None)

    num_sims   = config.mcts.num_simulations
    batch_size = getattr(config.mcts, "batch_size", 1)
    use_gpu    = (not args.no_gpu
                  and getattr(config, "inference", None)
                  and getattr(config.inference, "use_gpu", False))

    ckpt = args.checkpoint or find_latest_checkpoint(config)

    print(f"\n{'═'*76}")
    print(f"  AlphaZero Real-Pipeline Profiler")
    print(f"{'═'*76}")
    print(f"  workers       : {args.workers}  (real OS processes via mp.Process)")
    print(f"  games/worker  : {args.games}")
    print(f"  num_sims      : {num_sims}")
    print(f"  batch_size    : {batch_size}")
    print(f"  inference     : {'GPU (DirectML)' if use_gpu else 'CPU'}")
    print(f"  max_game_len  : {config.selfplay.max_game_length}")
    print(f"  checkpoint    : {ckpt or '(none — random weights)'}")
    print()

    # ── GPU server (optional) ─────────────────────────────────────────────
    gpu_ctx: Optional[GPUServerContext] = None
    if use_gpu:
        try:
            net_for_gpu = load_network_for_report(config, ckpt)
            gpu_ctx = GPUServerContext(config, args.workers)
            gpu_ctx.start()
            gpu_ctx.push_weights(net_for_gpu)
            del net_for_gpu      # don't keep a live torch net in the parent
        except Exception as e:
            print(f"[profiler] GPU server failed ({e}) — falling back to CPU")
            use_gpu = False
            gpu_ctx = None

    # ── Spawn worker processes ────────────────────────────────────────────
    result_q  = mp.Queue()
    processes = []
    config_dict = config.to_dict()

    for wid in range(args.workers):
        req_q  = gpu_ctx.request_q                if gpu_ctx else None
        resp_q = gpu_ctx.response_qs.get(wid)     if gpu_ctx else None
        p = mp.Process(
            target=_worker_entry,
            args=(wid, config_dict, ckpt, args.games, result_q, req_q, resp_q),
            daemon=True,
        )
        p.start()
        processes.append(p)

    print(f"[profiler] Spawned {args.workers} worker process(es). "
          f"Check Task Manager — you should see {args.workers + 1 + (1 if use_gpu else 0)} "
          f"python processes.\n")

    # ── Collect results ───────────────────────────────────────────────────
    all_game_infos: List[dict]              = []
    all_timer_dicts: List[Dict[str, Timer]] = []
    workers_done = 0
    total_expected = args.workers * args.games

    t_wall_start = time.perf_counter()

    while workers_done < args.workers:
        try:
            msg = result_q.get(timeout=600)
        except _queue.Empty:
            print("[profiler] Timeout waiting for worker results — aborting.")
            break

        if msg["status"] == "done":
            workers_done += 1
            print(f"  [W{msg['worker_id']}] finished all {args.games} game(s)")

        elif msg["status"] == "ok":
            gi = msg["game_info"]
            print(f"  [W{msg['worker_id']}] game {msg['game_num']+1}: "
                  f"{gi['termination']:<22} {gi['result_str']:>5}  "
                  f"{gi['length']:>3} moves  {gi['wall_s']:.1f}s")
            all_game_infos.append(gi)
            all_timer_dicts.append(msg["timers"])

        elif msg["status"] == "error":
            print(f"  [W{msg['worker_id']}] game {msg['game_num']+1} ERROR: {msg['error']}")
            if msg.get("traceback"):
                print(msg["traceback"])

    total_wall_s = time.perf_counter() - t_wall_start

    # ── Clean up ──────────────────────────────────────────────────────────
    for p in processes:
        p.join(timeout=10)
        if p.is_alive():
            p.kill()
    if gpu_ctx:
        gpu_ctx.stop()

    if not all_game_infos:
        print("\n[profiler] No games completed — nothing to report.")
        return

    # ── Merge and report ──────────────────────────────────────────────────
    merged = merge_timers(all_timer_dicts)
    total_moves = sum(g["length"] for g in all_game_infos)
    total_sims  = total_moves * num_sims

    print_phase_table(merged, total_wall_s, total_sims, total_moves)
    print_network_section(merged.get("network_infer"), batch_size, use_gpu)
    print_board_ops(merged)
    print_encoding(merged)
    print_game_stats(all_game_infos, total_wall_s, num_sims, args.workers)
    print_hints(merged, total_wall_s, use_gpu, batch_size)

    # ── One-liner summary ─────────────────────────────────────────────────
    sps = total_sims / total_wall_s if total_wall_s else 0
    net_t = merged.get("network_infer", Timer())
    net_pct = 100 * net_t.total_s / total_wall_s if total_wall_s else 0
    print(f"  {'─'*50}")
    print(f"  Sims/sec : {sps:.0f}  |  "
          f"Network : {net_pct:.0f}% of wall  |  "
          f"Games : {len(all_game_infos)}")
    print(f"  {'─'*50}\n")


if __name__ == "__main__":
    # Must be "spawn" for CUDA / DirectML and to match training behaviour
    if mp.get_start_method(allow_none=True) is None:
        try:
            mp.set_start_method("spawn")
        except RuntimeError:
            pass
    main()