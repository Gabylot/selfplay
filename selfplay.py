"""Self-play game generation and replay buffer for AlphaZero chess.

Parallel self-play:
  - ParallelSelfPlay manages a pool of worker processes.
  - The same worker pool handles both self-play AND eval games.
  - During eval the main loop stops pushing self-play tasks;
    workers drain their queue and receive eval tasks instead.

NOTE 1: each stored position is a 4-tuple (state, policy, value, legal_move_mask)
instead of a 3-tuple. This lets training.py mask illegal moves out of the
policy-loss softmax. See ReplayBuffer.load() for backward-compat handling of
old buffer files saved before this change.

NOTE 2: when GPU inference is enabled (config.inference.use_gpu), gating eval
(network A vs network B) is routed through the centralized dual-network GPU
server via InferenceClient(network_id=0 / 1) instead of loading both networks
onto CPU. Reference eval (network vs alpha-beta) also uses network_id=0 when
GPU inference is active. This mirrors the self-play path and avoids the CPU
fallback silently running whenever use_gpu is set.

NOTE 3: when GPU inference is enabled and ``config.inference.use_shared_memory``
is True (default), per-worker shared-memory buffers are allocated to eliminate
pickle/pipe overhead for state and policy/value arrays. Only ~100 bytes of
metadata travel through the queue per inference request.
"""

import numpy as np
import chess
from collections import deque
from typing import List, Tuple, Optional
import multiprocessing as mp
import queue
import time
import io
import os

from encoding import board_to_tensor, get_legal_move_mask, NUM_ACTIONS
from network import AlphaZeroNet
from mcts import MCTS
from inference_client import InferenceClient
from shared_memory_transport import SharedMemoryTransport


# ─────────────────────────────────────────────────────────────────────────────
# Replay Buffer
# ─────────────────────────────────────────────────────────────────────────────

class ReplayBuffer:
    """FIFO replay buffer with O(1) random access.

    Uses a plain ``list`` with manual size management instead of
    ``collections.deque``.  ``deque`` does NOT support O(1) random
    access — ``deque[i]`` is O(n/2) on average, which makes
    ``sample_batch`` extremely slow for large buffers (100k positions).
    A list gives true O(1) indexing.
    """

    def __init__(self, max_size=100000):
        self.max_size = max_size
        self.buffer = []          # list, not deque — O(1) random access
        self._write_idx = 0       # circular write pointer
        self.total_games = 0
        self.total_positions = 0

    def add_game(self, game_data):
        """Add a completed game's positions to the buffer.

        game_data: list of (state, policy, value, legal_move_mask) tuples.
        """
        for s, p, v, m in game_data:
            if len(self.buffer) < self.max_size:
                # Buffer not yet full — append
                self.buffer.append((s, p, v, m))
                self._write_idx = (self._write_idx + 1) % self.max_size
            else:
                # Buffer full — overwrite oldest (circular buffer)
                self.buffer[self._write_idx] = (s, p, v, m)
                self._write_idx = (self._write_idx + 1) % self.max_size
            self.total_positions += 1
        self.total_games += 1

    def sample_batch(self, batch_size):
        n = len(self.buffer)
        ix = np.random.choice(n, size=min(batch_size, n), replace=False)
        return (np.array([self.buffer[i][0] for i in ix]),
                np.array([self.buffer[i][1] for i in ix]),
                np.array([self.buffer[i][2] for i in ix], dtype=np.float32),
                np.array([self.buffer[i][3] for i in ix], dtype=np.float32))

    def __len__(self): return len(self.buffer)

    def get_outcome_distribution(self):
        if not self.buffer: return {'white_wins':0,'black_wins':0,'draws':0}
        n  = min(1000, len(self.buffer))
        ix = np.random.choice(len(self.buffer), size=n, replace=False)
        ww=bw=dr=0
        for i in ix:
            v=self.buffer[i][2]
            if v>0.5: ww+=1
            elif v<-0.5: bw+=1
            else: dr+=1
        s=len(self.buffer)/n
        return {'white_wins':int(ww*s),'black_wins':int(bw*s),'draws':int(dr*s)}

    # ── Serialization ──

    def save(self, path):
        """Save buffer to a compressed .npz file."""
        if not self.buffer:
            return
        n = len(self.buffer)
        states   = np.stack([self.buffer[i][0] for i in range(n)])
        policies = np.stack([self.buffer[i][1] for i in range(n)])
        values   = np.array([self.buffer[i][2] for i in range(n)], dtype=np.float32)
        masks    = np.stack([self.buffer[i][3] for i in range(n)])
        np.savez_compressed(
            path,
            states=states, policies=policies, values=values, masks=masks,
            total_games=self.total_games,
            total_positions=self.total_positions,
        )

    @classmethod
    def load(cls, path, max_size=100000):
        """Load buffer from a .npz file. Returns a new ReplayBuffer, or None on failure.

        Backward compatibility: buffers saved before legal-move masking was
        added won't have a 'masks' array. In that case we fill in all-ones
        masks (i.e. no masking effect) for those old positions rather than
        crashing or discarding the buffer. This is a soft transition — those
        old positions just won't get the masking benefit until they age out
        of the buffer naturally.
        """
        try:
            data = np.load(path, allow_pickle=False)
            buf = cls(max_size=max_size)
            states   = data['states']
            policies = data['policies']
            values   = data['values']
            n = len(values)
            if 'masks' in data:
                masks = data['masks']
            else:
                print(f"[WARN] {path} has no saved legal-move masks (old format). "
                      f"Filling with all-ones masks for {n} positions — these will "
                      f"train without masking until they age out of the buffer.")
                masks = np.ones((n, NUM_ACTIONS), dtype=np.float32)
            for i in range(n):
                buf.buffer.append((states[i], policies[i], float(values[i]), masks[i]))
            buf.total_games      = int(data.get('total_games', n))
            buf.total_positions  = int(data.get('total_positions', n))
            return buf
        except Exception as e:
            print(f"[WARN] Could not load replay buffer from {path}: {e}")
            return None


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_temperature(move_number, threshold=30, temp_high=1.0, temp_low=0.1):
    return temp_high if move_number < threshold else temp_low


def adjudicate_by_material(board, piece_values, graded=False, scaling=9.0):
    # Use piece_map() to iterate only occupied squares (~32) instead of all 64.
    w=b=0
    for sq, p in board.piece_map().items():
        v=piece_values.get(p.symbol().upper(),0)
        if p.color==chess.WHITE: w+=v
        else: b+=v
    diff=w-b
    if graded:
        if diff==0: return 0.0
        return float(np.tanh(diff / scaling))
    if diff>0: return 1.0
    if diff<0: return -1.0
    return 0.0


def material_point_difference(board, piece_values):
    """Return raw material point difference (white - black) at current position."""
    # Use piece_map() to iterate only occupied squares (~32) instead of all 64.
    w=b=0
    for sq, p in board.piece_map().items():
        v=piece_values.get(p.symbol().upper(),0)
        if p.color==chess.WHITE: w+=v
        else: b+=v
    return w-b


# ─────────────────────────────────────────────────────────────────────────────
# Core game loop (corrected)
# ─────────────────────────────────────────────────────────────────────────────

def play_one_game(mcts_engine, max_game_length=150, adjudicate_material=True,
                  piece_values=None, temp_threshold=30, temp_high=1.0, temp_low=0.1,
                  temperature_override=None, verbose=False, on_move=None,
                  adjudicate_graded=True, adjudicate_scaling=9.0):
    if piece_values is None:
        piece_values = {'P': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9}

    board = chess.Board()
    game_states = []
    mcts_stats_list = []
    move_count = 0
    termination = "unknown"
    outcome = 0.0
    root = None  # Tree recycling

    # Original loop – stop only on automatic game‑overs (not claimable draws)
    while not board.is_game_over() and not board.is_repetition(3) and move_count < max_game_length:
        if root is None:
            root = mcts_engine.get_root(board)

        visit_policy, best_move, stats = mcts_engine.search(root)
        move_candidates = mcts_engine.get_root_child_stats(root)

        temp = (temperature_override if temperature_override is not None
                else get_temperature(move_count, temp_threshold, temp_high, temp_low))
        visit_policy, selected_move = mcts_engine.select_move_with_temperature(root, temp)

        if selected_move is None:
            selected_move = best_move
            if selected_move is None:
                # This should never happen after fixing MCTS expansion (no claim_draw=True)
                with open("mcts_empty_root.log", "a") as f:
                    f.write(f"Empty root at move {move_count}: {board.fen()}\n"
                            f"Legal moves: {[m.uci() for m in board.legal_moves]}\n"
                            f"Children after search: {len(root.children)}\n\n")
                break

        st = board_to_tensor(board)
        cp = 1.0 if board.turn == chess.WHITE else -1.0
        legal_mask = get_legal_move_mask(board)
        game_states.append((st, visit_policy.copy(), cp, legal_mask))
        mcts_stats_list.append(stats)

        mcts_move_data = {'selected_move': selected_move.uci(), 'candidates': move_candidates}
        board.push(selected_move)
        move_count += 1
        if on_move:
            on_move(board.fen(), selected_move.uci(), move_count, mcts_move_data)

        root = mcts_engine.recycle_tree(root, selected_move)

    material_diff = material_point_difference(board, piece_values)

    # Determine outcome and termination reason
    if board.is_game_over(claim_draw=True):
        r = board.result(claim_draw=True)
        if r == "1-0":
            outcome, termination = 1.0, "checkmate" if board.is_checkmate() else "other"
        elif r == "0-1":
            outcome, termination = -1.0, "checkmate" if board.is_checkmate() else "other"
        else:
            outcome = 0.0
            if board.is_fivefold_repetition():
                termination = "fivefold_repetition"
            elif board.is_repetition(3):
                termination = "repetition"
            elif board.is_seventyfive_moves():
                termination = "seventyfive_moves"
            elif board.is_stalemate():
                termination = "stalemate"
            elif board.is_fifty_moves():
                termination = "fifty_moves"
            elif board.is_insufficient_material():
                termination = "insufficient_material"
            else:
                # Only reachable if the game ended but none of the above apply
                # (e.g., variant ending) – label generically
                termination = "draw"
    elif move_count >= max_game_length:
        termination = "max_length"
        if adjudicate_material:
            outcome = adjudicate_by_material(board, piece_values,
                                             graded=adjudicate_graded,
                                             scaling=adjudicate_scaling)
            if outcome > 0:
                termination = "material_white"
            elif outcome < 0:
                termination = "material_black"
    else:
        # Fallback – should not happen normally (kept for safety)
        outcome = 0.0
        termination = "unknown"
        with open("unknown_termination_log.txt", "a") as f:
            f.write(f"UNKNOWN at move {move_count}: {board.fen()} | "
                    f"halfmove_clock={board.halfmove_clock} | "
                    f"can_claim_3fold={board.can_claim_threefold_repetition()} | "
                    f"can_claim_50={board.can_claim_fifty_moves()} | "
                    f"is_50={board.is_fifty_moves()} | "
                    f"is_repetition(2)={board.is_repetition(2)} | "
                    f"is_repetition(3)={board.is_repetition(3)} | "
                    f"is_insufficient={board.is_insufficient_material()}\n")

    # Build result string for display
    if board.is_game_over(claim_draw=True):
        result_str = board.result(claim_draw=True)
    elif outcome > 0:
        result_str = "1-0"
    elif outcome < 0:
        result_str = "0-1"
    else:
        result_str = "1/2-1/2"

    game_data = [(s, p, outcome * pl, m) for s, p, pl, m in game_states]
    avg_depth = float(np.mean([s.get('avg_depth', 0) for s in mcts_stats_list])) if mcts_stats_list else 0.0

    return game_data, {
        'result': outcome,
        'result_str': result_str,
        'length': move_count,
        'termination': termination,
        'avg_mcts_depth': avg_depth,
        'num_positions': len(game_data),
        'material_diff': material_diff,
    }


def self_play_game(network, config, on_move=None):
    mcts_engine = MCTS(
        network=network,
        num_simulations=config.mcts.num_simulations,
        c_puct=config.mcts.c_puct,
        dirichlet_alpha=config.mcts.dirichlet_alpha,
        dirichlet_epsilon=config.mcts.dirichlet_epsilon,
        batch_size=getattr(config.mcts,'batch_size',1),
        c_virtual_loss=getattr(config.mcts,'c_virtual_loss',0.5),
        # ---- NEW: pass game‑length and adjudication parameters ----
        max_game_length=config.selfplay.max_game_length,
        adjudicate_material=config.selfplay.adjudicate_material,
        piece_values=config.selfplay.piece_values,
        adjudicate_graded=getattr(config.selfplay, 'adjudicate_graded', True),
        adjudicate_scaling=getattr(config.selfplay, 'adjudicate_scaling', 9.0),
    )
    return play_one_game(
        mcts_engine=mcts_engine,
        max_game_length=config.selfplay.max_game_length,
        adjudicate_material=config.selfplay.adjudicate_material,
        piece_values=config.selfplay.piece_values,
        temp_threshold=config.selfplay.temperature_threshold,
        temp_high=config.selfplay.temperature_high,
        temp_low=config.selfplay.temperature_low,
        on_move=on_move,
        adjudicate_graded=getattr(config.selfplay,'adjudicate_graded',True),
        adjudicate_scaling=getattr(config.selfplay,'adjudicate_scaling',9.0),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Worker process
# ─────────────────────────────────────────────────────────────────────────────

def _worker_process(worker_id, task_queue, result_queue, config_dict, shutdown_event,
                    request_queue=None, response_queue=None, shared_buffers=None):
    import torch, sys, os
    from config import Config
    from network import AlphaZeroNet
    from mcts import MCTS

    # Limit Torch to a single intra-op thread per worker process.
    # With 8+ worker processes each running MCTS with small inference
    # batches, multi-threading causes contention and cache thrashing.
    # Single-thread is faster for small batches and avoids oversubscription.
    torch.set_num_threads(1)

    config = Config(config_dict)
    pv = config.selfplay.piece_values
    piece_values = dict(pv) if hasattr(pv, 'items') else pv

    use_gpu = (getattr(config, 'inference', None)
               and getattr(config.inference, 'use_gpu', False))
    inference_client = None
    if use_gpu and request_queue is not None and response_queue is not None:
        # network_id=0 by default; gating eval below spins up a second
        # client bound to network_id=1 for the "best" network.
        # Pass shared_buffers for zero-copy IPC when available.
        inference_client = InferenceClient(
            worker_id, request_queue, response_queue, network_id=0,
            shared_buffers=shared_buffers,
        )

    def make_net():
        n = AlphaZeroNet(
            num_residual_blocks=config.network.num_residual_blocks,
            num_filters=config.network.num_filters,
            num_policy_channels=config.network.num_policy_channels,
            num_value_channels=config.network.num_value_channels,
            value_fc_size=config.network.value_fc_size,
        )
        n.eval(); return n

    net_a = make_net(); net_b = make_net()

    def load(net, wb):
        buf = io.BytesIO(wb)
        net.load_state_dict(torch.load(buf, map_location='cpu', weights_only=True))
        net.eval()

    def mcts(net, noise):
        return MCTS(
            network=net,
            num_simulations=config.mcts.num_simulations,
            c_puct=config.mcts.c_puct,
            dirichlet_alpha=config.mcts.dirichlet_alpha if noise else 0.0,
            dirichlet_epsilon=config.mcts.dirichlet_epsilon if noise else 0.0,
            batch_size=getattr(config.mcts,'batch_size',1),
            c_virtual_loss=getattr(config.mcts,'c_virtual_loss',0.5),
            # ---- NEW: pass game‑length and adjudication parameters ----
            max_game_length=config.selfplay.max_game_length,
            adjudicate_material=config.selfplay.adjudicate_material,
            piece_values=config.selfplay.piece_values,
            adjudicate_graded=getattr(config.selfplay, 'adjudicate_graded', True),
            adjudicate_scaling=getattr(config.selfplay, 'adjudicate_scaling', 9.0),
        )

    while not shutdown_event.is_set():
        try: task = task_queue.get(timeout=2.0)
        except queue.Empty: continue
        if task is None: break

        t = task.get('type', 'selfplay')

        if t == 'selfplay':
            if inference_client is not None:
                eng = mcts(inference_client, noise=True)
            else:
                load(net_a, task['weights'])
                eng = mcts(net_a, noise=True)

            result_queue.put({
                'worker_id': worker_id, 'type': 'live_start',
                'game_type': 'selfplay', 'match_info': None,
            })
            fens = []; ucis = []; mdata = []
            def on_sp(fen, uci, mn, ms=None):
                fens.append(fen); ucis.append(uci); mdata.append(ms)
                result_queue.put({
                    'worker_id': worker_id, 'type': 'live_move',
                    'fen': fen, 'move': uci, 'move_number': mn, 'mcts_stats': ms,
                })
            gd, gi = play_one_game(
                eng, config.selfplay.max_game_length,
                config.selfplay.adjudicate_material, piece_values,
                config.selfplay.temperature_threshold,
                config.selfplay.temperature_high, config.selfplay.temperature_low,
                on_move=on_sp,
                adjudicate_graded=getattr(config.selfplay, 'adjudicate_graded', True),
                adjudicate_scaling=getattr(config.selfplay, 'adjudicate_scaling', 9.0),
            )
            result_queue.put({
                'worker_id': worker_id, 'type': 'live_end',
                'result': gi['result_str'], 'termination': gi['termination'],
            })
            result_queue.put({
                'worker_id': worker_id, 'type': 'selfplay',
                'game_data': [(s.tolist(), p.tolist(), float(v), m.tolist()) for s, p, v, m in gd],
                'game_info': gi, 'fens': fens, 'moves': ucis, 'mcts_stats': mdata,
            })

        elif t == 'eval':
            a_is_white = task['a_is_white']
            eval_type = task['eval_type']
            game_label = task.get('game_label', '')
            fens = []; ucis = []; mdata = []

            gt_label = 'gating' if eval_type == 'gating' else 'reference'
            result_queue.put({
                'worker_id': worker_id, 'type': 'live_start',
                'game_type': gt_label, 'match_info': game_label,
            })

            def on_ev_live(fen, uci, mn, ms=None):
                fens.append(fen); ucis.append(uci); mdata.append(ms)
                result_queue.put({
                    'worker_id': worker_id, 'type': 'live_move',
                    'fen': fen, 'move': uci, 'move_number': mn, 'mcts_stats': ms,
                })

            if eval_type == 'gating':
                # ── GPU path: route both nets through the dual-network server ──
                if inference_client is not None:
                    client_a = InferenceClient(worker_id, request_queue, response_queue, network_id=0,
                                               shared_buffers=shared_buffers)
                    client_b = InferenceClient(worker_id, request_queue, response_queue, network_id=1,
                                               shared_buffers=shared_buffers)
                    ea = mcts(client_a, False); eb = mcts(client_b, False)
                else:
                    load(net_a, task['weights_a']); load(net_b, task['weights_b'])
                    ea = mcts(net_a, False); eb = mcts(net_b, False)

                board = chess.Board(); mc = 0
                root_a = None; root_b = None
                while not board.is_game_over() and mc < config.selfplay.max_game_length:
                    is_a = (board.turn == chess.WHITE) == a_is_white
                    e = ea if is_a else eb
                    if is_a:
                        if root_a is None: root_a = e.get_root(board)
                        r = root_a
                    else:
                        if root_b is None: root_b = e.get_root(board)
                        r = root_b
                    e.search(r)
                    _, mv = e.select_move_with_temperature(r, 0.1)
                    if mv is None: break
                    ms = {
                        'selected_move': mv.uci(),
                        'candidates': [{'move': c['move'], 'N': c['N'], 'W': c['W'], 'Q': c['Q'], 'P': c['P']}
                                       for c in e.get_root_child_stats(r)[:8]]
                    }
                    board.push(mv); mc += 1
                    on_ev_live(board.fen(), mv.uci(), mc, ms)
                    if is_a:
                        root_a = e.recycle_tree(r, mv)
                        root_b = None
                    else:
                        root_b = e.recycle_tree(r, mv)
                        root_a = None
                game_result = board.result() if board.is_game_over() else '*'

            else:  # reference
                sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
                from evaluation import alpha_beta_best_move

                # ── GPU path: network A via the GPU server (network_id=0) ──
                if inference_client is not None:
                    client_a = InferenceClient(worker_id, request_queue, response_queue, network_id=0,
                                               shared_buffers=shared_buffers)
                    ea = mcts(client_a, False)
                else:
                    load(net_a, task['weights_a'])
                    ea = mcts(net_a, False)

                board = chess.Board(); mc = 0
                root_net = None
                while not board.is_game_over() and mc < config.selfplay.max_game_length:
                    net_turn = (board.turn == chess.WHITE) == a_is_white
                    if net_turn:
                        if root_net is None: root_net = ea.get_root(board)
                        r = root_net
                        ea.search(r)
                        _, mv = ea.select_move_with_temperature(r, 0.1)
                        if mv is None: break
                        ms = {
                            'selected_move': mv.uci(),
                            'candidates': [{'move': c['move'], 'N': c['N'], 'W': c['W'], 'Q': c['Q'], 'P': c['P']}
                                           for c in ea.get_root_child_stats(r)[:8]]
                        }
                        board.push(mv); mc += 1
                        on_ev_live(board.fen(), mv.uci(), mc, ms)
                        root_net = ea.recycle_tree(r, mv)
                    else:
                        mv = alpha_beta_best_move(board, config.alpha_beta.depth)
                        if mv is None: break
                        board.push(mv); mc += 1
                        on_ev_live(board.fen(), mv.uci(), mc, None)
                        root_net = None
                game_result = board.result() if board.is_game_over() else '*'

            result_queue.put({
                'worker_id': worker_id, 'type': 'live_end',
                'result': game_result, 'termination': gt_label,
            })
            result_queue.put({
                'worker_id': worker_id, 'type': 'eval',
                'result': game_result, 'eval_type': eval_type,
                'a_is_white': a_is_white, 'game_label': game_label,
                'fens': fens, 'moves': ucis, 'mcts_stats': mdata,
            })

        elif t == 'profile':
            num_profile_games = task.get('num_games', 3)
            if inference_client is not None:
                eng = mcts(inference_client, noise=True)
            else:
                load(net_a, task['weights'])
                eng = mcts(net_a, noise=True)
            from worker_cpu_profiler import profile_in_worker
            summary = profile_in_worker(eng, config, piece_values, num_profile_games)
            result_queue.put({
                'worker_id': worker_id, 'type': 'profile_done',
                'summary': summary,
            })

    result_queue.put({'worker_id': worker_id, 'done': True})


# ─────────────────────────────────────────────────────────────────────────────
# Manager
# ─────────────────────────────────────────────────────────────────────────────

class ParallelSelfPlay:
    def __init__(self, config, num_workers=8, stats_queue=None):
        self.config = config
        self.num_workers = num_workers
        self._workers = []
        self._task_qs = []
        self._result_q = mp.Queue()
        self._shutdown = mp.Event()

        self._use_gpu = (getattr(config, 'inference', None)
                         and getattr(config.inference, 'use_gpu', False))
        self._gpu_server_process = None
        self._stats_queue = stats_queue  # optional mp.Queue for benchmark timing
        self._gpu_stats = None

        # Shared-memory transport for zero-copy IPC
        self._use_shared_memory = False
        self._shared_transport = None

        if self._use_gpu:
            self._request_q = mp.Queue()
            self._weight_q = mp.Queue()
            self._response_qs = {}
            self._gpu_ready = mp.Event()
            self._gpu_shutdown = mp.Event()

            # Check if shared memory is enabled in config
            inf_cfg = getattr(config, 'inference', None)
            self._use_shared_memory = (
                inf_cfg is not None and
                getattr(inf_cfg, 'use_shared_memory', True)
            )

            if self._use_shared_memory:
                # Buffer must be large enough for the largest possible batch.
                # MCTS batch_size can exceed inference.max_batch (the latter
                # only limits timer-based single-request aggregation, not
                # pre-stacked batch requests from MCTS).
                inf_max_batch = getattr(inf_cfg, 'max_batch', 64) if inf_cfg else 64
                mcts_batch_size = getattr(config.mcts, 'batch_size', 1)
                max_batch = max(inf_max_batch, mcts_batch_size)
                state_dtype_str = getattr(inf_cfg, 'state_dtype', 'float16') if inf_cfg else 'float16'
                import numpy as _np
                state_dtype = _np.float16 if state_dtype_str == 'float16' else _np.float32
                self._shared_transport = SharedMemoryTransport(
                    num_workers=num_workers,
                    max_batch=max_batch,
                    state_dtype=state_dtype,
                )
                self._shared_transport.create_buffers()
                print(f"[INFO] Shared-memory IPC enabled "
                      f"(dtype={state_dtype_str}, max_batch={max_batch})")

    def start(self):
        if self._use_gpu:
            for i in range(self.num_workers):
                self._response_qs[i] = mp.Queue(maxsize=256)
            self._start_gpu_server()

        cd = self.config.to_dict()
        for i in range(self.num_workers):
            tq = mp.Queue(maxsize=4)
            self._task_qs.append(tq)
            kwargs = {}
            if self._use_gpu:
                kwargs['request_queue'] = self._request_q
                kwargs['response_queue'] = self._response_qs[i]
                if self._use_shared_memory and self._shared_transport is not None:
                    kwargs['shared_buffers'] = self._shared_transport.get_worker_buffers(i)
            p = mp.Process(target=_worker_process,
                           args=(i, tq, self._result_q, cd, self._shutdown),
                           kwargs=kwargs, daemon=True)
            p.start()
            self._workers.append(p)

    def _start_gpu_server(self):
        from gpu_server import GPUInferenceServer
        shared_bufs = None
        if self._use_shared_memory and self._shared_transport is not None:
            shared_bufs = self._shared_transport.get_all_worker_buffers()
        server = GPUInferenceServer(
            config=self.config,
            request_queue=self._request_q,
            response_queues=self._response_qs,
            weight_queue=self._weight_q,
            ready_event=self._gpu_ready,
            shutdown_event=self._gpu_shutdown,
            shared_buffers=shared_bufs,
            stats_queue=self._stats_queue,
        )
        self._gpu_server_process = mp.Process(target=server.run, daemon=True)
        self._gpu_server_process.start()
        print("[INFO] Waiting for GPU server to warm up shaders...")
        self._gpu_ready.wait()
        print("[INFO] GPU server ready")

    def _serialize_weights(self, network):
        import torch
        buf = io.BytesIO(); torch.save(network.state_dict(), buf); return buf.getvalue()

    def push_selfplay(self, network):
        wb = self._serialize_weights(network)

        if self._use_gpu:
            while not self._weight_q.empty():
                try: self._weight_q.get_nowait()
                except: pass
            # network_id 0 = the network self-play (and reference eval) uses.
            self._weight_q.put((0, wb))

        for tq in self._task_qs:
            while not tq.empty():
                try: tq.get_nowait()
                except: pass
            task = {'type': 'selfplay'}
            if not self._use_gpu:
                task['weights'] = wb
            try: tq.put_nowait(task)
            except queue.Full: pass

    def push_weights(self, network): self.push_selfplay(network)

    def push_weights_to_gpu(self, network):
        """Push updated weights to the GPU server without disturbing task queues.

        Called after training so the GPU inference server picks up the latest
        network weights on its next iteration.  Unlike ``push_selfplay``, this
        does NOT clear or refill the per-worker task queues — in-flight
        self-play games continue with their current task while the GPU server
        seamlessly switches to the new weights for subsequent inference
        requests.

        No-op when GPU inference is disabled.
        """
        if not self._use_gpu:
            return
        wb = self._serialize_weights(network)
        while not self._weight_q.empty():
            try: self._weight_q.get_nowait()
            except: pass
        # network_id 0 = the network self-play (and reference eval) uses.
        self._weight_q.put((0, wb))

    def push_eval_weights(self, wb_latest, wb_best):
        """Push both networks to the GPU server ahead of a gating eval round.

        network_id 0 = latest (network A), network_id 1 = best (network B).
        No-op if GPU inference is disabled — in that case each eval task
        carries its own weights ('weights_a'/'weights_b') and workers load
        them locally instead.
        """
        if not self._use_gpu:
            return
        self._weight_q.put((0, wb_latest))
        self._weight_q.put((1, wb_best))

    def dispatch_eval_games(self, tasks):
        done = 0
        for i, task in enumerate(tasks):
            wid = i % self.num_workers
            try: self._task_qs[wid].put(task, timeout=60.0); done += 1
            except queue.Full: pass
        return done

    def collect_one(self, timeout=300.0):
        try: return self._result_q.get(timeout=timeout)
        except queue.Empty: return None

    def collect_available(self):
        out = []
        while True:
            try: out.append(self._result_q.get_nowait())
            except queue.Empty: break
        return out

    def dispatch_profile(self, network, num_games=3, worker_id=None):
        wb = self._serialize_weights(network)
        task = {'type': 'profile', 'weights': wb, 'num_games': num_games}
        if worker_id is not None:
            indices = [worker_id]
        else:
            indices = range(self.num_workers)
        for wid in indices:
            try: self._task_qs[wid].put_nowait(task)
            except (queue.Full, IndexError): pass

    def drain(self):
        while True:
            try: self._result_q.get_nowait()
            except queue.Empty: break

    def stop(self):
        self._shutdown.set()
        for tq in self._task_qs:
            try: tq.put_nowait(None)
            except: pass
        for p in self._workers:
            p.join(timeout=10)
            if p.is_alive(): p.kill()
        self._workers.clear(); self._task_qs.clear()

        if self._use_gpu and self._gpu_server_process is not None:
            self._gpu_shutdown.set()
            try: self._request_q.put_nowait(None)
            except: pass
            self._gpu_server_process.join(timeout=10)
            if self._gpu_server_process.is_alive():
                self._gpu_server_process.kill()
            # Drain the server's timing stats (benchmark mode)
            if self._stats_queue is not None:
                try:
                    self._gpu_stats = self._stats_queue.get(timeout=5)
                except queue.Empty:
                    self._gpu_stats = None
            self._gpu_server_process = None

    def get_gpu_stats(self):
        """Return the GPU server timing dict, or None if unavailable."""
        return self._gpu_stats
