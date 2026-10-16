"""Self-play game generation and replay buffer for AlphaZero.

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
onto CPU. Reference eval (network vs reference player) also uses network_id=0
when GPU inference is active. This mirrors the self-play path and avoids the
CPU fallback silently running whenever use_gpu is set.

NOTE 3: when GPU inference is enabled and ``config.inference.use_shared_memory``
is True (default), per-worker shared-memory buffers are allocated to eliminate
pickle/pipe overhead for state and policy/value arrays. Only ~100 bytes of
metadata travel through the queue per inference request.

All game-specific mechanics (chess, connect4, ...) are accessed through
``game_api``; see game_api.py.
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

import game_api
from encoding import board_to_tensor, get_legal_move_mask, NUM_ACTIONS
from network import AlphaZeroNet
from mcts import MCTS
from inference_client import InferenceClient
from shared_memory_transport import SharedMemoryTransport
from forced_mate import ForceMateMCTS, reset_memo


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
        # Emit in logical FIFO order (oldest first) so the circular write
        # pointer can be reset to 0 on load and still overwrite the oldest
        # positions first.  Without this the on-disk order is arbitrary once
        # the buffer has wrapped, and reloading silently scrambles age order.
        if n < self.max_size:
            order = range(n)
        else:
            order = list(range(self._write_idx, n)) + list(range(0, self._write_idx))
        states   = np.stack([self.buffer[i][0] for i in order])
        policies = np.stack([self.buffer[i][1] for i in order])
        values   = np.array([self.buffer[i][2] for i in order], dtype=np.float32)
        masks    = np.stack([self.buffer[i][3] for i in order])
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
                masks = np.ones((n, game_api.num_actions()), dtype=np.float32)
            if n > max_size:
                # Keep the most recent positions (saved oldest→newest).
                print(f"[WARN] {path} holds {n} positions, exceeds max_size "
                      f"{max_size}; keeping the most recent {max_size}.")
                states   = states[-max_size:]
                policies = policies[-max_size:]
                values   = values[-max_size:]
                masks    = masks[-max_size:]
                n = max_size
            for i in range(n):
                buf.buffer.append((states[i], policies[i], float(values[i]), masks[i]))
            # Buffer[0] is now the oldest position and n <= max_size, so the
            # circular write pointer restarts at 0 (overwrites oldest first
            # once the buffer refills).  Note: appending directly above does
            # not advance _write_idx, hence the explicit reset.
            buf._write_idx       = 0
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

# Piece letters accepted in a ``start_pieces`` spec (case = color).
_START_PIECE_TYPES = "PNBRQK"


def _parse_start_pieces(pieces):
    """Parse a piece spec like 'KRkr' (uppercase = white, lowercase = black)
    into a list of (piece_symbol, color).  Raises ValueError on bad input."""
    placed = []
    for ch in pieces:
        if ch.upper() not in _START_PIECE_TYPES:
            raise ValueError(f"invalid piece in start spec {pieces!r}: {ch!r}"
                             " (use P N B R Q K, case = color)")
        color = chess.WHITE if ch.isupper() else chess.BLACK
        placed.append((ch.upper(), color))
    for color, sym in ((chess.WHITE, "K"), (chess.BLACK, "K")):
        if sum(1 for s, c in placed if s == sym and c == color) != 1:
            raise ValueError(f"start spec {pieces!r} must contain exactly one "
                             f"{'white' if color else 'black'} king")
    if not placed:
        raise ValueError("empty start spec")
    return placed


def _split_piece_specs(spec):
    """Split a start-pieces value into individual specs so several endgames
    can be mixed, e.g. ``'KQk,KRk'`` (commas, semicolons or pipes all work).
    Returns a list of non-empty trimmed specs; raises ValueError when the
    value contains nothing usable."""
    if spec is None:
        return []
    if not isinstance(spec, str):
        raise ValueError(f"start_pieces must be a string, got {type(spec).__name__}")
    parts = [p.strip() for p in spec.replace(';', ',').replace('|', ',').split(',')
             if p.strip()]
    if not parts:
        raise ValueError(f"empty start_pieces spec: {spec!r}")
    return parts


# Token(s) that stand for the standard chess starting position inside a
# pieces mix, so the agent keeps playing full games and doesn't unlearn them.
_STANDARD_TOKENS = frozenset({"standard", "start", "full", "chess"})


def _parse_piece_mix(specs):
    """Parse raw spec strings into ``[(piece_spec, weight)]``.

    Each entry may carry an optional ``:weight`` suffix (e.g. ``'KQk:10'``,
    ``'standard:80'``).  Weights are relative to each other, so percentages
    like ``standard:80,KQk:10,KRk:10`` give 80% / 10% / 10%; a plain spec
    (no suffix) gets weight 1.  The token ``standard`` (aliases: ``start``,
    ``full``, ``chess``) means the standard starting position.  Raises
    ValueError on malformed weights or invalid piece specs."""
    import re as _re
    out = []
    for raw in specs:
        if raw in _STANDARD_TOKENS:
            out.append(("standard", 1.0))
            continue
        weight = 1.0
        name = raw
        m = _re.match(r"^(?P<name>.+):(?P<w>[\d.]+)%?$", raw.strip())
        if m:
            name = m.group("name").strip()
            weight = float(m.group("w"))
        if weight <= 0:
            raise ValueError(f"pieces weight must be > 0, got {raw!r}")
        if name in _STANDARD_TOKENS:
            out.append(("standard", weight))
        else:
            _parse_start_pieces(name)  # fail fast on a bad spec
            out.append((name, weight))
    if not out:
        raise ValueError("empty pieces mix")
    return out


def _pick_start_board_kind(spec, rng=None, randomize_colors=False):
    """Like ``_pick_start_board`` but also reports what kind of start was
    chosen: ``('standard', board)`` for the normal starting position or
    ``('endgame', board)`` when the board was generated from a piece set.
    The caller can use the kind to e.g. keep endgame-starting games at a
    fixed low temperature."""
    import random as _random
    r = rng or _random
    entries = _parse_piece_mix(_split_piece_specs(spec))
    if len(entries) > 1:
        total = sum(w for _, w in entries)
        pick = r.uniform(0.0, total)
        acc = 0.0
        name = entries[-1][0]
        for n, w in entries:
            acc += w
            if pick <= acc:
                name = n
                break
    else:
        name = entries[0][0]
    if name == "standard":
        return "standard", chess.Board()
    return "endgame", random_start_board(name, rng=r,
                                         randomize_colors=randomize_colors)


def _pick_start_board(spec, rng=None, randomize_colors=False):
    """Pick one entry at random from a (possibly mixed, possibly weighted)
    start-pieces value and build a fresh board from it.

    ``'KQk,KRk'`` produces a ~50/50 mix of queen and rook endgames with the
    colour assignment flipped per game when ``randomize_colors`` is set.
    ``'standard:80,KQk:10,KRk:10'`` keeps 80% of games on the normal start
    and splits the rest.  The ``standard`` entry ignores ``randomize_colors``
    (its colours are fixed by the rules of chess)."""
    kind, board = _pick_start_board_kind(spec, rng=rng,
                                         randomize_colors=randomize_colors)
    return board


def _fen_from_placement(by_sq):
    """Build a FEN board-string from {square: piece_char}, white to move."""
    rows = []
    for r in range(7, -1, -1):
        row_parts = []
        empties = 0
        for f in range(8):
            ch = by_sq.get(r * 8 + f)
            if ch is None:
                empties += 1
            else:
                if empties:
                    row_parts.append(str(empties))
                    empties = 0
                row_parts.append(ch)
        if empties:
            row_parts.append(str(empties))
        rows.append(''.join(row_parts))
    return '/'.join(rows)


def random_start_board(pieces, rng=None, randomize_colors=False):
    """Build a random legal starting board containing exactly ``pieces``
    (e.g. ``'KRkr'`` for king+rook vs king+rook).  The side to move is chosen
    randomly among the legal choices.  A fresh board is returned on every
    call, so each game starts from a different position.

    With ``randomize_colors=True``, for asymmetric specs (e.g. ``'KQQk'``)
    each game randomly decides which colour gets the strong pieces: only the
    assignment of colours to the spec is (uniformly) flipped per game, never
    the piece counts, so both sides get equal training time on both roles."""
    import random as _random
    r = rng or _random
    placed = _parse_start_pieces(pieces)
    n = len(placed)

    for _ in range(400):
        by_sq = {}
        for (sym, color), sq in zip(placed, r.sample(range(64), n)):
            by_sq[sq] = sym if color == chess.WHITE else sym.lower()

        # Pawns may not stand on the first or last rank.
        if any(ch.upper() == 'P' and chess.square_rank(sq) in (0, 7)
               for sq, ch in by_sq.items()):
            continue
        wking = next((sq for sq, ch in by_sq.items() if ch == 'K'), None)
        bking = next((sq for sq, ch in by_sq.items() if ch == 'k'), None)
        if wking is None or bking is None:
            continue
        # Kings must not be adjacent (that would be an illegal check).
        if max(abs(wking & 7 - bking & 7), abs((wking >> 3) - (bking >> 3))) <= 1:
            continue
        # Two bishops of the same colour must sit on opposite-coloured
        # squares: same-coloured bishops can never force mate, so those
        # starts are never generated (square colour = (file + rank) % 2).
        bishops_ok = True
        for bishop_sym in ('B', 'b'):
            parities = [((sq & 7) + (sq >> 3)) & 1
                        for sq, ch in by_sq.items() if ch == bishop_sym]
            if len(set(parities)) != len(parities):
                bishops_ok = False
                break
        if not bishops_ok:
            continue

        # Build the board straight from the FEN so the (strict) backend
        # validates it.  A turn is only usable if the resulting position is
        # accepted (i.e. no king exposed to check in either direction).
        board_str = _fen_from_placement(by_sq)
        legal_boards = []
        for turn in (chess.WHITE, chess.BLACK):
            try:
                cand = chess.Board(
                    f"{board_str} {'w' if turn == chess.WHITE else 'b'} - - 0 1")
            except ValueError:
                continue
            if not cand.legal_moves:
                continue
            legal_boards.append((turn, cand))
        if legal_boards:
            turn, board = r.choice(legal_boards)
            # Randomly flip the colour-assignment across the board (case of
            # every piece mirrored) *and* the side to move, so the whole
            # position is a colour-mirror of otherwise identical material:
            # for an asymmetric spec like 'KQQk' either side gets the queens.
            if randomize_colors and r.random() < 0.5:
                mirrored = {sq: ch.swapcase() for sq, ch in by_sq.items()}
                mirrored_str = _fen_from_placement(mirrored)
                other = chess.BLACK if turn == chess.WHITE else chess.WHITE
                try:
                    flipped = chess.Board(
                        f"{mirrored_str} "
                        f"{'w' if other == chess.WHITE else 'b'} - - 0 1")
                except ValueError:
                    flipped = None
                if flipped is not None and flipped.legal_moves:
                    board = flipped
            return board

    raise ValueError(f"could not generate a legal position for pieces {pieces!r}")


def play_one_game(mcts_engine, max_game_length=150, adjudicate_material=True,
                  piece_values=None, temp_threshold=30, temp_high=1.0, temp_low=0.1,
                  temperature_override=None, verbose=False, on_move=None,
                  adjudicate_graded=True, adjudicate_scaling=9.0,
                  endgame_temperature=None, endgame_piece_gate=6,
                  endgame_num_simulations=0, fen=None, start_pieces=None,
                  start_colors_random=False):
    if piece_values is None:
        piece_values = {'P': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9}

    # Per-game memo reset for the forced-mate proof search (no-op when
    # the override is disabled).  Keeps the memo bounded in long-lived
    # worker processes that play many games.
    reset_memo()

    start = {}
    if fen:
        start['fen'] = fen
    if start_pieces:
        start['start_pieces'] = start_pieces
        start['randomize_colors'] = start_colors_random
    board, start_kind = game_api.new_state(start)
    game_states = []
    mcts_stats_list = []
    move_count = 0
    termination = "unknown"
    outcome = 0.0
    root = None  # Tree recycling
    base_num_simulations = getattr(mcts_engine, 'num_simulations', 0)

    # Original loop – stop only on automatic game‑overs (not claimable draws)
    while (not game_api.is_terminal(board)
           and not game_api.is_draw_claim(board)
           and move_count < max_game_length):
        n_pieces = game_api.piece_count(board)   # None for piece-less games
        # Deeper simulation budget in sparse endgames (chess only;
        # ``n_pieces`` is None for games without pieces).
        if endgame_num_simulations > 0 and n_pieces is not None:
            mcts_engine.num_simulations = (endgame_num_simulations
                                           if n_pieces <= endgame_piece_gate
                                           else base_num_simulations)
        if root is None:
            root = mcts_engine.get_root(board)

        visit_policy, best_move, stats = mcts_engine.search(root)
        move_candidates = mcts_engine.get_root_child_stats(root)

        if temperature_override is not None:
            temp = temperature_override
        elif start_kind == "endgame":
            # Games that begin from a piece-set start are already in the
            # endgame, so there is no high-temperature opening stage: play
            # them at the fixed low temperature from move one.
            temp = temp_low
        elif endgame_temperature is not None and n_pieces is not None and n_pieces <= endgame_piece_gate:
            temp = endgame_temperature
        else:
            temp = get_temperature(move_count, temp_threshold, temp_high, temp_low)
        visit_policy, selected_move = mcts_engine.select_move_with_temperature(root, temp)

        if selected_move is None:
            selected_move = best_move
            if selected_move is None:
                # This should never happen after fixing MCTS expansion (no claim_draw=True)
                with open("mcts_empty_root.log", "a") as f:
                    f.write(f"Empty root at move {move_count}: {game_api.describe(board)}\n"
                            f"Legal moves: {[game_api.move_str(m) for m in game_api.legal_moves(board)]}\n"
                            f"Children after search: {len(root.children)}\n\n")
                break

        st = game_api.encode(board)
        cp = game_api.player_sign(board)
        legal_mask = game_api.mask(board)
        game_states.append((st, visit_policy.copy(), cp, legal_mask))
        mcts_stats_list.append(stats)

        mcts_move_data = {
            'selected_move': game_api.move_str(selected_move),
            'candidates':     move_candidates,
            # Augment the per-move snapshot with search-effort so the GUI's
            # "why this move?" panel can report how deep / how much MCTS ran.
            'depth':          round(float(stats.get('avg_depth') or 0), 2),
            'sims':           int(stats.get('num_simulations')
                                  or sum(c['N'] for c in move_candidates)),
        }
        game_api.apply(board, selected_move)
        move_count += 1
        if on_move:
            on_move(game_api.describe(board), game_api.move_str(selected_move), move_count, mcts_move_data)

        root = mcts_engine.recycle_tree(root, selected_move)

    material_diff = None
    mat_fn = getattr(game_api.get_game(), 'material_diff', None)
    if mat_fn is not None:
        material_diff = mat_fn(board, piece_values)

    # Determine outcome and termination reason via the game adapter.
    _res = game_api.final_result(
        board, move_count, max_game_length,
        adjudicate_material=adjudicate_material, piece_values=piece_values,
        adjudicate_graded=adjudicate_graded, adjudicate_scaling=adjudicate_scaling)
    outcome = _res['outcome']
    termination = _res['termination']
    result_str = _res['result_str']

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
    # Use the forced-mate override for endgame conversion when enabled
    # (see forced_mate.py).  The proof search is gated to sparse boards
    # (<= force_mate_gate_pieces pieces) and cheap (sub-ms).  It is a
    # chess-only feature, so it is disabled for other games.
    force_mate = (getattr(config.selfplay, 'force_mate', False)
                  and game_api.is_game('chess'))
    mcts_cls = ForceMateMCTS if force_mate else MCTS
    mcts_kwargs = {}
    if force_mate:
        mcts_kwargs = {
            'max_force_plies': getattr(config.selfplay, 'force_mate_max_plies', 5),
            'min_pieces_gate': getattr(config.selfplay, 'force_mate_gate_pieces', 8),
            'deep_plies': getattr(config.selfplay, 'force_mate_deep_plies', 9),
            'deep_gate_pieces': getattr(config.selfplay, 'force_mate_deep_gate_pieces', 4),
        }
    mcts_engine = mcts_cls(
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
        force_mate_in_one=getattr(config.mcts, 'force_mate_in_one', True),
        inference_pipeline=getattr(config.inference, 'client_concurrency', 1)
            if getattr(config, 'inference', None) else 1,
        **mcts_kwargs,
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
        endgame_temperature=getattr(config.selfplay, 'endgame_temperature', None),
        endgame_piece_gate=getattr(config.selfplay, 'endgame_piece_gate', 6),
        endgame_num_simulations=getattr(config.selfplay, 'endgame_num_simulations', 0),
        fen=getattr(config.selfplay, 'start_fen', None),
        start_pieces=getattr(config.selfplay, 'start_pieces', None),
        start_colors_random=getattr(config.selfplay, 'start_colors_random', False),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Worker process
# ─────────────────────────────────────────────────────────────────────────────

def _game_result_str(state, move_count, max_game_length):
    """Display result string for an eval match ('*' if the game lives on)."""
    if game_api.is_terminal(state):
        return game_api.final_result(
            state, move_count, max_game_length)['result_str']
    return '*'


def _mcts_child_stats(engine, root, mv, topn=8):
    return {
        'selected_move': game_api.move_str(mv),
        'candidates': [{'move': c['move'], 'N': c['N'], 'W': c['W'],
                        'Q': c['Q'], 'P': c['P']}
                       for c in engine.get_root_child_stats(root)[:topn]],
    }


def play_two_engine_match(engine_a, engine_b, a_is_first, max_game_length,
                          temperature=0.1, on_move=None):
    """Play one game between two MCTS engines of the active game.

    Returns ``(state, move_count, result_str)``.  Each engine keeps its own
    search-tree root; the opponent's cached root is invalidated after every
    move (a position always changes sides).
    """
    state, _ = game_api.new_state()
    move_count = 0
    root_a = None
    root_b = None
    while not game_api.is_terminal(state) and move_count < max_game_length:
        is_a = (game_api.player_parity(state) == 0) == a_is_first
        engine = engine_a if is_a else engine_b
        if is_a:
            if root_a is None:
                root_a = engine.get_root(state)
            r = root_a
        else:
            if root_b is None:
                root_b = engine.get_root(state)
            r = root_b
        engine.search(r)
        _, mv = engine.select_move_with_temperature(r, temperature)
        if mv is None:
            break
        ms = _mcts_child_stats(engine, r, mv)
        game_api.apply(state, mv)
        move_count += 1
        if on_move:
            on_move(game_api.describe(state), game_api.move_str(mv), move_count, ms)
        if is_a:
            root_a = engine.recycle_tree(r, mv)
            root_b = None
        else:
            root_b = engine.recycle_tree(r, mv)
            root_a = None
    return state, move_count, _game_result_str(state, move_count, max_game_length)


def play_engine_vs_reference(engine, that_first, max_game_length,
                             temperature=0.1, depth=2, on_move=None):
    """Play one game between an MCTS engine and the game's reference player.

    Returns ``(state, move_count, result_str)``.
    """
    state, _ = game_api.new_state()
    move_count = 0
    root_net = None
    while not game_api.is_terminal(state) and move_count < max_game_length:
        net_turn = (game_api.player_parity(state) == 0) == that_first
        if net_turn:
            if root_net is None:
                root_net = engine.get_root(state)
            r = root_net
            engine.search(r)
            _, mv = engine.select_move_with_temperature(r, temperature)
            if mv is None:
                break
            ms = _mcts_child_stats(engine, r, mv)
            game_api.apply(state, mv)
            move_count += 1
            if on_move:
                on_move(game_api.describe(state), game_api.move_str(mv), move_count, ms)
            root_net = engine.recycle_tree(r, mv)
        else:
            mv = game_api.reference_move(state, depth=depth)
            if mv is None:
                break
            game_api.apply(state, mv)
            move_count += 1
            if on_move:
                on_move(game_api.describe(state), game_api.move_str(mv), move_count, None)
            root_net = None
    return state, move_count, _game_result_str(state, move_count, max_game_length)

def _set_game_from_config_dict(config_dict):
    """Make worker processes adopt the config's active game name."""
    name = game_api.active_game_name()
    if isinstance(config_dict, dict):
        section = config_dict.get('game')
        if isinstance(section, dict):
            name = section.get('name', name)
        elif isinstance(section, str):
            name = section
    game_api.set_game(name)


def _worker_process(worker_id, task_queue, result_queue, config_dict, shutdown_event,
                    request_queue=None, response_queue=None, shared_buffers=None,
                    concurrency=1, selfplay_instances=None, eval_instance=None,
                    games_per_worker=1):
    """Dispatch entry point.

    games_per_worker == 1 keeps the original single-game-per-worker loop
    (byte-identical behaviour).  games_per_worker > 1 runs that many
    concurrent games via ``_worker_loop_multi``, each with its own private
    GPU inference endpoint.
    """
    _set_game_from_config_dict(config_dict)
    if max(1, int(games_per_worker or 1)) <= 1:
        _worker_loop_single(
            worker_id, task_queue, result_queue, config_dict, shutdown_event,
            request_queue=request_queue, response_queue=response_queue,
            shared_buffers=shared_buffers, concurrency=concurrency)
        return
    _worker_loop_multi(
        worker_id, task_queue, result_queue, config_dict, shutdown_event,
        request_queue=request_queue, selfplay_instances=selfplay_instances,
        eval_instance=eval_instance, concurrency=concurrency,
        games_per_worker=games_per_worker)


def _worker_loop_single(worker_id, task_queue, result_queue, config_dict, shutdown_event,
                        request_queue=None, response_queue=None, shared_buffers=None,
                        concurrency=1):
    import torch, sys, os
    from config import Config
    from network import AlphaZeroNet
    from mcts import MCTS

    # Limit Torch to a single intra-op thread per worker process.
    # With 8+ worker processes each running MCTS with small inference
    # batches, multi-threading causes contention and cache thrashing.
    # Single-thread is faster for small batches and avoids oversubscription.
    torch.set_num_threads(1)

    _set_game_from_config_dict(config_dict)
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
            shared_buffers=shared_buffers, concurrency=concurrency,
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

    # The forced-mate override is wired ONLY into the selfplay branch below
    # (force_mate=True).  Eval branches (gating / reference) keep it disabled
    # so they measure true strength.
    force_mate_enabled = (getattr(config.selfplay, 'force_mate', False)
                          and game_api.is_game('chess'))

    force_mate_in_one = getattr(config.mcts, 'force_mate_in_one', True)

    def mcts(net, noise, force_mate=False):
        cls = ForceMateMCTS if force_mate else MCTS
        mcts_kwargs = {}
        if force_mate:
            mcts_kwargs = {
                'max_force_plies': getattr(config.selfplay, 'force_mate_max_plies', 5),
                'min_pieces_gate': getattr(config.selfplay, 'force_mate_gate_pieces', 8),
                'deep_plies': getattr(config.selfplay, 'force_mate_deep_plies', 9),
                'deep_gate_pieces': getattr(config.selfplay, 'force_mate_deep_gate_pieces', 4),
            }
        return cls(
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
            force_mate_in_one=force_mate_in_one,
            inference_pipeline=getattr(config.inference, 'client_concurrency', 1)
                if getattr(config, 'inference', None) else 1,
            **mcts_kwargs,
        )

    while not shutdown_event.is_set():
        try: task = task_queue.get(timeout=2.0)
        except queue.Empty: continue
        if task is None: break

        t = task.get('type', 'selfplay')

        try:
            if t == 'selfplay':
                if inference_client is not None:
                    eng = mcts(inference_client, noise=True, force_mate=force_mate_enabled)
                else:
                    load(net_a, task['weights'])
                    eng = mcts(net_a, noise=True, force_mate=force_mate_enabled)

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
                    endgame_temperature=getattr(config.selfplay, 'endgame_temperature', None),
                    endgame_piece_gate=getattr(config.selfplay, 'endgame_piece_gate', 6),
                    endgame_num_simulations=getattr(config.selfplay, 'endgame_num_simulations', 0),
                    fen=getattr(config.selfplay, 'start_fen', None),
                    start_pieces=getattr(config.selfplay, 'start_pieces', None),
                    start_colors_random=getattr(config.selfplay, 'start_colors_random', False),
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
                                                   shared_buffers=shared_buffers, concurrency=concurrency)
                        client_b = InferenceClient(worker_id, request_queue, response_queue, network_id=1,
                                                   shared_buffers=shared_buffers, concurrency=concurrency)
                        ea = mcts(client_a, False); eb = mcts(client_b, False)
                    else:
                        load(net_a, task['weights_a']); load(net_b, task['weights_b'])
                        ea = mcts(net_a, False); eb = mcts(net_b, False)

                    _state, _mc, game_result = play_two_engine_match(
                        ea, eb, a_is_white, config.selfplay.max_game_length,
                        on_move=on_ev_live)

                else:  # reference
                    # ── GPU path: network A via the GPU server (network_id=0) ──
                    if inference_client is not None:
                        client_a = InferenceClient(worker_id, request_queue, response_queue, network_id=0,
                                                   shared_buffers=shared_buffers, concurrency=concurrency)
                        ea = mcts(client_a, False)
                    else:
                        load(net_a, task['weights_a'])
                        ea = mcts(net_a, False)

                    _state, _mc, game_result = play_engine_vs_reference(
                        ea, a_is_white, config.selfplay.max_game_length,
                        depth=config.alpha_beta.depth, on_move=on_ev_live)

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
        except Exception:
            # A crashed task used to kill the whole worker process silently,
            # leaving the manager waiting forever for a result that never came.
            # Report it and keep the loop alive so one bad game can't stall
            # the whole training run.
            import traceback
            tb = traceback.format_exc()
            print(f"[Worker {worker_id}] task '{t}' crashed:\n{tb}", flush=True)
            try:
                result_queue.put({
                    'worker_id': worker_id, 'type': 'task_error',
                    'task_type': t, 'error': tb,
                })
            except Exception:
                pass

    result_queue.put({'worker_id': worker_id, 'done': True})


def _worker_loop_multi(worker_id, task_queue, result_queue, config_dict, shutdown_event,
                       request_queue=None, selfplay_instances=None, eval_instance=None,
                       concurrency=1, games_per_worker=2):
    """Concurrent-games worker loop.

    A dispatcher thread reads the (multiprocessing) task queue and routes:
      - 'selfplay' tasks  → an in-process job queue drained by ``games_per_worker``
        subgame threads, each bound to its own private GPU inference endpoint
        (own response queue + shared-memory buffer), so the games are fully
        independent — no thread-safety issues and no visit-flattening penalty.
      - 'eval' / 'profile'→ run on the dispatcher thread using the worker's
        dedicated eval endpoint (never colliding with self-play endpoints).

    Live messages carry the real ``worker_id`` (for the GUI grid) plus the
    subgame ordinal ``sub`` so main.py can feature exactly one game per worker.
    """
    import threading, queue as _q, torch, sys, os
    from config import Config
    from network import AlphaZeroNet
    from mcts import MCTS

    torch.set_num_threads(1)

    _set_game_from_config_dict(config_dict)
    config = Config(config_dict)
    pv = config.selfplay.piece_values
    piece_values = dict(pv) if hasattr(pv, 'items') else pv

    use_gpu = (getattr(config, 'inference', None)
               and getattr(config.inference, 'use_gpu', False))

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

    force_mate_enabled = (getattr(config.selfplay, 'force_mate', False)
                          and game_api.is_game('chess'))
    force_mate_in_one = getattr(config.mcts, 'force_mate_in_one', True)

    def mcts(net, noise, force_mate=False):
        cls = ForceMateMCTS if force_mate else MCTS
        mcts_kwargs = {}
        if force_mate:
            mcts_kwargs = {
                'max_force_plies': getattr(config.selfplay, 'force_mate_max_plies', 5),
                'min_pieces_gate': getattr(config.selfplay, 'force_mate_gate_pieces', 8),
                'deep_plies': getattr(config.selfplay, 'force_mate_deep_plies', 9),
                'deep_gate_pieces': getattr(config.selfplay, 'force_mate_deep_gate_pieces', 4),
            }
        return cls(
            network=net,
            num_simulations=config.mcts.num_simulations,
            c_puct=config.mcts.c_puct,
            dirichlet_alpha=config.mcts.dirichlet_alpha if noise else 0.0,
            dirichlet_epsilon=config.mcts.dirichlet_epsilon if noise else 0.0,
            batch_size=getattr(config.mcts, 'batch_size', 1),
            c_virtual_loss=getattr(config.mcts, 'c_virtual_loss', 0.5),
            max_game_length=config.selfplay.max_game_length,
            adjudicate_material=config.selfplay.adjudicate_material,
            piece_values=config.selfplay.piece_values,
            adjudicate_graded=getattr(config.selfplay, 'adjudicate_graded', True),
            adjudicate_scaling=getattr(config.selfplay, 'adjudicate_scaling', 9.0),
            force_mate_in_one=force_mate_in_one,
            inference_pipeline=getattr(config.inference, 'client_concurrency', 1)
                if getattr(config, 'inference', None) else 1,
            **mcts_kwargs,
        )

    def make_client(vid, rq, buf, nid=0):
        return InferenceClient(vid, request_queue, rq, network_id=nid,
                               shared_buffers=buf, concurrency=concurrency)

    # One private client per concurrent self-play game.
    sp_clients = []
    if use_gpu and selfplay_instances:
        for (vid, rq, buf) in selfplay_instances:
            sp_clients.append(make_client(vid, rq, buf, 0))

    # Dedicated eval endpoint clients (network 0 = latest, 1 = best).
    eval_client_a = None
    eval_client_b = None
    if use_gpu and eval_instance is not None:
        vid, rq, buf = eval_instance
        eval_client_a = make_client(vid, rq, buf, 0)
        eval_client_b = make_client(vid, rq, buf, 1)

    G = max(2, int(games_per_worker or 1))
    jobs = _q.Queue()
    subgame_nets = {}   # per-subgame local nets (non-GPU path)

    def run_selfplay_game(g, task, client):
        if client is not None:
            eng = mcts(client, noise=True, force_mate=force_mate_enabled)
        else:
            net = subgame_nets.get(g)
            if net is None:
                net = make_net(); subgame_nets[g] = net
            load(net, task['weights'])
            eng = mcts(net, noise=True, force_mate=force_mate_enabled)

        result_queue.put({
            'worker_id': worker_id, 'type': 'live_start', 'sub': g,
            'game_type': 'selfplay', 'match_info': None,
        })
        fens = []; ucis = []; mdata = []
        def on_sp(fen, uci, mn, ms=None):
            fens.append(fen); ucis.append(uci); mdata.append(ms)
            result_queue.put({
                'worker_id': worker_id, 'type': 'live_move', 'sub': g,
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
            endgame_temperature=getattr(config.selfplay, 'endgame_temperature', None),
            endgame_piece_gate=getattr(config.selfplay, 'endgame_piece_gate', 6),
            endgame_num_simulations=getattr(config.selfplay, 'endgame_num_simulations', 0),
            fen=getattr(config.selfplay, 'start_fen', None),
            start_pieces=getattr(config.selfplay, 'start_pieces', None),
            start_colors_random=getattr(config.selfplay, 'start_colors_random', False),
        )
        result_queue.put({
            'worker_id': worker_id, 'type': 'live_end', 'sub': g,
            'result': gi['result_str'], 'termination': gi['termination'],
        })
        result_queue.put({
            'worker_id': worker_id, 'type': 'selfplay',
            'game_data': [(s.tolist(), p.tolist(), float(v), m.tolist()) for s, p, v, m in gd],
            'game_info': gi, 'fens': fens, 'moves': ucis, 'mcts_stats': mdata,
        })

    def subgame_loop(g, client):
        while not shutdown_event.is_set():
            try:
                task = jobs.get(timeout=2.0)
            except _q.Empty:
                continue
            if task is None:
                break
            try:
                run_selfplay_game(g, task, client)
            except Exception:
                # A dead subgame thread used to vanish SILENTLY: the worker
                # kept idling, the main process waited for results forever,
                # and nothing anywhere said why.  Report and keep the loop
                # alive so one bad game can't quietly kill throughput.
                import traceback
                tb = traceback.format_exc()
                print(f"[Worker {worker_id}] subgame {g} crashed:\n{tb}", flush=True)
                try:
                    result_queue.put({
                        'worker_id': worker_id, 'type': 'subgame_error',
                        'sub': g, 'error': tb,
                    })
                except Exception:
                    pass

    def run_eval(task):
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
            if eval_client_a is not None:
                ea = mcts(eval_client_a, False); eb = mcts(eval_client_b, False)
            else:
                load(net_a, task['weights_a']); load(net_b, task['weights_b'])
                ea = mcts(net_a, False); eb = mcts(net_b, False)

            _state, _mc, game_result = play_two_engine_match(
                ea, eb, a_is_white, config.selfplay.max_game_length,
                on_move=on_ev_live)

        else:  # reference
            if eval_client_a is not None:
                ea = mcts(eval_client_a, False)
            else:
                load(net_a, task['weights_a'])
                ea = mcts(net_a, False)

            _state, _mc, game_result = play_engine_vs_reference(
                ea, a_is_white, config.selfplay.max_game_length,
                depth=config.alpha_beta.depth, on_move=on_ev_live)

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

    def run_profile(task):
        num_profile_games = task.get('num_games', 3)
        if eval_client_a is not None:
            eng = mcts(eval_client_a, noise=True)
        else:
            load(net_a, task['weights'])
            eng = mcts(net_a, noise=True)
        from worker_cpu_profiler import profile_in_worker
        summary = profile_in_worker(eng, config, piece_values, num_profile_games)
        result_queue.put({
            'worker_id': worker_id, 'type': 'profile_done',
            'summary': summary,
        })

    # ── Dispatcher: feed self-play jobs, run eval/profile inline ──
    threads = []
    for g in range(G):
        client = sp_clients[g] if sp_clients else None
        th = threading.Thread(target=subgame_loop, args=(g, client), daemon=True)
        th.start()
        threads.append(th)

    while not shutdown_event.is_set():
        try:
            task = task_queue.get(timeout=2.0)
        except _q.Empty:
            continue
        if task is None:
            break
        t = task.get('type', 'selfplay')
        if t == 'selfplay':
            try:
                jobs.put(task)
            except Exception:
                import traceback
                print(f"[Worker {worker_id}] failed to queue selfplay task:\n"
                      f"{traceback.format_exc()}", flush=True)
        elif t == 'eval':
            try:
                run_eval(task)
            except Exception:
                import traceback
                tb = traceback.format_exc()
                print(f"[Worker {worker_id}] eval task crashed:\n{tb}", flush=True)
                try:
                    result_queue.put({'worker_id': worker_id,
                                      'type': 'task_error', 'task_type': t,
                                      'error': tb})
                except Exception:
                    pass
        elif t == 'profile':
            try:
                run_profile(task)
            except Exception:
                import traceback
                tb = traceback.format_exc()
                print(f"[Worker {worker_id}] profile task crashed:\n{tb}", flush=True)
                try:
                    result_queue.put({'worker_id': worker_id,
                                      'type': 'task_error', 'task_type': t,
                                      'error': tb})
                except Exception:
                    pass

    for _ in range(G):
        try: jobs.put_nowait(None)
        except Exception: pass
    result_queue.put({'worker_id': worker_id, 'done': True})


# ─────────────────────────────────────────────────────────────────────────────
# Manager
# ─────────────────────────────────────────────────────────────────────────────

class ParallelSelfPlay:
    def __init__(self, config, num_workers=8, stats_queue=None):
        self.config = config
        self.num_workers = num_workers
        self._games_per_worker = max(1, int(getattr(
            getattr(config, 'selfplay', None), 'games_per_worker', 1)))
        self._block = self._games_per_worker + 1       # G self-play + 1 eval endpoint
        self._total_endpoints = self.num_workers * self._block
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
                client_concurrency = getattr(inf_cfg, 'client_concurrency', 1) if inf_cfg else 1
                state_dtype_str = getattr(inf_cfg, 'state_dtype', 'float16') if inf_cfg else 'float16'
                import numpy as _np
                state_dtype = _np.float16 if state_dtype_str == 'float16' else _np.float32
                policy_dtype_str = getattr(inf_cfg, 'policy_dtype', 'float32') if inf_cfg else 'float32'
                policy_dtype = _np.float16 if policy_dtype_str == 'float16' else _np.float32
                if self._games_per_worker > 1:
                    # One private endpoint per concurrent game (+1 dedicated eval
                    # endpoint per worker).  Each endpoint only ever carries a single
                    # in-flight request of at most `mcts_batch_size` states, so its
                    # buffer is sized to that (inference.max_batch is only a server-side
                    # aggregation cap, not a per-endpoint buffer need).
                    num_ep = self._total_endpoints
                    per_ep_batch = mcts_batch_size
                    self._shared_transport = SharedMemoryTransport(
                        num_workers=num_ep,
                        max_batch=per_ep_batch,
                        state_dtype=state_dtype,
                        num_slots=max(1, client_concurrency),
                        policy_dtype=policy_dtype,
                    )
                    self._shared_transport.create_buffers()
                    print(f"[INFO] Shared-memory IPC enabled (multi-game): "
                          f"endpoints={num_ep}, max_batch/endpoint={per_ep_batch}, "
                          f"dtype={state_dtype_str}, policy_dtype={policy_dtype_str}, "
                          f"slots={client_concurrency}")
                else:
                    self._shared_transport = SharedMemoryTransport(
                        num_workers=num_workers,
                        max_batch=max_batch,
                        state_dtype=state_dtype,
                        num_slots=max(1, client_concurrency),
                        policy_dtype=policy_dtype,
                    )
                    self._shared_transport.create_buffers()
                    print(f"[INFO] Shared-memory IPC enabled "
                          f"(dtype={state_dtype_str}, max_batch={max_batch}, "
                          f"slots={client_concurrency})")

    def start(self):
        if self._use_gpu:
            for i in range(self._total_endpoints):
                self._response_qs[i] = mp.Queue(maxsize=256)
            self._start_gpu_server()

        cd = self.config.to_dict()
        for i in range(self.num_workers):
            tq = mp.Queue(maxsize=max(4, 2 * self._games_per_worker))
            self._task_qs.append(tq)
            kwargs = {'games_per_worker': self._games_per_worker}
            if self._use_gpu:
                kwargs['request_queue'] = self._request_q
                inf_cfg = getattr(self.config, 'inference', None)
                kwargs['concurrency'] = (
                    getattr(inf_cfg, 'client_concurrency', 1) if inf_cfg else 1
                )
                if self._games_per_worker > 1:
                    # Private endpoint per subgame + one dedicated eval endpoint.
                    base = i * self._block
                    sp_inst = []
                    for g in range(self._games_per_worker):
                        vid = base + g
                        buf = (self._shared_transport.get_worker_buffers(vid)
                               if self._use_shared_memory and self._shared_transport is not None else None)
                        sp_inst.append((vid, self._response_qs[vid], buf))
                    evid = base + self._games_per_worker
                    evbuf = (self._shared_transport.get_worker_buffers(evid)
                             if self._use_shared_memory and self._shared_transport is not None else None)
                    kwargs['selfplay_instances'] = sp_inst
                    kwargs['eval_instance'] = (evid, self._response_qs[evid], evbuf)
                else:
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

        G = self._games_per_worker
        for tq in self._task_qs:
            while not tq.empty():
                try: tq.get_nowait()
                except: pass
            task = {'type': 'selfplay'}
            if not self._use_gpu:
                task['weights'] = wb
            # Seed games_per_worker concurrent games per worker.
            for _ in range(G):
                try: tq.put_nowait(task)
                except queue.Full: break

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
        # Give all workers a SHARED grace period, then kill.  A per-worker
        # join(10) would cost 10s * num_workers (120s with 12 workers) because
        # a worker blocked inside MCTS / an inference round-trip cannot observe
        # the shutdown event until its game step returns.  Idle workers exit
        # immediately on the ``None`` sentinel above, so a short shared window
        # is enough; the rest are killed.
        deadline = time.perf_counter() + 3.0
        for p in self._workers:
            remaining = deadline - time.perf_counter()
            p.join(timeout=remaining if remaining > 0 else 0)
            if p.is_alive(): p.kill()
        self._workers.clear()

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

        # Release IPC resources.  multiprocessing.Queue leaks a feeder thread
        # and a pipe per queue (and mp.Array leaks a /dev/shm segment) until
        # close() is called; without this a long-training run that restarts
        # self-play leaks fds/shared memory until the process exits.
        all_queues = []
        if self._use_gpu:
            all_queues.extend(self._response_qs.values())
            all_queues.extend([self._request_q, self._weight_q])
        all_queues.extend(self._task_qs)
        all_queues.append(self._result_q)
        for q in all_queues:
            try: q.cancel_join_thread()
            except Exception: pass
            try: q.close()
            except Exception: pass
        self._task_qs.clear()

        if self._shared_transport is not None:
            try: self._shared_transport.close()
            except Exception: pass
            self._shared_transport = None
        if self._use_gpu:
            self._response_qs.clear()

    def get_gpu_stats(self):
        """Return the GPU server timing dict, or None if unavailable."""
        return self._gpu_stats
