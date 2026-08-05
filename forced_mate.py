"""Exact forced-mate proof solver and MCTS override for endgame conversion.

Self-play games frequently stall in won endgames (KQ/KR vs K) because the
value head learns these are drawn — games that reach them mostly end in
draws (``max_length``/``fifty_moves``).  This is target poisoning:
correct positions get mislabeled as drawn by the engine's own shuffling.

This module provides:

- ``forced_mate(board, plies)`` — an exact minimax proof search (memoized)
  that returns a move forcing checkmate within ``plies`` plies (or None).
  It is gated to sparse positions (<= ``min_pieces_gate`` pieces) by the
  caller, keeping the cost sub-millisecond.
- ``ForceMateMCTS`` — an ``mcts.MCTS`` subclass whose move selection
  overrides MCTS when a forced mate in 1..``max_force_plies`` exists,
  returning a one-hot policy for the forcing move instead of the
  visit-based distribution.  This both (a) converts the game and
  (b) feeds the value head +1/-1 targets plus a clean one-hot policy
  for the forcing line instead of a poisoned draw label.

The solver is validated against naive brute-force minimax in
``verify_forced_mate_broad.py`` (see ``forced_mate`` vs ``brute_can_force``).
"""

import numpy as np
import chess
from typing import Optional

from mcts import MCTS, NUM_ACTIONS, move_to_policy_index


# Module-level memo shared across calls.  Reset per game with
# ``reset_memo()`` to avoid unbounded growth in long-lived workers.
_memo: dict = {}


def reset_memo() -> None:
    """Clear the forced-mate memo.  Call once per game."""
    _memo.clear()


def forced_mate(board: chess.Board, plies: int) -> Optional[chess.Move]:
    """Return a move that forces checkmate within ``plies`` plies, or None.

    ``plies`` is the number of half-moves available to the side to move
    (1, 3, or 5 in practice).  A position is a forced mate in 1 if any
    legal move delivers checkmate.  A position is a forced mate in ``plies``
    if some legal move leaves the opponent in a position where *every*
    legal reply leaves us in a forced mate in ``plies - 2``.

    Memoized on ``(board.fen(), plies)``.
    """
    if board.is_game_over() or plies <= 0:
        return None
    key = (board.fen(), plies)
    if key in _memo:
        return _memo[key]
    if plies == 1:
        for m in board.legal_moves:
            nb = board.copy(stack=False)
            nb.push(m)
            if nb.is_checkmate():
                _memo[key] = m
                return m
        _memo[key] = None
        return None
    for m in board.legal_moves:
        nb = board.copy(stack=False)
        nb.push(m)
        if nb.is_checkmate():
            _memo[key] = m
            return m
        if nb.is_game_over():
            continue                      # stalemate/draw after m -> not mate
        ok = True
        for r in nb.legal_moves:
            rb = nb.copy(stack=False)
            rb.push(r)
            if rb.is_game_over():
                ok = False
                break                     # opponent escaped or mated us
            if forced_mate(rb, plies - 2) is None:
                ok = False
                break                     # opponent has a non-mating reply
        if ok:
            _memo[key] = m
            return m
    _memo[key] = None
    return None


class ForceMateMCTS(MCTS):
    """MCTS whose move selection is overridden by an exact forced-mate proof.

    Only active on sparse boards (<= ``min_pieces_gate`` pieces).  When a
    forced mate within ``max_force_plies`` exists, the forcing move is
    returned with a one-hot policy instead of the MCTS visit distribution.
    Otherwise falls back to the base MCTS behavior.

    Attributes:
        forced: total number of times the override fired.
        forced_log: list of ``(move, plies, fen)`` tuples for diagnostics.
    """

    def __init__(self, *args, max_force_plies: int = 5,
                 min_pieces_gate: int = 8,
                 deep_plies: int = 9, deep_gate_pieces: int = 4, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_force_plies = max_force_plies
        self.min_pieces_gate = min_pieces_gate
        self.deep_plies = deep_plies
        self.deep_gate_pieces = deep_gate_pieces
        self.forced = 0
        self.forced_log = []

    def select_move_with_temperature(self, root, temperature):
        b = root.board
        n_pieces = len(b.piece_map())
        if n_pieces <= self.min_pieces_gate:
            # Deeper proof search for very sparse positions (<= deep_gate_pieces
            # pieces), where the search tree is small enough to stay cheap.
            # This converts longer endgames (e.g. KR vs K mate-in-4/5) that the
            # shallow 5-ply limit misses.
            max_plies = self.max_force_plies
            if n_pieces <= self.deep_gate_pieces:
                max_plies = max(max_plies, self.deep_plies)
            for plies in range(1, max_plies + 1, 2):
                fm = forced_mate(b, plies)
                if fm is not None:
                    idx = move_to_policy_index(fm, b)
                    vp = np.zeros(NUM_ACTIONS, dtype=np.float32)
                    vp[idx] = 1.0
                    self.forced += 1
                    self.forced_log.append((fm, plies, b.fen()))
                    return vp, fm
        return super().select_move_with_temperature(root, temperature)
