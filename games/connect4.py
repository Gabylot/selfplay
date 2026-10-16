"""Connect Four game module for the game API.

Board: 6 rows x 7 columns.  Player 0 drops a piece into a column; it falls
to the lowest empty cell.  First to make a horizontal/vertical/diagonal
line of 4 wins; a full board without a winner is a draw.

Encoding (zonquilized, always from the current player's view):
  plane 0: current player's pieces
  plane 1: opponent's pieces
  plane 2: side to move (all 1.0)
  plane 3: move count / max plies

Action space: H*W = 42 cells.  The only legal actions are the 7 lowest
empty cells, one per non-full column; action index = row*W + col of the
landing cell.
"""

import numpy as np
import random

name = "connect4"
board_h, board_w = 6, 7
num_planes = 4
policy_channels = 1
num_actions = board_h * board_w  # 42

IN_A_ROW = 4
MOVE_COUNT_NORM = board_h * board_w


def _in_a_row(grid, r, c, player) -> bool:
    """True if the piece at (r, c) completes a 4-in-a-row for ``player``."""
    v = 1 if player == 0 else 2
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        cnt = 1
        rr, cc = r + dr, c + dc
        while 0 <= rr < board_h and 0 <= cc < board_w and grid[rr, cc] == v:
            cnt += 1
            rr += dr
            cc += dc
        rr, cc = r - dr, c - dc
        while 0 <= rr < board_h and 0 <= cc < board_w and grid[rr, cc] == v:
            cnt += 1
            rr -= dr
            cc -= dc
        if cnt >= IN_A_ROW:
            return True
    return False


class State:
    __slots__ = ("grid", "player", "ply")

    def __init__(self, grid=None, player=0, ply=0):
        if grid is None:
            grid = np.zeros((board_h, board_w), dtype=np.int8)
        self.grid = grid
        self.player = player  # 0 or 1, side to move
        self.ply = ply

    def copy(self, stack=True):
        return State(self.grid.copy(), self.player, self.ply)

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo):
        return self.copy()

    def legal_columns(self):
        cols = []
        for c in range(board_w):
            if self.grid[0, c] == 0:
                cols.append(c)
        return cols

    def is_full(self) -> bool:
        return bool((self.grid[0] == 0).sum() == 0)

    def push(self, col) -> bool:
        """Place the mover's piece in ``col``.  Returns True if that wins."""
        grid = self.grid
        r = -1
        for rr in range(board_h - 1, -1, -1):
            if grid[rr, col] == 0:
                r = rr
                break
        if r == -1:
            raise ValueError("column full")
        grid[r, col] = 1 if self.player == 0 else 2
        won = _in_a_row(grid, r, col, self.player)
        self.ply += 1
        self.player = 1 - self.player
        return won

    def winner(self):
        """0/1 for the winning player, -1 for a full-board draw, else None."""
        for r in range(board_h):
            for c in range(board_w):
                v = self.grid[r, c]
                if v and _in_a_row(self.grid, r, c, v - 1):
                    return v - 1
        return -1 if self.is_full() else None


def encode(state):
    tensor = np.zeros((num_planes, board_h, board_w), dtype=np.float32)
    p1 = 1 if state.player == 0 else 2
    p2 = 3 - p1
    g = state.grid
    tensor[0] = g == p1
    tensor[1] = g == p2
    tensor[2] = 1.0
    tensor[3] = state.ply / MOVE_COUNT_NORM
    return tensor


def mask(state):
    m = np.zeros(num_actions, dtype=np.float32)
    for c in state.legal_columns():
        r = 0
        while r < board_h and state.grid[r, c] == 0:
            r += 1
        m[(r - 1) * board_w + c] = 1.0
    return m


def legal_moves(state):
    return state.legal_columns()


def to_index(state, col):
    r = 0
    while r < board_h and state.grid[r, col] == 0:
        r += 1
    return (r - 1) * board_w + col


def index_to_move(state, action_idx):
    return action_idx % board_w


def move_key(move):
    return int(move)


def move_str(move):
    return str(int(move))


def apply(state, move):
    state.push(int(move))


def state_ply(state):
    return state.ply


def is_terminal(state):
    return state.winner() is not None


def terminal_value(state):
    """Terminal value from the side-to-move's perspective."""
    w = state.winner()
    if w is None:
        return None
    if w == -1:
        return 0.0
    # whoever just moved made the line; it is now the loser's turn
    return -1.0


def adjudicate_value(state, **kwargs):
    return 0.0


def player_parity(state):
    return state.player


def player_sign(state):
    return 1.0 if state.player == 0 else -1.0


def describe(state):
    chars = {0: " ", 1: "X", 2: "O"}
    rows = []
    for r in range(board_h):
        rows.append("|" + "|".join(chars[int(v)] for v in state.grid[r]) + "|")
    rows.append(" " + " ".join(str(c) for c in range(board_w)) + " ")
    return "\n".join(rows)


def piece_count(state):
    return None


def new_state(start=None):
    start = start or {}
    state = State()
    start_kind = "standard"
    if start.get("start_pieces"):
        raise ValueError(
            f"connect4 does not support start_pieces={start['start_pieces']!r}")
    return state, start_kind


def final_result(state, move_count, max_game_length,
                 adjudicate_material=True, piece_values=None,
                 adjudicate_graded=True, adjudicate_scaling=9.0,
                 **_ignored):
    """Adjudication from the first-player (X) perspective."""
    w = state.winner()
    if w is not None:
        if w == -1:
            outcome, termination, result_str = 0.0, "full_board_draw", "1/2-1/2"
        else:
            outcome = 1.0 if w == 0 else -1.0
            termination = "win" if w == 0 else "win_second"
            result_str = "1-0" if w == 0 else "0-1"
    elif move_count >= max_game_length:
        outcome, termination, result_str = 0.0, "max_length", "1/2-1/2"
    else:
        outcome, termination, result_str = 0.0, "unknown", "1/2-1/2"
    return {"outcome": outcome, "termination": termination,
            "result_str": result_str}


# Column preference for the baseline player: center columns first.
_COLUMN_ORDER = [3, 2, 4, 1, 5, 0, 6]


def reference_move(state, **kwargs):
    """Greedy baseline: win now, block opponent's win now, then center-love."""
    cols = state.legal_columns()

    # 1. Play an immediate winning move.
    for c in cols:
        trial = state.copy()
        if trial.push(c):
            return c

    # 2. Block an opponent's immediate winning move.
    for c in cols:
        trial = state.copy()
        trial.push(c)
        for c2 in trial.legal_columns():
            trial2 = trial.copy()
            if trial2.push(c2):
                return c

    # 3. Otherwise prefer the most central available column.
    for c in _COLUMN_ORDER:
        if c in cols:
            return c
    return cols[0]