"""Chess game module for the game API.

Wraps the existing chess backend (the ``chess`` module = Rust fastchess
shim) and the ``encoding`` module so the generic self-play pipeline can
drive chess through ``game_api``.
"""

import numpy as np
import chess

import encoding

name = "chess"
board_h = board_w = 8
num_planes = encoding.NUM_PLANES
num_actions = encoding.NUM_ACTIONS
policy_channels = 73

# Index of the constant side-to-move plane (see encoding._encode_global_planes).
stm_plane = encoding.PLANE_SIDE_TO_MOVE


def encode(state):
    return encoding.board_to_tensor(state)


def mask(state):
    return encoding.get_legal_move_mask(state)


def legal_moves(state):
    return state.legal_moves


def to_index(state, move):
    return encoding.move_to_policy_index(move, state)


def index_to_move(state, action_idx):
    return encoding.policy_index_to_move(action_idx, state)


def move_key(move):
    return (move.from_square, move.to_square, move.promotion)


def move_str(move):
    return move.uci()


def apply(state, move):
    state.push(move)


def state_ply(state):
    return state.ply()


def is_terminal(state):
    return state.is_game_over()


def is_draw_claim(state):
    return state.is_repetition(3)


def is_fifty_moves(state):
    return state.is_fifty_moves()


def terminal_value(state):
    """Terminal value from the side-to-move's perspective.

    Returns None only for non-terminal positions.  Callable draws are NOT
    returned as None: the backend treats threefold/50-move as automatic
    claims (result() already reports 1/2-1/2 for them), so this maps them to
    0.0 like other draws.
    """
    result = state.result()
    if result == "*":
        return None
    if result == "1-0":
        return 1.0 if state.turn == chess.WHITE else -1.0
    if result == "0-1":
        return -1.0 if state.turn == chess.WHITE else 1.0
    return 0.0


def adjudicate_value(state, piece_values=None, graded=False, scaling=9.0):
    """Material-based adjudication.

    NOTE: matches the engine's historical material adjudication (white
    perspective, no turn flip) so training signals don't change.  Connect4
    has no material and always returns 0.0.

    When graded, the material margin is clamped to one queen's value before
    the tanh: a runaway advantage (e.g. +30 pawns) must not push the value
    toward ~1.0, which would make it indistinguishable from a forced mate.
    With scaling=9.0 (the pipeline default; every engine call site explicitly
    passes it) the maximum is tanh(9/9) = ~0.762 (a lone queen).
    """
    if piece_values is None:
        piece_values = {"P": 1, "N": 3, "B": 3, "R": 5, "Q": 9}
    w = b = 0
    for _sq, p in state.piece_map().items():
        v = piece_values.get(p.symbol().upper(), 0)
        if p.color == chess.WHITE:
            w += v
        else:
            b += v
    diff = w - b
    if graded:
        # One-queen cap: material guides, but never outshines checkmate (1.0).
        cap = piece_values.get("Q", 9)
        diff = max(-cap, min(cap, diff))
        return 0.0 if diff == 0 else float(np.tanh(diff / scaling))
    return 1.0 if diff > 0 else (-1.0 if diff < 0 else 0.0)


def player_parity(state):
    return 0 if state.turn == chess.WHITE else 1


def player_sign(state):
    return 1.0 if state.turn == chess.WHITE else -1.0


def describe(state):
    return state.fen()


def piece_count(state):
    return len(state.piece_map())


def material_diff(state, piece_values=None):
    """Raw material difference (white - black) for diagnostics."""
    if piece_values is None:
        piece_values = {"P": 1, "N": 3, "B": 3, "R": 5, "Q": 9}
    w = b = 0
    for _sq, p in state.piece_map().items():
        v = piece_values.get(p.symbol().upper(), 0)
        if p.color == chess.WHITE:
            w += v
        else:
            b += v
    return w - b


def new_state(start=None):
    start = start or {}
    fen = start.get("fen")
    pieces = start.get("start_pieces")
    randomize = start.get("randomize_colors")

    start_kind = "standard"
    if fen:
        board = chess.Board(fen)
    elif pieces:
        # Lazy import: selfplay imports game_api at module scope, so pulling
        # it in here at import time would create a cycle.
        from selfplay import _pick_start_board_kind

        start_kind, board = _pick_start_board_kind(pieces,
                                                   randomize_colors=randomize)
    else:
        board = chess.Board()
    return board, start_kind


def final_result(state, move_count, max_game_length,
                 adjudicate_material=True, piece_values=None,
                 adjudicate_graded=True, adjudicate_scaling=9.0,
                 **_ignored):
    """Adjudication for a finished game.

    ``outcome`` follows the first-player (White) perspective: +1 White won,
    -1 Black won, 0 draw.
    """
    if piece_values is None:
        piece_values = {"P": 1, "N": 3, "B": 3, "R": 5, "Q": 9}

    if state.is_game_over(claim_draw=True):
        r = state.result(claim_draw=True)
        if r == "1-0":
            outcome, termination = 1.0, ("checkmate" if state.is_checkmate() else "other")
        elif r == "0-1":
            outcome, termination = -1.0, ("checkmate" if state.is_checkmate() else "other")
        else:
            outcome = 0.0
            if state.is_fivefold_repetition():
                termination = "fivefold_repetition"
            elif state.is_repetition(3):
                termination = "repetition"
            elif state.is_seventyfive_moves():
                termination = "seventyfive_moves"
            elif state.is_stalemate():
                termination = "stalemate"
            elif state.is_fifty_moves():
                termination = "fifty_moves"
            elif state.is_insufficient_material():
                termination = "insufficient_material"
            else:
                termination = "draw"
        result_str = r
    elif move_count >= max_game_length:
        termination = "max_length"
        if adjudicate_material:
            outcome = adjudicate_value(state, piece_values,
                                       graded=adjudicate_graded,
                                       scaling=adjudicate_scaling)
            if outcome > 0:
                termination = "material_white"
            elif outcome < 0:
                termination = "material_black"
            else:
                outcome = 0.0
        else:
            outcome = 0.0
        result_str = ("1-0" if outcome > 0
                      else "0-1" if outcome < 0 else "1/2-1/2")
    else:
        outcome = 0.0
        termination = "unknown"
        with open("unknown_termination_log.txt", "a") as f:
            f.write(f"UNKNOWN at move {move_count}: {state.fen()} | "
                    f"halfmove_clock={state.halfmove_clock} | "
                    f"can_claim_3fold={state.can_claim_threefold_repetition()} | "
                    f"can_claim_50={state.can_claim_fifty_moves()} | "
                    f"is_50={state.is_fifty_moves()} | "
                    f"is_repetition(2)={state.is_repetition(2)} | "
                    f"is_repetition(3)={state.is_repetition(3)} | "
                    f"is_insufficient={state.is_insufficient_material()}\n")
        result_str = "1/2-1/2"

    return {"outcome": outcome, "termination": termination,
            "result_str": result_str}


def reference_move(state, depth=2):
    """Eval-vs-reference opponent: fixed-depth alpha-beta on python-chess."""
    from evaluation import alpha_beta_best_move

    return alpha_beta_best_move(state, depth)