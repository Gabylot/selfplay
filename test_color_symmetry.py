"""Color-symmetry invariance test for board_to_tensor.

For every history board H_k of a played position A, compute its 180-degree
rotated, color-swapped partner base R_k (white<->black pieces, squares
sq -> 63-sq).  The player-relative encoding applied to R_k must produce
IDENTICAL piece/history/repetition planes to encoding A's H_k.

Because A is (typically) white to move and its reciprocal is black to move,
only these planes are ALLOWED to differ between the two full tensors:
    * PLANE_SIDE_TO_MOVE      (1.0 vs 0.0 — the player's absolute colour)
    * the 4 player-relative castling planes
      (their *meaning* swaps between sides under colour reversal,
       so plane-for-plane they differ by construction)

If ANY other plane differs, then black-to-move positions are encoded
asymmetrically relative to white-to-move positions -> systematic colour bias.

B is built with castling cleared (-) because fastchess's FEN validator rejects
some legal FENs where castling rights and the current piece placement diverge;
its EP square is cleared too (EP is not encoded as a plane).  Piece placement,
side to move, halfmove clock and fullmove number are preserved exactly.
"""

import numpy as np
import chess

from encoding import (
    board_to_tensor, PLANE_SIDE_TO_MOVE,
    PLANE_CASTLING_P1_K, PLANE_CASTLING_P1_Q,
    PLANE_CASTLING_P2_K, PLANE_CASTLING_P2_Q,
    PLANES_PER_HISTORY, NUM_PLANES, _encode_single_position,
    _encode_global_planes,
)

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

# Legal colour/piece-meaningful opening sequences (white first, alternating).
SEQUENCES = [
    ["e2e4", "d7d5", "g1f3", "b8c6"],            # 1.e4 d5 Nf3 Nc6
    ["d2d4", "d7d5", "b1c3", "g8f6"],            # 1.d4 d5 Nc3 Nf6
    ["e2e4", "d7d5", "e4e5", "b8c6", "g1f3"],   # 1.e4 d5 e5 Nc6 Nf3
    ["f2f4", "f7f5"],                           # 1.f4 f5 (K+P mirror)
    ["g1f3", "d7d5", "e2e3", "g8f6"],           # 1.Nf3 d5 e3 Nf6
    ["c2c4", "e7e5", "g1f3", "g8f6"],           # 1.c4 e5 Nf3 Nf6 (English)
    ["e2e4", "c7c5", "g1f3", "e7e6", "d2d4"],   # 1.e4 c5 Nf3 e6 d4 (Sicilian)
]

# Expected-allowed-differ planes: colour flag + the 4 player-relative
# castling planes (semantics swap under colour reversal).
ALLOWED_DIFFER = {PLANE_SIDE_TO_MOVE,
                  PLANE_CASTLING_P1_K, PLANE_CASTLING_P1_Q,
                  PLANE_CASTLING_P2_K, PLANE_CASTLING_P2_Q}


def rotate_color_invert_fen(fen: str) -> str:
    """Return the 180-deg rotated, colour-swapped FEN of ``fen``.

    Pieces keep their path-relative colour: every piece is placed on the
    square ``63 - square`` and swapped from white to black (and vice versa).
    Side-to-move is flipped. Castling and en-passant are cleared (see module
    docstring). Halfmove clock and fullmove number are preserved.
    """
    parts = fen.strip().split()
    board, turn = parts[0], parts[1]

    rows = board.split('/')
    # 180-deg rotation = reverse row order AND reverse each row, then
    # swap case (colour).  row0 = rank 8 -> becomes rank 1, etc.
    new_rows = []
    for row in reversed(rows):
        nr = ''.join(ch.swapcase() if ch.isalpha() else ch
                     for ch in reversed(row))
        new_rows.append(nr)

    new_turn = 'b' if turn == 'w' else 'w'
    return "/".join(new_rows) + f" {new_turn} - - {parts[-2]} {parts[-1]}"


def collect_history_fens(board) -> list:
    """Return the current + up to 7 older position FENs (or None for pads).

    Mirrors board_to_tensor's history walk: index 0 is the current position,
    index k is the position k plies ago (empty board when fewer plies exist).
    """
    b = board.copy(stack=True)
    num = len(b.move_stack)
    fens = []
    for ply in range(8):
        if ply <= num:
            fens.append(b.fen())
            if ply < num:
                b.pop()
        else:
            fens.append(None)
    return fens


def tensor_from_fens(fens) -> np.ndarray:
    """Encode a list of history FENs with the engine's exact plane layout."""
    t = np.zeros((NUM_PLANES, 8, 8), dtype=np.float32)
    for ply, f in enumerate(fens):
        if f is not None:
            _encode_single_position(chess.Board(f), ply * PLANES_PER_HISTORY, t)
    _encode_global_planes(chess.Board(fens[0]), t)
    return t


def test_seq(seq, label):
    a = chess.Board(START_FEN)
    for u in seq:
        a.push(chess.Move.from_uci(u))

    fens_a = collect_history_fens(a)
    fens_b = [rotate_color_invert_fen(f) if f is not None else None for f in fens_a]

    t_a = tensor_from_fens(fens_a)
    t_b = tensor_from_fens(fens_b)

    # Sanity: current A and its reciprocal must be opposite side to move.
    cur_a = chess.Board(fens_a[0])
    cur_b = chess.Board(fens_b[0])
    if cur_b.turn == cur_a.turn:
        print(f"[SKIP] {label}: rotate_color_invert_fen did not flip the turn")
        return True

    differing = [p for p in range(NUM_PLANES)
                 if not np.array_equal(t_a[p], t_b[p])]
    bad = [p for p in differing if p not in ALLOWED_DIFFER]

    status = "OK" if not bad else "MISMATCH"
    print(f"[{status}] {label}  ({', '.join(seq)})")
    print(f"       A: {fens_a[0]}")
    print(f"       B: {fens_b[0]}")
    if bad:
        print(f"       DIFFERING planes (beyond colour/castling): {bad}")
        for p in bad[:6]:
            d = np.where(t_a[p] != t_b[p])
            cells = [(int(r), int(c)) for r, c in zip(d[0], d[1])]
            print(f"         plane {p}: {len(cells)} cells differ: {cells[:6]}")
    else:
        # Confirm the flips that ARE expected happened, else it's a weak test.
        if PLANE_SIDE_TO_MOVE not in differing:
            print(f"         WARNING: expecting colour plane to differ, it did not")
        print(f"       colour plane differs: {PLANE_SIDE_TO_MOVE in differing} | "
              f"castling planes differ: "
              f"{[p for p in differing if p in ALLOWED_DIFFER]}")
    return not bad


def main():
    ok = True
    for i, seq in enumerate(SEQUENCES):
        r = test_seq(seq, f"seq {i+1}")
        ok = ok and r

    if ok:
        print("\nALL PASS: board_to_tensor piece/history/repetition planes "
              "are colour-symmetric")
    else:
        print("\nFAIL: colour asymmetry detected in board_to_tensor")


if __name__ == "__main__":
    main()