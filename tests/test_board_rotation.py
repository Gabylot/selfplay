"""Tests for the 180-degree 'board flip' (player-relative rotation) that the
encoding applies when Black is to move.

The input tensor is always oriented to the CURRENT player (P1): pieces are
stored player-relative (P1 piece-type planes 0-5, P2 piece-type planes 6-11),
and when Black is to move the whole board is rotated 180 degrees
(``sq -> 63 - sq``) so the current player's home rank is at array row 0.

These tests verify, independently of any hardcoded (r, c):
  1. ``_rotate_square`` is the exact 180-deg map ``sq -> 63 - sq`` (an
     involution) with the expected anchor pairings.
  2. ``board_to_tensor``: all 32 pieces in the CURRENT history group sit at
     their rotated coordinates, in the correct P1/P2 plane, for both
     white-to-move (no rotation) and black-to-move (rotation) positions —
     cross-checked against a reference built from the raw ``piece_map()``.
  3. The flip is the ONLY color-dependent transformation: rotating the board
     and swapping colours yields exactly the same cells.
  4. History groups alternate rotation with position parity.
  5. Edge cases: captures, promotions, and the colour plane stay consistent.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import chess

from encoding import (
    board_to_tensor, _rotate_square, PIECE_PLANE,
    PLANES_PER_HISTORY, NUM_PLANES, PLANE_SIDE_TO_MOVE,
    PLANE_CASTLING_P1_K, PLANE_CASTLING_P1_Q,
    PLANE_CASTLING_P2_K, PLANE_CASTLING_P2_Q,
)


def _make(moves):
    """Build a board by pushing UCI moves."""
    b = chess.Board()
    for m in moves:
        b.push(chess.Move.from_uci(m))
    return b


def _square_name(sq):
    return chr(ord('a') + (sq & 7)) + str((sq >> 3) + 1)


def expected_cells(board, group=0):
    """Reference: the set of (plane, row, col) that the position ``group``
    plies ago should place each piece at, per the player-relative + rotate
    rule, across the 12 piece planes.

    ``group`` only selects which 14-plane block to test; the piece placement
    itself is identical across history groups (each stores its own position).
    """
    rotate = (board.turn == chess.BLACK)
    p1 = board.turn
    out = set()
    for sq, pc in board.piece_map().items():
        r, f = sq >> 3, sq & 7
        if rotate:
            r, f = 7 - r, 7 - f
        plane = (6 if pc.color != p1 else 0) + pc.piece_type - 1
        out.add((plane, r, f))
    return out


def group_slice(group):
    return slice(group * PLANES_PER_HISTORY, group * PLANES_PER_HISTORY + 12)


# ---------------------------------------------------------------------------
# 1. _rotate_square is the exact 180-deg map and an involution
# ---------------------------------------------------------------------------

class TestRotateSquare:
    def test_is_63_minus_square(self):
        for sq in range(64):
            assert _rotate_square(sq) == 63 - sq, f"sq {sq}"

    def test_involution(self):
        for sq in range(64):
            assert _rotate_square(_rotate_square(sq)) == sq, f"sq {sq}"

    def test_anchor_pairings(self):
        # 180-deg rotation: corners swap, file e<->d, rank k <-> 9-k.
        pairs = [('a1', 'h8'), ('a8', 'h1'), ('h1', 'a8'), ('h8', 'a1'),
                 ('e1', 'd8'), ('e8', 'd1'), ('e2', 'd7'), ('d7', 'e2'),
                 ('d4', 'e5'), ('e5', 'd4'), ('b6', 'g3'), ('g7', 'b2')]
        for frm, to in pairs:
            got = _square_name(_rotate_square(chess.parse_square(frm)))
            assert got == to, f"{frm} -> {got} != {to}"


# ---------------------------------------------------------------------------
# 2 & 3. Exact piece placement under white-to-move and black-to-move
# ---------------------------------------------------------------------------

class TestFlipPiecePlacement:

    def _collect(self, planes):
        """Set of (plane, row, col) == 1.0 within a (N, 8, 8) slice."""
        return {(p, r, c) for p in range(planes.shape[0])
                for r in range(8) for c in range(8)
                if planes[p, r, c] == 1.0}

    def test_white_to_move_no_rotation(self):
        b = _make(['e2e4', 'e7e5', 'g1f3', 'b8c6'])  # 4 plies -> white to move
        assert b.turn == chess.WHITE
        t = board_to_tensor(b)
        exp = expected_cells(b)
        got = self._collect(t[:12])
        assert got == exp, (f"white-to-move mismatch: extra {got - exp}, "
                            f"missing {exp - got}")
        assert len(got) == len(b.piece_map()) == 32

    def test_black_to_move_rotation(self):
        b = _make(['e2e4'])  # 1 ply -> black to move (flip)
        assert b.turn == chess.BLACK
        t = board_to_tensor(b)
        exp = expected_cells(b)
        got = self._collect(t[:12])
        assert got == exp, (f"black-to-move rotation mismatch: "
                            f"extra {got - exp}, missing {exp - got}")
        assert len(got) == 32

    def test_capture_reduces_to_31(self):
        # 1.e4 d5 exd5 : black pawn captured -> 31 pieces, black to move.
        b = _make(['e2e4', 'd7d5', 'e4d5'])
        assert b.turn == chess.BLACK
        t = board_to_tensor(b)
        assert self._collect(t[:12]) == expected_cells(b)
        assert len(self._collect(t[:12])) == 31

    def test_promotion_lands_on_rotated_promotion_square(self):
        # White pawn promotes e7-e8=Q; black to move -> queen flips to (0,3).
        b = chess.Board("1k4r1/4P3/8/8/8/8/8/4K3 w - - 0 1")
        b.push(chess.Move.from_uci('e7e8q'))
        assert b.turn == chess.BLACK
        t = board_to_tensor(b)
        assert self._collect(t[:12]) == expected_cells(b)
        # White queen (P2) on rotated e8 -> (7-7, 7-4) = (0, 3), P2-queen plane 10.
        assert t[10, 0, 3] == 1.0

    def test_flip_then_colorswap_is_identical_placement(self):
        """Rotate the board AND swap colours: every piece maps onto the same
        set of (plane, row, col) as the original black-to-move encoding."""

        def swapped_cells(board):
            """Reference: (plane, row, col) each piece maps to when the board
            is rotated 180 deg, colours swapped, and treated as white-to-move
            (P1=white).  Computed directly from the raw piece map."""
            out = set()
            for sq, pc in board.piece_map().items():
                r, f = sq >> 3, sq & 7
                r, f = 7 - r, 7 - f
                # swapped colour becomes white  <=> was black
                plane = (0 if pc.color == chess.BLACK else 6) + pc.piece_type - 1
                out.add((plane, r, f))
            return out

        b = _make(['d2d4', 'd7d5', 'c2c4', 'g8f6', 'b1c3'])  # 5 plies
        assert b.turn == chess.BLACK
        t = board_to_tensor(b)
        got_black = self._collect(t[:12])
        assert got_black == swapped_cells(b), (
            "rotate+colorswap should land identically")


# ---------------------------------------------------------------------------
# 4. History alternation
# ---------------------------------------------------------------------------

class TestHistoryFlipAlternation:

    def _collect_group(self, t, group):
        return {(p, r, c) for p in range(12)
                for r in range(8) for c in range(8)
                if t[group * PLANES_PER_HISTORY + p, r, c] == 1.0}

    def test_history_groups_alternate_flip_by_parity(self):
        # Plies 0,2,4,6: white to move (no flip).  Plies 1,3,5,7: black to
        # move (flip).  Every group must match the reference computed from the
        # position that was current at that ply.
        moves = ['e2e4', 'd7d5', 'g1f3', 'g8f6', 'b1c3', 'b8c6']  # 6 plies
        b = _make(moves)
        t = board_to_tensor(b)

        # Reconstruct each historical board by popping the move stack.
        bb = b.copy(stack=True)
        for group in range(8):
            if group <= len(moves):
                exp = expected_cells(bb)
                got = self._collect_group(t, group)
                assert got == exp, f"group {group} mismatch"
                if group < len(moves):
                    bb.pop()

    def test_every_historical_position_still_has_32_pieces(self):
        moves = ['e2e4', 'd7d5', 'g1f3', 'g8f6', 'c2c4', 'e7e6']  # 6 plies
        b = _make(moves)
        t = board_to_tensor(b)
        # Valid positions: current (ply 0) .. ply num_moves (the start).
        filled = len(moves) + 1
        for group in range(filled):
            n = t[group_slice(group)].sum()
            assert n == 32.0, f"group {group} has {n} pieces, expected 32"
        for group in range(filled, 8):
            assert t[group_slice(group)].sum() == 0.0, \
                f"pre-game group {group} should be empty"


# ---------------------------------------------------------------------------
# 5. Colour plane + castling stay consistent with the flip
# ---------------------------------------------------------------------------

class TestFlipAuxiliaryPlanes:

    def test_colour_plane_flips_with_side_to_move(self):
        b = _make(['e2e4'])  # black to move
        t = board_to_tensor(b)
        assert t[PLANE_SIDE_TO_MOVE].sum() == 0.0, "black to move: colour plane 0"
        b2 = _make(['e2e4', 'e7e5'])  # white to move
        t2 = board_to_tensor(b2)
        assert t2[PLANE_SIDE_TO_MOVE].all() == 1.0, "white to move: colour plane 1"

    def test_castling_player_relative_after_flip(self):
        # Black to move, black still has K+Q rights -> they land in P1 planes.
        b = _make(['e2e4', 'e7e5', 'g1f3'])
        assert b.turn == chess.BLACK
        t = board_to_tensor(b)
        assert t[PLANE_CASTLING_P1_K].all() == 1.0, "P1 (black) kingside"
        assert t[PLANE_CASTLING_P1_Q].all() == 1.0, "P1 (black) queenside"
        assert t[PLANE_CASTLING_P2_K].all() == 1.0, "P2 (white) kingside"
        assert t[PLANE_CASTLING_P2_Q].all() == 1.0, "P2 (white) queenside"

    def test_no_ghost_pieces_at_rotated_corner(self):
        b = _make(['e2e4'])  # 1 ply -> black to move (flip)
        t = board_to_tensor(b)
        # Black king e8 -> rotated (7-7, 7-4) = (0, 3), P1-king plane 5.
        assert t[5, 0, 3] == 1.0, "black king e8 rotates to P1-king (0,3)"
        # Black rook a8 -> rotated (7-7, 7-0) = (0, 7) = h1, P1-rook plane 3.
        assert t[3, 0, 7] == 1.0, "black rook a8 rotates to P1-rook (0,7)"
        assert t[5, 0, 0] == 0.0, "no king should sit at (0,0)"


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])
