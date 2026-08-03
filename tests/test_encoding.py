"""Tests for the 119-plane AlphaZero board encoding.

Verifies:
1. Correct tensor shape (119, 8, 8)
2. Correct plane assignments for piece positions (P1/P2, player-oriented)
3. Board rotation when Black is to move (180-degree flip)
4. Player-relative castling planes
5. Per-timestep repetition planes
6. History planes: positions from previous plies appear correctly
7. History planes: positions before game start are empty
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import chess
from encoding import (
    board_to_tensor, NUM_PLANES, PLANES_PER_HISTORY, NUM_HISTORY_STEPS,
    PIECE_PLANE, PLANE_SIDE_TO_MOVE,
    PLANE_CASTLING_P1_K, PLANE_CASTLING_P1_Q, PLANE_CASTLING_P2_K, PLANE_CASTLING_P2_Q,
    PLANE_REPETITION_P1, PLANE_REPETITION_P2,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_board_move_stack(moves: list) -> chess.Board:
    """Create a board from a list of UCI move strings, preserving move stack."""
    board = chess.Board()
    for m in moves:
        board.push(chess.Move.from_uci(m))
    return board


def piece_planes(piece_type, color):
    """Return the plane index within a 14-plane group for a given piece.

    Planes 0-5 = P1 (current player) pieces, 6-11 = P2 (opponent) pieces.
    """
    return PIECE_PLANE[(piece_type, color)]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBoardToTensorShape:
    """The tensor must have shape (119, 8, 8)."""

    def test_shape_initial_position(self):
        board = chess.Board()
        tensor = board_to_tensor(board)
        assert tensor.shape == (119, 8, 8), f"Expected (119, 8, 8), got {tensor.shape}"

    def test_shape_mid_game(self):
        board = _make_board_move_stack(['e2e4', 'e7e5', 'g1f3', 'b8c6'])
        tensor = board_to_tensor(board)
        assert tensor.shape == (119, 8, 8), f"Expected (119, 8, 8), got {tensor.shape}"

    def test_type_is_float32(self):
        board = chess.Board()
        tensor = board_to_tensor(board)
        assert tensor.dtype == np.float32, f"Expected float32, got {tensor.dtype}"


class TestPiecePlanes:
    """Piece positions should appear in the correct planes (planes 0-5 P1, 6-11 P2)."""

    def test_white_to_move_no_flip(self):
        """White to move: no rotation, P1=White, P2=Black."""
        board = chess.Board()
        tensor = board_to_tensor(board)
        # e2 = rank 1, file 4. White pawn is P1-pawn plane 0.
        assert tensor[0, 1, 4] == 1.0, "White pawn at e2 should be in P1-pawn plane 0, position (1,4)"
        # e7 = rank 6, file 4. Black pawn is P2-pawn plane 6.
        assert tensor[6, 6, 4] == 1.0, "Black pawn at e7 should be in P2-pawn plane 6, position (6,4)"
        # e1 = rank 0, file 4. White king is P1-king plane 5.
        assert tensor[5, 0, 4] == 1.0, "White king at e1 should be in P1-king plane 5"

    def test_black_to_move_rotated(self):
        """Black to move: board rotated 180 degrees, P1=Black, P2=White."""
        board = _make_board_move_stack(['e2e4'])
        tensor = board_to_tensor(board)
        # Black to move. Black's e7 pawn (abs rank 6, file 4) rotates to
        # (7-6, 7-4) = (1, 3), and is a P1 piece -> P1-pawn plane 0.
        assert tensor[0, 1, 3] == 1.0, "Black e7 pawn should be at rotated (1,3) in P1-pawn plane 0"
        # White's e4 pawn (abs rank 3, file 4) rotates to (7-3, 7-4) = (4, 3),
        # P2 piece -> P2-pawn plane 6.
        assert tensor[6, 4, 3] == 1.0, "White e4 pawn should be at rotated (4,3) in P2-pawn plane 6"
        # Black king e8 (abs rank 7, file 4) rotates to (0, 3), P1-king plane 5.
        assert tensor[5, 0, 3] == 1.0, "Black king e8 should be at rotated (0,3) in P1-king plane 5"

    def test_piece_positions_sum(self):
        """Total 1s in piece planes should be 32 (16 per side)."""
        board = chess.Board()
        tensor = board_to_tensor(board)
        # Sum across all 12 piece planes in the current position group
        piece_plane_sum = tensor[0:12].sum()
        assert piece_plane_sum == 32.0, f"Expected 32 pieces, got {piece_plane_sum}"

    def test_no_pieces_in_empty_planes(self):
        """Planes for pieces not on the board should be all zeros."""
        board = chess.Board()  # No promoted pieces yet
        tensor = board_to_tensor(board)
        # Check that plane sum for the first group is exactly 32 (no extra pieces)
        piece_plane_sum = tensor[0:12].sum()
        assert piece_plane_sum == 32.0, f"Extra pieces detected: sum={piece_plane_sum}"


class TestSideToMove:
    """Side-to-move (colour) plane should be 1.0 for white, 0.0 for black."""

    def test_white_to_move(self):
        board = chess.Board()
        tensor = board_to_tensor(board)
        # White to move -> colour plane 112 should be all 1s
        assert tensor[PLANE_SIDE_TO_MOVE].all() == 1.0, "White to move: colour plane should be all 1s"

    def test_black_to_move(self):
        board = _make_board_move_stack(['e2e4'])
        tensor = board_to_tensor(board)
        # Black to move -> colour plane 112 should be all 0s
        assert tensor[PLANE_SIDE_TO_MOVE].sum() == 0.0, "Black to move: colour plane should be all 0s"


class TestCastlingPlanes:
    """Castling rights should be player-relative (P1 = current player)."""

    def test_initial_position_all_castling(self):
        board = chess.Board()
        tensor = board_to_tensor(board)
        # White to move: P1 = White, P2 = Black. All four rights available.
        assert tensor[PLANE_CASTLING_P1_K].all() == 1.0, "P1 kingside castling should be 1s initially"
        assert tensor[PLANE_CASTLING_P1_Q].all() == 1.0, "P1 queenside castling should be 1s initially"
        assert tensor[PLANE_CASTLING_P2_K].all() == 1.0, "P2 kingside castling should be 1s initially"
        assert tensor[PLANE_CASTLING_P2_Q].all() == 1.0, "P2 queenside castling should be 1s initially"

    def test_black_to_move_black_rights_in_p1(self):
        """With Black to move, Black's castling rights go into the P1 planes."""
        # Position where Black still has castling rights and is to move.
        board = _make_board_move_stack(['e2e4', 'e7e5', 'g1f3'])
        tensor = board_to_tensor(board)
        # Black to move: P1 = Black. Black still has both castling rights.
        assert tensor[PLANE_CASTLING_P1_K].all() == 1.0, "P1 (Black) kingside should be available"
        assert tensor[PLANE_CASTLING_P1_Q].all() == 1.0, "P1 (Black) queenside should be available"
        # White (P2) still has both rights too.
        assert tensor[PLANE_CASTLING_P2_K].all() == 1.0, "P2 (White) kingside should be available"
        assert tensor[PLANE_CASTLING_P2_Q].all() == 1.0, "P2 (White) queenside should be available"

    def test_no_castling_after_king_move(self):
        """White loses castling rights after moving king."""
        board = _make_board_move_stack(['e2e4', 'e7e5', 'f1c4', 'b8c6', 'g1f3', 'g8f6', 'e1g1'])
        tensor = board_to_tensor(board)
        # After 0-0, it's Black to move, so P1 = Black (still has rights)
        # and P2 = White (who has lost both castling rights).
        assert tensor[PLANE_CASTLING_P1_K].all() == 1.0, "P1 (Black) kingside should be available"
        assert tensor[PLANE_CASTLING_P1_Q].all() == 1.0, "P1 (Black) queenside should be available"
        assert tensor[PLANE_CASTLING_P2_K].sum() == 0.0, "P2 (White) kingside should be 0 after king moved"
        assert tensor[PLANE_CASTLING_P2_Q].sum() == 0.0, "P2 (White) queenside should be 0 after king moved"


class TestRepetitionPlanes:
    """Per-timestep repetition planes (12 = >=2 occurrences, 13 = >=3)."""

    def test_no_repetition(self):
        board = chess.Board()
        tensor = board_to_tensor(board)
        assert tensor[PLANE_REPETITION_P1].sum() == 0.0, "No repetition: plane 12 should be 0"
        assert tensor[PLANE_REPETITION_P2].sum() == 0.0, "No repetition: plane 13 should be 0"

    def test_twofold_repetition(self):
        # 1. Nf3 Nf6 2. Ng1 Ng8 -> back to start (2nd occurrence)
        board = _make_board_move_stack(['g1f3', 'g8f6', 'f3g1', 'f6g8'])
        assert board.is_repetition(2), "Should detect 2-fold repetition"
        tensor = board_to_tensor(board)
        assert tensor[PLANE_REPETITION_P1].all() == 1.0, "2-fold: plane 12 should be 1"
        assert tensor[PLANE_REPETITION_P2].sum() == 0.0, "2-fold: plane 13 should be 0"

    def test_threefold_repetition(self):
        # 3 cycles of Nf3/Nf6/Ng1/Ng8 -> 3rd occurrence of start
        board = _make_board_move_stack(
            ['g1f3', 'g8f6', 'f3g1', 'f6g8', 'g1f3', 'g8f6', 'f3g1', 'f6g8'])
        assert board.is_repetition(3), "Should detect 3-fold repetition"
        tensor = board_to_tensor(board)
        assert tensor[PLANE_REPETITION_P1].all() == 1.0, "3-fold: plane 12 should be 1"
        assert tensor[PLANE_REPETITION_P2].all() == 1.0, "3-fold: plane 13 should be 1"


class TestHistoryPlanes:
    """The previous 7 positions should be encoded in planes 14-111."""

    def test_history_has_correct_positions(self):
        """After 4 plies, the history should show the correct positions."""
        moves = ['e2e4', 'e7e5', 'g1f3', 'b8c6']
        board = _make_board_move_stack(moves)
        tensor = board_to_tensor(board)

        # After 4 plies (even number), it's white's turn again.
        # Current position (group 0, planes 0-13): white to move, no flip.
        # White knight at f3 (rank 2, file 5) -> P1-knight plane 1.
        assert tensor[1, 2, 5] == 1.0, "White knight should be at f3 in current position"

        # Position 1 ply ago (group 1, planes 14-27): black to move -> rotated.
        # After g1f3, before b8c6. Black knight b8 (abs rank 7, file 1) rotates
        # to (0, 6), P1-knight plane 1.
        assert tensor[14 + 1, 0, 6] == 1.0, "Black knight should be at rotated (0,6) 1 ply ago"

        # Position 2 plies ago (group 2, planes 28-41): white to move -> no flip.
        # After e7e5, before g1f3. White knight g1 (rank 0, file 6) -> P1-knight plane 1.
        assert tensor[28 + 1, 0, 6] == 1.0, "White knight should be at g1 2 plies ago"

        # Position 3 plies ago (group 3, planes 42-55): black to move -> rotated.
        # After e2e4, before e7e5. Black knight b8 (abs rank 7, file 1) rotates
        # to (0, 6), P1-knight plane 1.
        assert tensor[42 + 1, 0, 6] == 1.0, "Black knight should be at rotated (0,6) 3 plies ago"

        # Position 4 plies ago (group 4, planes 56-69): initial position, white to move.
        # White knight g1 (rank 0, file 6) -> P1-knight plane 1.
        assert tensor[56 + 1, 0, 6] == 1.0, "White knight should be at g1 4 plies ago"

        # Positions 5-7 (groups 5-7, planes 70-111): before game start -> empty.
        for i in range(5, 8):
            offset = i * PLANES_PER_HISTORY
            assert tensor[offset:offset+12].sum() == 0.0, \
                f"Position {i} plies ago (before game): should be empty board"

    def test_history_knight_movement(self):
        """Verify that a knight's position changes correctly across history."""
        moves = ['e2e4', 'e7e5', 'g1f3']
        board = _make_board_move_stack(moves)
        tensor = board_to_tensor(board)

        # Current: black to move, rotated. White knight at f3 (abs rank 2, file 5)
        # rotates to (7-2, 7-5) = (5, 2). White is P2 -> P2-knight plane 7.
        assert tensor[7, 5, 2] == 1.0, "White knight should be at rotated (5,2) in P2-knight plane 7"

        # 1 ply ago: white to move, no flip. Knight at g1 (rank 0, file 6).
        # P1-knight plane 1.
        assert tensor[14 + 1, 0, 6] == 1.0, "White knight should be at g1 1 ply ago"

        # 2 plies ago: black to move, rotated. Knight at g1 (abs rank 0, file 6)
        # rotates to (7, 1). White is P2 -> P2-knight plane 7.
        assert tensor[28 + 7, 7, 1] == 1.0, "White knight should be at rotated (7,1) 2 plies ago"

    def test_history_repetition_detection(self):
        """When a position repeats, the history planes should show identical patterns."""
        # Create a simple 3-fold repetition: 1.e4 e5 2.Nf3 Nc6 3.Ng1 Nb8 4.Nf3 Nc6
        moves = ['e2e4', 'e7e5', 'g1f3', 'b8c6', 'f3g1', 'c6b8', 'g1f3', 'b8c6']
        board = _make_board_move_stack(moves)
        tensor = board_to_tensor(board)

        # Current position (group 0) should match position 4 plies ago (group 4)
        # Both have e4, e5, Nf3, Nc6 on the board, same side to move.
        # Compare only the 12 piece planes -- the repetition planes differ
        # (current is the 2nd occurrence, 4 plies ago is the 1st).
        current_group = tensor[0:12]
        four_plies_ago_group = tensor[4 * PLANES_PER_HISTORY: 4 * PLANES_PER_HISTORY + 12]

        assert np.array_equal(current_group, four_plies_ago_group), \
            "Current position should match position 4 plies ago (repetition)"

    def test_each_plane_group_has_32_pieces(self):
        """Each 14-plane group representing a valid position should have 32 pieces."""
        moves = ['e2e4', 'd7d5', 'g1f3', 'g8f6']
        board = _make_board_move_stack(moves)
        tensor = board_to_tensor(board)

        # Group 0 (current) through group 4 (before first move): all have 32 pieces
        for i in range(5):
            offset = i * PLANES_PER_HISTORY
            piece_sum = tensor[offset:offset+12].sum()
            assert piece_sum == 32.0, \
                f"Group {i} ({i} plies ago) should have 32 pieces, got {piece_sum}"

        # Groups 5-7: before game start -> empty board (0 pieces)
        for i in range(5, 8):
            offset = i * PLANES_PER_HISTORY
            piece_sum = tensor[offset:offset+12].sum()
            assert piece_sum == 0.0, \
                f"Group {i} (before game) should have 0 pieces, got {piece_sum}"


class TestEdgeCases:

    def test_empty_board_before_game(self):
        """Planes for positions before game start should be empty."""
        board = _make_board_move_stack(['e2e4'])
        tensor = board_to_tensor(board)

        # Group 0: current (1 ply)
        # Group 1: initial position
        # Groups 2-7: before game start -> empty

        # Initial position (group 1): has pieces
        assert tensor[PLANES_PER_HISTORY:PLANES_PER_HISTORY+12].sum() > 0, "Initial position should have pieces"

        # Before game start (groups 2-7): empty
        for i in range(2, 8):
            offset = i * PLANES_PER_HISTORY
            assert tensor[offset:offset+12].sum() == 0.0, \
                f"Group {i} should be empty (before game start)"

    def test_castling_after_rook_moves(self):
        """Only one side loses castling after rook moves."""
        board = _make_board_move_stack(['a2a4', 'a7a5', 'h2h4', 'h7h5', 'a1a3', 'a8a6', 'a3b3'])
        tensor = board_to_tensor(board)

        # White to move: P1 = White. White lost queenside (a-rook moved) but keeps kingside.
        assert tensor[PLANE_CASTLING_P1_K].all() == 1.0, "P1 kingside should remain after a-rook moves"
        assert tensor[PLANE_CASTLING_P1_Q].sum() == 0.0, "P1 queenside should be lost after a-rook moves"

        # Black (P2) lost queenside but keeps kingside.
        assert tensor[PLANE_CASTLING_P2_K].all() == 1.0, "P2 kingside should remain"
        assert tensor[PLANE_CASTLING_P2_Q].sum() == 0.0, "P2 queenside should be lost after a-rook moves"

    def test_captured_pieces_disappear(self):
        """A captured piece should not appear in the current position."""
        # 1.e4 d5 2.exd5
        moves = ['e2e4', 'd7d5', 'e4d5']
        board = _make_board_move_stack(moves)
        tensor = board_to_tensor(board)

        # After 3 plies it's Black to move, so the board is rotated.
        # White pawn at d5 (abs rank 4, file 3) rotates to (7-4, 7-3) = (3, 4).
        # White is P2 -> P2-pawn plane 6.
        assert tensor[6, 3, 4] == 1.0, "White pawn should be at rotated (3,4) in P2-pawn plane 6"

        # Black pawn at d5 should be gone (captured). P1-pawn plane 0 at (3,4).
        assert tensor[0, 3, 4] == 0.0, "Black pawn at d5 should have been captured"

    def test_queen_promotion(self):
        """After promotion, queen should appear in the right plane."""
        # Pre-promotion FEN: White pawn e7, White king e1, Black king b8, Black rook g8
        fen_before = "1k4r1/4P3/8/8/8/8/8/4K3 w - - 0 1"
        board = chess.Board(fen_before)

        # Push promotion move: e7-e8=Q
        board.push(chess.Move.from_uci('e7e8q'))
        tensor = board_to_tensor(board)

        # After promotion it's Black to move, so the board is rotated.
        # White queen at e8 (abs rank 7, file 4) rotates to (0, 3).
        # White is P2 -> P2-queen plane 10.
        assert tensor[10, 0, 3] == 1.0, "White queen should be at rotated (0,3) in P2-queen plane 10"


class TestBoardToTensorBatch:

    def test_batch_shape(self):
        board = chess.Board()
        tensor = board_to_tensor(board)
        batch = tensor[np.newaxis, ...]
        assert batch.shape == (1, 119, 8, 8), f"Expected (1, 119, 8, 8), got {batch.shape}"


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])