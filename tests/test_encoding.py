"""Tests for the 136-plane AlphaZero board encoding.

Verifies:
1. Correct tensor shape (136, 8, 8)
2. Correct plane assignments for piece positions
3. Side-to-move plane encoding
4. Castling rights encoding
5. History planes: positions from previous plies appear correctly
6. History planes: positions before game start are empty
7. Repetition detection: identical positions across history produce matching plane patterns
8. All positions from a played game have correct history
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import chess
from encoding import (
    board_to_tensor, NUM_PLANES, PLANES_PER_HISTORY, NUM_HISTORY_STEPS,
    PIECE_PLANE, PLANE_SIDE_TO_MOVE,
    PLANE_CASTLING_WK, PLANE_CASTLING_WQ, PLANE_CASTLING_BK, PLANE_CASTLING_BQ,
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
    """Return the plane index within a 17-plane group for a given piece."""
    return PIECE_PLANE[(piece_type, color)]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBoardToTensorShape:
    """The tensor must have shape (136, 8, 8)."""

    def test_shape_initial_position(self):
        board = chess.Board()
        tensor = board_to_tensor(board)
        assert tensor.shape == (136, 8, 8), f"Expected (136, 8, 8), got {tensor.shape}"

    def test_shape_mid_game(self):
        board = _make_board_move_stack(['e2e4', 'e7e5', 'g1f3', 'b8c6'])
        tensor = board_to_tensor(board)
        assert tensor.shape == (136, 8, 8), f"Expected (136, 8, 8), got {tensor.shape}"

    def test_type_is_float32(self):
        board = chess.Board()
        tensor = board_to_tensor(board)
        assert tensor.dtype == np.float32, f"Expected float32, got {tensor.dtype}"


class TestPiecePlanes:
    """Piece positions should appear in the correct planes (planes 0-5 white, 6-11 black)."""

    def test_white_pawn_at_e2(self):
        board = chess.Board()
        tensor = board_to_tensor(board)
        # e2 = rank 1, file 4
        # White pawn is plane 0 in the first 17-plane group (current position)
        assert tensor[0, 1, 4] == 1.0, "White pawn at e2 should be in plane 0, position (1,4)"

    def test_black_pawn_at_e7(self):
        board = chess.Board()
        tensor = board_to_tensor(board)
        # e7 = rank 6, file 4
        # Black pawn is plane 6 in the first 17-plane group
        assert tensor[6, 6, 4] == 1.0, "Black pawn at e7 should be in plane 6, position (6,4)"

    def test_white_king_at_e1(self):
        board = chess.Board()
        tensor = board_to_tensor(board)
        # e1 = rank 0, file 4
        # White king is plane 5
        assert tensor[5, 0, 4] == 1.0, "White king at e1 should be in plane 5"

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
    """Side-to-move plane should be 1.0 for white, 0.0 for black."""

    def test_white_to_move(self):
        board = chess.Board()
        tensor = board_to_tensor(board)
        # White to move -> plane 12 in first group should be all 1s
        assert tensor[12].all() == 1.0, "White to move: plane 12 should be all 1s"

    def test_black_to_move(self):
        board = _make_board_move_stack(['e2e4'])
        tensor = board_to_tensor(board)
        # Black to move -> plane 12 in first group should be all 0s
        assert tensor[12].sum() == 0.0, "Black to move: plane 12 should be all 0s"


class TestCastlingPlanes:
    """Castling rights should be encoded in planes 13-16."""

    def test_initial_position_all_castling(self):
        board = chess.Board()
        tensor = board_to_tensor(board)
        assert tensor[13].all() == 1.0, "WK castling should be 1s initially"
        assert tensor[14].all() == 1.0, "WQ castling should be 1s initially"
        assert tensor[15].all() == 1.0, "BK castling should be 1s initially"
        assert tensor[16].all() == 1.0, "BQ castling should be 1s initially"

    def test_no_castling_after_king_move(self):
        """White loses castling rights after moving king."""
        board = _make_board_move_stack(['e2e4', 'e7e5', 'f1c4', 'b8c6', 'd1f3', 'g8f6', 'e1g1'])
        tensor = board_to_tensor(board)
        # After 0-0, white has lost both castling rights
        assert tensor[13].sum() == 0.0, "WK castling should be 0 after king moved"
        assert tensor[14].sum() == 0.0, "WQ castling should be 0 after king moved"
        # Black should still have castling rights
        assert tensor[15].all() == 1.0, "BK castling should still be available"
        assert tensor[16].all() == 1.0, "BQ castling should still be available"


class TestHistoryPlanes:
    """The previous 7 positions should be encoded in planes 17-135."""

    def test_history_has_correct_positions(self):
        """After 4 plies, the history should show the correct positions."""
        moves = ['e2e4', 'e7e5', 'g1f3', 'b8c6']
        board = _make_board_move_stack(moves)
        tensor = board_to_tensor(board)

        # After 4 plies (even number), it's white's turn again
        # Current position (plane group 0, planes 0-16):
        assert tensor[12].sum() == 64.0, "Position after 4 plies: white to move"

        # Position 1 ply ago (plane group 1, planes 17-33):
        # After g1f3, before b8c6 -> black to move (odd plies = black)
        assert tensor[17 + 12].sum() == 0.0, "1 ply ago: black should be to move"

        # Position 2 plies ago (plane group 2, planes 34-50):
        # After e7e5, before g1f3 -> white to move (even plies = white)
        assert tensor[34 + 12].sum() == 64.0, "2 plies ago: white should be to move"

        # Position 3 plies ago (plane group 3, planes 51-67):
        # After e2e4, before e7e5 -> black to move (odd plies = black)
        assert tensor[51 + 12].sum() == 0.0, "3 plies ago: black should be to move"

        # Position 4 plies ago (plane group 4, planes 68-84):
        # Before e2e4 -> initial position, white to move
        assert tensor[68 + 12].sum() == 64.0, "4 plies ago: initial position, white to move"

        # Positions 5-7 (plane groups 5-7, planes 85-135):
        # Before game start -> empty board
        for i in range(5, 8):
            offset = i * 17
            assert tensor[offset:offset+12].sum() == 0.0, \
                f"Position {i} plies ago (before game): should be empty board"

    def test_history_knight_movement(self):
        """Verify that a knight's position changes correctly across history."""
        moves = ['e2e4', 'e7e5', 'g1f3']
        board = _make_board_move_stack(moves)
        tensor = board_to_tensor(board)

        # Current: white knight at f3 (rank 2, file 5). Knight plane = 1 (within group)
        # White knight plane in group 0 (current): plane 1
        assert tensor[1, 2, 5] == 1.0, "White knight should be at f3 in current position"

        # 1 ply ago: knight at g1 (rank 0, file 6). Group 1 offset = 17.
        assert tensor[17 + 1, 0, 6] == 1.0, "White knight should be at g1 1 ply ago"

        # 2 plies ago: knight at g1 as well (initial). Group 2 offset = 34.
        assert tensor[34 + 1, 0, 6] == 1.0, "White knight should be at g1 2 plies ago"

    def test_history_repetition_detection(self):
        """When a position repeats, the history planes should show identical patterns."""
        # Create a simple 3-fold repetition: 1.e4 e5 2.Nf3 Nc6 3.Ng1 Nb8 4.Nf3 Nc6
        # After move 8 (ply 8), the position matches move 4 (ply 4):
        #   Ply 8: r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w
        #   Ply 4: r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w
        moves = ['e2e4', 'e7e5', 'g1f3', 'b8c6', 'f3g1', 'c6b8', 'g1f3', 'b8c6']
        board = _make_board_move_stack(moves)
        tensor = board_to_tensor(board)

        # Current position (group 0) should match position 4 plies ago (group 4)
        # Both have e4, e5, Nf3, Nc6 on the board
        current_group = tensor[0:17]
        four_plies_ago_group = tensor[4 * 17: 4 * 17 + 17]

        assert np.array_equal(current_group, four_plies_ago_group), \
            "Current position should match position 4 plies ago (repetition)"

    def test_each_plane_group_has_32_pieces(self):
        """Each 17-plane group representing a valid position should have 32 pieces."""
        moves = ['e2e4', 'd7d5', 'g1f3', 'g8f6']
        board = _make_board_move_stack(moves)
        tensor = board_to_tensor(board)

        # Group 0 (current) through group 4 (before first move): all have 32 pieces
        for i in range(5):
            offset = i * 17
            piece_sum = tensor[offset:offset+12].sum()
            assert piece_sum == 32.0, \
                f"Group {i} ({i} plies ago) should have 32 pieces, got {piece_sum}"

        # Groups 5-7: before game start -> empty board (0 pieces)
        for i in range(5, 8):
            offset = i * 17
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
        assert tensor[17:29].sum() > 0, "Initial position should have pieces"

        # Before game start (groups 2-7): empty
        for i in range(2, 8):
            offset = i * 17
            assert tensor[offset:offset+12].sum() == 0.0, \
                f"Group {i} should be empty (before game start)"

    def test_castling_after_rook_moves(self):
        """Only one side loses castling after rook moves."""
        board = _make_board_move_stack(['a2a4', 'a7a5', 'h2h4', 'h7h5', 'a1a3', 'a8a6', 'a3b3'])
        tensor = board_to_tensor(board)

        # White lost queenside castling (a-rook moved) but keeps kingside (h-rook untouched)
        assert tensor[13].all() == 1.0, "WK should remain after a-rook moves"
        assert tensor[14].sum() == 0.0, "WQ should be lost after a-rook moves"

        # Black lost queenside castling (a-rook moved) but keeps kingside (h-rook untouched)
        assert tensor[15].all() == 1.0, "BK should remain"
        assert tensor[16].sum() == 0.0, "BQ should be lost after a-rook moves"

    def test_captured_pieces_disappear(self):
        """A captured piece should not appear in the current position."""
        # 1.e4 d5 2.exd5
        moves = ['e2e4', 'd7d5', 'e4d5']
        board = _make_board_move_stack(moves)
        tensor = board_to_tensor(board)

        # Black pawn at d5 should be gone (captured)
        # d5 = rank 4, file 3. Black pawn plane = 6 (within current group)
        assert tensor[6, 4, 3] == 0.0, "Black pawn at d5 should have been captured"

        # White pawn at d5 should exist (white pawn plane 0)
        # d5 = rank 4, file 3
        assert tensor[0, 4, 3] == 1.0, "White pawn should be at d5"

    def test_queen_promotion(self):
        """After promotion, queen should appear in the right plane."""
        # Set up a position with a white pawn about to promote
        board = chess.Board()
        board.clear()  # Clear all pieces
        board.set_piece_at(chess.E7, chess.Piece(chess.PAWN, chess.WHITE))
        board.set_piece_at(chess.E8, chess.Piece(chess.ROOK, chess.BLACK))
        board.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
        board.turn = chess.WHITE

        # Push promotion move: e7-e8=Q
        board.push(chess.Move.from_uci('e7e8q'))
        tensor = board_to_tensor(board)

        # White queen at e8 (rank 7, file 4)
        # White queen is plane 4 in current group
        assert tensor[4, 7, 4] == 1.0, "White queen should be at e8 after promotion"


class TestBoardToTensorBatch:

    def test_batch_shape(self):
        board = chess.Board()
        tensor = board_to_tensor(board)
        batch = tensor[np.newaxis, ...]
        assert batch.shape == (1, 136, 8, 8), f"Expected (1, 136, 8, 8), got {batch.shape}"


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])