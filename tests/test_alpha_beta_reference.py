"""Regression tests for the depth-N alpha-beta reference opponent.

The reference engine previously had a sign bug in its move selection that
made BOTH colors effectively avoid winning material, e.g. from a position
with a free queen capture available at depth 4 it would refuse to take it.
These tests lock in correct tactical behaviour (material-winning moves and
mate-in-1 detection) so the "reference" opponent is actually a meaningful
measurement signal.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import chess
from evaluation import alpha_beta_best_move, alpha_beta_search


def test_white_captures_free_queen():
    """White to move should take the undefended queen on a2 (Rxa2)."""
    board = chess.Board("7k/8/8/8/8/8/q7/R6K w - - 0 1")
    move = alpha_beta_best_move(board, depth=4)
    assert move is not None
    assert move.uci() == "a1a2", f"expected Rxa2 (a1a2), got {move.uci()}"


def test_black_captures_hanging_rook():
    """Black to move should take the undefended rook on a1 (Qxa1)."""
    board = chess.Board("7k/8/8/8/8/8/q7/R6K b - - 0 1")
    move = alpha_beta_best_move(board, depth=4)
    assert move is not None
    assert move.uci() == "a2a1", f"expected Qxa1 (a2a1), got {move.uci()}"


def test_mate_in_one_back_rank():
    """White to move should deliver back-rank mate Rb1-b8#."""
    board = chess.Board("7k/6pp/8/8/8/8/8/1R5K w - - 0 1")
    move = alpha_beta_best_move(board, depth=4)
    assert move is not None
    assert move.uci() == "b1b8", f"expected Rb8# (b1b8), got {move.uci()}"
    # Play the move and confirm it is actually checkmate
    board.push(move)
    assert board.is_checkmate()


def test_search_returns_side_to_move_perspective():
    """alpha_beta_search returns positive for the side to move when winning."""
    # Black to move with a queen on a2 vs bare white king h1: black is winning.
    board = chess.Board("7k/8/8/8/8/8/q7/7K b - - 0 1")
    val = alpha_beta_search(board, depth=2, alpha=-float("inf"), beta=float("inf"))
    assert val > 0, f"expected positive value (black up a queen), got {val}"

    # White to move in the same position is losing => negative value.
    board = chess.Board("7k/8/8/8/8/8/q7/7K w - - 0 1")
    val = alpha_beta_search(board, depth=2, alpha=-float("inf"), beta=float("inf"))
    assert val < 0, f"expected negative value (white down a queen), got {val}"


def test_alpha_beta_vs_alpha_beta_runs():
    """A full game between two depth-4 alpha-beta engines must terminate
    without error and produce a standard result string."""
    from evaluation import play_game_alpha_beta_vs_alpha_beta
    result, move_count = play_game_alpha_beta_vs_alpha_beta(depth=3, max_moves=60)
    # "*" is returned when the move cap is hit before the game ends
    # (material-only engines often shuffle without reaching checkmate).
    assert result in ("1-0", "0-1", "1/2-1/2", "*")
    assert move_count > 0
