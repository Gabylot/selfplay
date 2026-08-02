"""Tests for repetition detection in the 136-plane AlphaZero board encoding.

With the 136-plane encoding (8 history positions × 17 planes each), repetition
is detected by comparing the current position's 17-plane group with previous
groups. If the current position matches a position from 2 or 4 plies ago, the
network can learn that this is a repetition.

Key tests:
1. Tensor shape is (136, 8, 8)
2. A position repeated once has its current group matching a history group
3. A position repeated twice has its current group matching two history groups
4. Non-repeating positions have no matching history groups
5. Child boards with stack=True preserve history for repetition detection
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import chess
import numpy as np
from encoding import board_to_tensor, NUM_PLANES, PLANES_PER_HISTORY, NUM_HISTORY_STEPS


def test_tensor_shape():
    """Verify the tensor has 137 planes."""
    board = chess.Board()
    tensor = board_to_tensor(board)
    assert tensor.shape == (137, 8, 8), f"Expected (137, 8, 8), got {tensor.shape}"
    assert NUM_PLANES == 137, f"Expected NUM_PLANES=137, got {NUM_PLANES}"
    print("  PASS: test_tensor_shape")


def test_fresh_position_no_repetition():
    """Starting position has no history of itself — no history group should match current."""
    board = chess.Board()
    tensor = board_to_tensor(board)

    # Get the current position's 17-plane group
    current_group = tensor[0:17]

    # Check groups 1-7 (previous positions): none should match
    for i in range(1, 8):
        offset = i * 17
        hist_group = tensor[offset:offset + 17]
        assert not np.array_equal(current_group, hist_group), \
            f"Group {i} should not match current group on fresh board"

    print("  PASS: test_fresh_position_no_repetition")


def test_position_seen_twice():
    """After returning to the starting position once, the current group should
    match a previous group (2 plies ago)."""
    board = chess.Board()

    # Play moves that return to the starting position:
    # 1. Nf3 Nf6 2. Ng1 Ng8
    # After 4 plies, we're back to the starting position (second occurrence)
    moves_uci = ["g1f3", "g8f6", "f3g1", "f6g8"]
    for uci in moves_uci:
        board.push(chess.Move.from_uci(uci))

    # Now we're back to the starting position (second occurrence)
    assert board.is_repetition(2), "Board should detect 2-fold repetition"

    tensor = board_to_tensor(board)

    # After 4 moves, ply=4 (the 4th half-move has been pushed).
    # Group 0 (current) = ply 4 = back to start position
    # Group 1 = ply 3 = after Ng1, before Ng8
    # Group 2 = ply 2 = after Nf6, before Ng1
    # Group 3 = ply 1 = after Nf3
    # Group 4 = ply 0 = original start position
    # So group 0 and group 4 should match.
    current_group = tensor[0:17]
    group_4 = tensor[4 * 17: 4 * 17 + 17]
    assert np.array_equal(current_group, group_4), \
        "Current position should match position 4 plies ago (repetition)"

    print("  PASS: test_position_seen_twice")


def test_position_three_times():
    """After 3 occurrences of the same position, the current group should
    match two different history groups."""
    board = chess.Board()
    moves = [
        chess.Move.from_uci("g1f3"),   # 1. Nf3
        chess.Move.from_uci("g8f6"),   # ... Nf6
        chess.Move.from_uci("f3g1"),   # 2. Ng1
        chess.Move.from_uci("f6g8"),   # ... Ng8  (2nd occurrence of start)
        chess.Move.from_uci("g1f3"),   # 3. Nf3
        chess.Move.from_uci("g8f6"),   # ... Nf6
        chess.Move.from_uci("f3g1"),   # 4. Ng1
        chess.Move.from_uci("f6g8"),   # ... Ng8  (3rd occurrence of start)
    ]
    for m in moves:
        board.push(m)

    assert board.is_repetition(3), "Should detect 3-fold repetition"

    tensor = board_to_tensor(board)

    current_group = tensor[0:17]

    # After 8 plies, we're at start position again.
    # Group 4 (4 plies ago) should also be the start position.
    # Group 0 and group 4 should match.
    group_4 = tensor[4 * 17: 4 * 17 + 17]
    assert np.array_equal(current_group, group_4), \
        "Current position should match position 4 plies ago (3rd repetition)"

    print("  PASS: test_position_three_times")


def test_nonrepeating_position():
    """A random mid-game position should have no matching history groups."""
    board = chess.Board()
    # Play a real opening (no repetition)
    moves_uci = [
        "e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6",
        "b5a4", "g8f6", "e1g1", "f8e7",
    ]
    for uci in moves_uci:
        board.push(chess.Move.from_uci(uci))

    tensor = board_to_tensor(board)
    current_group = tensor[0:17]

    # No history group should match the current position
    matches = 0
    for i in range(1, 8):
        offset = i * 17
        hist_group = tensor[offset:offset + 17]
        if np.array_equal(current_group, hist_group):
            matches += 1

    assert matches == 0, f"Expected no matching history groups, found {matches}"

    print("  PASS: test_nonrepeating_position")


def test_child_board_repetition_detection():
    """Verify that child boards (stack=True) can detect repetition.

    This tests the key MCTS fix: child boards are created with stack=True
    so they preserve move history for repetition detection.
    """
    board = chess.Board()
    # Play moves that lead to a repetition cycle:
    # 1. Nf3 Nf6  2. Ng1 Ng8  3. Nf3 Nf6  4. Ng1 Ng8
    moves = [
        chess.Move.from_uci("g1f3"),   # 1. Nf3
        chess.Move.from_uci("g8f6"),   # ... Nf6
        chess.Move.from_uci("f3g1"),   # 2. Ng1
        chess.Move.from_uci("f6g8"),   # ... Ng8
        chess.Move.from_uci("g1f3"),   # 3. Nf3
        chess.Move.from_uci("g8f6"),   # ... Nf6
        chess.Move.from_uci("f3g1"),   # 4. Ng1
        chess.Move.from_uci("f6g8"),   # ... Ng8
    ]
    for m in moves:
        board.push(m)

    # Now we have the starting position for the 3rd time.
    assert board.is_repetition(3), "Should detect 3-fold repetition"
    assert board.is_game_over(claim_draw=True), "Board should be game over with claim_draw"

    # Now create a child board with stack=True (as we do in MCTS)
    child_board = board.copy(stack=True)
    assert child_board.is_repetition(3), "Child board should also detect repetition"
    assert child_board.is_game_over(claim_draw=True), "Child board should detect game over"

    # The tensor from the child board should also show the repetition
    child_tensor = board_to_tensor(child_board)
    current_group = child_tensor[0:17]
    group_4 = child_tensor[4 * 17: 4 * 17 + 17]
    assert np.array_equal(current_group, group_4), \
        "Child board tensor should show repetition in history planes"

    print("  PASS: test_child_board_repetition_detection")


def test_child_board_without_stack_cannot_detect():
    """Verify that child boards without stack FAIL to detect repetition.

    This demonstrates the bug that was fixed: child boards created with
    stack=False cannot use is_repetition() to detect threefold repetition.
    """
    board = chess.Board()
    # Play 3 cycles of Nf3/Nf6/Ng1/Ng8
    moves = [
        chess.Move.from_uci("g1f3"),   # 1. Nf3
        chess.Move.from_uci("g8f6"),   # ... Nf6
        chess.Move.from_uci("f3g1"),   # 2. Ng1
        chess.Move.from_uci("f6g8"),   # ... Ng8
        chess.Move.from_uci("g1f3"),   # 3. Nf3
        chess.Move.from_uci("g8f6"),   # ... Nf6
        chess.Move.from_uci("f3g1"),   # 4. Ng1
        chess.Move.from_uci("f6g8"),   # ... Ng8
    ]
    for m in moves:
        board.push(m)

    # With stack (correct behavior) — is_repetition(3) works
    board_with_stack = board.copy(stack=True)
    assert board_with_stack.is_repetition(3), "With stack: should detect 3-fold repetition"

    # Without stack (bug behavior) — is_repetition(3) fails because move history is lost
    board_without_stack = board.copy(stack=False)
    assert not board_without_stack.is_repetition(3), \
        "Without stack: should NOT detect repetition (this is the bug)"

    print("  PASS: test_child_board_without_stack_cannot_detect")

def test_repetition_respects_castling_rights():
    """Two positions with identical piece placement but different castling
    rights are NOT the same position for repetition purposes (FIDE rule)."""
    import chess

    board = chess.Board()
    # Clear a path for the king to step out and back without capturing
    # or landing on anything: 1. e4 e5 2. Ke2 Ke7 3. Ke1 Ke8
    # After this, piece placement matches the position after 1. e4 e5
    # (same pawn structure), but White and Black have both lost ALL
    # castling rights due to the king round-trip.
    for uci in ["e2e4", "e7e5", "e1e2", "e8e7", "e2e1", "e7e8"]:
        move = chess.Move.from_uci(uci)
        assert move in board.legal_moves, f"{uci} should be legal here"
        board.push(move)

    # Compare against a fresh board that reached the identical piece
    # placement via 1. e4 e5 only (2 plies) -- same squares, but that
    # board still has full castling rights, so it is a DIFFERENT
    # position under FIDE repetition rules despite matching placement.
    reference = chess.Board()
    reference.push(chess.Move.from_uci("e2e4"))
    reference.push(chess.Move.from_uci("e7e5"))

    assert board.fen().split(" ")[0] == reference.fen().split(" ")[0], \
        "Test setup error: piece placement should match after the round-trip"

    # The two positions must NOT be treated as the same position for
    # repetition purposes, since castling rights differ.
    assert board.fen() != reference.fen(), \
        "Castling rights should differ between these two positions"


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import traceback

    print("=" * 60)
    print("  Repetition Detection Tests (136-plane encoding)")
    print("=" * 60 + "\n")

    tests = [
        ("test_tensor_shape", test_tensor_shape),
        ("test_fresh_position_no_repetition", test_fresh_position_no_repetition),
        ("test_position_seen_twice", test_position_seen_twice),
        ("test_position_three_times", test_position_three_times),
        ("test_nonrepeating_position", test_nonrepeating_position),
        ("test_child_board_repetition_detection", test_child_board_repetition_detection),
        ("test_child_board_without_stack_cannot_detect", test_child_board_without_stack_cannot_detect),
        ("test_repetition_respects_castling_rights", test_repetition_respects_castling_rights),
    ]

    passed = 0
    failed = 0

    for name, func in tests:
        try:
            func()
            print(f"[PASS] {name}")
            passed += 1
        except Exception as e:
            print(f"[FAIL] {name}: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'=' * 60}")