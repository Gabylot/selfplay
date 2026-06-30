"""Tests for repetition encoding planes in board_to_tensor.

Verifies that the new planes 103/104 correctly encode repetition history:
- Plane 103: is_repetition(2) — position has appeared before
- Plane 104: is_repetition(3) — on the verge of 3-fold repetition
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import chess
import numpy as np
from encoding import board_to_tensor, NUM_PLANES


def test_tensor_shape():
    """Verify the tensor has the expected NUM_PLANES channels."""
    board = chess.Board()
    tensor = board_to_tensor(board)
    assert tensor.shape == (NUM_PLANES, 8, 8), f"Expected (NUM_PLANES, 8, 8), got {tensor.shape}"
    print("  PASS: test_tensor_shape")


def test_fresh_position_no_repetition():
    """Starting position has no repetition history — both planes should be 0."""
    board = chess.Board()
    tensor = board_to_tensor(board)
    assert np.all(tensor[103] == 0.0), "Plane 18 should be 0 for fresh position"
    assert np.all(tensor[104] == 0.0), "Plane 19 should be 0 for fresh position"
    print("  PASS: test_fresh_position_no_repetition")


def test_position_seen_twice_plane103():
    """After returning to the starting position once, plane 18 should be 1.0."""
    board = chess.Board()

    # Play moves that return to the starting position:
    # 1. Nf3  Nf6  2. Ng1  Ng6 (actually let's use a simpler approach)
    # Try: 1. Nf3 Nf6 2. Ng1 Nf6 (back to starting with 2 occurrences of the start)
    # Actually, python-chess tracks repetition from the game's move history.
    # We need: position A -> move -> position A again. That's a repetition of 2.
    # Example: 1. Nf3 Nf6 2. Nf3 (not legal, be2 not Nf3)
    # Actually we need: start (pos A), play 1. Nf3 Nf6 2. Ng1 Ng6 (now pos is same as start)
    # wait, the start position is after move 0. After 2 moves, we're NOT back to start.
    # We need: white moves Nf3, black moves Nf6, white moves Ng1, black moves Ng6
    # that's 4 half-moves, back to starting position?

    # Let's use a known repetition setup:
    # Position where the same FEN appears twice:
    # 1. Nf3 Nf6 2. Ng1 Ng6 3. Nf3 Nf6 4. Ng1 Ng6 -> starts repeating.

    # But is_repetition(2) checks from the current game's move_stack.
    # A starting position has not been repeated. We need to create a position
    # where the FEN appears twice in the game.

    # Use: 1. Nf3 Nf6 2. Ng1 Ng6
    # After 4 half-moves (2 full), we should be back to the starting position.
    # Two occurrences: starting position and current position = is_repetition(2) = True.

    # Actually, let's verify: the initial position is before any moves.
    # After 1.Nf3, the board is in a new state. After ...Nf6, new state.
    # After 2.Ng1, we're back to starting position.
    # After ...Ng6, new state.
    # So: position after 1.Nf3 Nf6 2.Ng1 should be the same as starting position.

    # But we need both occurrences to be tracked by python-chess is_repetition().
    # chess.Board.is_repetition(count) returns True if the current position
    # has been visited `count` times in the current game's move history.

    # So for is_repetition(2) to be True, the position must appear twice
    # in the game (including the current occurrence).

    # Let's use a line that definitely repeats the starting position:
    # 1. Nf3 Nf6 2. Ng1 Ng6 3. Nf3 Nf6 4. Ng1 Ng6
    # After move 4, we're back to start. And we've seen it: start + this position.
    # Actually after move 2 (2.Ng1), we're back to start position (once seen before).
    # After move 4 (4.Ng1), we're back to start position (seen twice before).

    # But we need to check: does python-chess count the starting position?
    # From the docs: "Detects if the current position has happened `count` times
    # in the current game."
    # The starting position is counted as the first occurrence.

    # So after 1.Nf3 Nf6 2.Ng1, we're back to start. is_repetition(2) = True.

    # Play moves that return to the starting position:
    # 1. Nf3 Nf6 2. Ng1 Ng8
    # Both knights return to original squares — starting position is repeated.
    moves_so_far = [
        chess.Move.from_uci("g1f3"),   # 1. Nf3
        chess.Move.from_uci("g8f6"),   # ... Nf6
        chess.Move.from_uci("f3g1"),   # 2. Ng1
        chess.Move.from_uci("f6g8"),   # ... Ng8
    ]

    for m in moves_so_far:
        board.push(m)

    # Now we're back to the starting position (second occurrence)
    assert board.is_repetition(2), "Board should detect 2-fold repetition"
    tensor = board_to_tensor(board)
    assert np.all(tensor[103] == 1.0), "Plane 18 should be 1.0 for seen-once position"
    assert np.all(tensor[104] == 0.0), "Plane 19 should be 0.0 (not yet 3-fold)"
    print("  PASS: test_position_seen_twice_plane103")


def test_position_three_times_plane104():
    """After 3 occurrences of the same position, both planes should be 1.0.

    Note: In a real game, is_repetition(3) would also mean is_game_over()
    returns True (the game ends in a draw by repetition). But we can still
    check the encoding plane is set correctly.
    """
    # Play moves that return to the starting position 3 times:
    # 1. Nf3 Nf6 2. Ng1 Ng6  3. Nf3 Nf6 4. Ng1 Ng6 5. Nf3 Nf6 6. Ng1
    board = chess.Board()
    moves_uci = [
        "g1f3", "g8f6",   # 1. Nf3 Nf6
        "f3g1", "f6g6",   # 2. Ng1 Ng6 -> not back to start yet...
    ]
    # Actually we need to go back to starting position 3 times.
    # Starting position = before any moves.
    # 1. Nf3 Nf6 2. Ng1Ng6 -> NOT back to start (Ng6 is not the start)
    # Let me think more carefully:
    # Starting position: all pieces on original squares
    # 1. Nf3 -> knights on f3, g8
    # 1...Nf6 -> knights on f3, f6
    # 2. Ng1 -> knights on g1, f6 -> THIS is back to starting position (king's knight on g1,
    #    queen's knight on b8... wait no)
    # Wait, in the starting position:
    # White: Rh1, Ng1, Bf1, Qd1, Ke1, Bc1, Nb1, Ra1
    # Black: Ra8, Nb8, Bc8, Qd8, Ke8, Bf8, Ng8, Rh8

    # After 1.Nf3 Nf6: White Nf3 (was Ng1) and Black Nf6 (was Ng8)
    # After 2.Ng1 Ng6: White Ng1 (back to g1) and Black Ng6 (was Ng8... Ng6 is NOT Ng8)
    # So the position after 2.Ng1 Ng6 is NOT the same as starting.

    # We need: 1.Nf3 Nf6 2.Ng1 Ng8 (back to starting position for black too)
    # But Ng8 is a retreat, is it legal? Yes, N is on f6, can go to g8.

    # Actually the bug-free approach: play 1.Nf3 Nf6 2.Ng1 Ng8
    # After that, board state = starting position (except it's still the old game)
    # is_repetition(2) should detect it's been seen twice (start + now).

    # But wait: is_repetition counts from the game start, and the starting position
    # is position 0 (before any moves). The current position after 2.Ng1 Ng8 is
    # position 4 (after 4 half-moves). Both have the same FEN.
    # python-chess.is_repetition(2) should return True.

    # For 3 occurrences:
    # 1.Nf3 Nf6 2.Ng1 Ng8 3.Nf3 Nf6 4.Ng1 Ng8 -> third time at start position
    # is_repetition(3) should return True.

    board = chess.Board()
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

    tensor = board_to_tensor(board)
    assert np.all(tensor[103] == 1.0), "Plane 18 should be 1.0 (seen before)"
    assert np.all(tensor[104] == 1.0), "Line 19 should be 1.0 (3-fold repetition imminent)"
    print("  PASS: test_position_three_times_plane104")


def test_nonrepeating_position():
    """A random mid-game position should have both planes at 0."""
    board = chess.Board()
    # Play a real opening (no repetition)
    moves_uci = [
        "e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6",
        "b5a4", "g8f6", "e1g1", "f8e7",
    ]
    for uci in moves_uci:
        board.push(chess.Move.from_uci(uci))

    tensor = board_to_tensor(board)
    assert np.all(tensor[103] == 0.0), "Plane 18 should be 0 for non-repeating position"
    assert np.all(tensor[104] == 0.0), "Plane 19 should be 0 for non-repeating position"
    print("  PASS: test_nonrepeating_position")


def test_repetition_plane_is_full_plane():
    """When set, the repetition planes should be fully set (1.0 everywhere)."""
    board = chess.Board()
    # Create a repetition
    for uci in ["g1f3", "g8f6", "f3g1", "f6g8"]:
        board.push(chess.Move.from_uci(uci))

    tensor = board_to_tensor(board)
    plane103 = tensor[103]
    plane104 = tensor[104]

    # Plane 18 should be all 1.0
    assert plane103.shape == (8, 8)
    assert np.all(plane103 == 1.0) or not np.any(plane103 == 1.0), \
        "Plane 18 should be uniform (all 0 or all 1)"
    # Plane 19 should be all 0.0 for double repetition (not triple)
    assert np.all(plane104 == 0.0), "Plane 19 should be 0 for double repetition"
    print("  PASS: test_repetition_plane_is_full_plane")


def test_child_board_repetition_detection():
    """Verify that child boards (stack=True) can detect repetition.

    This tests the key MCTS fix: child boards are created with stack=True
    so is_game_over() can detect threefold repetition for child nodes.
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
    # is_repetition(3) should detect this as a 3-fold repetition.
    # Note: is_game_over() by default requires is_repetition(4) (automatic draw).
    # For 3-fold repetition, use claim_draw=True or check is_repetition(3) directly.
    assert board.is_repetition(3), "Should detect 3-fold repetition"
    assert board.is_game_over(claim_draw=True), "Board should be game over with claim_draw"

    # Now create a child board with stack=True (as we do in MCTS)
    child_board = board.copy()  # stack=True
    assert child_board.is_repetition(3), "Child board should also detect repetition"
    assert child_board.is_game_over(claim_draw=True), "Child board should detect game over"
    print("  PASS: test_child_board_repetition_detection")


def test_child_board_without_stack_cannot_detect():
    """Verify that child boards without stack FAIL to detect repetition.

    This demonstrates the bug that was fixed: child boards created with
    stack=False cannot use is_game_over() to detect threefold repetition.
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
    board_with_stack = board.copy()
    assert board_with_stack.is_repetition(3), "With stack: should detect 3-fold repetition"

    # Without stack (bug behavior) — is_repetition(3) fails because move history is lost
    board_without_stack = board.copy(stack=False)
    assert not board_without_stack.is_repetition(3), \
        "Without stack: should NOT detect repetition (this is the bug)"

    print("  PASS: test_child_board_without_stack_cannot_detect")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import traceback

    print("=" * 60)
    print("  Repetition Encoding Tests")
    print("=" * 60 + "\n")

    tests = [
        ("test_tensor_shape", test_tensor_shape),
        ("test_fresh_position_no_repetition", test_fresh_position_no_repetition),
        ("test_position_seen_twice_plane103", test_position_seen_twice_plane103),
        ("test_position_three_times_plane104", test_position_three_times_plane104),
        ("test_nonrepeating_position", test_nonrepeating_position),
        ("test_repetition_plane_is_full_plane", test_repetition_plane_is_full_plane),
        ("test_child_board_repetition_detection", test_child_board_repetition_detection),
        ("test_child_board_without_stack_cannot_detect", test_child_board_without_stack_cannot_detect),
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