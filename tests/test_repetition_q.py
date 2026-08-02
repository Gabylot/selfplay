"""Tests: MCTS Q-values for threefold repetition (after Rust backend fix).

PURPOSE
=======
These tests verify that the fastchess Rust backend correctly detects
repetition and that MCTS assigns Q=0.0 for 3-fold repetition positions.

After the fix to `fastchess/src/lib.rs`:
  - `count_repetitions()` uses an incremental hash cache (no off-by-one)
  - `pop()` rebuilds from `initial_pos` (not `Chess::default()`)
  - `is_game_over()` treats 3-fold and 50-move as automatic draws
  - `result()` returns "1/2-1/2" for 3-fold and 50-move
  - Custom FEN boards get correct repetition detection
  - `stack=false` copies cannot detect repetition (python-chess semantics)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import chess
import numpy as np
from mcts import MCTS, MCTSNode
from encoding import NUM_ACTIONS


# ─────────────────────────────────────────────────────────────────────────────
# Mock network
# ─────────────────────────────────────────────────────────────────────────────

class MockNetwork:
    """Uniform policy, zero value -- terminal detection must bypass network."""

    def predict(self, state):
        policy = np.ones(NUM_ACTIONS, dtype=np.float32) / NUM_ACTIONS
        return policy, 0.0

    def predictBatch(self, states_batch):
        n = states_batch.shape[0]
        policies = np.ones((n, NUM_ACTIONS), dtype=np.float32) / NUM_ACTIONS
        values = np.zeros(n, dtype=np.float32)
        return policies, values


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_2fold_position():
    """Create a board where the starting position has been seen twice.

    Moves: 1. Nf3 Nf6 2. Ng1 Ng8
    After 4 plies, we're back to the starting position (2nd occurrence).
    """
    board = chess.Board()
    for uci in ["g1f3", "g8f6", "f3g1", "f6g8"]:
        board.push(chess.Move.from_uci(uci))
    return board


def _make_3fold_position():
    """Create a board where the starting position has been seen 3 times.

    Moves: 1. Nf3 Nf6 2. Ng1 Ng8 3. Nf3 Nf6 4. Ng1 Ng8
    After 8 plies, we're back to the starting position (3rd occurrence).
    """
    board = chess.Board()
    for uci in ["g1f3", "g8f6", "f3g1", "f6g8",
                "g1f3", "g8f6", "f3g1", "f6g8"]:
        board.push(chess.Move.from_uci(uci))
    return board


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: count_repetitions() is correct (no off-by-one)
# ─────────────────────────────────────────────────────────────────────────────

def test_count_repetitions_correct():
    """Verify count_repetitions() returns correct values at each ply."""
    print("\n" + "=" * 60)
    print("Test 1: count_repetitions() correct (no off-by-one)")
    print("=" * 60)

    # After 1 move: position seen once -> rep(2) should be False
    board = chess.Board()
    board.push(chess.Move.from_uci("e2e4"))
    rep2 = board.is_repetition(2)
    print(f"  After 1 move (e2e4): is_repetition(2) = {rep2} (expected False)")
    assert rep2 == False, (
        f"After 1 move, is_repetition(2) should be False, got {rep2}"
    )

    # After 2 moves: position seen once -> rep(2) should be False
    board = chess.Board()
    board.push(chess.Move.from_uci("e2e4"))
    board.push(chess.Move.from_uci("e7e5"))
    rep2 = board.is_repetition(2)
    print(f"  After 2 moves (e2e4 e7e5): is_repetition(2) = {rep2} (expected False)")
    assert rep2 == False, (
        f"After 2 moves, is_repetition(2) should be False, got {rep2}"
    )

    # After 4 plies (Nf3 Nf6 Ng1 Ng8): start pos seen twice -> rep(2)=True, rep(3)=False
    board = _make_2fold_position()
    rep2 = board.is_repetition(2)
    rep3 = board.is_repetition(3)
    print(f"  After 4 plies (2-fold): rep(2)={rep2} (expected True), rep(3)={rep3} (expected False)")
    assert rep2 == True, f"2-fold: rep(2) should be True, got {rep2}"
    assert rep3 == False, f"2-fold: rep(3) should be False, got {rep3}"

    # After 5 plies: position after Nf3 seen twice -> rep(2)=True, rep(3)=False
    board = chess.Board()
    for uci in ["g1f3", "g8f6", "f3g1", "f6g8", "g1f3"]:
        board.push(chess.Move.from_uci(uci))
    rep2 = board.is_repetition(2)
    rep3 = board.is_repetition(3)
    print(f"  After 5 plies: rep(2)={rep2} (expected True), rep(3)={rep3} (expected False)")
    assert rep2 == True, f"After 5 plies: rep(2) should be True, got {rep2}"
    assert rep3 == False, f"After 5 plies: rep(3) should be False, got {rep3}"

    # After 8 plies (actual 3-fold): rep(3) should be True
    board = _make_3fold_position()
    rep3 = board.is_repetition(3)
    print(f"  After 8 plies (3-fold): rep(3)={rep3} (expected True)")
    assert rep3 == True, f"3-fold: rep(3) should be True, got {rep3}"

    print("  PASS: count_repetitions() is correct (no off-by-one)")


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: Custom FEN repetition detection works
# ─────────────────────────────────────────────────────────────────────────────

def test_custom_fen_repetition_detected():
    """Verify repetition detection works for boards created from custom FENs."""
    print("\n" + "=" * 60)
    print("Test 2: Custom FEN repetition detection")
    print("=" * 60)

    # K+N vs K, knight shuffles Nd4-c6-d4-c6, king shuffles Ke8-d7-e8-d7
    board = chess.Board("4k3/8/8/8/3N4/8/8/4K3 w - - 0 1")
    moves = ["d4c6", "e8d7", "c6d4", "d7e8",  # 1st cycle
             "d4c6", "e8d7", "c6d4", "d7e8"]  # 2nd cycle

    for uci in moves:
        move = chess.Move.from_uci(uci)
        board.push(move)

    print(f"  Initial FEN: 4k3/8/8/8/3N4/8/8/4K3 w - - 0 1")
    print(f"  After 8 plies: {board.fen()}")

    rep2 = board.is_repetition(2)
    rep3 = board.is_repetition(3)
    print(f"  is_repetition(2) = {rep2} (expected True)")
    print(f"  is_repetition(3) = {rep3} (expected True)")

    assert rep3 == True, (
        f"Custom FEN 3-fold should be detected, got rep(3)={rep3}"
    )
    print("  PASS: Custom FEN repetition correctly detected")


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: Self-play game ends at correct ply (not premature)
# ─────────────────────────────────────────────────────────────────────────────

def test_correct_draw_termination():
    """Verify self-play games end at the correct ply for 3-fold repetition."""
    print("\n" + "=" * 60)
    print("Test 3: Correct draw termination timing")
    print("=" * 60)

    from selfplay import play_one_game

    class RepeatingMCTS:
        """Always plays Nf3/Nf6/Ng1/Ng8 to force repetition."""
        def __init__(self):
            self._cycle = ["g1f3", "g8f6", "f3g1", "f6g8"]
            self._idx = 0

        def get_root(self, board):
            return MCTSNode(board.copy())

        def search(self, root):
            return np.zeros(NUM_ACTIONS, dtype=np.float32), None, {"avg_depth": 0}

        def get_root_child_stats(self, root):
            return []

        def select_move_with_temperature(self, root, temperature):
            uci = self._cycle[self._idx % 4]
            self._idx += 1
            move = chess.Move.from_uci(uci)
            if move not in root.board.legal_moves:
                moves = list(root.board.legal_moves)
                if moves:
                    move = moves[0]
                else:
                    return np.zeros(NUM_ACTIONS, dtype=np.float32), None
            return np.zeros(NUM_ACTIONS, dtype=np.float32), move

        def recycle_tree(self, root, move):
            return None

    mcts_engine = RepeatingMCTS()
    game_data, game_info = play_one_game(
        mcts_engine,
        max_game_length=150,
        adjudicate_material=False,
        temp_threshold=30,
    )

    print(f"  Game length: {game_info['length']} half-moves")
    print(f"  Termination: {game_info['termination']}")
    print(f"  Result: {game_info['result_str']}")

    # With the fix, the game should end at 8 plies (actual 3-fold)
    assert game_info["termination"] == "repetition", (
        f"Expected 'repetition', got '{game_info['termination']}'"
    )
    assert game_info["length"] == 8, (
        f"Game should end at 8 plies (actual 3-fold), got {game_info['length']}"
    )
    assert game_info["result_str"] == "1/2-1/2", (
        f"Result should be '1/2-1/2', got '{game_info['result_str']}'"
    )
    print("  PASS: Game ends at correct ply (8) with result '1/2-1/2'")


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: MCTS returns Q=0 for 3-fold repetition (when visited)
# ─────────────────────────────────────────────────────────────────────────────

def test_3fold_repetition_child_q_when_visited():
    """When MCTS visits a child that is a 3-fold repetition, it should
    assign Q=0.0 (draw)."""
    print("\n" + "=" * 60)
    print("Test 4: 3-fold rep child Q=0 when visited")
    print("=" * 60)

    board = _make_3fold_position()
    assert board.is_repetition(3), "Should be 3-fold repetition"

    mcts = MCTS(MockNetwork(), num_simulations=10,
                dirichlet_alpha=0.0, dirichlet_epsilon=0.0)
    root = mcts.get_root(board)
    value = mcts._expand_node(root)

    print(f"  _expand_node returned: {value}")
    print(f"  Root expanded: {root.is_expanded}")
    print(f"  Root children: {len(root.children)}")
    print(f"  _game_over_cached: {root._game_over_cached}")

    assert abs(value) < 0.01, f"3-fold rep value should be 0.0, got {value}"
    assert not root.is_expanded or len(root.children) == 0, \
        "3-fold position should not have children"
    print("  PASS: 3-fold repetition detected as terminal with Q=0.0")

    # Also verify via _get_terminal_value
    node = MCTSNode(board=board)
    val = mcts._get_terminal_value(node)
    assert abs(val) < 0.01, f"_get_terminal_value should be 0.0, got {val}"
    print(f"  PASS: _get_terminal_value(3-fold) = {val}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: stack=false copies cannot detect repetition
# ─────────────────────────────────────────────────────────────────────────────

def test_stack_false_no_repetition():
    """Verify that copy(stack=False) cannot detect repetition.

    This matches python-chess semantics: a stack-less copy has no move
    history and thus cannot know about prior occurrences.
    """
    print("\n" + "=" * 60)
    print("Test 5: stack=false cannot detect repetition")
    print("=" * 60)

    board = _make_3fold_position()
    assert board.is_repetition(3), "Original should detect 3-fold"

    # stack=True preserves history
    board_stack = board.copy(stack=True)
    assert board_stack.is_repetition(3), "stack=True should detect 3-fold"
    print("  PASS: copy(stack=True) detects repetition")

    # stack=False loses history
    board_nostack = board.copy(stack=False)
    assert not board_nostack.is_repetition(3), \
        "stack=False should NOT detect 3-fold"
    assert not board_nostack.is_repetition(2), \
        "stack=False should NOT detect 2-fold"
    print("  PASS: copy(stack=False) cannot detect repetition")


# ─────────────────────────────────────────────────────────────────────────────
# Test 6: pop() on custom FEN board does not corrupt state
# ─────────────────────────────────────────────────────────────────────────────

def test_pop_custom_fen_no_corruption():
    """Verify that pop() on a board created from a custom FEN correctly
    restores the previous position (not the standard starting position).
    """
    print("\n" + "=" * 60)
    print("Test 6: pop() on custom FEN board")
    print("=" * 60)

    # Create a board from a custom FEN
    fen = "4k3/8/8/8/3N4/8/8/4K3 w - - 0 1"
    board = chess.Board(fen)
    original_fen = board.fen()

    # Push a move and then pop it
    move = chess.Move.from_uci("d4c6")
    board.push(move)
    fen_after_push = board.fen()
    print(f"  Original FEN: {original_fen}")
    print(f"  After push (d4c6): {fen_after_push}")

    popped = board.pop()
    fen_after_pop = board.fen()
    print(f"  After pop: {fen_after_pop}")
    print(f"  Popped move: {popped.uci()}")

    # The FEN after pop should match the original (except for halfmove clock
    # and fullmove number which may differ slightly in FEN formatting)
    # Compare piece placement and side to move
    original_parts = original_fen.split()
    popped_parts = fen_after_pop.split()

    assert original_parts[0] == popped_parts[0], (
        f"Piece placement corrupted after pop!\n"
        f"  Original: {original_parts[0]}\n"
        f"  After pop: {popped_parts[0]}"
    )
    assert original_parts[1] == popped_parts[1], (
        f"Side to move corrupted after pop!\n"
        f"  Original: {original_parts[1]}\n"
        f"  After pop: {popped_parts[1]}"
    )
    print("  PASS: pop() correctly restores custom FEN position")

    # Test multiple push/pop cycles
    board2 = chess.Board(fen)
    for i in range(3):
        board2.push(chess.Move.from_uci("d4c6"))
        board2.push(chess.Move.from_uci("e8d7"))
        board2.pop()
        board2.pop()
        assert board2.fen().split()[0] == original_fen.split()[0], (
            f"Position corrupted after push/pop cycle {i+1}"
        )
    print("  PASS: Multiple push/pop cycles preserve custom FEN state")


# ─────────────────────────────────────────────────────────────────────────────
# Test 7: is_game_over() detects 3-fold automatically
# ─────────────────────────────────────────────────────────────────────────────

def test_is_game_over_detects_3fold():
    """Verify is_game_over() returns True for 3-fold repetition (automatic)."""
    print("\n" + "=" * 60)
    print("Test 7: is_game_over() detects 3-fold automatically")
    print("=" * 60)

    # 2-fold should NOT be game over
    board = _make_2fold_position()
    assert not board.is_game_over(), "2-fold should NOT be game over"
    print("  PASS: 2-fold is not game over")

    # 3-fold SHOULD be game over (automatic)
    board = _make_3fold_position()
    assert board.is_game_over(), "3-fold should be game over (automatic)"
    print("  PASS: 3-fold is game over (automatic)")

    # result() should return "1/2-1/2" for 3-fold
    result = board.result()
    assert result == "1/2-1/2", (
        f"result() should return '1/2-1/2' for 3-fold, got '{result}'"
    )
    print(f"  PASS: result() = '{result}' for 3-fold")


# ─────────────────────────────────────────────────────────────────────────────
# Test 8: is_repetition at each ply (knight shuffle)
# ─────────────────────────────────────────────────────────────────────────────

def test_repetition_at_each_ply():
    """Trace is_repetition(2) and is_repetition(3) at each ply to verify
    correct behavior after the fix."""
    print("\n" + "=" * 60)
    print("Test 8: is_repetition at each ply (knight shuffle)")
    print("=" * 60)

    board = chess.Board()
    moves = ["g1f3", "g8f6", "f3g1", "f6g8",
             "g1f3", "g8f6", "f3g1", "f6g8"]

    print(f"  {'Ply':4s} {'Move':8s} {'rep(2)':8s} {'rep(3)':8s} {'Description'}")
    print(f"  {'---':4s} {'---':8s} {'---':8s} {'---':8s} {'---'}")

    # Ply 0 (initial position)
    rep2_0 = board.is_repetition(2)
    rep3_0 = board.is_repetition(3)
    print(f"  {'0':4s} {'(start)':8s} {str(rep2_0):8s} {str(rep3_0):8s} Start position (1st occurrence)")
    assert not rep2_0, "Ply 0: rep(2) should be False"
    assert not rep3_0, "Ply 0: rep(3) should be False"

    for i, uci in enumerate(moves):
        board.push(chess.Move.from_uci(uci))
        ply = i + 1
        rep2 = board.is_repetition(2)
        rep3 = board.is_repetition(3)

        if ply <= 3:
            desc = f"After {uci} (1st time)"
            expected_rep2 = False
            expected_rep3 = False
        elif ply == 4:
            desc = "Start pos 2nd time (actual 2-fold)"
            expected_rep2 = True
            expected_rep3 = False
        elif ply <= 7:
            desc = f"After {uci} (2nd time, actual 2-fold)"
            expected_rep2 = True
            expected_rep3 = False
        elif ply == 8:
            desc = "Start pos 3rd time (actual 3-fold)"
            expected_rep2 = True
            expected_rep3 = True

        status = "OK" if (rep2 == expected_rep2 and rep3 == expected_rep3) else "FAIL!"
        print(f"  {ply:4d} {uci:8s} {str(rep2):8s} {str(rep3):8s} {desc} [{status}]")

        assert rep2 == expected_rep2, (
            f"Ply {ply}: rep(2) should be {expected_rep2}, got {rep2}"
        )
        assert rep3 == expected_rep3, (
            f"Ply {ply}: rep(3) should be {expected_rep3}, got {rep3}"
        )

    print("\n  PASS: All repetition values correct at every ply")


# ─────────────────────────────────────────────────────────────────────────────
# Test 9: Batched MCTS handles 3-fold as terminal
# ─────────────────────────────────────────────────────────────────────────────

def test_batched_mcts_3fold_terminal():
    """Verify batched MCTS (batch_size > 1) also detects 3-fold as terminal."""
    print("\n" + "=" * 60)
    print("Test 9: Batched MCTS 3-fold terminal detection")
    print("=" * 60)

    board = _make_3fold_position()
    mcts = MCTS(MockNetwork(), num_simulations=50, batch_size=8,
                dirichlet_alpha=0.0, dirichlet_epsilon=0.0)

    root = mcts.get_root(board)
    visit_policy, best_move, stats = mcts.search(root)

    print(f"  best_move: {best_move}")
    print(f"  root children: {len(root.children)}")
    print(f"  root._game_over_cached: {root._game_over_cached}")

    assert len(root.children) == 0 or not root.is_expanded, \
        "3-fold position should not be expanded"
    print("  PASS: Batched MCTS correctly handles 3-fold as terminal")


# ─────────────────────────────────────────────────────────────────────────────
# Test 10: Direct terminal value for 3-fold
# ─────────────────────────────────────────────────────────────────────────────

def test_terminal_value_3fold_direct():
    """Directly verify _get_terminal_value returns 0.0 for a 3-fold position."""
    print("\n" + "=" * 60)
    print("Test 10: _get_terminal_value for 3-fold rep")
    print("=" * 60)

    board = _make_3fold_position()
    mcts = MCTS(MockNetwork(), num_simulations=10,
                dirichlet_alpha=0.0, dirichlet_epsilon=0.0)

    node = MCTSNode(board)
    val = mcts._get_terminal_value(node)
    print(f"  3-fold position: _get_terminal_value = {val}")
    assert abs(val) < 0.01, f"Expected 0.0, got {val}"
    print("  PASS: 3-fold repetition -> Q=0.0")


# ─────────────────────────────────────────────────────────────────────────────
# Test 11: 50-move rule is automatic
# ─────────────────────────────────────────────────────────────────────────────

def test_fifty_move_automatic():
    """Verify 50-move rule is treated as automatic draw."""
    print("\n" + "=" * 60)
    print("Test 11: 50-move rule automatic draw")
    print("=" * 60)

    board = chess.Board("4k3/8/8/8/8/8/8/4K3 w - - 100 51")
    assert board.is_fifty_moves(), "Position should trigger 50-move rule"
    assert board.is_game_over(), "50-move should be game over (automatic)"
    result = board.result()
    assert result == "1/2-1/2", (
        f"result() should return '1/2-1/2' for 50-move, got '{result}'"
    )
    print(f"  is_game_over: {board.is_game_over()}")
    print(f"  result: {result}")
    print("  PASS: 50-move rule is automatic draw")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import traceback

    print("=" * 60)
    print("  MCTS Repetition Q-Value Tests (post-fix)")
    print("=" * 60)

    tests = [
        ("test_count_repetitions_correct",
         test_count_repetitions_correct),
        ("test_custom_fen_repetition_detected",
         test_custom_fen_repetition_detected),
        ("test_correct_draw_termination",
         test_correct_draw_termination),
        ("test_3fold_repetition_child_q_when_visited",
         test_3fold_repetition_child_q_when_visited),
        ("test_stack_false_no_repetition",
         test_stack_false_no_repetition),
        ("test_pop_custom_fen_no_corruption",
         test_pop_custom_fen_no_corruption),
        ("test_is_game_over_detects_3fold",
         test_is_game_over_detects_3fold),
        ("test_repetition_at_each_ply",
         test_repetition_at_each_ply),
        ("test_batched_mcts_3fold_terminal",
         test_batched_mcts_3fold_terminal),
        ("test_terminal_value_3fold_direct",
         test_terminal_value_3fold_direct),
        ("test_fifty_move_automatic",
         test_fifty_move_automatic),
    ]

    passed = 0
    failed = 0

    for name, func in tests:
        try:
            func()
            passed += 1
        except Exception as e:
            print(f"\n[FAIL] {name}: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'=' * 60}")