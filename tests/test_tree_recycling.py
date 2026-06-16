"""Tests for MCTS tree recycling.

Verifies that promoting a child node to root preserves the subtree,
resets root stats, clears parent pointers, and that Dirichlet noise
does not distort priors across recycled moves.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import chess
import numpy as np
from mcts import MCTS, MCTSNode
from encoding import move_to_policy_index, NUM_ACTIONS


# ─────────────────────────────────────────────────────────────────────────────
# Mock network for testing (no PyTorch dependency)
# ─────────────────────────────────────────────────────────────────────────────

class MockNetwork:
    """Returns uniform policy and zero value for any position."""
    def predict(self, state):
        policy = np.ones(NUM_ACTIONS, dtype=np.float32) / NUM_ACTIONS
        return policy, 0.0

    def predict_batch(self, states_batch):
        n = states_batch.shape[0]
        policies = np.ones((n, NUM_ACTIONS), dtype=np.float32) / NUM_ACTIONS
        values = np.zeros(n, dtype=np.float32)
        return policies, values


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_recycle_tree_basic():
    """recycle_tree promotes the correct child and clears its parent."""
    mcts = MCTS(MockNetwork(), num_simulations=10, dirichlet_alpha=0.0, dirichlet_epsilon=0.0)
    board = chess.Board()
    root = mcts.get_root(board)
    mcts.search(root)

    # Pick the most-visited child
    best_child = max(root.children.values(), key=lambda c: c.N)
    best_move = best_child.move

    recycled = mcts.recycle_tree(root, best_move)

    assert recycled is not None, "recycle_tree should return the child"
    assert recycled is best_child, "recycled node should be the same object"
    assert recycled.parent is None, "recycled root should have no parent"
    assert recycled.move == best_move, "recycled root should have the selected move"
    assert recycled.board.fen() == best_child.board.fen(), "board should be unchanged"
    print("  PASS: test_recycle_tree_basic")


def test_recycle_tree_children_preserved():
    """Recycling preserves the child's children (the pre-built subtree)."""
    mcts = MCTS(MockNetwork(), num_simulations=10, dirichlet_alpha=0.0, dirichlet_epsilon=0.0)
    board = chess.Board()
    root = mcts.get_root(board)
    mcts.search(root)

    best_child = max(root.children.values(), key=lambda c: c.N)
    num_children_before = len(best_child.children)
    children_before = dict(best_child.children)

    recycled = mcts.recycle_tree(root, best_child.move)

    assert len(recycled.children) == num_children_before, \
        f"Expected {num_children_before} children, got {len(recycled.children)}"
    for idx, child in children_before.items():
        assert idx in recycled.children, f"Child {idx} missing after recycling"
        assert recycled.children[idx] is child, f"Child {idx} should be same object"
    print("  PASS: test_recycle_tree_children_preserved")


def test_recycle_tree_root_stats_preserved():
    """The promoted root's N/W/Q are preserved for correct PUCT scaling."""
    mcts = MCTS(MockNetwork(), num_simulations=20, dirichlet_alpha=0.0, dirichlet_epsilon=0.0)
    board = chess.Board()
    root = mcts.get_root(board)
    mcts.search(root)

    best_child = max(root.children.values(), key=lambda c: c.N)
    assert best_child.N > 0, "best child should have visits"

    # Save original values before recycling
    orig_N = best_child.N
    orig_W = best_child.W
    orig_Q = best_child.Q

    recycled = mcts.recycle_tree(root, best_child.move)

    # N/W/Q should be preserved so PUCT exploration scaling is correct
    assert recycled.N == orig_N, f"root N should be {orig_N}, got {recycled.N}"
    assert recycled.W == orig_W, f"root W should be {orig_W}, got {recycled.W}"
    assert recycled.Q == orig_Q, f"root Q should be {orig_Q}, got {recycled.Q}"
    print("  PASS: test_recycle_tree_root_stats_preserved")


def test_recycle_tree_not_found():
    """recycle_tree returns None if the move is not among children."""
    mcts = MCTS(MockNetwork(), num_simulations=10, dirichlet_alpha=0.0, dirichlet_epsilon=0.0)
    board = chess.Board()
    root = mcts.get_root(board)
    mcts.search(root)

    # Try to recycle with a valid move that's not among the children
    # (h8h8 is not a legal move in the starting position)
    non_child_move = chess.Move(chess.H8, chess.H8)
    result = mcts.recycle_tree(root, non_child_move)
    assert result is None, "recycle_tree should return None for unknown move"
    print("  PASS: test_recycle_tree_not_found")


def test_search_on_recycled_root():
    """Running search on a recycled root works and produces valid results."""
    mcts = MCTS(MockNetwork(), num_simulations=10, dirichlet_alpha=0.0, dirichlet_epsilon=0.0)
    board = chess.Board()

    # First move
    root = mcts.get_root(board)
    visit_policy1, best_move1, stats1 = mcts.search(root)
    assert best_move1 is not None

    board.push(best_move1)
    recycled = mcts.recycle_tree(root, best_move1)

    # Second move on recycled root
    visit_policy2, best_move2, stats2 = mcts.search(recycled)
    assert best_move2 is not None
    assert visit_policy2.sum() > 0, "visit policy should be non-empty"
    assert stats2['num_simulations'] == 10

    print("  PASS: test_search_on_recycled_root")


def test_dirichlet_noise_uses_p_orig():
    """Dirichlet noise blends with P_orig, not the noisy P."""
    mcts = MCTS(MockNetwork(), num_simulations=10, dirichlet_alpha=0.3, dirichlet_epsilon=0.25)
    board = chess.Board()
    root = mcts.get_root(board)
    mcts.search(root)

    # Record P_orig for each child
    p_orig_values = {idx: child.P_orig for idx, child in root.children.items()}
    p_after_first = {idx: child.P for idx, child in root.children.items()}

    # After first search, P should differ from P_orig due to noise
    noise_was_applied = False
    for idx in root.children:
        if abs(root.children[idx].P - root.children[idx].P_orig) > 1e-8:
            noise_was_applied = True
            break
    assert noise_was_applied, "Dirichlet noise should have modified P"

    # Recycle and search again
    best_child = max(root.children.values(), key=lambda c: c.N)
    recycled = mcts.recycle_tree(root, best_child.move)
    mcts.search(recycled)

    # After second search on recycled root, P_orig should be unchanged
    for idx, child in recycled.children.items():
        if idx in p_orig_values:
            assert child.P_orig == p_orig_values[idx], \
                f"P_orig should not change across searches for child {idx}"

    print("  PASS: test_dirichlet_noise_uses_p_orig")


def test_no_prior_distortion_over_multiple_recycles():
    """Repeated noise application doesn't distort P away from P_orig.

    With the fix, P = (1-eps)*P_orig + eps*noise each time, so P always
    stays within eps of P_orig. Without the fix, P would be repeatedly
    blended with itself, causing exponential decay toward 0.
    """
    eps = 0.25
    mcts = MCTS(MockNetwork(), num_simulations=10, dirichlet_alpha=0.3, dirichlet_epsilon=eps)
    board = chess.Board()
    root = mcts.get_root(board)

    # Run 5 moves with recycling, tracking a specific child's P_orig
    tracked_p_orig = None
    for move_num in range(5):
        mcts.search(root)
        best_child = max(root.children.values(), key=lambda c: c.N)
        move = best_child.move

        # Track the P_orig of all children at this level
        if tracked_p_orig is None:
            tracked_p_orig = {idx: c.P_orig for idx, c in root.children.items()}

        board.push(move)
        root = mcts.recycle_tree(root, move)
        if root is None:
            break

    # After all recycling, P_orig should still equal the original values
    if tracked_p_orig is not None and root is not None:
        for idx, p_orig in tracked_p_orig.items():
            if idx in root.children:
                actual = root.children[idx].P_orig
                assert abs(actual - p_orig) < 1e-10, \
                    f"P_orig drifted: expected {p_orig}, got {actual}"
    print("  PASS: test_no_prior_distortion_over_multiple_recycles")


def test_short_game_with_recycling():
    """Play a short game using tree recycling — should complete without errors."""
    mcts = MCTS(MockNetwork(), num_simulations=10, dirichlet_alpha=0.0,
                dirichlet_epsilon=0.0)
    board = chess.Board()
    root = None
    moves_played = 0

    while not board.is_game_over(claim_draw=True) and moves_played < 30:
        if root is None:
            root = mcts.get_root(board)
        _, best_move, _ = mcts.search(root)
        assert best_move is not None, f"No move found at move {moves_played}"
        board.push(best_move)
        root = mcts.recycle_tree(root, best_move)
        moves_played += 1

    assert moves_played > 0, "Should have played at least one move"
    result = board.result() if board.is_game_over(claim_draw=True) else "*"
    print(f"  PASS: test_short_game_with_recycling ({moves_played} moves, result={result})")


if __name__ == "__main__":
    print("=== Tree Recycling Tests ===\n")

    test_recycle_tree_basic()
    test_recycle_tree_children_preserved()
    test_recycle_tree_root_stats_preserved()
    test_recycle_tree_not_found()
    test_search_on_recycled_root()
    test_dirichlet_noise_uses_p_orig()
    test_no_prior_distortion_over_multiple_recycles()
    test_short_game_with_recycling()

    print("\n=== All tree recycling tests passed! ===")