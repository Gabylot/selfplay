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
    N_before_second_search = recycled.N
    visit_policy2, best_move2, stats2 = mcts.search(recycled)
    assert best_move2 is not None
    assert visit_policy2.sum() > 0, "visit policy should be non-empty"

    # Under fixed-budget recycling, search() only runs enough NEW
    # simulations to top the root up to num_simulations total.
    expected_new_sims = max(0, mcts.num_simulations - N_before_second_search)
    assert stats2['num_simulations'] == expected_new_sims, (
        f"Expected {expected_new_sims} new sims (root already had "
        f"{N_before_second_search}/{mcts.num_simulations}), got {stats2['num_simulations']}"
    )
    # And the root should now be topped up to the full budget
    assert recycled.N == mcts.num_simulations, (
        f"root.N should reach num_simulations ({mcts.num_simulations}) after search, "
        f"got {recycled.N}"
    )

    print("  PASS: test_search_on_recycled_root")


def test_dirichlet_noise_uses_p_orig():
    """Dirichlet noise blends with P_orig, not the noisy P."""
    mcts = MCTS(MockNetwork(), num_simulations=10, dirichlet_alpha=0.3, dirichlet_epsilon=0.25)
    board = chess.Board()
    root = mcts.get_root(board)
    mcts.search(root)

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
    # Snapshot the recycled root's own children (tracked by identity -- the same
    # policy index can denote different moves at a different ply).
    p_orig_recycled = {id(c): c.P_orig for c in recycled.children.values()}
    mcts.search(recycled)

    # After second search on recycled root, P_orig should be unchanged
    for child in recycled.children.values():
        assert child.P_orig == p_orig_recycled[id(child)], \
            "P_orig should not change across searches"

    print("  PASS: test_dirichlet_noise_uses_p_orig")


def test_no_prior_distortion_over_multiple_recycles():
    """Re-applied Dirichlet noise re-anchors P to the untouched P_orig.

    On every search, ``P = (1-eps)*P_orig + eps*noise`` where ``P_orig`` is the
    node's ORIGINAL prior.  Hence (a) ``P_orig`` must never change, and (b)
    re-running a search with the SAME noise draw must reproduce the SAME ``P``
    -- proving the noise blends with ``P_orig``, not with the already-noised
    ``P`` (the old bug decayed P toward the noise on each re-search).  Both
    must hold across repeated tree recycles.

    Nodes are tracked by identity: the same policy *index* denotes different
    moves at different plies, so comparing priors by index across positions is
    invalid.
    """
    eps = 0.25
    mcts = MCTS(MockNetwork(), num_simulations=10, dirichlet_alpha=0.3, dirichlet_epsilon=eps)
    board = chess.Board()
    root = mcts.get_root(board)

    for _ in range(5):
        # Same seed => same noise draw => P must be reproduced exactly.
        np.random.seed(12345)
        mcts.search(root)
        p_first = {id(c): c.P for c in root.children.values()}
        p_orig = {id(c): c.P_orig for c in root.children.values()}

        np.random.seed(12345)
        mcts.search(root)
        for c in root.children.values():
            assert c.P_orig == p_orig[id(c)], "P_orig must never change"
            assert abs(c.P - p_first[id(c)]) < 1e-9, (
                "re-applied noise must re-anchor to P_orig, not blend with "
                "the already-noised P")

        if not root.children:
            break
        best_child = max(root.children.values(), key=lambda c: c.N)
        move = best_child.move
        board.push(move)
        new_root = mcts.recycle_tree(root, move)
        if new_root is None:
            break
        assert new_root is best_child
        root = new_root
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