"""Verify the force_mate_in_one MCTS toggle behavior.

Tests that:
- force_mate_in_one=True  forces the mate move even when it is the
  LEAST-visited child (one-hot policy on the mate move).
- force_mate_in_one=False disables the override, so normal visit-based
  selection picks the most-visited (non-mate) move instead.
"""
import numpy as np
import chess

from mcts import MCTS
from encoding import NUM_ACTIONS, move_to_policy_index


class MockNet:
    """Uniform-policy, zero-value network."""
    def predict(self, state):
        return np.ones(NUM_ACTIONS, dtype=np.float32) / NUM_ACTIONS, 0.0


# Valid back-rank mate-in-one: white plays Re1-e8#
FEN = "6k1/5ppp/8/8/8/8/8/4R1K1 w - - 0 1"


def build_root_with_mate_child(mcts, board):
    """Expand root and mark the mate child as the LEAST visited.

    This makes the distinction clear:
    - force_mate_in_one=True  -> mate move forced despite low visits
    - force_mate_in_one=False -> normal visit-based selection picks
                                 the most-visited (non-mate) move
    """
    root = mcts.get_root(board)
    mcts._expand_node(root)
    mate_move = None
    for m in board.legal_moves:
        nb = board.copy()
        nb.push(m)
        if nb.is_checkmate():
            mate_move = m
            break
    assert mate_move is not None, "No mate-in-one found in test position"
    for idx, child in root.children.items():
        if (child.move.from_square == mate_move.from_square and
                child.move.to_square == mate_move.to_square and
                child.move.promotion == mate_move.promotion):
            child.N = 1
            child.W = 1.0
            child.Q = 1.0
        else:
            child.N = 5
            child.W = 5.0
            child.Q = 1.0
    return root, mate_move


def main():
    board = chess.Board(FEN)

    # ── ON: mate-in-one forced as one-hot ──
    m_on = MCTS(MockNet(), num_simulations=1, force_mate_in_one=True)
    r_on, mate_move = build_root_with_mate_child(m_on, board)
    vp_on, mv_on = m_on.select_move_with_temperature(r_on, 0.0)
    on_ok = (mv_on == mate_move and vp_on.sum() == 1.0 and
             vp_on[move_to_policy_index(mate_move, board)] == 1.0)
    print(f"[ON ] mate={mate_move} selected={mv_on} "
          f"onehot_sum={vp_on.sum():.1f} -> {'OK' if on_ok else 'FAIL'}")

    # ── OFF: no forced mate (visit-based distribution) ──
    m_off = MCTS(MockNet(), num_simulations=1, force_mate_in_one=False)
    r_off, mate_move = build_root_with_mate_child(m_off, board)
    vp_off, mv_off = m_off.select_move_with_temperature(r_off, 0.0)
    # The mate child is the LEAST visited, so normal greedy selection
    # must NOT pick it — proving the mate-in-one forcing is disabled.
    off_not_forced_mate = (mv_off != mate_move)
    print(f"[OFF] mate={mate_move} selected={mv_off} "
          f"policy_sum={vp_off.sum():.1f} "
          f"not_forced_mate={'OK' if off_not_forced_mate else 'FAIL'}")

    ok = on_ok and off_not_forced_mate
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())