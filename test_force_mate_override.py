"""End-to-end smoke test: ForceMateMCTS override converts sparse endgames.

Tests two scenarios with a zero-value uniform-policy network (value head
thinks draws — the target-poisoning worst case):

1. KQ vs K far corner (3 pieces): forced mate in 2, found by the shallow
   5-ply search.
2. KR vs K near corner (3 pieces): a mate that can require 4+ moves,
   requiring the deep 9-plies / <=4-pieces search to prove.

Both should terminate in checkmate with the override firing.
"""
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import chess
import sys

from forced_mate import ForceMateMCTS, reset_memo, forced_mate


class MockNet:
    """Deterministic uniform-policy, zero-value network (value head thinks draws)."""
    def predict(self, state):
        return np.ones(4672, dtype=np.float32) / 4672, 0.0

    def predictBatch(self, states):
        n = len(states)
        return (np.tile(np.ones(4672, dtype=np.float32) / 4672, (n, 1)),
                np.zeros(n, dtype=np.float32))


def play(fen, max_len=60):
    reset_memo()
    fm = ForceMateMCTS(MockNet(), num_simulations=8, batch_size=1,
                       dirichlet_alpha=0.0, dirichlet_epsilon=0.0,
                       max_game_length=max_len, adjudicate_material=False,
                       max_force_plies=5, min_pieces_gate=8,
                       deep_plies=9, deep_gate_pieces=4)
    board = chess.Board(fen)
    root = None
    moves = 0
    while (not board.is_game_over()
           and not board.is_repetition(3)
           and moves < max_len):
        if root is None:
            root = fm.get_root(board)
        fm.search(root)
        _, selected = fm.select_move_with_temperature(root, 0.1)
        if selected is None:
            selected = max(root.children, key=lambda c: c.N).move
        board.push(selected)
        moves += 1
        root = fm.recycle_tree(root, selected)
    return board, moves, fm


def main():
    # ── Scenario 1: KQ vs K far (shallow search converts) ──
    kq = '8/8/8/8/2Q5/8/4K3/7k w - - 0 1'
    b1, n1, fm1 = play(kq)
    print(f"[KQ]  result={b1.result()} checkmate={b1.is_checkmate()} "
          f"len={n1} forced={fm1.forced}")
    ok1 = b1.is_checkmate() and fm1.forced >= 1

    # ── Scenario 2: KR vs K — a mate that needs the deep 9-ply search ──
    # (the shallow 5-ply search returns None here, the 9-ply one finds it)
    kr = '8/8/8/5R2/8/7k/4K3/8 w - - 0 1'
    reset_memo()
    shallow = forced_mate(chess.Board(kr), 5)
    reset_memo()
    deep = forced_mate(chess.Board(kr), 9)
    print(f"[KR]  shallow(5pl)={shallow}  deep(9pl)={deep}")
    b2, n2, fm2 = play(kr)
    print(f"[KR]  result={b2.result()} checkmate={b2.is_checkmate()} "
          f"len={n2} forced={fm2.forced}  deep_used={'deep' if any(p > 5 for _, p, _ in fm2.forced_log) else 'shallow-only'}")
    ok2 = b2.is_checkmate() and fm2.forced >= 1

    print(f"\nRESULT: {'PASS' if (ok1 and ok2) else 'FAIL'}")
    sys.exit(0 if (ok1 and ok2) else 1)


if __name__ == '__main__':
    main()