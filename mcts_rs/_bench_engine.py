"""Microbenchmark: RustMCTS vs MCTS full search loop (no GPU, fake net).

Measures the end-to-end search() wall time for the same board with the same
deterministic network, isolating the MCTS-core speedup (selection + backprop).
board_to_tensor is still called (real encoding), so this includes the encode
cost — the part we did NOT move to Rust.
"""
import os, sys, time
import numpy as np
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (os.path.dirname(os.path.abspath(__file__)), _ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)
import chess

from mcts import MCTS
from rust_mcts import RustMCTS
from encoding import NUM_ACTIONS


class FakeNet:
    def predict(self, board):
        p = np.zeros(NUM_ACTIONS, dtype=np.float32)
        p[0] = 0.6; p[1] = 0.4
        return p, 0.1
    def predictBatch(self, states):
        n = len(states)
        p = np.zeros((n, NUM_ACTIONS), dtype=np.float32)
        p[:, 0] = 0.6; p[:, 1] = 0.4
        return p, np.full(n, 0.1, dtype=np.float32)


def play_games(engine_cls, sims, batch, n_games, fen, net=None):
    t0 = time.perf_counter()
    for g in range(n_games):
        board = chess.Board(fen)
        eng = engine_cls(network=FakeNet(), num_simulations=sims,
                         batch_size=batch, dirichlet_alpha=0,
                         dirichlet_epsilon=0)
        root = eng.get_root(board)
        moves = 0
        while not board.is_game_over() and moves < 40:
            vp, best, st = eng.search(root)
            if best is None:
                break
            board.push(best)
            root = eng.recycle_tree(root, best)
            moves += 1
    return time.perf_counter() - t0, moves


def bench(sims=64, batch=16, games=4):
    fen = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
    t_py, _ = play_games(MCTS, sims, batch, games, fen)
    t_rs, _ = play_games(RustMCTS, sims, batch, games, fen)
    print(f"games={games} sims={sims} batch={batch}")
    print(f"  MCTS     : {t_py:.3f}s")
    print(f"  RustMCTS : {t_rs:.3f}s")
    print(f"  Speedup  : {t_py/t_rs:.2f}x")


if __name__ == '__main__':
    bench(sims=int(os.environ.get('BENCH_SIMS', 64)),
          batch=int(os.environ.get('BENCH_BATCH', 16)),
          games=int(os.environ.get('BENCH_GAMES', 4)))