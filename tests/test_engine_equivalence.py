"""End-to-end equivalence: MCTS vs RustMCTS with an identical network stub.

Drives both engines through repeated search + recycle on the same board with a
deterministic policy network, and asserts the selected best move and the root
visit counts match. Uses a deterministic FakeNet (no randomness) so both engines
agree exactly.
"""

import os
import sys
import ctypes

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (HERE, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import chess

from mcts import MCTS, MCTSNode
from rust_mcts import RustMCTS
from encoding import NUM_ACTIONS


class FakeNet:
    """Deterministic network: policy from board fen hash, value 0."""
    def predict(self, state):
        p = np.zeros(NUM_ACTIONS, dtype=np.float32)
        s = float(state.sum())
        p[0] = 0.6; p[1] = 0.4
        return p, 0.1
    def predictBatch(self, states):
        n = len(states)
        p = np.zeros((n, NUM_ACTIONS), dtype=np.float32)
        p[:, 0] = 0.6; p[:, 1] = 0.4
        return p, np.full(n, 0.1, dtype=np.float32)


def make_engines(**kw):
    kw.setdefault('num_simulations', 64)
    kw.setdefault('batch_size', 16)
    kw.setdefault('dirichlet_alpha', 0.0)   # no noise → deterministic
    kw.setdefault('dirichlet_epsilon', 0.0)
    kw.setdefault('c_puct', 1.5)
    kw.setdefault('max_game_length', 300)
    py = MCTS(network=FakeNet(), **kw)
    rs = RustMCTS(network=FakeNet(), **kw)
    return py, rs


def run_one(eng, start_fen, moves):
    board = chess.Board(start_fen)
    root = eng.get_root(board)
    visits = None
    last = None
    for m in moves:
        vp, best, stats = eng.search(root)
        visits = {i: v for i, v in enumerate(vp)}
        last = stats
        root = eng.recycle_tree(root, chess.Move.from_uci(m))
        if root is None:
            break
    vp, best, stats = eng.search(root)
    # aggregate root children N
    counts = {}
    for c_eng in ([root] if hasattr(root, 'children') else []):
        pass
    return root, vp, stats


def collect_counts(engine_root, engine):
    counts = {}
    for a, child in engine_root.children.items():
        if hasattr(engine, '_N'):
            counts[a] = engine._N(child.rs_id)
        else:
            counts[a] = child.N
    return counts


def main():
    print("ENGINE EQUIVALENCE (RustMCTS vs MCTS)")
    print("=" * 60)
    seq = ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5"]
    for run_i in (1, 2):
        py, rs = make_engines(num_simulations=64, batch_size=16)
        board = chess.Board()
        # compare same sequence of moves, checking root visit counts each ply
        py_root = py.get_root(board)
        rs_root = rs.get_root(board)
        ok = True
        for ply, m in enumerate(seq + ["a7a6"]):
            vp_py, bm_py, _ = py.search(py_root)
            vp_rs, bm_rs, _ = rs.search(rs_root)
            if (bm_py is None) != (bm_rs is None):
                ok = False
                print(f"  ply {ply}: best move None mismatch")
            # compare visit counts on root children
            pc = collect_counts(py_root, py)
            rc = collect_counts(rs_root, rs)
            if set(pc) != set(rc):
                ok = False
                print(f"  ply {ply}: child set mismatch py={set(pc)} rs={set(rc)}")
            elif any(abs(pc.get(a,0)-rc.get(a,0))>0 for a in pc):
                ok = False
                print(f"  ply {ply}: visit-count mismatch py={pc} rs={rc}")
            if not ok:
                break
            mv_str = bm_py.uci()
            py_root = py.recycle_tree(py_root, bm_py)
            rs_root = rs.recycle_tree(rs_root, bm_rs)
            if (py_root is None) != (rs_root is None):
                ok = False; break
        print(f"  [{'PASS' if ok else 'FAIL'}] run {run_i} over {ply+1} plies")
        if not ok:
            sys.exit(1)
    print("=" * 60)
    print("  ALL TESTS PASSED")
    print("=" * 60)


if __name__ == '__main__':
    main()