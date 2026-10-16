"""Equivalence test: Rust `mcts_rs` arena vs. pure-Python reference model.

Both models receive identical operation sequences (add_child with priors,
mark_expanded, select_batch, backprop with random values) and must select the
same leaves and reach identical N/W at every node.

Mapping: the Rust arena assigns node ids in creation order (root = 0, then each
add_child returns the next id). The Python model creates a RefNode for the same
operation, so we keep ``rust_to_ref[id]`` — both are driven in lockstep.
"""

import os
import sys
import ctypes
import random

HERE = os.path.dirname(os.path.abspath(__file__))
RS = os.path.join(HERE, '..', 'mcts_rs')
for p in (HERE, RS, os.path.dirname(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import _refmodel as rm


def load_rust():
    dll_path = os.path.join(RS, 'target', 'release', 'mcts_rs.dll')
    lib = ctypes.CDLL(dll_path)
    lib.mcts_rs_init()
    lib.mcts_rs_reset()
    lib.mcts_rs_root.restype = ctypes.c_int
    lib.mcts_rs_add_child.argtypes = [ctypes.c_int, ctypes.c_float]
    lib.mcts_rs_add_child.restype = ctypes.c_int
    lib.mcts_rs_mark_expanded.argtypes = [ctypes.c_int, ctypes.c_bool]
    lib.mcts_rs_select_batch.argtypes = [
        ctypes.c_int, ctypes.c_float, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_int)]
    lib.mcts_rs_select_batch.restype = ctypes.c_int
    lib.mcts_rs_backprop.argtypes = [ctypes.c_int, ctypes.c_double]
    lib.mcts_rs_get_n.argtypes = [ctypes.c_int]
    lib.mcts_rs_get_n.restype = ctypes.c_longlong
    lib.mcts_rs_get_w.argtypes = [ctypes.c_int]
    lib.mcts_rs_get_w.restype = ctypes.c_double
    lib.mcts_rs_recycle.argtypes = [ctypes.c_int]
    return lib


class Driver:
    """Drives both models in lockstep."""

    def __init__(self, lib, rng):
        self.lib = lib
        self.rng = rng
        self.rust_to_ref = {}   # rust node id -> RefNode

    def add(self, parent_rust, prior):
        rust_id = self.lib.mcts_rs_add_child(parent_rust, prior)
        ref = rm.RefNode(self.rust_to_ref[parent_rust], prior)
        self.rust_to_ref[parent_rust].children.append(ref)
        self.rust_to_ref[rust_id] = ref
        return rust_id

    def expand(self, rust_id):
        self.lib.mcts_rs_mark_expanded(rust_id, True)
        self.rust_to_ref[rust_id].is_expanded = True

    def select_batch(self, root_rust, batch_size):
        arr = (ctypes.c_int * batch_size)()
        depths = (ctypes.c_int * batch_size)()
        n = self.lib.mcts_rs_select_batch(
            root_rust, rm.C_PUCT, batch_size, arr, batch_size, depths)
        rust_leaves = [arr[i] for i in range(n)]
        # reference
        ref_leaves = rm.collect_batch(self.rust_to_ref[root_rust], batch_size)
        # compare by node identity (both are created in identical order)
        rust_ref = [self.rust_to_ref[r] for r in rust_leaves]
        assert len(rust_ref) == len(ref_leaves), \
            f"leaf-count mismatch: rust={len(rust_ref)} ref={len(ref_leaves)}"
        assert all(a is b for a, b in zip(rust_ref, ref_leaves)), \
            "leaf selection mismatch between Rust and reference"
        return rust_leaves, ref_leaves

    def backprop(self, rust_leaf, value):
        self.lib.mcts_rs_backprop(rust_leaf, value)
        rm.backprop(self.rust_to_ref[rust_leaf], value)

    def check_stats(self, n_nodes):
        for rust_id, ref in self.rust_to_ref.items():
            rn = self.lib.mcts_rs_get_n(rust_id)
            rw = self.lib.mcts_rs_get_w(rust_id)
            assert rn == ref.n, f"node {rust_id}: N {rn} != {ref.n}"
            assert abs(rw - ref.w) < 1e-5, f"node {rust_id}: W {rw} != {ref.w}"


def run(seed=42, sims=400, batch=8, branch=6):
    rng = random.Random(seed)
    lib = load_rust()
    d = Driver(lib, rng)
    root = lib.mcts_rs_root()
    d.rust_to_ref[root] = rm.RefNode(None, 0.0)

    n_nodes = 1
    total = 0
    for step in range(sims):
        # Grow the tree deterministically-ish: expand a random unexpanded node
        # if available.
        leaves_rust, ref_leaves = d.select_batch(root, batch)
        if not leaves_rust:
            break
        for r, ref in zip(leaves_rust, ref_leaves):
            value = rng.uniform(-1.0, 1.0)
            d.backprop(r, value)
        total += len(leaves_rust)
        # Periodically expand the least-recently-visited unexpanded node to
        # deepen the tree (root included). Expansion assigns that node random
        # children -- keep the added-child priors identical across both models.
        if step % 3 == 0:
            candidate = None
            for rid, ref in d.rust_to_ref.items():
                if not ref.is_expanded:
                    candidate = rid
                    break
            if candidate is not None:
                nb = rng.randint(1, branch)
                for _ in range(nb):
                    d.add(candidate, rng.random() * 0.5 + 0.1)
                d.expand(candidate)

        if step % 50 == 49:
            d.check_stats(len(d.rust_to_ref))
            print(f"  ... step {step+1}: nodes={len(d.rust_to_ref)} "
                  f"leaves={total} OK")

    d.check_stats(len(d.rust_to_ref))
    return len(d.rust_to_ref), total


def main():
    print("RUST MCTS EQUIVALENCE TEST")
    print("=" * 60)
    for seed in (1, 7, 42):
        nodes, leaves = run(seed=seed)
        print(f"  [PASS] seed={seed}: {nodes} nodes, {leaves} leaf-backprops, "
              "N/W identical to reference")
    print("=" * 60)
    print("  ALL TESTS PASSED")
    print("=" * 60)


if __name__ == '__main__':
    main()