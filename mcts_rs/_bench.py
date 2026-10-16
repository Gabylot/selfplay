"""Microbenchmark: Rust `mcts_rs.select_batch` vs pure-Python `_collect_batch`.

Builds an identical deep, fully-expanded tree in the Rust arena and the Python
refmodel, then times many select_batch calls -- the hottest MCTS loop. This
measures representation/loop overhead only (no chess, no network).
"""
import os, sys, time, random
import ctypes

HERE = os.path.dirname(os.path.abspath(__file__))
RS = os.path.join(HERE, '..', 'mcts_rs')
for p in (HERE, RS):
    if p not in sys.path:
        sys.path.insert(0, p)
import _refmodel as rm


def make_rust_tree(lib, depth=4, fanout=4, rng=None):
    """Fully-expanded binary-ish tree; returns root restore + node id set."""
    rng = rng or random.Random(1)
    rust_to_ref = {}
    root = lib.mcts_rs_root()
    rust_to_ref[root] = rm.RefNode(None, 0.0)
    frontier = [root]
    for _ in range(depth):
        nxt = []
        for p in frontier:
            nb = rng.randint(2, fanout)
            for _ in range(nb):
                prior = rng.random() * 0.9 + 0.1
                r = lib.mcts_rs_add_child(p, prior)
                ref = rm.RefNode(rust_to_ref[p], prior)
                rust_to_ref[p].children.append(ref)
                rust_to_ref[r] = ref
                nxt.append(r)
            lib.mcts_rs_mark_expanded(p, True)
            rust_to_ref[p].is_expanded = True
        frontier = nxt
    return root, rust_to_ref


def make_py_tree(depth=4, fanout=4, rng=None):
    rng = rng or random.Random(1)
    root = rm.RefNode(None, 0.0)
    frontier = [root]
    for _ in range(depth):
        nxt = []
        for p in frontier:
            nb = rng.randint(2, fanout)
            for _ in range(nb):
                c = rm.RefNode(p, rng.random() * 0.9 + 0.1)
                p.children.append(c)
                nxt.append(c)
            p.is_expanded = True
        frontier = nxt
    return root


def bench_rust(lib, root, batch, iters):
    arr = (ctypes.c_int * batch)()
    depths = (ctypes.c_int * batch)()
    for _ in range(30):
        lib.mcts_rs_select_batch(root, rm.C_PUCT, batch, arr, batch, depths)
    t = time.perf_counter()
    for _ in range(iters):
        lib.mcts_rs_select_batch(root, rm.C_PUCT, batch, arr, batch, depths)
    return time.perf_counter() - t


def bench_py(root, batch, iters):
    for _ in range(30):
        rm.collect_batch(root, batch)
    t = time.perf_counter()
    for _ in range(iters):
        rm.collect_batch(root, batch)
    return time.perf_counter() - t


def main():
    depth, fanout, batch, iters = int(os.environ.get('BENCH_DEPTH', 4)), 4, 32, 3000
    lib = ctypes.CDLL(os.path.join(RS, 'target', 'release', 'mcts_rs.dll'))
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

    seed = 5
    lib.mcts_rs_reset()
    rust_root, _ = make_rust_tree(lib, depth, fanout, random.Random(seed))
    py_root = make_py_tree(depth, fanout, random.Random(seed))

    rust_t = bench_rust(lib, rust_root, batch, iters)
    py_t = bench_py(py_root, batch, iters)

    print(f"select_batch: {iters} iters, batch={batch}, depth={depth} fanout={fanout}")
    print(f"  Python _collect_batch: {py_t*1000:9.2f} ms")
    print(f"  Rust   mcts_rs:        {rust_t*1000:9.2f} ms")
    if py_t > 0:
        print(f"  Speedup: {py_t/rust_t:.1f}x")


if __name__ == '__main__':
    main()