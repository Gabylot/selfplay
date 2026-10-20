"""Opening-diversity probe via the GPU inference server (parallel workers).

Starts the real GPUInferenceServer, pushes the checkpoint weights, then runs
NUM_WORKERS processes that each pull (game_id, fen, seed) tasks and play full
self-play games (net vs itself) through an InferenceClient, batched on the GPU.

Buckets:
  baseline_standard : fen=None (the standard start the run always uses)
  openings_asis     : a set of balanced opening FENs
  openings_mirror   : each opening's colour-mirror (White still to move)

Reports White's decisive share per bucket.

Usage: python probe_opening_diversity_gpu.py --baseline 24 --games 3 --workers 8
"""
import argparse
import multiprocessing as mp
import queue
import sys
import time
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, r"f:\python\selfplay")

OPENINGS = [
    "r1bqkbnr/1ppp1ppp/p1n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4",
    "rnbqkb1r/ppp2ppp/4pn2/3p4/2PP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 0 4",
    "rnbqkb1r/pp2pppp/3p1n2/8/3NP3/8/PPP2PPP/RNBQKB1R w KQkq - 0 5",
    "rnbqkb1r/ppp1pppp/5n2/3p4/8/5NP1/PPPPPPBP/RNBQK2R w KQkq - 0 4",
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    "rnbqkb1r/ppp2ppp/4pn2/3p4/3PP3/2N5/PPP2PPP/R1BQKBNR w KQkq - 0 4",
    "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 6 5",
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 3",
    "rnbqkb1r/1p2pppp/p2p1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 6",
    "rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPPKPPPP/RNBQ1BNR w kq - 2 3",
    "r1bqkbnr/pp1n1ppp/2p1p3/3pP3/3P4/8/PPPN1PPP/R1BQKBNR w KQkq - 1 5",
    "rnbqkbnr/pp2pppp/2p5/3p4/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 3",
]


def reflect(fen):
    """True chess colour-mirror: flip ranks (a1<->a8), swap colours of pieces,
    swap side to move, swap castling rights, mirror the en-passant rank.

    NOTE: do NOT also reverse files -- reversing both ranks and files is a
    180-degree rotation (a diagonal reflection), which is NOT a chess colour
    symmetry and produces king/queen-swapped positions.
    """
    parts = fen.split()
    rows = parts[0].split("/")
    out = ["".join(c.lower() if c.isupper() else c.upper() for c in r)
           for r in reversed(rows)]
    parts[0] = "/".join(out)
    parts[1] = "b" if parts[1] == "w" else "w"
    if parts[2] != "-":
        t = parts[2].translate(str.maketrans("KQkq", "kqKQ"))
        parts[2] = "".join(ch for ch in "KQkq" if ch in t)
    if parts[3] != "-":
        parts[3] = parts[3][0] + str(9 - int(parts[3][1]))
    return " ".join(parts)


# NOTE: the old SymClient / hflip+rankflip policy permutations were removed
# with the frame-version-2 change.  The network is now colour-neutral by
# construction (rank-flip encoder + mover-relative policy), so no inference-
# time symmetry averaging is needed.  ``reflect`` above is the true chess
# colour mirror used to build the mirrored-opening bucket.
def _worker(worker_id, request_q, response_q, shared_buffers, config_dict, task_q, result_q, sym=False):
    from selfplay import _set_game_from_config_dict
    from config import Config
    from mcts import MCTS
    from inference_client import InferenceClient
    from selfplay import play_one_game
    import numpy as np

    _set_game_from_config_dict(config_dict)
    config = Config(config_dict)
    pv = config.selfplay.piece_values
    piece_values = dict(pv) if hasattr(pv, "items") else pv
    concurrency = getattr(config.inference, "client_concurrency", 1)
    client = InferenceClient(worker_id, request_q, response_q, network_id=0,
                             shared_buffers=shared_buffers, concurrency=concurrency)

    def make_mcts():
        return MCTS(
            network=client,
            num_simulations=config.mcts.num_simulations,
            c_puct=config.mcts.c_puct,
            dirichlet_alpha=config.mcts.dirichlet_alpha,
            dirichlet_epsilon=config.mcts.dirichlet_epsilon,
            batch_size=config.mcts.batch_size,
            c_virtual_loss=config.mcts.c_virtual_loss,
            max_game_length=config.selfplay.max_game_length,
            adjudicate_material=config.selfplay.adjudicate_material,
            piece_values=piece_values,
            adjudicate_graded=config.selfplay.adjudicate_graded,
            adjudicate_scaling=config.selfplay.adjudicate_scaling,
            inference_pipeline=getattr(config.inference, "client_concurrency", 1),
        )

    while True:
        try:
            task = task_q.get(timeout=2.0)
        except queue.Empty:
            continue
        if task is None:
            break
        gid, fen, seed = task
        try:
            np.random.seed(seed)
            _, info = play_one_game(
                make_mcts(),
                max_game_length=config.selfplay.max_game_length,
                adjudicate_material=config.selfplay.adjudicate_material,
                piece_values=piece_values,
                temp_threshold=config.selfplay.temperature_threshold,
                temp_high=config.selfplay.temperature_high,
                temp_low=config.selfplay.temperature_low,
                adjudicate_graded=config.selfplay.adjudicate_graded,
                adjudicate_scaling=config.selfplay.adjudicate_scaling,
                fen=fen)
            result_q.put((gid, info["result"], info["termination"], info["length"]))
        except Exception as e:
            import traceback
            result_q.put((gid, None, "ERROR:" + traceback.format_exc(), 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="output/default/checkpoints/latest.pt")
    ap.add_argument("--baseline", type=int, default=24)
    ap.add_argument("--games", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--sims", type=int, default=None)
    ap.add_argument("--seed0", type=int, default=500000)
    ap.add_argument("--random", action="store_true",
                    help="skip loading the checkpoint (use random init weights)")
    ap.add_argument("--sym", action="store_true",
                    help="deprecated no-op (network is colour-neutral by "
                         "construction since frame version 2)")
    args = ap.parse_args()

    import torch
    import io
    import numpy as np
    from config import get_config
    from network import create_model_from_config, load_checkpoint
    from gpu_server import GPUInferenceServer
    from shared_memory_transport import SharedMemoryTransport

    config = get_config()
    if args.sims:
        config.mcts.num_simulations = args.sims
    config_dict = config.to_dict()

    net = create_model_from_config(config)
    if not args.random:
        load_checkpoint(args.ckpt, net)
    net.eval()
    print(f"checkpoint={'(random init)' if args.random else args.ckpt}  "
          f"sims={config.mcts.num_simulations}  "
          f"max_len={config.selfplay.max_game_length}  workers={args.workers}")

    request_q = mp.Queue()
    weight_q = mp.Queue()
    ready = mp.Event()
    shutdown = mp.Event()
    response_qs = {i: mp.Queue(maxsize=256) for i in range(args.workers)}

    concurrency = getattr(config.inference, "client_concurrency", 1)
    max_buf_batch = max(config.inference.max_batch, config.mcts.batch_size)
    transport = SharedMemoryTransport(
        num_workers=args.workers, max_batch=max_buf_batch,
        state_dtype=np.float32, num_slots=max(1, concurrency),
    )
    transport.create_buffers()
    shared_buffers_all = transport.get_all_worker_buffers()
    worker_shared_buffers = [transport.get_worker_buffers(i) for i in range(args.workers)]

    server = GPUInferenceServer(
        config=config, request_queue=request_q, response_queues=response_qs,
        weight_queue=weight_q, ready_event=ready, shutdown_event=shutdown,
        shared_buffers=shared_buffers_all)
    server_proc = mp.Process(target=server.run, daemon=True)
    server_proc.start()

    buf = io.BytesIO()
    torch.save(net.state_dict(), buf)
    weight_q.put((0, buf.getvalue()))
    print("Waiting for GPU server...", flush=True)
    ready.wait()
    print("GPU server ready.")

    # Build tasks
    tasks = []  # (gid, fen, seed)
    bucket_of = {}
    seed = args.seed0
    for _ in range(args.baseline):
        gid = len(tasks)
        tasks.append((gid, None, seed)); seed += 1
        bucket_of[gid] = "baseline_standard"
    for fen in OPENINGS:
        for tag, f in (("openings_asis", fen), ("openings_mirror", reflect(fen))):
            for _ in range(args.games):
                gid = len(tasks)
                tasks.append((gid, f, seed)); seed += 1
                bucket_of[gid] = tag
    print(f"total games: {len(tasks)}")

    task_q = mp.Queue()
    result_q = mp.Queue()
    workers = []
    for wid in range(args.workers):
        p = mp.Process(target=_worker,
                       args=(wid, request_q, response_qs[wid], worker_shared_buffers[wid],
                             config_dict, task_q, result_q, args.sym), daemon=True)
        p.start()
        workers.append(p)
    for t in tasks:
        task_q.put(t)
    for _ in workers:
        task_q.put(None)

    meta = {t[0]: (t[1], t[2]) for t in tasks}
    results = {}
    t0 = time.time()
    done = 0
    while done < len(tasks):
        try:
            gid, res, term, length = result_q.get(timeout=600.0)
        except queue.Empty:
            print("TIMEOUT waiting for results"); break
        results[gid] = (res, term, length)
        done += 1
        if done % 10 == 0 or done == len(tasks):
            print(f"  {done}/{len(tasks)} games  ({time.time()-t0:.0f}s)")

    shutdown.set()
    try: request_q.put_nowait(None)
    except Exception: pass
    server_proc.join(timeout=5)
    if server_proc.is_alive():
        server_proc.kill()
    for p in workers:
        p.join(timeout=3)
        if p.is_alive():
            p.kill()

    buckets = {}
    terms = {}
    for gid, (res, term, length) in results.items():
        b = bucket_of[gid]
        buckets.setdefault(b, []).append(res)
        terms[term] = terms.get(term, 0) + 1
    print("terminations:", terms)

    print("\n" + "=" * 70)
    print(f"total wall: {time.time()-t0:.0f}s  ({len(tasks)/(time.time()-t0):.2f} games/s)")
    print(f"{'bucket':<22}{'n':>5}{'W':>5}{'B':>5}{'D':>5}{'W/(W+B)':>10}{'z':>8}")
    for name in ("baseline_standard", "openings_asis", "openings_mirror"):
        res = buckets.get(name, [])
        w = sum(1 for r in res if r is not None and r > 0.001)
        b = sum(1 for r in res if r is not None and r < -0.001)
        d = len(res) - w - b
        n = w + b
        wd = 100 * w / n if n else float("nan")
        z = (w - b) / (n ** 0.5) if n else 0.0
        print(f"{name:<22}{len(res):>5}{w:>5}{b:>5}{d:>5}{wd:>9.1f}%{z:>8.2f}")


if __name__ == "__main__":
    main()
