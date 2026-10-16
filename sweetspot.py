"""GPU sweet‑spot profiler with worker scaling.
   Can load a trained checkpoint and diagnoses policy‑shape mismatch.

   NOW USES SHARED-MEMORY IPC (matches the real training pipeline's
   `inference.use_shared_memory: true` path) instead of pickling full
   state arrays through mp.Queue. This removes a confound from the
   earlier benchmark: without shared memory, a single mp.Queue's
   feeder-thread serialization of large batches (e.g. 128 x 137x8x8
   float32 ≈ 28MB) becomes a second bottleneck stacked on top of the
   single-threaded GPU server, making worker-scaling numbers look worse
   than they would with the actual transport used in training.
"""

import argparse, time, numpy as np, chess, multiprocessing as mp, io, torch
from mcts import MCTS
from network import create_model_from_config
from config import get_config
from inference_client import InferenceClient
from gpu_server import GPUInferenceServer
from shared_memory_transport import SharedMemoryTransport


# ── Worker function (now accepts shared_buffers) ───────────────────────
def _worker_search(worker_id, request_q, response_q, config_dict, board_fen,
                   batch_size, num_sims, result_queue, shared_buffers=None):
    import chess
    from config import Config
    from mcts import MCTS
    from inference_client import InferenceClient

    config = Config(config_dict)
    board = chess.Board(board_fen)
    client = InferenceClient(worker_id, request_q, response_q, network_id=0,
                             shared_buffers=shared_buffers)

    mcts = MCTS(
        network=client,
        num_simulations=num_sims,
        batch_size=batch_size,
        c_puct=config.mcts.c_puct,
        dirichlet_alpha=config.mcts.dirichlet_alpha,
        dirichlet_epsilon=config.mcts.dirichlet_epsilon,
        c_virtual_loss=config.mcts.c_virtual_loss,
    )

    root = mcts.get_root(board)
    t0 = time.perf_counter()
    visit_policy, best_move, stats = mcts.search(root)
    elapsed = time.perf_counter() - t0

    visits = sorted([c.N for c in root.children.values() if c.N > 0], reverse=True)
    total = sum(visits)
    top3 = 100 * sum(visits[:3]) / total if total > 0 else 0

    result_queue.put({
        'worker_id': worker_id,
        'elapsed': elapsed,
        'top3': top3,
    })


def main():
    parser = argparse.ArgumentParser(description="GPU sweet-spot profiler")
    parser.add_argument('--checkpoint', type=str, default=None,
                        help="Path to a trained checkpoint .pt file")
    parser.add_argument('--no-shared-memory', action='store_true',
                        help="Fall back to queue-based (pickle) transport for comparison")
    args = parser.parse_args()

    CONFIGS = [
        ("b32_s1200",   32, 1200),
        ("b64_s1200",    64, 1200),
        ("b128_s1200",  128, 1200),
    ]
    WORKER_COUNTS = [1, 4, 8, 12]
    NUM_REPEATS = 3

    config = get_config()
    config_dict = config.to_dict()
    board_fen = chess.Board().fen()

    max_workers = max(WORKER_COUNTS)
    # Buffers must be sized for the largest batch_size any config will use,
    # since MCTS's pre-stacked batch request (predict_batch) can exceed
    # inference.max_batch — same reasoning as ParallelSelfPlay in selfplay.py.
    max_batch_needed = max(bs for _, bs, _ in CONFIGS)

    use_shared_memory = not args.no_shared_memory
    shared_transport = None
    shared_bufs_all = None

    if use_shared_memory:
        inf_cfg = getattr(config, 'inference', None)
        state_dtype_str = getattr(inf_cfg, 'state_dtype', 'float16') if inf_cfg else 'float16'
        state_dtype = np.float16 if state_dtype_str == 'float16' else np.float32

        print(f"Allocating shared-memory buffers "
              f"(workers={max_workers}, max_batch={max_batch_needed}, dtype={state_dtype_str})...")
        shared_transport = SharedMemoryTransport(
            num_workers=max_workers,
            max_batch=max_batch_needed,
            state_dtype=state_dtype,
        )
        shared_transport.create_buffers()
        shared_bufs_all = shared_transport.get_all_worker_buffers()
        print("Shared-memory buffers ready.\n")
    else:
        print("Shared memory DISABLED (--no-shared-memory) — using queue/pickle transport.\n")

    # ── Start GPU server ──────────────────────────────────────────────
    print("Starting GPU server...")
    request_q = mp.Queue()
    weight_q = mp.Queue()
    ready = mp.Event()
    shutdown = mp.Event()

    response_qs = {i: mp.Queue(maxsize=256) for i in range(max_workers)}

    server = GPUInferenceServer(
        config=config,
        request_queue=request_q,
        response_queues=response_qs,
        weight_queue=weight_q,
        ready_event=ready,
        shutdown_event=shutdown,
        shared_buffers=shared_bufs_all,   # None if --no-shared-memory
    )
    server_proc = mp.Process(target=server.run, daemon=True)
    server_proc.start()

    # ── Prepare network weights ────────────────────────────────────────
    net = create_model_from_config(config)          # architecture from config

    if args.checkpoint:
        print(f"Loading checkpoint from: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # ── DIAGNOSTIC: check policy head dimensions ──────────────────
        print("\n--- Policy layer shapes from checkpoint ---")
        policy_shapes = []
        for k, v in state_dict.items():
            if 'policy' in k:
                print(f"  {k}: {v.shape}")
                policy_shapes.append(v.shape)
        # Look for the final linear layer that outputs logits (usually policy_fc.weight)
        final_policy_dim = None
        for k, v in state_dict.items():
            if 'policy_fc.weight' in k:
                final_policy_dim = v.shape[0]  # output features
                break
        if final_policy_dim is None:
            # Fallback: find any weight with a large first dimension near 4672
            for k, v in state_dict.items():
                if 'policy' in k and 'weight' in k and v.ndim == 2:
                    if v.shape[0] > 4000 or v.shape[0] == 4672:
                        final_policy_dim = v.shape[0]
                        break
        if final_policy_dim is not None:
            print(f"\nDetected policy output dimension: {final_policy_dim}")
            if final_policy_dim != 4672:
                print("!!! WARNING: This is NOT the standard 4672-action space.")
                print("!!! The checkpoint's policy head will not work with the current move_to_policy_index encoding.")
                print("!!! Results will be meaningless. Use a standard network or adapt the encoding.")
        else:
            print("Could not determine policy output dimension; check the printed shapes above.")
        print("---\n")

        net.load_state_dict(state_dict)
        print("Checkpoint loaded successfully.\n")
    else:
        print("Using random (untrained) network weights.\n")

    net.eval()

    buf = io.BytesIO()
    torch.save(net.state_dict(), buf)
    weight_q.put((0, buf.getvalue()))

    ready.wait()
    print("GPU server ready.\n")

    # ── Run experiments ────────────────────────────────────────────────
    results = []
    for label, bs, sims in CONFIGS:
        for n_workers in WORKER_COUNTS:
            times_all = []
            top3_all = []

            print(f"Testing {label}, workers={n_workers} ({NUM_REPEATS} batches)...",
                  end="", flush=True)

            for rep in range(NUM_REPEATS):
                result_queue = mp.Queue()
                processes = []

                for wid in range(n_workers):
                    worker_shared_buf = (shared_transport.get_worker_buffers(wid)
                                         if use_shared_memory else None)
                    p = mp.Process(
                        target=_worker_search,
                        args=(wid, request_q, response_qs[wid], config_dict,
                              board_fen, bs, sims, result_queue, worker_shared_buf),
                        daemon=True,
                    )
                    p.start()
                    processes.append(p)

                for p in processes:
                    p.join(timeout=120)
                    if p.is_alive():
                        p.kill()

                worker_elapsed = [0.0] * n_workers
                worker_top3 = [0.0] * n_workers
                while not result_queue.empty():
                    try:
                        res = result_queue.get_nowait()
                    except:
                        break
                    wid = res['worker_id']
                    worker_elapsed[wid] = res['elapsed']
                    worker_top3[wid] = res['top3']

                avg_worker_time = np.mean(worker_elapsed) if n_workers > 0 else 0
                avg_top3 = np.mean(worker_top3)

                times_all.append(avg_worker_time)
                top3_all.append(avg_top3)
                print(".", end="", flush=True)

            print()

            mean_time = np.mean(times_all)
            mean_top3 = np.mean(top3_all)
            std_top3 = np.std(top3_all)

            per_worker_sps = sims / mean_time if mean_time > 0 else 0
            aggregate_sps = n_workers * per_worker_sps

            results.append((label, bs, sims, n_workers, mean_time,
                            per_worker_sps, aggregate_sps, mean_top3, std_top3))

    # ── Cleanup ─────────────────────────────────────────────────────────
    shutdown.set()
    try:
        request_q.put_nowait(None)
    except:
        pass
    server_proc.join(timeout=5)
    if server_proc.is_alive():
        server_proc.kill()

    # ── Print table ─────────────────────────────────────────────────────
    print("\n" + "=" * 100)
    weight_type = "CHECKPOINT" if args.checkpoint else "RANDOM"
    transport_type = "SHARED-MEMORY" if use_shared_memory else "QUEUE/PICKLE"
    print(f"  GPU Sweet‑Spot with Worker Scaling ({weight_type} network, {transport_type} transport)")
    print("=" * 100)
    print(f"{'Config':<12} {'Workers':>7} {'Avg Time(s)':>11} {'Per-Worker sims/s':>17} {'Aggregate sims/s':>16} {'Top-3%':>8}  {'Top-3%±σ':>10}")
    print("-" * 100)
    for (label, bs, sims, nw, t, pwsps, asps, top3, std) in results:
        print(f"{label:<12} {nw:>7} {t:>11.3f} {pwsps:>17.0f} {asps:>16.0f} {top3:>7.1f}%  {top3:>5.1f}%±{std:.1f}%")
    print("-" * 100)
    print("Note: Aggregate sims/s = total simulations per second across all workers.")


if __name__ == "__main__":
    main()