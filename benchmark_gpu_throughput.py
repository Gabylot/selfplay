"""Benchmark GPU server throughput with simulated workers.

Spawns N worker processes that each send batch_size inference requests
to the GPU server in a tight loop for a fixed duration.  Measures GPU
utilization, throughput, and aggregation stats.

This benchmark directly tests the cross-worker batch aggregation:
  - With aggregation: N workers x batch_size=32 -> 1 forward pass of Nx32
  - Without aggregation (old design): N workers x batch_size=32 -> N forward passes of 32

Usage:
    python benchmark_gpu_throughput.py [--workers N] [--duration S] [--batch-size B]

Example:
    python benchmark_gpu_throughput.py --workers 8 --duration 10 --batch-size 32
"""

import argparse
import io
import time
import queue
import multiprocessing as mp
import numpy as np
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import get_config
from network import AlphaZeroNet, create_model_from_config, load_checkpoint
from encoding import NUM_PLANES, NUM_ACTIONS
from gpu_server import GPUInferenceServer, warmup_shaders
from inference_client import InferenceClient
from shared_memory_transport import SharedMemoryTransport


# ─────────────────────────────────────────────────────────────────────────────
# Worker process
# ─────────────────────────────────────────────────────────────────────────────

def _benchmark_worker(worker_id, request_queue, response_queue, shared_buffers,
                      batch_size, duration, ready_event, shutdown_event,
                      result_queue, concurrency):
    """Send inference requests in a tight loop for `duration` seconds.

    Each worker creates an InferenceClient and calls predict_batch() with
    dummy states.  The GPU server aggregates requests from all workers
    into mega-batches.

    When ``concurrency > 1`` the worker issues ``concurrency`` in-flight
    requests (via ``predict_batch_async``) before collecting results,
    pipelining a small batch of requests per worker so the server always
    has work queued behind the ongoing GPU forward.
    """
    # Limit threads to avoid contention (same as production workers)
    torch.set_num_threads(1)

    # Wait for GPU server to be ready
    ready_event.wait()

    client = InferenceClient(
        worker_id, request_queue, response_queue, network_id=0,
        shared_buffers=shared_buffers, concurrency=concurrency,
    )

    # Create dummy states (random data -- we're measuring throughput, not accuracy)
    states = np.random.randn(batch_size, NUM_PLANES, 8, 8).astype(np.float32)

    # Send requests in a tight loop, keeping `concurrency` in flight
    requests_sent = 0
    t_start = time.perf_counter()
    deadline = t_start + duration

    inflight = []
    while time.perf_counter() < deadline and not shutdown_event.is_set():
        # Keep `concurrency` in flight: top up slots, then block on the
        # oldest result (frees a slot for the next send above).
        while len(inflight) < concurrency:
            inflight.append(client.predict_batch_async(states))
            requests_sent += 1
        req_id = inflight.pop(0)
        client.wait_result(req_id)

    # Drain any remaining in-flight requests
    for req_id in inflight:
        client.wait_result(req_id)

    elapsed = time.perf_counter() - t_start
    result_queue.put({
        'worker_id': worker_id,
        'requests_sent': requests_sent,
        'samples_sent': requests_sent * batch_size,
        'elapsed': elapsed,
        'throughput': requests_sent * batch_size / elapsed if elapsed > 0 else 0,
    })


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark runner
# ─────────────────────────────────────────────────────────────────────────────

def run_benchmark(num_workers=12, duration=10, batch_size=256,
                  use_shared_memory=True, concurrency=1):
    """Run the GPU throughput benchmark.

    Parameters
    ----------
    num_workers : int
        Number of simulated worker processes.
    duration : float
        Duration in seconds to run the benchmark.
    batch_size : int
        Batch size per inference request (matches MCTS batch_size).
    use_shared_memory : bool
        Whether to use shared-memory IPC (production setting).
    concurrency : int
        In-flight requests per worker.  >1 enables client pipelining: each
        worker keeps ``concurrency`` requests queued on the server so the
        GPU forward never starves behind request round-trips.
    """
    print("=" * 72)
    print("  GPU THROUGHPUT BENCHMARK")
    print("=" * 72)
    print(f"\n  Workers: {num_workers}")
    print(f"  Duration: {duration}s")
    print(f"  Batch size per request: {batch_size}")
    print(f"  Concurrency (in-flight/worker): {concurrency}")
    print(f"  Shared memory: {use_shared_memory}")
    print(f"  Max batch (aggregation): {get_config().inference.max_batch}")
    print(f"  Max wait ms: {get_config().inference.max_wait_ms}")

    # ── Setup ──
    config = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create network (for weight serialization)
    network = create_model_from_config(config)
    network.to(device)
    network.eval()

    # Load checkpoint if available
    from pathlib import Path
    output_dir = Path(config.main.output_dir) / config.main.run_name
    ckpt_path = output_dir / "checkpoints" / "latest.pt"
    if ckpt_path.exists():
        print(f"\n  Loading checkpoint: {ckpt_path}")
        load_checkpoint(str(ckpt_path), network)
    else:
        print(f"\n  No checkpoint found -- using random weights")

    # Serialize weights for GPU server
    buf = io.BytesIO()
    torch.save(network.state_dict(), buf)
    weights_bytes = buf.getvalue()

    # Create IPC infrastructure
    request_queue = mp.Queue()
    response_queues = {i: mp.Queue(maxsize=256) for i in range(num_workers)}
    weight_queue = mp.Queue()
    ready_event = mp.Event()
    shutdown_event = mp.Event()
    stats_queue = mp.Queue()

    # Shared memory transport
    shared_buffers_all = None
    worker_shared_buffers = [None] * num_workers
    if use_shared_memory:
        inf_max_batch = config.inference.max_batch
        max_buf_batch = max(inf_max_batch, batch_size)
        transport = SharedMemoryTransport(
            num_workers=num_workers,
            max_batch=max_buf_batch,
            state_dtype=np.float32,
            num_slots=concurrency,
        )
        transport.create_buffers()
        shared_buffers_all = transport.get_all_worker_buffers()
        for i in range(num_workers):
            worker_shared_buffers[i] = transport.get_worker_buffers(i)
        print(f"  Shared memory: enabled (dtype=float32, max_batch={max_buf_batch})")

    # Create GPU server
    server = GPUInferenceServer(
        config=config,
        request_queue=request_queue,
        response_queues=response_queues,
        weight_queue=weight_queue,
        ready_event=ready_event,
        shutdown_event=shutdown_event,
        shared_buffers=shared_buffers_all,
        stats_queue=stats_queue,
    )
    gpu_process = mp.Process(target=server.run, daemon=True)
    gpu_process.start()

    # Push weights to GPU server
    weight_queue.put((0, weights_bytes))

    # Wait for GPU server to be ready
    print("\n  Waiting for GPU server shader pre-warming...")
    ready_event.wait()
    print("  GPU server ready")

    # ── Spawn workers ──
    result_queue = mp.Queue()
    workers = []
    for i in range(num_workers):
        p = mp.Process(
            target=_benchmark_worker,
            args=(i, request_queue, response_queues[i], worker_shared_buffers[i],
                  batch_size, duration, ready_event, shutdown_event, result_queue,
                  concurrency),
            daemon=True,
        )
        p.start()
        workers.append(p)

    # ── Run benchmark ──
    print(f"\n  Running benchmark for {duration}s...")
    wall_start = time.perf_counter()

    # Wait for all workers to finish
    for p in workers:
        p.join(timeout=duration + 30)

    wall_total = time.perf_counter() - wall_start

    # ── Shutdown GPU server ──
    shutdown_event.set()
    try:
        request_queue.put_nowait(None)
    except Exception:
        pass
    gpu_process.join(timeout=10)
    if gpu_process.is_alive():
        gpu_process.kill()

    # ── Collect results ──
    worker_results = []
    while True:
        try:
            r = result_queue.get_nowait()
            worker_results.append(r)
        except queue.Empty:
            break

    gpu_stats = None
    try:
        gpu_stats = stats_queue.get(timeout=5)
    except queue.Empty:
        pass

    # ── Print report ──
    _print_report(worker_results, gpu_stats, wall_total, num_workers, batch_size, duration)

    # Cleanup
    for p in workers:
        if p.is_alive():
            p.kill()


def _print_report(worker_results, gpu_stats, wall_total, num_workers, batch_size, duration):
    """Print the benchmark report."""
    print("\n" + "=" * 72)
    print("  BENCHMARK RESULTS")
    print("=" * 72)

    # ── Worker stats ──
    total_requests = sum(r['requests_sent'] for r in worker_results)
    total_samples = sum(r['samples_sent'] for r in worker_results)
    total_throughput = total_samples / wall_total if wall_total > 0 else 0

    print(f"\n  Wall time: {wall_total:.2f}s")
    print(f"  Workers: {len(worker_results)}/{num_workers} reported")
    print(f"  Total requests: {total_requests}")
    print(f"  Total samples: {total_samples}")
    print(f"  Throughput: {total_throughput:.0f} samples/sec")
    print(f"  Throughput: {total_throughput / 1e6:.2f}M samples/sec")

    # Per-worker breakdown
    print(f"\n  {'Worker':<10} {'Requests':<12} {'Samples':<12} {'Throughput':<15} {'Latency':<10}")
    print("  " + "-" * 58)
    for r in sorted(worker_results, key=lambda x: x['worker_id']):
        throughput = r['throughput']
        latency_ms = (r['elapsed'] / r['requests_sent'] * 1000) if r['requests_sent'] > 0 else 0
        print(f"  W{r['worker_id']:<9} {r['requests_sent']:<12} {r['samples_sent']:<12} "
              f"{throughput:<15.0f} {latency_ms:<10.1f}ms")

    # ── GPU server stats ──
    if gpu_stats:
        print("\n" + "=" * 72)
        print("  GPU SERVER STATS")
        print("=" * 72)

        srv_total = gpu_stats.get('server_total', 0) or 0
        gpu_forward = gpu_stats.get('gpu_forward_time', 0)
        wait_time = gpu_stats.get('wait_time', 0)
        agg_wait = gpu_stats.get('aggregation_wait_time', 0)
        shm_read = gpu_stats.get('shm_read_time', 0)
        shm_write = gpu_stats.get('shm_write_time', 0)
        weight_drain = gpu_stats.get('weight_drain_time', 0)
        weight_load = gpu_stats.get('weight_load_time', 0)
        prewarm = gpu_stats.get('prewarm_time', 0)

        # GPU utilization = forward time / (total - prewarm)
        # Exclude prewarm since it's one-time startup
        active_time = srv_total - prewarm
        gpu_util = (gpu_forward / active_time * 100) if active_time > 0 else 0

        print(f"\n  Server total run time: {srv_total:.3f}s")
        print(f"  Shader pre-warm: {prewarm:.3f}s")
        print(f"  Active time (excl. prewarm): {active_time:.3f}s")
        print()
        print(f"  GPU forward time: {gpu_forward:.3f}s  ({gpu_forward/active_time*100:.1f}% of active)")
        print(f"  Wait time (main loop): {wait_time:.3f}s  ({wait_time/active_time*100:.1f}% of active)")
        print(f"  Aggregation wait time: {agg_wait:.3f}s  ({agg_wait/active_time*100:.1f}% of active)")
        print(f"  SHM read time: {shm_read:.3f}s  ({shm_read/active_time*100:.1f}% of active)")
        print(f"  SHM write time: {shm_write:.3f}s  ({shm_write/active_time*100:.1f}% of active)")
        print(f"  Weight drain: {weight_drain:.3f}s")
        print(f"  Weight load: {weight_load:.3f}s")

        print(f"\n  *** GPU UTILIZATION: {gpu_util:.1f}% ***")
        print(f"  (forward_time / active_time)")

        # Aggregation stats
        total_reqs = gpu_stats.get('total_requests', 0)
        fwd_passes = gpu_stats.get('aggregated_batches', 0)
        samples_processed = gpu_stats.get('samples_processed', 0)
        agg_cycles = gpu_stats.get('aggregation_cycles', 0)

        avg_batch = samples_processed / fwd_passes if fwd_passes > 0 else 0
        gpu_throughput = samples_processed / active_time if active_time > 0 else 0

        print(f"\n  Total requests received: {total_reqs}")
        print(f"  Forward passes: {fwd_passes}")
        print(f"  Aggregation cycles: {agg_cycles}")
        print(f"  Samples processed: {samples_processed}")
        print(f"  Avg batch size: {avg_batch:.1f}")
        print(f"  GPU throughput: {gpu_throughput:.0f} samples/sec")
        print(f"  GPU throughput: {gpu_throughput / 1e6:.2f}M samples/sec")

        # Efficiency analysis
        print(f"\n  {'-' * 50}")
        print(f"  ANALYSIS")
        print(f"  {'-' * 50}")
        if avg_batch > 0:
            old_passes = total_reqs  # Old design: 1 forward pass per request
            old_samples_per_sec = total_reqs * avg_batch / active_time if active_time > 0 else 0
            speedup = fwd_passes / old_passes if old_passes > 0 else 0
            print(f"  Without aggregation: {old_passes} forward passes of ~{batch_size}")
            print(f"  With aggregation:    {fwd_passes} forward passes of ~{avg_batch:.0f}")
            print(f"  Forward pass reduction: {old_passes}/{fwd_passes} = {1/speedup:.2f}x fewer passes" if speedup > 0 else "")
            print(f"  Expected speedup: {avg_batch/batch_size:.1f}x (avg_batch/input_batch)")

        if gpu_util < 50:
            print(f"\n  [!] GPU utilization is {gpu_util:.1f}% -- GPU is idle most of the time.")
            print(f"     Possible causes:")
            print(f"     - Workers are CPU-bound (MCTS selection is slow)")
            print(f"     - Too few workers for the GPU's throughput")
            print(f"     - IPC overhead (shared memory read/write)")
        elif gpu_util < 80:
            print(f"\n  [!] GPU utilization is {gpu_util:.1f}% -- decent but could be better.")
        else:
            print(f"\n  [OK] GPU utilization is {gpu_util:.1f}% -- excellent!")

    else:
        print("\n  [WARN] No GPU server stats available")

    print("\n" + "=" * 72)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GPU throughput benchmark")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of simulated worker processes (default: 8)")
    parser.add_argument("--duration", type=float, default=10.0,
                        help="Duration in seconds (default: 10)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size per inference request (default: 32)")
    parser.add_argument("--no-shm", action="store_true",
                        help="Disable shared memory (use queue/pickle transport)")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="In-flight requests per worker (client pipelining, default: 1)")
    args = parser.parse_args()

    run_benchmark(
        num_workers=args.workers,
        duration=args.duration,
        batch_size=args.batch_size,
        use_shared_memory=not args.no_shm,
        concurrency=args.concurrency,
    )


if __name__ == "__main__":
    # Ensure spawn start method (required for DirectML + shared memory)
    if mp.get_start_method(allow_none=True) is None:
        try:
            mp.set_start_method('spawn')
        except RuntimeError:
            pass

    main()