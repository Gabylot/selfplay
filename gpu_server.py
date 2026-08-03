"""Centralized GPU inference server for AlphaZero chess.

Runs in a dedicated process, owns the GPU (torch_directml), and serves
inference requests from all worker processes.  Includes shader pre-warming
to eliminate DirectML's lazy-compilation latency spikes.

Supports **two** networks simultaneously (network_id 0 and 1) so that
gating evaluation (network A vs network B) can run entirely on the GPU
server without round-tripping weights.

Protocol
--------
Weight  : (network_id, raw_bytes)           — network_id ∈ {0, 1}

Request (queue mode):
    (worker_id, request_id, network_id, state)
      - state ndim == 3 (20,8,8):   single request (timer-aggregated)
      - state ndim == 4 (N,20,8,8):  batch request (processed immediately)

Request (shared-memory mode):
    (worker_id, request_id, network_id, batch_size)
      - batch_size == 1:  single request (timer-aggregated)
      - batch_size > 1:   batch request (processed immediately)
    The actual state data lives in the worker's shared request buffer.

Response (queue mode):
    (request_id, policy, value)
      - single response:  policy (4672,) ndarray, value float
      - batch response:   policy (N,4672) ndarray, values (N,) ndarray

Response (shared-memory mode):
    (request_id, batch_size)
    The actual policy/value data lives in the worker's shared response buffer.

Shutdown : None sentinel in request_queue
"""

import io
import time
import queue
import numpy as np
import torch
import torch.nn.functional as F

from network import AlphaZeroNet
from encoding import NUM_PLANES
from shared_memory_transport import SharedMemoryTransport, WorkerSharedBuffers


# ─────────────────────────────────────────────────────────────────────────────
# Shader pre-warming
# ─────────────────────────────────────────────────────────────────────────────

def warmup_shaders(network, device, batch_sizes=(1, 8, 16, 32, 64, 128)):
    """Run dummy forward passes to force DirectML to compile & cache shaders.

    After this function returns, inference at the given batch sizes should
    consistently hit the cached fast path (~1-2 ms) instead of randomly
    recompiling (which can take hundreds of ms).
    """
    network.eval()
    with torch.no_grad():
        for bs in batch_sizes:
            dummy = torch.randn(bs, NUM_PLANES, 8, 8, device=device)
            for _ in range(5):  # enough to stabilize the shader cache
                _ = network(dummy)


# ─────────────────────────────────────────────────────────────────────────────
# Server
# ─────────────────────────────────────────────────────────────────────────────

class GPUInferenceServer:
    """Centralized GPU inference server with dual-network support.

    Parameters
    ----------
    config : Config
        Must contain ``config.inference.max_batch`` and
        ``config.inference.max_wait_ms``.
    request_queue : mp.Queue
        Workers put ``(worker_id, request_id, network_id, state)`` tuples here.
    response_queues : dict[int, mp.Queue]
        Per-worker queues.  Server puts ``(request_id, policy, value)``
        tuples here so each worker only receives its own results.
    weight_queue : mp.Queue
        Main process puts ``(network_id, raw_bytes)`` tuples here.  The server
        drains this queue every iteration to always use the latest weights
        for each network.
    ready_event : mp.Event
        Set after shader pre-warming is complete.
    shutdown_event : mp.Event
        Checked each iteration; when set the server exits its loop.
    shared_buffers : dict[int, WorkerSharedBuffers], optional
        Per-worker shared-memory buffers.  When provided, state data is
        read from / results written to shared memory instead of being
        pickled through the queues.  When ``None`` (default), the original
        queue-based transport is used.
    """

    def __init__(self, config, request_queue, response_queues,
                 weight_queue, ready_event, shutdown_event,
                 shared_buffers=None, stats_queue=None):
        self.config = config
        self.request_queue = request_queue
        self.response_queues = response_queues
        self.weight_queue = weight_queue
        self.ready_event = ready_event
        self.shutdown_event = shutdown_event
        self.shared_buffers = shared_buffers  # dict[worker_id → buffers]
        self.stats_queue = stats_queue        # optional mp.Queue for benchmark timing

        inf_cfg = getattr(config, 'inference', None)
        self.max_batch = getattr(inf_cfg, 'max_batch', 64) if inf_cfg else 64
        self.max_wait_ms = getattr(inf_cfg, 'max_wait_ms', 3.0) if inf_cfg else 3.0
        prewarm_sizes = getattr(inf_cfg, 'prewarm_batch_sizes', [1, 8, 16, 32, 64, 128]) if inf_cfg else [1, 8, 16, 32, 64, 128]
        self.prewarm_sizes = prewarm_sizes

        # ── Timing instrumentation (benchmark mode only) ──
        self.stats = {
            'prewarm_time': 0.0,
            'wait_time': 0.0,
            'gpu_forward_time': 0.0,
            'shm_read_time': 0.0,
            'shm_write_time': 0.0,
            'weight_drain_time': 0.0,
            'weight_load_time': 0.0,
            'total_requests': 0,
            'aggregated_batches': 0,  # = number of forward passes
            'samples_processed': 0,
            'aggregation_wait_time': 0.0,  # time spent collecting requests
            'aggregation_cycles': 0,       # number of _handle_request calls
            'server_total': 0.0,
        }

    # ── Entry point (called in a subprocess) ────────────────────────────

    def run(self):
        """Main loop — blocks until shutdown."""
        import torch_directml

        device = torch_directml.device()
        t_start = time.perf_counter()
        print(f"[GPU-Server] DirectML device: {torch_directml.device_name(0)}")

        # Build two networks (network_id 0 = primary/latest,
        # network_id 1 = best/secondary for gating evaluation)
        def _build_net():
            n = AlphaZeroNet(
                num_residual_blocks=self.config.network.num_residual_blocks,
                num_filters=self.config.network.num_filters,
                num_policy_channels=self.config.network.num_policy_channels,
                num_value_channels=self.config.network.num_value_channels,
                value_fc_size=self.config.network.value_fc_size,
            ).to(device)
            n.eval()
            return n

        net_a = _build_net()
        net_b = _build_net()
        nets = {0: net_a, 1: net_b}

        # Pre-warm shaders for both networks
        print(f"[GPU-Server] Pre-warming shaders for batch sizes: {self.prewarm_sizes}")
        t0 = time.perf_counter()
        for label, net in [("net_a", net_a), ("net_b", net_b)]:
            warmup_shaders(net, device, self.prewarm_sizes)

            # Verify the fast path is stable after warming
            with torch.no_grad():
                for bs in self.prewarm_sizes:
                    dummy = torch.randn(bs, NUM_PLANES, 8, 8, device=device)
                    times = []
                    for _ in range(5):
                        t = time.perf_counter()
                        net(dummy)
                        times.append((time.perf_counter() - t) * 1000)
                    print(f"[GPU-Server] {label} bs={bs}: min={min(times):.1f}ms max={max(times):.1f}ms")

        t_prewarm = time.perf_counter() - t0
        self.stats['prewarm_time'] = t_prewarm
        elapsed = t_prewarm * 1000
        print(f"[GPU-Server] Shader pre-warming done in {elapsed:.0f} ms")
        self.ready_event.set()

        # Load any weights already in the queue
        self._drain_weight_queue(nets, device)

        # ── Main inference loop ──
        while not self.shutdown_event.is_set():
            # Drain weight queue first (non-blocking)
            t0 = time.perf_counter()
            self._drain_weight_queue(nets, device)
            self.stats['weight_drain_time'] += time.perf_counter() - t0

            # Get one request (blocking, with timeout to allow weight checks)
            try:
                t0 = time.perf_counter()
                req = self.request_queue.get(timeout=0.5)
                self.stats['wait_time'] += time.perf_counter() - t0
            except queue.Empty:
                continue

            if req is None:
                # Shutdown sentinel
                self._send_shutdown_to_workers()
                self._report_stats(t_start)
                return

            self._handle_request(req, nets, device, net_a)

        self._report_stats(t_start)
        print("[GPU-Server] Shutting down")

    def _handle_request(self, req, nets, device, net_a):
        """Handle a request by aggregating with subsequent requests.

        ALL requests (single and batch) participate in aggregation.
        The server collects requests until ``total_samples >= max_batch``
        or ``max_wait_ms`` expires, then processes them in one forward
        pass per network_id.

        This replaces the old two-path design where batch requests were
        processed immediately (bypassing aggregation).  Now 8 workers
        each sending 32 samples produce 1-2 forward passes of 128-256
        instead of 8 separate passes of 32.

        Non-blocking short-poll: uses ``get_nowait()`` and breaks
        immediately on ``queue.Empty``, so a single worker with an empty
        queue pays zero wait — not the full ``max_wait_ms``.
        This avoids the Windows timer resolution issue where
        ``get(timeout=0.001)`` takes ~15ms.
        """
        self.stats['aggregation_cycles'] += 1

        # Parse the first request
        first = self._parse_request(req)
        if first is None:
            return

        self.stats['total_requests'] += 1
        pending = [first]
        total_samples = first['batch_size']

        # ── Collect more requests (non-blocking short-poll) ──
        # Use get_nowait() instead of get(timeout=0.001) to avoid Windows
        # timer resolution issue where get(timeout=0.001) takes ~15ms.
        # get_nowait() returns immediately, so the short-poll pattern
        # (break on empty queue) adds zero latency.
        deadline = time.monotonic() + self.max_wait_ms / 1000.0
        t_agg_start = time.perf_counter()

        while total_samples < self.max_batch:
            if time.monotonic() >= deadline:
                break  # timer expired

            try:
                req2 = self.request_queue.get_nowait()

                if req2 is None:
                    # Shutdown sentinel — fire what we have first
                    break

                item2 = self._parse_request(req2)
                if item2 is None:
                    continue

                self.stats['total_requests'] += 1

                # If adding this would exceed max_batch, fire current batch
                # first, then start a new batch with this item.
                if total_samples + item2['batch_size'] > self.max_batch:
                    self._process_aggregated(nets, device, pending, net_a)
                    pending = [item2]
                    total_samples = item2['batch_size']
                    # Reset deadline for the new batch
                    deadline = time.monotonic() + self.max_wait_ms / 1000.0
                else:
                    pending.append(item2)
                    total_samples += item2['batch_size']

            except queue.Empty:
                break  # ← KEY: break immediately on empty queue (zero latency)

        self.stats['aggregation_wait_time'] += time.perf_counter() - t_agg_start

        if pending:
            self._process_aggregated(nets, device, pending, net_a)

    def _parse_request(self, req):
        """Parse a raw request tuple into a structured dict.

        Returns a dict with keys:
            worker_id, request_id, network_id, is_shm, batch_size,
            states (ndarray or None), buf (WorkerSharedBuffers or None)

        For shared-memory requests, states are NOT read here — they are
        read later in ``_process_aggregated`` when ready to process.
        This is safe because workers block synchronously on
        ``predict``/``predict_batch`` and won't overwrite their buffer
        until the server responds.
        """
        worker_id, request_id, network_id, fourth = req
        is_shm = isinstance(fourth, int)

        if is_shm:
            batch_size = fourth
            buf = self.shared_buffers.get(worker_id) if self.shared_buffers else None
            if buf is None:
                return None
            return {
                'worker_id': worker_id,
                'request_id': request_id,
                'network_id': network_id,
                'is_shm': True,
                'batch_size': batch_size,
                'states': None,
                'buf': buf,
            }
        else:
            states = fourth  # ndarray
            if states.ndim == 3:
                batch_size = 1
            else:
                batch_size = len(states)
            return {
                'worker_id': worker_id,
                'request_id': request_id,
                'network_id': network_id,
                'is_shm': False,
                'batch_size': batch_size,
                'states': states,
                'buf': None,
            }

    # ── Internal helpers ────────────────────────────────────────────────

    def _drain_weight_queue(self, nets, device):
        """Load the most recent weights for each network from the weight queue.

        Accepts both the new ``(network_id, raw_bytes)`` format and the
        legacy raw-bytes format (treated as network_id 0 for backward
        compatibility).
        """
        latest = {0: None, 1: None}
        while True:
            try:
                item = self.weight_queue.get_nowait()
                if item is None:
                    continue
                # Backward compat: raw bytes (old format) → network_id 0
                if isinstance(item, tuple) and len(item) == 2:
                    network_id, wb = item
                else:
                    network_id, wb = 0, item
                latest[network_id] = wb
            except queue.Empty:
                break

        for nid in (0, 1):
            if latest[nid] is not None:
                t0 = time.perf_counter()
                buf = io.BytesIO(latest[nid])
                state_dict = torch.load(buf, map_location='cpu', weights_only=True)
                nets[nid].load_state_dict(state_dict)
                nets[nid].eval()
                self.stats['weight_load_time'] += time.perf_counter() - t0

    def _process_aggregated(self, nets, device, pending, net_a):
        """Process aggregated requests in one forward pass per network_id.

        ``pending`` is a list of dicts returned by ``_parse_request``.

        Steps:
        1. Group by ``network_id`` so each group uses the correct network.
        2. For each group, read states from shared memory (shm) or use the
           ndarray (queue mode), then concatenate into one mega-batch.
        3. Run one ``net(mega_batch)`` forward pass.
        4. Split results by ``offset`` and write/send to each worker.

        The offset tracking is the highest-risk part — an off-by-one would
        silently hand worker A the policy/value meant for worker B.  The
        correctness test in ``tests/test_gpu_aggregation.py`` validates
        this with distinguishable synthetic states.
        """
        # Group by network_id, preserving insertion order
        groups = {}
        order = []
        for item in pending:
            nid = item['network_id']
            if nid not in groups:
                groups[nid] = []
                order.append(nid)
            groups[nid].append(item)

        # Process each group with the appropriate network
        for nid in order:
            group = groups[nid]
            net = nets.get(nid, net_a)

            # ── Read/collect states from all requests in this group ──
            states_list = []
            for item in group:
                if item['is_shm']:
                    t0 = time.perf_counter()
                    states = SharedMemoryTransport.read_states(
                        item['buf'], item['batch_size']
                    )
                    self.stats['shm_read_time'] += time.perf_counter() - t0
                else:
                    states = item['states']
                    # Ensure 4D (single requests come as 3D)
                    if states.ndim == 3:
                        states = states[np.newaxis, ...]
                states_list.append(states)

            # ── Concatenate into mega-batch ──
            if len(states_list) == 1:
                mega_states = states_list[0]
            else:
                mega_states = np.concatenate(states_list, axis=0)

            # ── Forward pass ──
            t0 = time.perf_counter()
            states_t = torch.from_numpy(mega_states).float().to(device)

            with torch.no_grad():
                policy_logits, values = net(states_t)
                policies = F.softmax(policy_logits, dim=1).cpu().numpy()
                values = values.squeeze(-1).cpu().numpy()
            self.stats['gpu_forward_time'] += time.perf_counter() - t0
            self.stats['samples_processed'] += len(mega_states)
            self.stats['aggregated_batches'] += 1

            # ── Distribute results to per-worker response queues ──
            offset = 0
            for item in group:
                bs = item['batch_size']
                item_policies = policies[offset:offset + bs]
                item_values = values[offset:offset + bs]
                offset += bs

                worker_id = item['worker_id']
                request_id = item['request_id']

                # ── Shared-memory response path ──
                if (item['is_shm'] and self.shared_buffers is not None
                        and worker_id in self.shared_buffers):
                    buf = self.shared_buffers[worker_id]
                    t0 = time.perf_counter()
                    SharedMemoryTransport.write_policies_values(
                        buf, item_policies, item_values
                    )
                    self.stats['shm_write_time'] += time.perf_counter() - t0
                    try:
                        self.response_queues[worker_id].put_nowait(
                            (request_id, bs)
                        )
                    except Exception:
                        pass
                    continue

                # ── Queue response path ──
                try:
                    if bs == 1:
                        self.response_queues[worker_id].put_nowait(
                            (request_id, item_policies[0], float(item_values[0]))
                        )
                    else:
                        self.response_queues[worker_id].put_nowait(
                            (request_id, item_policies, item_values)
                        )
                except Exception:
                    pass  # queue full or closed — worker likely dead

            # Safety assertion: offset must equal total samples in this group
            assert offset == sum(it['batch_size'] for it in group), \
                f"Offset tracking error: offset={offset} != total={sum(it['batch_size'] for it in group)}"

    def _send_shutdown_to_workers(self):
        """Send a sentinel to each worker's response queue so they unblock."""
        for wq in self.response_queues.values():
            try:
                wq.put_nowait(None)
            except Exception:
                pass

    def _report_stats(self, t_start):
        """Send accumulated timing stats to the main process (benchmark mode).

        No-op when ``stats_queue`` is None (normal training/eval runs).
        """
        self.stats['server_total'] = time.perf_counter() - t_start
        if self.stats_queue is not None:
            try:
                self.stats_queue.put(self.stats)
            except Exception:
                pass
