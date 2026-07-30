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
                 shared_buffers=None):
        self.config = config
        self.request_queue = request_queue
        self.response_queues = response_queues
        self.weight_queue = weight_queue
        self.ready_event = ready_event
        self.shutdown_event = shutdown_event
        self.shared_buffers = shared_buffers  # dict[worker_id → buffers]

        inf_cfg = getattr(config, 'inference', None)
        self.max_batch = getattr(inf_cfg, 'max_batch', 64) if inf_cfg else 64
        self.max_wait_ms = getattr(inf_cfg, 'max_wait_ms', 3.0) if inf_cfg else 3.0
        prewarm_sizes = getattr(inf_cfg, 'prewarm_batch_sizes', [1, 8, 16, 32, 64, 128]) if inf_cfg else [1, 8, 16, 32, 64, 128]
        self.prewarm_sizes = prewarm_sizes

    # ── Entry point (called in a subprocess) ────────────────────────────

    def run(self):
        """Main loop — blocks until shutdown."""
        import torch_directml

        device = torch_directml.device()
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

        elapsed = (time.perf_counter() - t0) * 1000
        print(f"[GPU-Server] Shader pre-warming done in {elapsed:.0f} ms")
        self.ready_event.set()

        # Load any weights already in the queue
        self._drain_weight_queue(nets, device)

        # ── Main inference loop ──
        while not self.shutdown_event.is_set():
            # Drain weight queue first (non-blocking)
            self._drain_weight_queue(nets, device)

            # Get one request (blocking, with timeout to allow weight checks)
            try:
                req = self.request_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if req is None:
                # Shutdown sentinel
                self._send_shutdown_to_workers()
                return

            self._handle_request(req, nets, device, net_a)

        print("[GPU-Server] Shutting down")

    def _handle_request(self, req, nets, device, net_a):
        """Dispatch a single request, handling both shared-memory and queue modes.

        In shared-memory mode the 4th element of *req* is an ``int``
        (batch_size); in queue mode it is an ``np.ndarray`` (state).
        """
        worker_id, request_id, network_id, fourth = req

        # ── Determine if this is a shared-memory request ──
        is_shm = isinstance(fourth, int)

        if is_shm:
            batch_size = fourth
            buf = self.shared_buffers.get(worker_id)
            if buf is None:
                # No shared buffers for this worker — can't proceed
                return
            states = SharedMemoryTransport.read_states(buf, batch_size)
        else:
            batch_size = None
            states = fourth  # ndarray

        net = nets.get(network_id, net_a)

        # ── Batch request (ndim == 4 or batch_size > 1): process immediately ──
        is_batch = (batch_size is not None and batch_size > 1) or \
                   (batch_size is None and states.ndim == 4)

        if is_batch:
            self._process_single_batch(
                net, device, worker_id, request_id, states
            )
            return

        # ── Single request: extract state and collect more for timer batching ──
        state = states[0] if is_shm else states  # (NUM_PLANES, 8, 8)
        batch = [(worker_id, request_id, network_id, state)]
        deadline = time.monotonic() + self.max_wait_ms / 1000.0

        while len(batch) < self.max_batch:
            remaining = max(0.0, deadline - time.monotonic())
            if remaining <= 0:
                break  # timer expired
            try:
                req2 = self.request_queue.get(timeout=min(remaining, 0.001))
                if req2 is None:
                    # Shutdown — fire what we have first
                    break

                w2, rid2, nid2, f2 = req2
                is_shm2 = isinstance(f2, int)

                if is_shm2:
                    buf2 = self.shared_buffers.get(w2)
                    if buf2 is None:
                        continue
                    states2 = SharedMemoryTransport.read_states(buf2, f2)

                    if f2 > 1:
                        # Batch request arrived during our window
                        if batch:
                            self._process_batch(nets, device, batch)
                        self._process_single_batch(
                            nets.get(nid2, net_a), device, w2, rid2, states2
                        )
                        batch = []
                        break
                    # Single request — add to accumulation
                    batch.append((w2, rid2, nid2, states2[0]))
                else:
                    # Queue-mode request
                    if f2.ndim == 4:
                        # Batch request arrived during our window
                        if batch:
                            self._process_batch(nets, device, batch)
                        self._process_single_batch(
                            nets.get(nid2, net_a), device, w2, rid2, f2
                        )
                        batch = []
                        break
                    batch.append((w2, rid2, nid2, f2))

            except queue.Empty:
                break

        if batch:
            self._process_batch(nets, device, batch)

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
                buf = io.BytesIO(latest[nid])
                state_dict = torch.load(buf, map_location='cpu', weights_only=True)
                nets[nid].load_state_dict(state_dict)
                nets[nid].eval()

    def _process_single_batch(self, net, device, worker_id, request_id, states):
        """Process a pre-stacked batch request ``(N, {NUM_PLANES}, 8, 8)`` immediately.

        In shared-memory mode, writes results to the worker's shared response
        buffers and sends only ``(request_id, batch_size)`` through the queue.
        In queue mode, sends ``(request_id, policies, values)`` through the queue.
        """
        states_t = torch.from_numpy(states).float().to(device)

        with torch.no_grad():
            policy_logits, values = net(states_t)
            policies = F.softmax(policy_logits, dim=1).cpu().numpy()
            values = values.squeeze(-1).cpu().numpy()

        n = len(policies)

        # ── Shared-memory response path ──
        if self.shared_buffers is not None and worker_id in self.shared_buffers:
            buf = self.shared_buffers[worker_id]
            SharedMemoryTransport.write_policies_values(buf, policies, values)
            try:
                self.response_queues[worker_id].put_nowait(
                    (request_id, n)
                )
            except Exception:
                pass  # queue full or closed — worker likely dead
            return

        # ── Queue response path ──
        try:
            self.response_queues[worker_id].put_nowait(
                (request_id, policies, values)
            )
        except Exception:
            pass  # queue full or closed — worker likely dead

    def _process_batch(self, nets, device, batch):
        """Run GPU forward passes and distribute results.

        ``batch`` is a list of ``(worker_id, request_id, network_id, state)``
        tuples where each ``state.shape == ({NUM_PLANES}, 8, 8)`` (individual
        requests aggregated by the timer).

        Requests are **grouped by ``network_id``** so each group is processed
        with the correct network.  This allows timer-batched single requests
        from different networks (e.g. gating eval) to be handled correctly.

        In shared-memory mode, each worker's results are written to their
        shared response buffer and only ``(request_id, 1)`` is sent through
        the queue.  In queue mode, ``(request_id, policy, value)`` is sent.
        """
        # Group by network_id, preserving insertion order
        groups = {}
        order = []
        for item in batch:
            w, rid, nid, st = item
            if nid not in groups:
                groups[nid] = []
                order.append(nid)
            groups[nid].append(item)

        # Process each group with the appropriate network
        for nid in order:
            group = groups[nid]
            net = nets.get(nid, list(nets.values())[0])
            states = np.stack([r[3] for r in group], axis=0)  # (N, {NUM_PLANES}, 8, 8)
            states_t = torch.from_numpy(states).float().to(device)

            with torch.no_grad():
                policy_logits, values = net(states_t)
                policies = F.softmax(policy_logits, dim=1).cpu().numpy()
                values = values.squeeze(-1).cpu().numpy()

            # Distribute results to per-worker response queues
            for i, (worker_id, request_id, _, _) in enumerate(group):
                # ── Shared-memory response path ──
                if (self.shared_buffers is not None
                        and worker_id in self.shared_buffers):
                    buf = self.shared_buffers[worker_id]
                    # Write single result (batch_size=1) to shared memory
                    SharedMemoryTransport.write_policies_values(
                        buf,
                        policies[i:i+1],  # (1, 4672)
                        values[i:i+1],    # (1,)
                    )
                    try:
                        self.response_queues[worker_id].put_nowait(
                            (request_id, 1)
                        )
                    except Exception:
                        pass
                    continue

                # ── Queue response path ──
                try:
                    self.response_queues[worker_id].put_nowait(
                        (request_id, policies[i], float(values[i]))
                    )
                except Exception:
                    pass  # queue full or closed — worker likely dead

    def _send_shutdown_to_workers(self):
        """Send a sentinel to each worker's response queue so they unblock."""
        for wq in self.response_queues.values():
            try:
                wq.put_nowait(None)
            except Exception:
                pass