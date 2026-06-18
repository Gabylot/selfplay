"""Centralized GPU inference server for AlphaZero chess.

Runs in a dedicated process, owns the GPU (torch_directml), and serves
inference requests from all worker processes.  Includes shader pre-warming
to eliminate DirectML's lazy-compilation latency spikes.

Protocol
--------
Request  : (worker_id, request_id, state)
           - state ndim == 3 (20,8,8):   single request (timer-aggregated)
           - state ndim == 4 (N,20,8,8):  batch request (processed immediately)
Response : (request_id, policy, value)
           - single response:  policy (4672,) ndarray, value float
           - batch response:   policy (N,4672) ndarray, values (N,) ndarray
Weight   : raw bytes (serialized state_dict via torch.save)
Shutdown : None sentinel in request_queue
"""

import io
import time
import queue
import numpy as np
import torch
import torch.nn.functional as F

from network import AlphaZeroNet


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
            dummy = torch.randn(bs, 20, 8, 8, device=device, dtype=torch.float16)
            for _ in range(5):  # enough to stabilize the shader cache
                _ = network(dummy)


# ─────────────────────────────────────────────────────────────────────────────
# Server
# ─────────────────────────────────────────────────────────────────────────────

class GPUInferenceServer:
    """Centralized GPU inference server.

    Parameters
    ----------
    config : Config
        Must contain ``config.inference.max_batch`` and
        ``config.inference.max_wait_ms``.
    request_queue : mp.Queue
        Workers put ``(worker_id, request_id, state)`` tuples here.
    response_queues : dict[int, mp.Queue]
        Per-worker queues.  Server puts ``(request_id, policy, value)``
        tuples here so each worker only receives its own results.
    weight_queue : mp.Queue
        Main process puts raw weight bytes here.  The server drains this
        queue every iteration to always use the latest weights.
    ready_event : mp.Event
        Set after shader pre-warming is complete.
    shutdown_event : mp.Event
        Checked each iteration; when set the server exits its loop.
    """

    def __init__(self, config, request_queue, response_queues,
                 weight_queue, ready_event, shutdown_event):
        self.config = config
        self.request_queue = request_queue
        self.response_queues = response_queues
        self.weight_queue = weight_queue
        self.ready_event = ready_event
        self.shutdown_event = shutdown_event

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

        # Build network
        net = AlphaZeroNet(
            num_residual_blocks=self.config.network.num_residual_blocks,
            num_filters=self.config.network.num_filters,
            num_policy_channels=self.config.network.num_policy_channels,
            num_value_channels=self.config.network.num_value_channels,
            value_fc_size=self.config.network.value_fc_size,
        ).to(device)
        net.eval()
        net = net.half()

        # Pre-warm shaders
        print(f"[GPU-Server] Pre-warming shaders for batch sizes: {self.prewarm_sizes}")
        t0 = time.perf_counter()
        warmup_shaders(net, device, self.prewarm_sizes)

        # Verify the fast path is stable after warming
        with torch.no_grad():
            for bs in self.prewarm_sizes:
                dummy = torch.randn(bs, 20, 8, 8, device=device, dtype=torch.float16)
                times = []
                for _ in range(5):
                    t = time.perf_counter()
                    net(dummy)
                    times.append((time.perf_counter() - t) * 1000)
                print(f"[GPU-Server] bs={bs}: min={min(times):.1f}ms max={max(times):.1f}ms")

        elapsed = (time.perf_counter() - t0) * 1000
        print(f"[GPU-Server] Shader pre-warming done in {elapsed:.0f} ms")
        self.ready_event.set()

        # Load any weights already in the queue
        self._drain_weight_queue(net, device)

        # ── Main inference loop ──
        while not self.shutdown_event.is_set():
            # Drain weight queue first (non-blocking)
            self._drain_weight_queue(net, device)

            # Get one request (blocking, with timeout to allow weight checks)
            try:
                req = self.request_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if req is None:
                # Shutdown sentinel
                self._send_shutdown_to_workers()
                return

            worker_id, request_id, state = req

            # Differentiate: batch request (ndim == 4) vs single (ndim == 3)
            if state.ndim == 4:
                # ── Batch request: process immediately, no timer ──
                self._process_single_batch(net, device, worker_id, request_id, state)
            else:
                # ── Single request: collect more for timer-based batching ──
                batch = [(worker_id, request_id, state)]
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
                        w2, rid2, st2 = req2
                        if st2.ndim == 4:
                            # Another batch request arrived during our window.
                            # Process the current accumulation first, then handle
                            # the batch request on the next loop iteration.
                            # Push it back by prepending (not possible reliably),
                            # so we just fall through and fire the timer batch,
                            # then put the batch request back by sending it again.
                            # Actually simpler: process the timer batch now,
                            # then handle the batch request immediately.
                            if batch:
                                self._process_batch(net, device, batch)
                            self._process_single_batch(net, device, w2, rid2, st2)
                            batch = []
                            break
                        batch.append((w2, rid2, st2))
                    except queue.Empty:
                        break

                if batch:
                    self._process_batch(net, device, batch)

        print("[GPU-Server] Shutting down")

    # ── Internal helpers ────────────────────────────────────────────────

    def _drain_weight_queue(self, net, device):
        """Load the most recent weights from the weight queue."""
        latest_bytes = None
        while True:
            try:
                wb = self.weight_queue.get_nowait()
                if wb is not None:
                    latest_bytes = wb
            except queue.Empty:
                break

        if latest_bytes is not None:
            buf = io.BytesIO(latest_bytes)
            state_dict = torch.load(buf, map_location='cpu', weights_only=True)
            net.load_state_dict(state_dict)
            net.eval()

    def _process_single_batch(self, net, device, worker_id, request_id, states):
        """Process a pre-stacked batch request ``(N, 20, 8, 8)`` immediately.

        Sends a single response ``(request_id, policies, values)`` back to
        the worker, where ``policies.shape == (N, 4672)``.
        """
        states_t = torch.from_numpy(states).float().to(device).half()

        with torch.no_grad():
            policy_logits, values = net(states_t)
            policies = F.softmax(policy_logits, dim=1).float().cpu().numpy()
            values = values.float().squeeze(-1).cpu().numpy()

        try:
            self.response_queues[worker_id].put_nowait(
                (request_id, policies, values)
            )
        except Exception:
            pass  # queue full or closed — worker likely dead

    def _process_batch(self, net, device, batch):
        """Run a single GPU forward pass and distribute results.

        ``batch`` is a list of ``(worker_id, request_id, state)`` tuples
        where each ``state.shape == (20, 8, 8)`` (individual requests
        aggregated by the timer).
        """
        states = np.stack([r[2] for r in batch], axis=0)  # (N, 20, 8, 8)
        states_t = torch.from_numpy(states).float().to(device).half()

        with torch.no_grad():
            policy_logits, values = net(states_t)
            policies = F.softmax(policy_logits, dim=1).float().cpu().numpy()
            values = values.float().squeeze(-1).cpu().numpy()

        # Distribute results to per-worker response queues
        for i, (worker_id, request_id, _) in enumerate(batch):
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