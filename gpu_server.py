"""Centralized GPU inference server for AlphaZero chess.

Runs in a dedicated process, owns the GPU (torch_directml), and serves
inference requests from all worker processes.  Includes shader pre-warming
to eliminate DirectML's lazy-compilation latency spikes.

Protocol
--------
Request  : (worker_id, request_id, state)   state is (20,8,8) float32 ndarray
Response : (request_id, policy, value)      policy (4672,) ndarray, value float
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
            dummy = torch.randn(bs, 20, 8, 8, device=device)
            # First pass triggers compilation
            _ = network(dummy)
            # Second pass confirms cached path
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

        # Pre-warm shaders
        print(f"[GPU-Server] Pre-warming shaders for batch sizes: {self.prewarm_sizes}")
        t0 = time.perf_counter()
        warmup_shaders(net, device, self.prewarm_sizes)
        elapsed = (time.perf_counter() - t0) * 1000
        print(f"[GPU-Server] Shader pre-warming done in {elapsed:.0f} ms")
        self.ready_event.set()

        # Load any weights already in the queue
        self._drain_weight_queue(net, device)

        # ── Main inference loop ──
        while not self.shutdown_event.is_set():
            # Drain weight queue first (non-blocking)
            self._drain_weight_queue(net, device)

            # Collect a batch of requests
            batch = []
            deadline = time.monotonic() + self.max_wait_ms / 1000.0

            while len(batch) < self.max_batch:
                remaining = max(0.0, deadline - time.monotonic())
                if remaining <= 0 and batch:
                    break  # timer expired, fire what we have
                try:
                    req = self.request_queue.get(timeout=min(remaining, 0.001))
                    if req is None:
                        # Shutdown sentinel
                        self._send_shutdown_to_workers()
                        return
                    batch.append(req)
                except queue.Empty:
                    if batch:
                        break  # timer expired
                    continue

            if not batch:
                continue

            # Run inference
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

    def _process_batch(self, net, device, batch):
        """Run a single GPU forward pass and distribute results."""
        # batch is list of (worker_id, request_id, state)
        states = np.stack([r[2] for r in batch], axis=0)  # (N, 20, 8, 8)
        states_t = torch.from_numpy(states).float().to(device)

        with torch.no_grad():
            policy_logits, values = net(states_t)
            policies = F.softmax(policy_logits, dim=1).cpu().numpy()
            values = values.squeeze(-1).cpu().numpy()

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