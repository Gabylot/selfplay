"""Centralized GPU inference server for AlphaZero chess.

Runs in a dedicated process, owns the GPU (torch_directml), and serves
inference requests from all worker processes.  Includes shader pre-warming
to eliminate DirectML's lazy-compilation latency spikes.

Supports dual networks: ``net_a`` (latest, receiving weights from ``weight_queue``)
and ``net_b`` (best, from ``weight_queue_b``).  Workers specify which network
to use via ``net_id`` (``'a'`` or ``'b'``) in their request tuples.

Protocol
--------
Request  : (worker_id, request_id, state)           - legacy (net_id='a')
           (worker_id, request_id, state, net_id)   - with network selection
            - state ndim == 3 (NUM_PLANES,8,8):   single request (timer-aggregated)
            - state ndim == 4 (N,NUM_PLANES,8,8):  batch request (processed immediately)
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
from encoding import NUM_PLANES



# ----------------------------------------------------------------------------
# Shader pre-warming
# ----------------------------------------------------------------------------

def warmup_shaders(network, device, batch_sizes=(1, 8, 16, 32, 64, 128)):
    """Run dummy forward passes to force DirectML to compile & cache shaders.

    After this function returns, inference at the given batch sizes should
    consistently hit the cached fast path (~1-2 ms) instead of randomly
    recompiling (which can take hundreds of ms).
    """
    network.eval()
    with torch.no_grad():
        for bs in batch_sizes:
            dummy = torch.randn(bs, NUM_PLANES, 8, 8, device=device, dtype=torch.float16)
            for _ in range(5):
                _ = network(dummy)


# ----------------------------------------------------------------------------
# Server
# ----------------------------------------------------------------------------

class GPUInferenceServer:
    """Centralized GPU inference server with dual-network support.

    Parameters
    ----------
    config : Config
        Must contain ``config.inference.max_batch`` and
        ``config.inference.max_wait_ms``.
    request_queue : mp.Queue
        Workers put ``(worker_id, request_id, state)`` or
        ``(worker_id, request_id, state, net_id)`` tuples here.
    response_queues : dict[int, mp.Queue]
        Per-worker queues.  Server puts ``(request_id, policy, value)``
        tuples here so each worker only receives its own results.
    weight_queue : mp.Queue
        Main process puts raw weight bytes for the *latest* network here.
        The server drains this queue every iteration.
    weight_queue_b : mp.Queue or None
        Main process puts raw weight bytes for the *best* network here.
        If None, only ``net_a`` is used.
    ready_event : mp.Event
        Set after shader pre-warming is complete.
    shutdown_event : mp.Event
        Checked each iteration; when set the server exits its loop.
    """

    def __init__(self, config, request_queue, response_queues,
                 weight_queue, weight_queue_b, ready_event, shutdown_event):
        self.config = config
        self.request_queue = request_queue
        self.response_queues = response_queues
        self.weight_queue = weight_queue
        self.weight_queue_b = weight_queue_b
        self.ready_event = ready_event
        self.shutdown_event = shutdown_event

        inf_cfg = getattr(config, 'inference', None)
        self.max_batch = getattr(inf_cfg, 'max_batch', 64) if inf_cfg else 64
        self.max_wait_ms = getattr(inf_cfg, 'max_wait_ms', 3.0) if inf_cfg else 3.0
        prewarm_sizes = getattr(inf_cfg, 'prewarm_batch_sizes', [1, 8, 16, 32, 64, 128]) if inf_cfg else [1, 8, 16, 32, 64, 128]
        self.prewarm_sizes = prewarm_sizes


    # -- Entry point (called in a subprocess) --------------------------------

    def run(self):
        """Main loop -- blocks until shutdown."""
        import torch_directml

        device = torch_directml.device()
        print(f"[GPU-Server] DirectML device: {torch_directml.device_name(0)}")

        # Build two networks
        net_a = AlphaZeroNet(
            num_residual_blocks=self.config.network.num_residual_blocks,
            num_filters=self.config.network.num_filters,
            num_policy_channels=self.config.network.num_policy_channels,
            num_value_channels=self.config.network.num_value_channels,
            value_fc_size=self.config.network.value_fc_size,
        ).to(device)
        net_a.eval()
        net_a = net_a.half()

        net_b = AlphaZeroNet(
            num_residual_blocks=self.config.network.num_residual_blocks,
            num_filters=self.config.network.num_filters,
            num_policy_channels=self.config.network.num_policy_channels,
            num_value_channels=self.config.network.num_value_channels,
            value_fc_size=self.config.network.value_fc_size,
        ).to(device)
        net_b.eval()
        net_b = net_b.half()

        print("[GPU-Server] Created dual networks (net_a=latest, net_b=best)")

        # Pre-warm shaders (just use net_a -- both are same architecture)
        print(f"[GPU-Server] Pre-warming shaders for batch sizes: {self.prewarm_sizes}")
        t0 = time.perf_counter()
        warmup_shaders(net_a, device, self.prewarm_sizes)

        # Verify the fast path is stable after warming
        with torch.no_grad():
            for bs in self.prewarm_sizes:
                dummy = torch.randn(bs, NUM_PLANES, 8, 8, device=device, dtype=torch.float16)
                times = []
                for _ in range(5):
                    t = time.perf_counter()
                    net_a(dummy)
                    times.append((time.perf_counter() - t) * 1000)
                print(f"[GPU-Server] bs={bs}: min={min(times):.1f}ms max={max(times):.1f}ms")

        elapsed = (time.perf_counter() - t0) * 1000
        print(f"[GPU-Server] Shader pre-warming done in {elapsed:.0f} ms")
        self.ready_event.set()

        # Load any weights already in the queues
        self._drain_weight_queues(net_a, net_b, device)

        # -- Main inference loop --
        while not self.shutdown_event.is_set():
            self._drain_weight_queues(net_a, net_b, device)

            try:
                req = self.request_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if req is None:
                self._send_shutdown_to_workers()
                return

            # Parse request -- support both 3-tuple (legacy) and 4-tuple (with net_id)
            if len(req) == 4:
                worker_id, request_id, state, net_id = req
            else:
                worker_id, request_id, state = req
                net_id = 'a'

            net = net_a if net_id == 'a' else net_b

            # Differentiate: batch request (ndim == 4) vs single (ndim == 3)
            if state.ndim == 4:
                self._process_single_batch(net, device, worker_id, request_id, state)
            else:
                # -- Single request: collect more for timer-based batching --
                batch = [(worker_id, request_id, state)]
                deadline = time.monotonic() + self.max_wait_ms / 1000.0

                while len(batch) < self.max_batch:
                    remaining = max(0.0, deadline - time.monotonic())
                    if remaining <= 0:
                        break
                    try:
                        req2 = self.request_queue.get(timeout=min(remaining, 0.001))
                        if req2 is None:
                            break

                        if len(req2) == 4:
                            w2, rid2, st2, nid2 = req2
                        else:
                            w2, rid2, st2 = req2
                            nid2 = 'a'

                        # If net_id differs, process current batch and handle the new one
                        if nid2 != net_id:
                            if batch:
                                self._process_batch(net, device, batch)
                            net2 = net_a if nid2 == 'a' else net_b
                            if st2.ndim == 4:
                                self._process_single_batch(net2, device, w2, rid2, st2)
                                batch = []  # already processed; prevent line 226 re-sending
                            else:
                                batch = [(w2, rid2, st2)]
                                net_id = nid2
                                net = net2
                                deadline = time.monotonic() + self.max_wait_ms / 1000.0
                            break
                        elif st2.ndim == 4:
                            if batch:
                                self._process_batch(net, device, batch)
                            self._process_single_batch(net, device, w2, rid2, st2)
                            batch = []
                            break
                        else:
                            batch.append((w2, rid2, st2))
                    except queue.Empty:
                        break

                if batch:
                    self._process_batch(net, device, batch)

        print("[GPU-Server] Shutting down")


    # -- Internal helpers ---------------------------------------------------

    def _drain_weight_queues(self, net_a, net_b, device):
        """Load the most recent weights from both weight queues."""
        # Drain net_a (latest) weights
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
            net_a.load_state_dict(state_dict)
            net_a.eval()

        # Drain net_b (best) weights -- only if a second queue exists
        if self.weight_queue_b is not None:
            latest_bytes_b = None
            while True:
                try:
                    wb = self.weight_queue_b.get_nowait()
                    if wb is not None:
                        latest_bytes_b = wb
                except queue.Empty:
                    break
            if latest_bytes_b is not None:
                buf = io.BytesIO(latest_bytes_b)
                state_dict = torch.load(buf, map_location='cpu', weights_only=True)
                net_b.load_state_dict(state_dict)
                net_b.eval()

    def _process_single_batch(self, net, device, worker_id, request_id, states):
        """Process a pre-stacked batch request ``(N, NUM_PLANES, 8, 8)`` immediately."""
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
            pass

    def _process_batch(self, net, device, batch):
        """Run a single GPU forward pass and distribute results.

        ``batch`` is a list of ``(worker_id, request_id, state)`` tuples
        where each ``state.shape == (NUM_PLANES, 8, 8)``.
        """
        states = np.stack([r[2] for r in batch], axis=0)
        states_t = torch.from_numpy(states).float().to(device).half()

        with torch.no_grad():
            policy_logits, values = net(states_t)
            policies = F.softmax(policy_logits, dim=1).float().cpu().numpy()
            values = values.float().squeeze(-1).cpu().numpy()

        for i, (worker_id, request_id, _) in enumerate(batch):
            try:
                self.response_queues[worker_id].put_nowait(
                    (request_id, policies[i], float(values[i]))
                )
            except Exception:
                pass

    def _send_shutdown_to_workers(self):
        for wq in self.response_queues.values():
            try:
                wq.put_nowait(None)
            except Exception:
                pass
