"""Centralized GPU inference server for AlphaZero chess.

Runs in a dedicated process, owns the GPU (torch_directml), and serves
inference requests from all worker processes.  Includes shader pre-warming
to eliminate DirectML's lazy-compilation latency spikes.

Supports **two** networks simultaneously (network_id 0 and 1) so that
gating evaluation (network A vs network B) can run entirely on the GPU
server without round-tripping weights.

Protocol
--------
Weight  : (network_id, raw_bytes)           -- network_id in {0, 1}

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
import threading
import contextlib
import numpy as np
import torch
import torch.nn.functional as F

from network import AlphaZeroNet
import game_api
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
        np_, bh, bw = network.num_planes, network.board_h, network.board_w
        for bs in batch_sizes:
            dummy = torch.randn(bs, np_, bh, bw, device=device)
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
        section = getattr(config, 'game', None)
        game_api.set_game(getattr(section, 'name', 'chess'))
        self.game_name = game_api.active_game_name()
        self.num_planes = game_api.num_planes()
        self.board_h = game_api.board_h()
        self.board_w = game_api.board_w()
        self.request_queue = request_queue
        self.response_queues = response_queues
        self.weight_queue = weight_queue
        self.ready_event = ready_event
        self.shutdown_event = shutdown_event
        self.shared_buffers = shared_buffers  # dict[worker_id -> buffers]
        self.stats_queue = stats_queue        # optional mp.Queue for benchmark timing

# ── Pipelining (optional, enabled by run()) ──
        # The host thread aggregates requests into mega-batches and enqueues
        # them here.  A dedicated forward-worker thread pulls a batch and does
        # SHM-read + numpy build + GPU forward + distribute, so the host's
        # aggregation of batch N overlaps the GPU + copy of batch N-1.
        #
        # Note: _work_queue and _model_lock are created lazily in run() because
        # queue.Queue holds a _thread.lock and threading.Lock is not picklable --
        # the server object is pickled into the GPU subprocess via mp.Process.
        # The forward-worker thread is only used inside run().
        self._work_queue = None
        self._forward_thread = None
        self._forward_threads = []
        self._model_lock = None  # created in run()
        self._busy_count = 0  # number of forward threads currently processing

        inf_cfg = getattr(config, 'inference', None)
        self.max_batch = getattr(inf_cfg, 'max_batch', 64) if inf_cfg else 64
        self.max_wait_ms = getattr(inf_cfg, 'max_wait_ms', 3.0) if inf_cfg else 3.0
        prewarm_sizes = getattr(inf_cfg, 'prewarm_batch_sizes', [1, 8, 16, 32, 64, 128]) if inf_cfg else [1, 8, 16, 32, 64, 128]
        self.prewarm_sizes = prewarm_sizes

        # ── Batch padding ──
        # Forward every aggregated batch at a FIXED size drawn from the
        # prewarmed set (smallest entry >= n).  Variable shapes caused MIOpen
        # algo churn + allocator sync stalls that inflated pass time 3-6x on
        # this ROCm stack; fixed shapes hit the warmed fast path every time.
        # Padding rows are zero-filled and sliced away after the forward.
        self._pad_sizes = sorted({int(s) for s in self.prewarm_sizes})
        # Persistent host staging buffers per padded size (forward thread only).
        self._pad_bufs = {}
        self._pad_max = self._pad_sizes[-1] if self._pad_sizes else self.max_batch

        # Policy response dtype: when "float16", the softmax output is cast to
        # f16 ON the GPU before the D2H download, halving PCIe response bytes;
        # SHM policy buffers then carry f16 and workers gather only the legal
        # priors they need (no full-array conversion anywhere).
        policy_dtype_str = getattr(inf_cfg, 'policy_dtype', 'float32') if inf_cfg else 'float32'
        self.policy_f16 = (policy_dtype_str == 'float16')

        # Dtype used for SHM reads + GPU upload (matches request buffers).
        self._upload_dtype = self._resolve_upload_dtype()

        self.stats = self._init_stats()

    def _resolve_upload_dtype(self):
        """Determine the dtype used to upload states to the GPU.

        Matches the shared-memory request buffers' state_dtype so SHM views
        can be copied straight into the staging buffer without conversion;
        f16 uploads are cast back to f32 on the device.
        """
        if self.shared_buffers:
            for buf in self.shared_buffers.values():
                return buf.state_dtype
        return np.dtype(np.float32)

    # ── Timing instrumentation (benchmark mode only) ──
    def _init_stats(self):
        return {
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
        """Main loop -- blocks until shutdown."""
        # The object arrives via spawn/unpickle, so __init__ does not re-run:
        # restore the active game (spawned interpreter defaults to chess).
        game_api.set_game(self.game_name)
        # Re-resolve: shared_buffers arrive via pickle, dtype may differ from
        # what __init__ saw in the parent process.
        self._upload_dtype = self._resolve_upload_dtype()
        device = torch.device("cuda")
        t_start = time.perf_counter()
        print(f"[GPU-Server] ROCm device: {torch.cuda.get_device_name(0)}")

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
                    dummy = torch.randn(bs, self.num_planes, self.board_h, self.board_w, device=device)
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

        # ── Start the forward-worker thread(s) (pipelining) ──
        # The host thread below aggregates requests into mega-batches and
        # enqueues them; forward threads do SHM-read + numpy build + GPU
        # forward + distribute.  Aggregation of batch N therefore overlaps
        # the GPU forward + copies of batch N-1.
        #
        # TWO forward threads: while thread A waits on its .cpu() sync +
        # distributes results, thread B's H2D + kernel launches already queue
        # behind it, so the GPU no longer idles between passes (~10ms of
        # per-pass latency was previously exposed as idle time).
        self._model_lock = threading.Lock()  # only main server process
        self._work_queue = queue.Queue(maxsize=32)  # created after fork/spawn
        self._forward_threads = []
        for _ in range(2):
            t = threading.Thread(
                target=self._forward_worker, args=(device,), daemon=True
            )
            t.start()
            self._forward_threads.append(t)
        self._forward_thread = self._forward_threads[0]  # compat handle

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
                self._stop_forward_worker()
                self._report_stats(t_start)
                return

            try:
                self._handle_request(req, nets, device, net_a)
            except Exception as e:
                # A failure aggregating/handling one request must not kill or
                # wedge the server loop.  Log it and keep serving.
                self.stats['host_errors'] = self.stats.get('host_errors', 0) + 1
                print(f"[GPU-Server] request handling error: {e!r}")
        self._send_shutdown_to_workers()
        self._stop_forward_worker()
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
        queue pays zero wait -- not the full ``max_wait_ms``.
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
        #
        # CRITICAL: use time.perf_counter() for the deadline, NOT
        # time.monotonic().  On Windows monotonic() uses GetTickCount64
        # which only advances in ~15.6ms ticks, so a 2ms deadline would
        # silently stretch to ~15.6ms (exactly the Windows timer issue
        # documented above, but on the deadline check instead of get()).
        # perf_counter() uses QueryPerformanceCounter (sub-ms resolution).
        #
        # Pipelined aggregation policy:
        #   - If the forward worker is BUSY with a previous batch, keep
        #     collecting here so we hand it one big batch once it drains --
        #     this rebuilds the batch-accumulation the synchronous path got
        #     "for free" from being blocked on the GPU forward.
        #   - If the forward worker is IDLE, break as soon as the queue is
        #     empty so the GPU never starves waiting for us to gather more.
        pipelined = (self._forward_thread is not None
                     and self._forward_thread.is_alive()
                     and self._work_queue is not None)
        deadline = time.perf_counter() + self.max_wait_ms / 1000.0
        t_agg_start = time.perf_counter()

        while total_samples < self.max_batch:
            try:
                req2 = self.request_queue.get_nowait()

                if req2 is None:
                    # Shutdown sentinel consumed mid-aggregation.  Make the
                    # outer loop stop too (don't rely on the sentinel being
                    # seen there) so cleanup still runs.
                    self.shutdown_event.set()
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
                    deadline = time.perf_counter() + self.max_wait_ms / 1000.0
                else:
                    pending.append(item2)
                    total_samples += item2['batch_size']

            except queue.Empty:
                if pipelined and self._busy_count > 0:
                    # A forward thread is still busy -- hold off firing a tiny
                    # batch and give workers a moment to enqueue more requests.
                    if time.perf_counter() >= deadline:
                        break  # don't wait forever
                    # Back off politely: a 0.1ms wake-up rate hammers the GIL
                    # 10,000x/s, which measurably starves the forward thread's
                    # Python-side kernel dispatches (pass time inflated ~4-6x).
                    # Requests arrive at ~10s/sec at most, so 1ms polling adds
                    # negligible assembly latency.
                    time.sleep(0.001)
                    continue
                break  # idle workers + empty queue -> fire immediately

        self.stats['aggregation_wait_time'] += time.perf_counter() - t_agg_start

        if pending:
            self._process_aggregated(nets, device, pending, net_a)

    def _parse_request(self, req):
        """Parse a raw request tuple into a structured dict.

        Returns a dict with keys:
            worker_id, request_id, network_id, is_shm, batch_size,
            slot, states (ndarray or None), buf (WorkerSharedBuffers or None)

        Request protocol (shared-memory mode):
            (worker_id, request_id, network_id, batch_size[, slot])
        The optional 5th element is the shared-buffer slot index used for
        client-side pipelining (multiple in-flight requests per worker).
        4-tuples default to slot 0 (single-request-in-flight behaviour).

        For shared-memory requests, states are NOT read here -- they are
        read later in ``_process_aggregated`` when ready to process.
        This is safe because each in-flight request owns its slot and a
        worker only reuses a slot after the server has responded.
        """
        if len(req) >= 5:
            worker_id, request_id, network_id, fourth, slot = req
        else:
            worker_id, request_id, network_id, fourth = req
            slot = 0
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
                'slot': slot,
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
                'slot': slot,
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
                # Backward compat: raw bytes (old format) -> network_id 0
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
                # Serialize against the forward-worker thread which may be
                # executing nets[nid] concurrently.
                lock = self._model_lock if self._model_lock is not None else contextlib.nullcontext()
                with lock:
                    nets[nid].load_state_dict(state_dict)
                    nets[nid].eval()
                self.stats['weight_load_time'] += time.perf_counter() - t0

    def _process_aggregated(self, nets, device, pending, net_a):
        """Process aggregated requests in one forward pass per network_id.

        ``pending`` is a list of dicts returned by ``_parse_request``.

        Steps:
        1. Group by ``network_id`` so each group uses the correct network.
        2. For each group, build the host-side mega-batch numpy array
           (reading shared memory on the host thread -- this overlaps the
           previous batch's GPU forward when pipelining is active).
        3. Either run the GPU forward + distribute inline (tests /
           single-threaded) or enqueue the group for the forward worker
           thread (pipelined mode active in ``run()``).

        The offset tracking is the highest-risk part -- an off-by-one would
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

        pipelined = (self._forward_thread is not None
                     and self._forward_thread.is_alive()
                     and self._work_queue is not None)

        # Process each group with the appropriate network
        for nid in order:
            group = groups[nid]
            net = nets.get(nid, net_a)
            if pipelined:
                # Bounded enqueue with inline fallback.  A plain blocking
                # put() wedges the WHOLE server if the forward worker has
                # died/stalled: the queue fills to maxsize and the host
                # thread blocks in put() forever, so no request ever gets a
                # response.  If we can't enqueue promptly, run the forward
                # inline on this thread -- the model lock serializes forwards,
                # and it un-sticks the system even if the worker is dead.
                #
                # The SHM reads + mega-batch build happen INSIDE the forward
                # thread: the host thread used to serially memcpy ~15MB per
                # pass, which starved both forward threads (~60% idle). Now
                # feeding memcpys parallelize across threads.
                try:
                    self._work_queue.put_nowait((net, group, None))
                except queue.Full:
                    self._forward_and_distribute(net, device, group)
            else:
                self._forward_and_distribute(net, device, group)

    def _build_mega(self, group):
        """Build the mega-batch numpy array for one group.

        Reads shared memory / gathers ndarrays into a contiguous
        ``(total, num_planes, board_h, board_w)`` array.  Runs on the
        forward-worker thread (pipelined mode) or the host thread (inline).
        """
        # Pre-allocate mega-batch and copy directly (avoids
        # intermediate list + np.concatenate overhead).
        total_in_group = sum(it['batch_size'] for it in group)
        mega_states = np.empty(
            (total_in_group, self.num_planes, self.board_h, self.board_w),
            dtype=self._upload_dtype
        )
        offset = 0
        for item in group:
            bs = item['batch_size']
            if item['is_shm']:
                t0 = time.perf_counter()
                # copy=False: hand back a zero-copy view of shared memory in
                # its native dtype; the slice-assign below performs the single
                # necessary copy (dtype conversion included when f16).
                states = SharedMemoryTransport.read_states(
                    item['buf'], bs, item['slot'], copy=False
                )
                self.stats['shm_read_time'] += time.perf_counter() - t0
            else:
                states = item['states']
                # Ensure 4D (single requests come as 3D)
                if states.ndim == 3:
                    states = states[np.newaxis, ...]
                if states.dtype != self._upload_dtype:
                    states = states.astype(self._upload_dtype)
            mega_states[offset:offset + bs] = states
            offset += bs
        return mega_states

    def _forward_and_distribute(self, net, device, group):
        """GPU forward + response distribution for one group.

        Runs on the forward-worker thread in pipelined mode, or inline on
        the host thread otherwise.  The model lock serializes this against
        ``load_state_dict`` calls from the host thread.
        """
        # ── Forward pass ──
        t0 = time.perf_counter()
        mega_states = self._build_mega(group)
        n = mega_states.shape[0]

        # Pick the smallest prewarmed size >= n (pad up; zero-fill).
        padded = self._pad_max
        for s in self._pad_sizes:
            if s >= n:
                padded = s
                break
        else:
            # Beyond the largest prewarmed size: round up to a multiple of it
            # so shapes still stay within a small fixed set.
            padded = ((n + self._pad_max - 1) // self._pad_max) * self._pad_max

        # Per-thread staging: two forward threads must not share buffers.
        key = (threading.get_ident(), padded, self._upload_dtype)
        buf = self._pad_bufs.get(key)
        if buf is None or buf.shape[0] != padded:
            buf = np.zeros((padded, mega_states.shape[1],
                            mega_states.shape[2], mega_states.shape[3]),
                           dtype=self._upload_dtype)
            self._pad_bufs[key] = buf
        buf[:n] = mega_states

        states_t = torch.from_numpy(buf).to(device)
        if self._upload_dtype == np.float16:
            # Cast on-device: halves the PCIe upload vs f32; the network
            # itself keeps its float32 weights.
            states_t = states_t.float()
        t_h2d = time.perf_counter()

        lock = self._model_lock if self._model_lock is not None else contextlib.nullcontext()
        with lock:
            with torch.inference_mode():
                policy_logits, values = net(states_t)
                pol = F.softmax(policy_logits[:n], dim=1)
                if self.policy_f16:
                    pol = pol.half()
                t_fwd = time.perf_counter()
        # Download OUTSIDE the model lock: the sync may wait ~10ms for the
        # GPU to drain; holding the lock here would serialize weight reloads
        # and (with 2 forward threads) stall the sibling's launches.
        policies = pol.cpu().numpy()
        values = values.squeeze(-1)[:n].cpu().numpy()
        t_d2h = time.perf_counter()
        self.stats['gpu_forward_time'] += time.perf_counter() - t0
        self.stats['samples_processed'] += n
        self.stats['aggregated_batches'] += 1
        # Sub-phase timing (diagnostics): H2D copy / network / D2H copy
        self.stats['t_h2d'] = self.stats.get('t_h2d', 0.0) + (t_h2d - t0)
        self.stats['t_fwd'] = self.stats.get('t_fwd', 0.0) + (t_fwd - t_h2d)
        self.stats['t_d2h'] = self.stats.get('t_d2h', 0.0) + (t_d2h - t_fwd)
        self.stats['n_passes_timed'] = self.stats.get('n_passes_timed', 0) + 1

        self._distribute(group, policies, values)

    def _put_response(self, worker_id: int, payload) -> None:
        """Deliver a response to a worker's queue without silent drops.

        A plain ``put_nowait`` wrapped in ``except: pass`` used to *silently
        discard* responses whenever a worker's bounded queue filled up --
        that worker then blocks forever in ``response_queue.get()`` with no
        diagnostic anywhere.  Retry briefly instead; only if the queue is
        still full after the timeout (worker genuinely stuck/dead) do we
        drop, and then loudly.
        """
        q = self.response_queues.get(worker_id)
        if q is None:
            return
        if isinstance(payload, tuple) and len(payload) >= 2 and payload[0] == 'ERR':
            req_id = payload[1]
        else:
            req_id = payload[0] if isinstance(payload, tuple) else -1
        try:
            q.put_nowait(payload)
            return
        except queue.Full:
            pass
        except Exception:
            # Closed interpreter / broken pipe on shutdown -- nothing to do,
            # but make it visible rather than silently swallowed.
            self.stats['dropped_responses'] = self.stats.get('dropped_responses', 0) + 1
            print(f"[GPU-Server] WARNING: response queue for worker {worker_id} "
                  f"unavailable (shutdown?); dropped response {req_id}")
            return
        try:
            q.put(payload, timeout=2.0)
        except queue.Full:
            self.stats['dropped_responses'] = self.stats.get('dropped_responses', 0) + 1
            print(f"[GPU-Server] WARNING: response queue FULL for worker "
                  f"{worker_id}; DROPPED response {req_id} after 2s retry -- "
                  f"that client will hang unless it recovers")

    def _distribute(self, group, policies, values):
        """Split a group's mega-batch results back to the individual workers."""
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
                    buf, item_policies, item_values, item['slot']
                )
                self.stats['shm_write_time'] += time.perf_counter() - t0
                self._put_response(worker_id, (request_id, bs))
                continue

            # ── Queue response path ──
            # Always send the (policies, values) arrays, even for bs == 1.
            # The client decides (via its batch-request id set) whether the
            # caller wanted the full arrays (predict_batch) or a single
            # (policy_vector, scalar) pair (predict).  Sending a bare scalar
            # value here broke the bs == 1 batched path (MCTS indexes
            # values_batch[k]).
            self._put_response(worker_id,
                               (request_id, item_policies, item_values))

        # Safety assertion: offset must equal total samples in this group
        assert offset == sum(it['batch_size'] for it in group), \
            f"Offset tracking error: offset={offset} != total={sum(it['batch_size'] for it in group)}"

    def _forward_worker(self, device):
        """Forward-worker thread: pulls groups and runs GPU + distribute.

        This decouples the GPU forward (with its .to/.cpu copies) from the
        host thread's request aggregation, so aggregation of batch N
        overlaps GPU + copies of batch N-1.  With two worker threads, one
        thread's sync/distribute overlaps the sibling's launches.
        """
        while True:
            job = self._work_queue.get()
            if job is None:
                break
            net, group, _unused = job  # mega build now happens inside _forward_and_distribute
            self._busy_count += 1
            try:
                self._forward_and_distribute(net, device, group)
            except Exception as e:
                # A transient GPU/DirectML error must NOT kill this thread.
                # Dropping one batch is far better than crashing the server --
                # but we now tell each waiting client explicitly so it can
                # raise instead of blocking until its timeout.
                self.stats['forward_errors'] = self.stats.get('forward_errors', 0) + 1
                n_samples = sum(it['batch_size'] for it in group)
                print(f"[GPU-Server] forward error (dropping batch {n_samples} samples): {e!r}")
                for item in group:
                    try:
                        self._put_response(
                            item['worker_id'],
                            ('ERR', item['request_id'], repr(e)))
                    except Exception:
                        pass
            finally:
                self._busy_count -= 1

    def _stop_forward_worker(self):
        """Signal the forward workers to stop and wait for them to drain."""
        if self._forward_threads and self._work_queue is not None:
            for _ in self._forward_threads:
                try:
                    self._work_queue.put_nowait(None)
                except queue.Full:
                    pass
            for t in self._forward_threads:
                t.join(timeout=5.0)
            self._forward_threads = []
            self._forward_thread = None

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
