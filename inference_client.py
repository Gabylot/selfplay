"""Lightweight GPU inference client for worker processes.

Drop-in replacement for ``AlphaZeroNet`` in MCTS calls.  Workers send
inference requests to the centralized GPU server and block until the
result arrives.

Optimization: ``predict_batch()`` sends the **entire batch as a single
message** ``(worker_id, request_id, network_id, batch_states)`` with shape
``(N, 20, 8, 8)``, eliminating per-sample IPC overhead.  The GPU server
detects the batch (ndim == 4) and processes it immediately without the
timer-based aggregation window.

**Shared-memory transport**: When ``shared_buffers`` is provided, state
and policy/value arrays are exchanged through pre-allocated ``mp.Array``
shared memory instead of being pickled through the queue.  Only ~100 bytes
of metadata travel through the queue per request, reducing IPC overhead by
~99%.  If ``shared_buffers`` is ``None`` the client falls back to the
original queue-based (pickle) transport.

Usage::

    client = InferenceClient(worker_id=0, request_queue=q, response_queue=rq)
    policy, value = client.predict(state)            # single (20,8,8)
    policies, values = client.predict_batch(states)  # batch (N,20,8,8)

    # For dual-network eval (e.g. gating), bind a client to a specific
    # network slot on the GPU server:
    client_b = InferenceClient(worker_id=0, request_queue=q,
                                response_queue=rq, network_id=1)

    # Shared-memory mode (optional, requires buffers from SharedMemoryTransport):
    from shared_memory_transport import SharedMemoryTransport
    transport = SharedMemoryTransport(num_workers=8, max_batch=64)
    transport.create_buffers()
    client = InferenceClient(worker_id=0, request_queue=q, response_queue=rq,
                             shared_buffers=transport.get_worker_buffers(0))
"""

import time
import queue
import itertools
import numpy as np
import multiprocessing as mp
import threading
from typing import Optional, List

import game_api
from shared_memory_transport import WorkerSharedBuffers, SharedMemoryTransport

# How long to wait for a single inference response before giving up.  The GPU
# server never silently drops a request in normal operation, so a timeout here
# means something is genuinely wrong (crashed forward thread, OOM, dead
# worker); raising is far better than blocking the worker forever.
_RESPONSE_TIMEOUT_S = 120.0

# Process-global request-id counter.  Multiple InferenceClients can share one
# response queue (gating eval creates two: network 0 and network 1), so ids
# MUST be unique across clients -- otherwise a response could cross clients and
# hand one network the other's policy/value.
_REQ_COUNTER = itertools.count(1)

# Process-global response parking lot.  Clients sharing one response queue can
# pop responses that belong to a sibling client; instead of caching those in a
# private dict (where the rightful owner would never see them and eventually
# time out), every client parks them here so the owner pops them on arrival.
_RESPONSE_CACHE = {}
_RESPONSE_CACHE_LOCK = threading.Lock()


class InferenceClient:
    """Drop-in replacement for ``AlphaZeroNet`` in MCTS.

    Implements the same ``predict`` / ``predict_batch`` interface so that
    ``MCTS`` (and any code that calls those methods) works unchanged.

    Parameters
    ----------
    worker_id : int
        Unique worker identifier (0-based).  Used to route responses.
    request_queue : mp.Queue
        Shared queue leading to the GPU inference server.
    response_queue : mp.Queue
        Per-worker queue where the server puts this worker's results.
    network_id : int
        Which network slot on the (dual-network) GPU server to query.
        0 = primary/latest, 1 = secondary/best (used for gating eval).
    shared_buffers : WorkerSharedBuffers, optional
        Pre-allocated shared-memory buffers for zero-copy state/policy
        exchange.  When provided, only metadata is sent through the queue
        and the actual arrays travel through shared memory.  When ``None``
        (default), the original queue-based pickle transport is used.
    concurrency : int
        Number of requests a single worker may have in flight before it
        must wait for a response.  Enables pipelining: the worker issues
        ``predict_batch_async()`` multiple times (each uses a distinct
        shared-buffer slot) and collects results with ``wait_result()``.
        The synchronous ``predict``/``predict_batch`` methods still block
        as before (they pipeline internally, so MCTS code is unchanged).
    """

    def __init__(self, worker_id: int, request_queue: mp.Queue,
                 response_queue: mp.Queue, network_id: int = 0,
                 shared_buffers: Optional[WorkerSharedBuffers] = None,
                 concurrency: int = 1):
        self.worker_id = worker_id
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.network_id = network_id
        self._shared_buffers = shared_buffers
        self._concurrency = max(1, concurrency)
        self._req_counter = 0
        # req_ids that are BATCH predictions (vs single-state).  Used to
        # return full arrays instead of (policy_vector, scalar_value).
        self._batch_req_ids = set()
        # req_id → shared-buffer slot used when the request was issued.
        # The client assigns each request a round-robin slot, so it can
        # locate the right buffer segment for a response without the server
        # echoing the slot back.
        self._slot_by_req = {}
        if self._shared_buffers is not None:
            self._num_slots = max(1, self._shared_buffers.num_slots)
            # Round-robin slot allocator: start slot 0, wrap when > slots.
            self._next_slot = 0
        else:
            self._num_slots = 1
            self._next_slot = 0
        # In-flight request count (shared-memory mode only).  num_slots are
        # the hard ceiling: more requests than that would round-robin onto a
        # slot whose previous payload the server has not read yet and thus
        # silently corrupt batches.  Callers keep inflight <= concurrency,
        # and SharedMemoryTransport sizes num_slots == client_concurrency,
        # so this only trips if that invariant is ever broken.
        self._inflight = 0
        # req_id -> {'indices': [...per-sample legal index arrays...],
        #            'shm_gather': bool} for gathered requests.  The server
        # packs priors for shm_gather requests; otherwise the response
        # carries full rows and this client gathers them on receive.
        self._gather_by_req = {}

    def _acquire_slot(self) -> int:
        """Return the slot to use for the next in-flight request."""
        slot = self._next_slot
        self._next_slot = (self._next_slot + 1) % self._num_slots
        return slot

    def _track_slot(self, req_id: int, slot: int):
        self._slot_by_req[req_id] = slot

    def _claim_slot_budget(self):
        """Raise if more requests are in flight than there are buffer slots.

        Prevents silent shared-memory slot corruption (see _inflight note
        in __init__); every entry point that writes a shared-memory slot
        must call this first.
        """
        if self._shared_buffers is not None and self._inflight >= self._num_slots:
            raise RuntimeError(
                f"InferenceClient(worker={self.worker_id}) reached its slot "
                f"limit: {self._inflight} requests already in flight vs "
                f"{self._num_slots} slots.  Collect pending results before "
                f"issuing further predict_batch_async/predict calls."
            )

    def _release_slot_budget(self):
        if self._inflight > 0:
            self._inflight -= 1

    # ── Public interface (matches AlphaZeroNet) ─────────────────────────

    def predictBatch(self, states: np.ndarray):
        """Alias for predict_batch() to match AlphaZeroNet interface."""
        return self.predict_batch(states)


    def predict(self, state: np.ndarray):
        """Predict policy and value for a single board state.

        Args:
            state: (20, 8, 8) numpy array

        Returns:
            policy: (4672,) numpy array of probabilities
            value: scalar float in [-1, 1]
        """
        req_id = self._next_id()

        if self._shared_buffers is not None:
            # ── Shared-memory path ──
            # Write the single state (as a 1-element batch) to shared memory,
            # then send only metadata through the queue.
            self._claim_slot_budget()
            self._inflight += 1
            states_batch = state[np.newaxis, ...]  # (1, 20, 8, 8)
            slot = self._acquire_slot()
            self._track_slot(req_id, slot)
            SharedMemoryTransport.write_states(
                self._shared_buffers, states_batch, slot
            )
            self.request_queue.put(
                (self.worker_id, req_id, self.network_id, 1, slot)  # batch_size=1, slot
            )
            try:
                return self._wait_response(req_id)
            finally:
                self._release_slot_budget()

        # ── Queue (pickle) path ──
        self.request_queue.put((self.worker_id, req_id, self.network_id, state))
        return self._wait_response(req_id)

    def predict_batch(self, states: np.ndarray):
        """Predict policy and value for a batch of board states (synchronous).

        **Optimization**: sends the entire stacked batch ``(N,20,8,8)``
        as a *single* message to the GPU server, eliminating per-sample
        IPC overhead.  The server detects ndim == 4 and processes it
        immediately without the timer-based aggregation window.

        When shared-memory buffers are configured, the state array is
        written to shared memory and only ~100 bytes of metadata travel
        through the queue.

        Args:
            states: (batch, 20, 8, 8) numpy array

        Returns:
            policies: (batch, 4672) numpy array of probabilities
            values: (batch,) numpy array of scalars
        """
        return self.wait_result(self.predict_batch_async(states))

    def predict_batch_async(self, states: np.ndarray) -> int:
        """Send a batch prediction without blocking.

        Writes *states* to the next available shared-buffer slot (or the queue)
        and returns a request-id that ``wait_result()`` accepts.  A worker may
        issue up to ``concurrency`` of these before collecting any results.

        Args:
            states: (batch, 20, 8, 8) numpy array

        Returns:
            request_id (int) — pass to ``wait_result()`` to get the result.
        """
        n = len(states)
        req_id = self._next_id()
        self._batch_req_ids.add(req_id)

        if self._shared_buffers is not None:
            # ── Shared-memory path ──
            self._claim_slot_budget()
            self._inflight += 1
            slot = self._acquire_slot()
            self._track_slot(req_id, slot)
            SharedMemoryTransport.write_states(
                self._shared_buffers, states, slot
            )
            self.request_queue.put(
                (self.worker_id, req_id, self.network_id, n, slot)
            )
        else:
            # ── Queue (pickle) path ──
            self.request_queue.put(
                (self.worker_id, req_id, self.network_id, states)
            )
        return req_id

    def predict_batch_gathered_async(self, states: np.ndarray,
                                     legal_indices) -> int:
        """Send a gathered-resolution batch prediction without blocking.

        Like ``predict_batch_async``, but the request also carries the per
        sample legal-move policy indices.  The GPU server then responds with
        ONLY the priors at those indices (packed f16) instead of the full
        4672-wide policy vector -- roughly a 100x smaller download.

        ``wait_result()`` on the returned id ALWAYS yields
        ``(priors_list, values)`` -- ``priors_list[k]`` is a numpy array
        aligned with ``legal_indices[k]`` -- regardless of which transport
        path handled the request (the client gathers full rows itself when
        the server did not).

        Args:
            states: (batch, planes, h, w) numpy array
            legal_indices: list of per-sample int arrays (each may be empty)

        Returns:
            request_id (int) — pass to ``wait_result()``.
        """
        n = len(states)
        req_id = self._next_id()
        self._batch_req_ids.add(req_id)

        # Per-request gather metadata: indices for client-side gathering +
        # whether the server did the gathering (shared-memory path only).
        self._gather_by_req[req_id] = {
            'indices': list(legal_indices),
            'shm_gather': False,
        }

        if self._shared_buffers is not None:
            self._claim_slot_budget()
            self._inflight += 1
            slot = self._acquire_slot()
            self._track_slot(req_id, slot)
            SharedMemoryTransport.write_states(
                self._shared_buffers, states, slot
            )
            gm = getattr(self._shared_buffers, 'gather_moves', 0)
            acceptable = (gm > 0
                          and all(len(idx) <= gm for idx in legal_indices))
            if acceptable:
                SharedMemoryTransport.write_gather_indices(
                    self._shared_buffers, legal_indices, slot
                )
                self._gather_by_req[req_id]['shm_gather'] = True
                self.request_queue.put(
                    (self.worker_id, req_id, self.network_id, n, slot, 1)
                )
                return req_id
            # Oversized or legacy buffers: ship as a normal full-policy SHM
            # request; this client gathers the response rows itself.
            self.request_queue.put(
                (self.worker_id, req_id, self.network_id, n, slot)
            )
            return req_id

        # ── Queue (pickle) fallback: send states, gather client-side ──
        self.request_queue.put(
            (self.worker_id, req_id, self.network_id, states)
        )
        return req_id

    def _gather_rows_from_full(self, policies, values, indices):
        """Gather per-sample legal priors from full policy rows (client side).

        Used when the response carries complete 4672-wide rows (queue mode
        or an oversized SHM request the server handled as full policy).
        """
        dtype = (self._shared_buffers.policy_dtype
                 if self._shared_buffers is not None else np.float16)
        priors_list = []
        for k, idx in enumerate(indices):
            if len(idx):
                priors_list.append(
                    np.asarray(policies[k], dtype=dtype)[
                        np.asarray(idx, dtype=np.int64)].copy())
            else:
                priors_list.append(np.zeros(0, dtype=dtype))
        return (priors_list, values)

    def _gather_response(self, resp_id, batch_size, slot):
        """Interpret a shared-memory response for a gathered request.

        Returns ``(priors_list, values)`` aligned to the per-sample index
        lists stored in ``self._gather_by_req[resp_id]``.  Pops the metadata.
        """
        meta = self._gather_by_req.pop(resp_id, None)
        indices = meta['indices'] if meta is not None else None
        values = SharedMemoryTransport.read_values(
            self._shared_buffers, batch_size, slot
        )
        if meta is None or indices is None:
            # Unknown metadata (should not happen): return raw rows.
            policies = SharedMemoryTransport.read_policies(
                self._shared_buffers, batch_size, slot
            )
            return (policies, values)
        if meta.get('shm_gather'):
            # Server packs priors via SUBSET-softmax over the legal-index
            # columns (exp(l_j) proportional).  Renormalize over each
            # sample's count to recover the same priors a full softmax +
            # renormalize-over-legal would produce.  Unused tail columns
            # (exp(0)=1 padding) are sliced away by the per-sample counts.
            priors = SharedMemoryTransport.read_gathered_priors(
                self._shared_buffers, [len(x) for x in indices], slot
            )
            renormed = []
            for p in priors:
                p32 = p.astype(np.float32)
                s = float(p32.sum())
                renormed.append(
                    p32 / s if s > 0.0 else p32)
            return (renormed, values)
        # Full rows were written to SHM; gather the legal priors here.
        policies = SharedMemoryTransport.read_policies(
            self._shared_buffers, batch_size, slot
        )
        return self._gather_rows_from_full(policies, values, indices)

    def wait_result(self, req_id: int):
        """Block until the response for *req_id* arrives and return it.

        If the request was issued with shared memory, the policy/value are
        read from this worker's shared response buffer.

        Returns:
            For a batch: (policies, values) arrays.
            For a single predict: (policy_vector, scalar_value).
        """
        try:
            return self._wait_response(req_id)
        finally:
            self._release_slot_budget()

    # ── Internal ────────────────────────────────────────────────────────

    def _next_id(self):
        return next(_REQ_COUNTER)

    def _park(self, req_id: int, payload):
        """Park an out-of-order (or sibling-client) response by request id."""
        with _RESPONSE_CACHE_LOCK:
            _RESPONSE_CACHE[req_id] = payload

    @staticmethod
    def _take_cached(req_id: int):
        with _RESPONSE_CACHE_LOCK:
            return _RESPONSE_CACHE.pop(req_id, None)

    def _cleanup_pending(self, req_id: int):
        self._slot_by_req.pop(req_id, None)
        self._batch_req_ids.discard(req_id)
        self._gather_by_req.pop(req_id, None)

    def _wait_response(self, req_id):
        """Block until a response matching *req_id* arrives.

        In shared-memory mode the server sends
        ``(req_id, batch_size, slot)`` and the actual policy/value data
        lives in the shared response buffers.  In queue mode the server
        sends ``(req_id, policy, value)`` directly.

        Responses for *other* request ids (our own pipelining, or sibling
        clients sharing this response queue) are parked in the global
        ``_RESPONSE_CACHE`` so their owners can pick them up -- a client
        must never swallow a response that belongs to someone else.
        """
        deadline = time.monotonic() + _RESPONSE_TIMEOUT_S
        while True:
            cached = self._take_cached(req_id)
            if cached is None:
                pass
            elif isinstance(cached, tuple) and len(cached) > 0 \
                    and isinstance(cached[0], str):
                # Marker tuples start with a str tag; raw data tuples parked
                # by a sibling client start with arrays.  Guarding on the
                # cached[0] type avoids numpy elementwise comparison crashes
                # for plain data tuples.
                if cached[0] == '__ERR__':
                    self._cleanup_pending(req_id)
                    raise RuntimeError(
                        f"GPU inference server error for request {req_id}: {cached[1]}"
                    )
                if cached[0] == '__SHM__':
                    # Parked by a sibling client (it could not know this
                    # client's shared-memory slot): perform our own SHM read
                    # using the slot mapping we kept from the request.
                    batch_size = cached[1]
                    slot = self._slot_by_req.pop(req_id, 0)
                    if req_id in self._gather_by_req:
                        out = self._gather_response(req_id, batch_size, slot)
                        self._batch_req_ids.discard(req_id)
                        return out
                    policies = SharedMemoryTransport.read_policies(
                        self._shared_buffers, batch_size, slot
                    )
                    values = SharedMemoryTransport.read_values(
                        self._shared_buffers, batch_size, slot
                    )
                    out = ((policies, values) if req_id in self._batch_req_ids
                           else (policies[0], float(values[0])))
                    self._batch_req_ids.discard(req_id)
                    return out
                if cached[0] == '__Q__':
                    # Parked raw queue-style response (from a sibling
                    # client): decide the return shape via OUR state.
                    _, policies, values = cached
                    self._slot_by_req.pop(req_id, None)
                    meta = self._gather_by_req.pop(req_id, None)
                    if meta is not None:
                        # Gathered request served through the queue path.
                        if req_id not in self._batch_req_ids:
                            raise RuntimeError(
                                f"InferenceClient(worker={self.worker_id}, "
                                f"net={self.network_id}): gathered request "
                                f"{req_id} did not go through the batch API"
                            )
                        self._batch_req_ids.discard(req_id)
                        return self._gather_rows_from_full(
                            policies, values, meta['indices']
                        )
                    if req_id in self._batch_req_ids:
                        self._batch_req_ids.discard(req_id)
                        return (policies, values)
                    return (policies[0], float(values[0]))
                raise RuntimeError(
                    f"InferenceClient(worker={self.worker_id}, "
                    f"net={self.network_id}): unknown marker {cached!r}"
                )
            else:
                # Plain (possibly out-of-order, sibling-parked) data response
                # such as (policies, values) or (policy, value).
                self._batch_req_ids.discard(req_id)
                self._slot_by_req.pop(req_id, None)
                return cached
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._slot_by_req.pop(req_id, None)
                self._batch_req_ids.discard(req_id)
                self._gather_by_req.pop(req_id, None)
                raise TimeoutError(
                    f"InferenceClient(worker={self.worker_id}, "
                    f"net={self.network_id}) timed out after "
                    f"{_RESPONSE_TIMEOUT_S:.0f}s waiting for response "
                    f"{req_id}; the GPU inference server may have dropped the "
                    f"request (check dropped_responses / forward_errors)."
                )
            try:
                resp = self.response_queue.get(timeout=remaining)
            except queue.Empty:
                continue
            if resp is None:
                # Server shutting down: bookkeeping consistent with the other
                # failure paths, then raise -- returning zeroes would inject
                # meaningless policy/values (and draw labels) into a live
                # game/search without anyone noticing.
                self._slot_by_req.pop(req_id, None)
                self._batch_req_ids.discard(req_id)
                self._gather_by_req.pop(req_id, None)
                raise RuntimeError(
                    f"InferenceClient(worker={self.worker_id}, "
                    f"net={self.network_id}): inference server shut down "
                    f"while request {req_id} was pending."
                )

            # Error response from the server (e.g. a batch dropped because the
            # forward pass raised).  The second element identifies the exact
            # request; only raise/shutdown bookkeeping for OUR request, park
            # errors for sibling/pipelined requests so they surface to their
            # rightful owner.
            if (isinstance(resp, tuple) and len(resp) == 3
                    and isinstance(resp[0], str) and resp[0] == 'ERR'):
                err_id = resp[1]
                if err_id == req_id:
                    self._slot_by_req.pop(req_id, None)
                    self._batch_req_ids.discard(req_id)
                    self._gather_by_req.pop(req_id, None)
                    raise RuntimeError(
                        f"GPU inference server error for request {req_id}: {resp[2]}"
                    )
                self._park(err_id, ('__ERR__', resp[2]))
                continue

            if self._shared_buffers is not None and len(resp) == 2:
                # ── Shared-memory response: (req_id, batch_size) ──
                resp_id, batch_size = resp
                if resp_id in self._slot_by_req or resp_id == req_id:
                    # Our own request (current or pipelined): we know the slot.
                    slot = self._slot_by_req.pop(resp_id, 0)
                    meta = self._gather_by_req.get(resp_id)
                    if meta is not None:
                        # ── Gathered (or gather-fallback) response ──
                        out = self._gather_response(resp_id, batch_size, slot)
                        self._batch_req_ids.discard(resp_id)
                        if resp_id == req_id:
                            return out
                        self._park(resp_id, out)
                        continue
                    policies = SharedMemoryTransport.read_policies(
                        self._shared_buffers, batch_size, slot
                    )
                    values = SharedMemoryTransport.read_values(
                        self._shared_buffers, batch_size, slot
                    )
                    if resp_id in self._batch_req_ids:
                        self._batch_req_ids.discard(resp_id)
                        out = (policies, values)
                    else:
                        out = (policies[0], float(values[0]))
                    if resp_id == req_id:
                        return out
                    self._park(resp_id, out)
                else:
                    # Request from a sibling client sharing this response
                    # queue: we cannot know its slot, so park a marker and
                    # let the owner do its own shared-memory read.
                    self._park(resp_id, ('__SHM__', batch_size))
            else:
                # ── Queue response: (req_id, policies, values) ──
                # The server always sends full arrays now; decide whether the
                # caller wanted the arrays (predict_batch) or a single
                # (policy_vector, scalar) pair (predict) via _batch_req_ids.
                resp_id, policies, values = resp
                meta = self._gather_by_req.pop(resp_id, None)
                if resp_id == req_id or resp_id in self._batch_req_ids:
                    if meta is not None:
                        # Gathered request served through the queue path:
                        # response carries full rows; gather client-side.
                        self._batch_req_ids.discard(resp_id)
                        out = self._gather_rows_from_full(
                            policies, values, meta['indices']
                        )
                    elif resp_id in self._batch_req_ids:
                        self._batch_req_ids.discard(resp_id)
                        out = (policies, values)
                    else:
                        out = (policies[0], float(values[0]))
                    if resp_id == req_id:
                        self._slot_by_req.pop(resp_id, None)
                        return out
                    # Out-of-order response — park it for its owner
                    self._slot_by_req.pop(resp_id, None)
                    self._park(resp_id, out)
                else:
                    # Sibling client's request: shapes are decided by its
                    # state, not ours — park raw for the owner to interpret.
                    self._park(resp_id, ('__Q__', policies, values))
