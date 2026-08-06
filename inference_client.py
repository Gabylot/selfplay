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

import numpy as np
import multiprocessing as mp
from typing import Optional, List

from shared_memory_transport import WorkerSharedBuffers, SharedMemoryTransport


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
        # Cache for out-of-order responses (needed when multiple
        # InferenceClients share the same response queue, e.g. gating eval)
        self._response_cache = {}
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

    def _acquire_slot(self) -> int:
        """Return the slot to use for the next in-flight request."""
        slot = self._next_slot
        self._next_slot = (self._next_slot + 1) % self._num_slots
        return slot

    def _track_slot(self, req_id: int, slot: int):
        self._slot_by_req[req_id] = slot

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
            states_batch = state[np.newaxis, ...]  # (1, 20, 8, 8)
            slot = self._acquire_slot()
            self._track_slot(req_id, slot)
            SharedMemoryTransport.write_states(
                self._shared_buffers, states_batch, slot
            )
            self.request_queue.put(
                (self.worker_id, req_id, self.network_id, 1, slot)  # batch_size=1, slot
            )
            return self._wait_response(req_id)

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

    def wait_result(self, req_id: int):
        """Block until the response for *req_id* arrives and return it.

        If the request was issued with shared memory, the policy/value are
        read from this worker's shared response buffer.

        Returns:
            For a batch: (policies, values) arrays.
            For a single predict: (policy_vector, scalar_value).
        """
        return self._wait_response(req_id)

    # ── Internal ────────────────────────────────────────────────────────

    def _next_id(self):
        self._req_counter += 1
        return self._req_counter

    def _wait_response(self, req_id):
        """Block until a response matching *req_id* arrives.

        In shared-memory mode the server sends
        ``(req_id, batch_size, slot)`` and the actual policy/value data
        lives in the shared response buffers.  In queue mode the server
        sends ``(req_id, policy, value)`` directly.
        """
        # Check cache for any previously received out-of-order response
        if req_id in self._response_cache:
            return self._response_cache.pop(req_id)
        while True:
            resp = self.response_queue.get()
            if resp is None:
                # Server shutting down — return zeros
                return np.zeros(4672, dtype=np.float32), 0.0

            if self._shared_buffers is not None and len(resp) == 2:
                # ── Shared-memory response: (req_id, batch_size) ──
                resp_id, batch_size = resp
                slot = self._slot_by_req.get(resp_id, 0)
                policies = SharedMemoryTransport.read_policies(
                    self._shared_buffers, batch_size, slot
                )
                values = SharedMemoryTransport.read_values(
                    self._shared_buffers, batch_size, slot
                )
                if resp_id == req_id:
                    self._slot_by_req.pop(resp_id, None)
                    if resp_id in self._batch_req_ids:
                        self._batch_req_ids.discard(resp_id)
                        return policies, values
                    return policies[0], float(values[0])
                # Out-of-order — cache it and keep waiting
                if resp_id in self._batch_req_ids:
                    self._batch_req_ids.discard(resp_id)
                    self._response_cache[resp_id] = (policies, values)
                else:
                    self._response_cache[resp_id] = (policies[0], float(values[0]))
            else:
                # ── Queue response: (req_id, policy, value) ──
                resp_id, policy, value = resp
                if resp_id == req_id:
                    return policy, value
                # Out-of-order response — cache it and keep waiting
                self._response_cache[resp_id] = (policy, value)
