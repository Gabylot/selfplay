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
from typing import Optional

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
    """

    def __init__(self, worker_id: int, request_queue: mp.Queue,
                 response_queue: mp.Queue, network_id: int = 0,
                 shared_buffers: Optional[WorkerSharedBuffers] = None):
        self.worker_id = worker_id
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.network_id = network_id
        self._shared_buffers = shared_buffers
        self._req_counter = 0
        # Cache for out-of-order responses (needed when multiple
        # InferenceClients share the same response queue, e.g. gating eval)
        self._response_cache = {}

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
            SharedMemoryTransport.write_states(self._shared_buffers, states_batch)
            self.request_queue.put(
                (self.worker_id, req_id, self.network_id, 1)  # batch_size=1
            )
            return self._wait_response(req_id)

        # ── Queue (pickle) path ──
        self.request_queue.put((self.worker_id, req_id, self.network_id, state))
        return self._wait_response(req_id)

    def predict_batch(self, states: np.ndarray):
        """Predict policy and value for a batch of board states.

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
        n = len(states)
        req_id = self._next_id()

        if self._shared_buffers is not None:
            # ── Shared-memory path ──
            SharedMemoryTransport.write_states(self._shared_buffers, states)
            self.request_queue.put(
                (self.worker_id, req_id, self.network_id, n)
            )
        else:
            # ── Queue (pickle) path ──
            # The server differentiates: ndim == 4  =>  batch request (immediate)
            #                     ndim == 3  =>  single request (timer-batched)
            self.request_queue.put(
                (self.worker_id, req_id, self.network_id, states)
            )

        # Wait for a single response containing the full batch.
        resp = self.response_queue.get()
        if resp is None:
            # Server shutting down — return zeros
            return np.zeros((n, 4672), dtype=np.float32), np.zeros(n, dtype=np.float32)

        if self._shared_buffers is not None and len(resp) == 2:
            # ── Shared-memory response: (req_id, batch_size) ──
            resp_id, batch_size = resp
            policies = SharedMemoryTransport.read_policies(
                self._shared_buffers, batch_size
            )
            values = SharedMemoryTransport.read_values(
                self._shared_buffers, batch_size
            )
            return policies, values

        # ── Queue response: (req_id, policies, values) ──
        resp_id, policies, values = resp
        return policies, values

    # ── Internal ────────────────────────────────────────────────────────

    def _next_id(self):
        self._req_counter += 1
        return self._req_counter

    def _wait_response(self, req_id):
        """Block until a response matching *req_id* arrives.

        In shared-memory mode the server sends ``(req_id, batch_size)``
        and the actual policy/value data lives in the shared response
        buffers.  In queue mode the server sends
        ``(req_id, policy, value)`` directly.
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
                if resp_id == req_id:
                    policies = SharedMemoryTransport.read_policies(
                        self._shared_buffers, batch_size
                    )
                    values = SharedMemoryTransport.read_values(
                        self._shared_buffers, batch_size
                    )
                    # Single predict: return (policy_vector, scalar_value)
                    return policies[0], float(values[0])
                # Out-of-order — copy from shared memory before it's overwritten
                policies = SharedMemoryTransport.read_policies(
                    self._shared_buffers, batch_size
                )
                values = SharedMemoryTransport.read_values(
                    self._shared_buffers, batch_size
                )
                self._response_cache[resp_id] = (policies[0], float(values[0]))
            else:
                # ── Queue response: (req_id, policy, value) ──
                resp_id, policy, value = resp
                if resp_id == req_id:
                    return policy, value
                # Out-of-order response — cache it and keep waiting
                self._response_cache[resp_id] = (policy, value)
