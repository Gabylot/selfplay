"""Lightweight GPU inference client for worker processes.

Drop-in replacement for ``AlphaZeroNet`` in MCTS calls.  Workers send
inference requests to the centralized GPU server and block until the
result arrives.

Usage::

    client = InferenceClient(worker_id=0, request_queue=q, response_queue=rq)
    policy, value = client.predict(state)            # single (20,8,8)
    policies, values = client.predict_batch(states)  # batch (N,20,8,8)
"""

import numpy as np
import multiprocessing as mp


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
    """

    def __init__(self, worker_id: int, request_queue: mp.Queue,
                 response_queue: mp.Queue):
        self.worker_id = worker_id
        self.request_queue = request_queue
        self.response_queue = response_queue
        self._req_counter = 0

    # ── Public interface (matches AlphaZeroNet) ─────────────────────────

    def predict(self, state: np.ndarray):
        """Predict policy and value for a single board state.

        Args:
            state: (20, 8, 8) numpy array

        Returns:
            policy: (4672,) numpy array of probabilities
            value: scalar float in [-1, 1]
        """
        req_id = self._next_id()
        self.request_queue.put((self.worker_id, req_id, state))
        return self._wait_response(req_id)

    def predict_batch(self, states: np.ndarray):
        """Predict policy and value for a batch of board states.

        Sends individual requests for each state.  The GPU server
        automatically batches them together because they arrive within
        the ``max_wait_ms`` window.

        Args:
            states: (batch, 20, 8, 8) numpy array

        Returns:
            policies: (batch, 4672) numpy array of probabilities
            values: (batch,) numpy array of scalars
        """
        n = len(states)
        req_ids = [self._next_id() for _ in range(n)]

        for i in range(n):
            self.request_queue.put((self.worker_id, req_ids[i], states[i]))

        # Collect all responses (may arrive in any order)
        results = {}
        while len(results) < n:
            resp = self.response_queue.get()
            if resp is None:
                # Server shutting down — fill remaining with zeros
                for rid in req_ids:
                    if rid not in results:
                        results[rid] = (np.zeros(4672, dtype=np.float32), 0.0)
                break
            resp_id, policy, value = resp
            results[resp_id] = (policy, value)

        # Reconstruct in the original request order
        policies = np.array([results[rid][0] for rid in req_ids])
        values = np.array([results[rid][1] for rid in req_ids], dtype=np.float32)
        return policies, values

    # ── Internal ────────────────────────────────────────────────────────

    def _next_id(self):
        self._req_counter += 1
        return self._req_counter

    def _wait_response(self, req_id):
        """Block until a response matching *req_id* arrives."""
        while True:
            resp = self.response_queue.get()
            if resp is None:
                # Server shutting down — return zeros
                return np.zeros(4672, dtype=np.float32), 0.0
            resp_id, policy, value = resp
            if resp_id == req_id:
                return policy, value
            # If out-of-order (shouldn't happen for single predict),
            # we still need to handle it — store and keep waiting.
            # In practice this path is never hit for predict() because
            # there's only one outstanding request at a time.