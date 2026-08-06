"""Shared-memory transport for GPU inference IPC.

Reduces the per-request IPC payload from ~1.7 MB (pickled numpy arrays)
down to ~100 bytes (metadata only) by pre-allocating ``mp.Array`` buffers
that are shared between each worker process and the GPU server process.

Protocol
--------
Each worker gets three shared buffers:

  * **request buffer**  — states  ``(max_batch, NUM_PLANES, 8, 8)`` dtype
  * **policy buffer**   — policies ``(max_batch, NUM_ACTIONS)`` dtype
  * **value buffer**    — values   ``(max_batch,)`` float32

The synchronous request/response pattern (client blocks until the server
replies) guarantees that no two processes read/write the same buffer
concurrently, so no explicit locking is needed.

Usage
-----
In the main process (before starting workers / GPU server)::

    transport = SharedMemoryTransport(num_workers=8, max_batch=64,
                                      state_dtype=np.float16)
    transport.create_buffers()          # allocate mp.Array objects
    worker_bufs = transport.get_worker_buffers(0)   # pass to worker 0
    server_bufs = transport.get_all_worker_buffers() # pass to GPU server

In a worker process::

    client = InferenceClient(..., shared_buffers=worker_bufs)

In the GPU server process::

    server = GPUInferenceServer(..., shared_buffers=server_bufs)

If ``shared_buffers`` is ``None`` the client/server transparently fall
back to the original queue-based (pickle) transport.
"""

import multiprocessing as mp
import numpy as np
from typing import Optional, Dict, List

from encoding import NUM_PLANES, NUM_ACTIONS


# Default dtype for state tensors in shared memory.
# float16 halves the buffer size with no loss for binary board planes
# and the single normalized move-count plane.
DEFAULT_STATE_DTYPE = np.float16


def _get_raw_array(arr):
    """Get the raw ctypes array from an mp.Array.

    When ``lock=False``, ``mp.Array`` returns the raw ctypes array
    directly.  When ``lock=True`` (default), it returns a
    ``SynchronizedArray`` wrapper that requires ``get_obj()``.
    """
    if hasattr(arr, 'get_obj'):
        return arr.get_obj()
    return arr


class WorkerSharedBuffers:
    """Bundle of shared-memory buffers for a single worker.

    Attributes
    ----------
    request_states : mp.Array
        Flat shared array backing the request state tensor
        ``(max_batch, NUM_PLANES, 8, 8)`` of *state_dtype*.
    response_policies : mp.Array
        Flat shared array backing the response policy tensor
        ``(max_batch, NUM_ACTIONS)`` of *state_dtype*.
    response_values : mp.Array
        Flat shared array backing the response value vector
        ``(max_batch,)`` float32.
    max_batch : int
        Maximum batch size the buffers can hold.
    state_dtype : np.dtype
        Numpy dtype used for states and policies.
    """

    def __init__(self, max_batch: int, state_dtype=np.float16, num_slots: int = 1):
        self.max_batch = max_batch
        self.state_dtype = np.dtype(state_dtype)
        self.num_slots = max(1, num_slots)
        slot_stride = max_batch

        state_elems = max_batch * NUM_PLANES * 8 * 8 * self.num_slots
        policy_elems = max_batch * NUM_ACTIONS * self.num_slots
        value_elems = max_batch * self.num_slots

        # mp.Array typecode: 'f' = float32, 'd' = float64, 'h' = float16
        # States can be float16 (binary board planes -- no precision loss).
        # Policies MUST be float32 -- the GPU outputs float32 and converting
        # to float16 during np.copyto is a major CPU bottleneck (~200 MB/s
        # vs >1 GB/s for same-type memcpy).
        if self.state_dtype == np.float16:
            state_tc = 'h'
        elif self.state_dtype == np.float32:
            state_tc = 'f'
        else:
            # Fall back to float32 for unsupported dtypes
            state_tc = 'f'
            self.state_dtype = np.float32

        self.request_states = mp.Array(state_tc, state_elems, lock=False)
        self.response_policies = mp.Array('f', policy_elems, lock=False)  # always float32
        self.response_values = mp.Array('f', value_elems, lock=False)

    # ── Numpy views (created per-process, not shared) ────────────────
    # Each slot is an independent, capacity-``max_batch`` segment so a worker
    # can have up to ``num_slots`` requests in flight simultaneously.

    def _slot_off(self, slot: int, per_elem: int) -> int:
        """Start offset (in elements) for ``slot`` given elements per row set."""
        return slot * self.max_batch * per_elem

    def request_states_np(self, batch_size: int, slot: int = 0) -> np.ndarray:
        """Return a numpy view of the first *batch_size* rows of *slot*'s
        request buffer, shaped ``(batch_size, NUM_PLANES, 8, 8)``."""
        n_ppp = NUM_PLANES * 8 * 8  # elems per sample
        off = self._slot_off(slot, n_ppp)
        flat = np.frombuffer(
            _get_raw_array(self.request_states),
            dtype=self.state_dtype,
        )
        return flat[off:off + batch_size * n_ppp].reshape(
            batch_size, NUM_PLANES, 8, 8
        )

    def response_policies_np(self, batch_size: int, slot: int = 0) -> np.ndarray:
        """Return a numpy view of the first *batch_size* rows of *slot*'s
        policy buffer, shaped ``(batch_size, NUM_ACTIONS)``.

        Always float32 -- the GPU outputs float32 and the buffer is always
        float32 regardless of state_dtype.
        """
        start = self._slot_off(slot, NUM_ACTIONS)
        flat = np.frombuffer(
            _get_raw_array(self.response_policies), dtype=np.float32,
        )
        return flat[start:start + batch_size * NUM_ACTIONS].reshape(
            batch_size, NUM_ACTIONS
        )

    def response_values_np(self, batch_size: int, slot: int = 0) -> np.ndarray:
        """Return a numpy view of the first *batch_size* elements of *slot*'s
        value buffer, shaped ``(batch_size,)``."""
        start = self._slot_off(slot, 1)
        flat = np.frombuffer(
            _get_raw_array(self.response_values), dtype=np.float32,
        )
        return flat[start:start + batch_size]


class SharedMemoryTransport:
    """Manages per-worker shared-memory buffers for the GPU inference pipeline.

    Created in the main process **before** workers and the GPU server are
    started.  The buffer objects (``mp.Array``) are picklable and can be
    passed as process arguments or through ``mp.Queue``.
    """

    def __init__(self, num_workers: int, max_batch: int = 64,
                 state_dtype=np.float16, num_slots: int = 1):
        self.num_workers = num_workers
        self.max_batch = max_batch
        self.state_dtype = np.dtype(state_dtype)
        self.num_slots = max(1, num_slots)
        self._buffers: Dict[int, WorkerSharedBuffers] = {}

    def create_buffers(self):
        """Allocate one ``WorkerSharedBuffers`` per worker."""
        for i in range(self.num_workers):
            self._buffers[i] = WorkerSharedBuffers(
                self.max_batch, self.state_dtype, self.num_slots
            )

    def get_worker_buffers(self, worker_id: int) -> WorkerSharedBuffers:
        """Return the shared buffers for *worker_id*.

        Call this in the main process and pass the result to the worker
        process (e.g. via ``mp.Process`` args or a task queue).
        """
        return self._buffers[worker_id]

    def get_all_worker_buffers(self) -> Dict[int, WorkerSharedBuffers]:
        """Return a dict mapping worker_id → buffers for the GPU server."""
        return dict(self._buffers)

    @staticmethod
    def write_states(buf: WorkerSharedBuffers, states: np.ndarray,
                     slot: int = 0):
        """Copy *states* into the request shared buffer for *slot*.

        ``states`` may be float32 or float16; it is cast to the buffer's
        dtype in-place.  No copy is made if the dtypes already match and
        the array is contiguous.
        """
        n = len(states)
        if n > buf.max_batch:
            raise ValueError(
                f"Batch size {n} exceeds shared buffer capacity {buf.max_batch}"
            )
        view = buf.request_states_np(n, slot)
        # np.copyto handles dtype conversion (float32 → float16) efficiently
        np.copyto(view, states)

    @staticmethod
    def write_policies_values(buf: WorkerSharedBuffers,
                              policies: np.ndarray, values: np.ndarray,
                              slot: int = 0):
        """Copy *policies* and *values* into the response shared buffers."""
        n = len(policies)
        if n > buf.max_batch:
            raise ValueError(
                f"Batch size {n} exceeds shared buffer capacity {buf.max_batch}"
            )
        pol_view = buf.response_policies_np(n, slot)
        np.copyto(pol_view, policies)

        val_view = buf.response_values_np(n, slot)
        np.copyto(val_view, values)

    @staticmethod
    def read_states(buf: WorkerSharedBuffers, batch_size: int,
                    slot: int = 0) -> np.ndarray:
        """Read states from the request shared buffer for *slot*.

        Returns a float32 array (the dtype the network expects).  When
        the buffer is already float32, this is a fast same-type copy.
        When the buffer is float16, a conversion is required (slower).
        """
        view = buf.request_states_np(batch_size, slot)
        if buf.state_dtype == np.float32:
            # Same dtype -- return a copy (safe, worker won't overwrite
            # until we respond, but copy avoids any aliasing issues)
            return view.copy()
        # float16 -> float32 conversion (CPU-bound)
        return view.astype(np.float32)

    @staticmethod
    def read_policies(buf: WorkerSharedBuffers, batch_size: int,
                      slot: int = 0) -> np.ndarray:
        """Read policies from the response shared buffer for *slot* as float32.

        Returns a copy so the caller owns the data before the server
        overwrites the shared buffer on the next request.
        """
        view = buf.response_policies_np(batch_size, slot)
        return view.copy()

    @staticmethod
    def read_values(buf: WorkerSharedBuffers, batch_size: int,
                    slot: int = 0) -> np.ndarray:
        """Read values from the response shared buffer for *slot* as float32."""
        return buf.response_values_np(batch_size, slot).copy()