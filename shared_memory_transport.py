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

import game_api


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
    request_indices : mp.Array
        Gathered-response request side: legal-move policy indices per
        sample ``(max_batch, gather_moves)`` uint16 (0 when fewer legal
        moves than ``gather_moves``).
    request_counts : mp.Array
        Number of valid indices per sample ``(max_batch,)`` uint16.
    max_batch : int
        Maximum batch size the buffers can hold.
    state_dtype : np.dtype
        Numpy dtype used for states and policies.
    gather_moves : int
        Width of the packed-legal-priors layout (min(256, num_actions)).
    """

    def __init__(self, max_batch: int, state_dtype=np.float16, num_slots: int = 1,
                 policy_dtype=np.float32):
        self.max_batch = max_batch
        self.state_dtype = np.dtype(state_dtype)
        self.policy_dtype = np.dtype(policy_dtype)
        self.num_slots = max(1, num_slots)
        slot_stride = max_batch
        self.num_planes = game_api.num_planes()
        self.num_actions = game_api.num_actions()
        # Width of the gathered-priors layout.  Chess positions have at most
        # 218 legal moves; 256 gives headroom and stays far below the full
        # action space (4672), so the gather response occupies only the first
        # ``gather_moves`` columns of the existing policy region.
        self.gather_moves = min(256, self.num_actions)

        state_elems = max_batch * self.num_planes * game_api.board_h() * game_api.board_w() * self.num_slots
        policy_elems = max_batch * self.num_actions * self.num_slots
        value_elems = max_batch * self.num_slots
        index_elems = max_batch * self.gather_moves * self.num_slots
        count_elems = max_batch * self.num_slots

        # mp.Array typecode: 'f' = float32, 'd' = float64, 'h' = float16
        # States can be float16 (binary board planes -- no precision loss).
        # Policies may be float16 when the GPU server casts its softmax
        # output to f16 BEFORE the D2H download -- then every copy in the
        # pipeline is a same-dtype memcpy and the PCIe response shrinks 2x.
        # (The old "always float32" rule existed because converting f32->f16
        # during np.copyto is slow; producing f16 on the GPU avoids that.)
        if self.state_dtype == np.float16:
            state_tc = 'h'
        elif self.state_dtype == np.float32:
            state_tc = 'f'
        else:
            # Fall back to float32 for unsupported dtypes
            state_tc = 'f'
            self.state_dtype = np.float32

        if self.policy_dtype == np.float16:
            policy_tc = 'h'
        else:
            policy_tc = 'f'
            self.policy_dtype = np.float32

        self.request_states = mp.Array(state_tc, state_elems, lock=False)
        self.response_policies = mp.Array(policy_tc, policy_elems, lock=False)
        self.response_values = mp.Array('f', value_elems, lock=False)
        self.request_indices = mp.Array('H', index_elems, lock=False)
        self.request_counts = mp.Array('H', count_elems, lock=False)

    def close(self):
        """Release the underlying shared-memory segments for this worker."""
        for arr in (self.request_states, self.response_policies,
                    self.response_values, self.request_indices,
                    self.request_counts):
            try:
                if hasattr(arr, 'close'):
                    arr.close()
            except Exception:
                pass

    # ── Numpy views (created per-process, not shared) ────────────────
    # Each slot is an independent, capacity-``max_batch`` segment so a worker
    # can have up to ``num_slots`` requests in flight simultaneously.

    def _slot_off(self, slot: int, per_elem: int) -> int:
        """Start offset (in elements) for ``slot`` given elements per row set."""
        return slot * self.max_batch * per_elem

    def request_states_np(self, batch_size: int, slot: int = 0) -> np.ndarray:
        """Return a numpy view of the first *batch_size* rows of *slot*'s
        request buffer, shaped ``(batch_size, num_planes, board_h, board_w)``."""
        n_ppp = self.num_planes * game_api.board_h() * game_api.board_w()
        off = self._slot_off(slot, n_ppp)
        flat = np.frombuffer(
            _get_raw_array(self.request_states),
            dtype=self.state_dtype,
        )
        return flat[off:off + batch_size * n_ppp].reshape(
            batch_size, self.num_planes, game_api.board_h(), game_api.board_w()
        )

    def response_policies_np(self, batch_size: int, slot: int = 0) -> np.ndarray:
        """Return a numpy view of the first *batch_size* rows of *slot*'s
policy buffer, shaped ``(batch_size, num_actions)``.

        Dtype matches the buffer's ``policy_dtype`` (float32 or float16).
        """
        start = self._slot_off(slot, self.num_actions)
        flat = np.frombuffer(
            _get_raw_array(self.response_policies), dtype=self.policy_dtype,
        )
        return flat[start:start + batch_size * self.num_actions].reshape(
            batch_size, self.num_actions
        )
        

    def request_indices_np(self, batch_size: int, slot: int = 0) -> np.ndarray:
        """Numpy view of *slot*'s request legal-index table
        (``(batch_size, gather_moves)`` uint16)."""
        off = self._slot_off(slot, self.gather_moves)
        flat = np.frombuffer(
            _get_raw_array(self.request_indices), dtype=np.uint16,
        )
        return flat[off:off + batch_size * self.gather_moves].reshape(
            batch_size, self.gather_moves
        )

    def request_counts_np(self, batch_size: int, slot: int = 0) -> np.ndarray:
        """Numpy view of *slot*'s request per-sample index counts
        (``(batch_size,)`` uint16)."""
        off = self._slot_off(slot, 1)
        flat = np.frombuffer(
            _get_raw_array(self.request_counts), dtype=np.uint16,
        )
        return flat[off:off + batch_size]

    def response_priors_np(self, batch_size: int, slot: int = 0) -> np.ndarray:
        """Numpy view of *slot*'s gathered-priors region: the first
        ``gather_moves`` columns of the policy region, shaped
        ``(batch_size, gather_moves)``.

        Dtype matches ``policy_dtype`` (float16 or float32) -- identical
        quantization to the full-policy download.
        """
        start = self._slot_off(slot, self.num_actions)
        flat = np.frombuffer(
            _get_raw_array(self.response_policies), dtype=self.policy_dtype,
        )
        return flat[start:start + batch_size * self.gather_moves].reshape(
            batch_size, self.gather_moves
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
                 state_dtype=np.float16, num_slots: int = 1,
                 policy_dtype=np.float32):
        self.num_workers = num_workers
        self.max_batch = max_batch
        self.state_dtype = np.dtype(state_dtype)
        self.policy_dtype = np.dtype(policy_dtype)
        self.num_slots = max(1, num_slots)
        self._buffers: Dict[int, WorkerSharedBuffers] = {}

    def create_buffers(self):
        """Allocate one ``WorkerSharedBuffers`` per worker."""
        for i in range(self.num_workers):
            self._buffers[i] = WorkerSharedBuffers(
                self.max_batch, self.state_dtype, self.num_slots,
                policy_dtype=self.policy_dtype
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

    def close(self):
        """Release all shared-memory segments held by this transport."""
        for buf in self._buffers.values():
            try:
                buf.close()
            except Exception:
                pass
        self._buffers.clear()

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
                    slot: int = 0, copy: bool = True) -> np.ndarray:
        """Read states from the request shared buffer for *slot*.

        With ``copy=False`` a zero-copy *view* in the buffer's NATIVE dtype is
        returned (float32 or float16) — only safe when the caller consumes it
        before the worker reuses this slot (the synchronous request/response
        protocol guarantees this).  The GPU server uploads f16 views directly
        and casts to f32 on-device, halving PCIe input bytes.
        With ``copy=True`` a float32 array is returned as before.
        """
        view = buf.request_states_np(batch_size, slot)
        if not copy:
            return view
        if buf.state_dtype == np.float32:
            return view.copy()
        # float16 -> float32 conversion (CPU-bound)
        return view.astype(np.float32)

    @staticmethod
    def read_policies(buf: WorkerSharedBuffers, batch_size: int,
                      slot: int = 0) -> np.ndarray:
        """Read policies from the response shared buffer for *slot*.

        Returns a copy in the buffer's own dtype (float32 or float16) — no
        conversion.  Consumers gather only the handful of legal-move priors
        they need, so f16 precision is fine and per-element conversion cost
        is paid on ~35 values instead of 4672.
        """
        view = buf.response_policies_np(batch_size, slot)
        return view.copy()

    @staticmethod
    def read_values(buf: WorkerSharedBuffers, batch_size: int,
                    slot: int = 0) -> np.ndarray:
        """Read values from the response shared buffer for *slot* as float32."""
        return buf.response_values_np(batch_size, slot).copy()

    # ── Gathered-policy (legal priors only) protocol ─────────────────────

    @staticmethod
    def write_gather_indices(buf: WorkerSharedBuffers,
                             legal_indices, slot: int = 0):
        """Write per-sample legal-move policy indices for a gather request.

        ``legal_indices`` is a list of per-sample integer arrays (variable
        length).  Each sample's count is written to the counts region and
        its indices zero-padded to ``gather_moves`` columns.

        Raises ValueError if any sample exceeds ``gather_moves`` indices.
        """
        gm = buf.gather_moves
        n = len(legal_indices)
        if n > buf.max_batch:
            raise ValueError(
                f"Gather request batch {n} exceeds buffer capacity {buf.max_batch}"
            )
        idx_view = buf.request_indices_np(n, slot)
        cnt_view = buf.request_counts_np(n, slot)
        for k, idx in enumerate(legal_indices):
            c = len(idx)
            if c > gm:
                raise ValueError(
                    f"Sample {k} has {c} legal moves > gather width {gm}; "
                    f"caller must split or fall back to the full-policy path"
                )
            if c:
                idx_view[k, :c] = idx
                if c < gm:
                    idx_view[k, c:] = 0
            cnt_view[k] = c

    @staticmethod
    def read_gather_requests(buf: WorkerSharedBuffers, batch_size: int,
                             slot: int = 0):
        """Server side: read a request's legal-index table and per-sample
        counts.  Returns ``(indices uint16 (bs, gather_moves), counts (bs,))``
        -- zero-copy views, server consumes them before responding."""
        return (buf.request_indices_np(batch_size, slot).copy(),
                buf.request_counts_np(batch_size, slot).copy())

    @staticmethod
    def write_priors_values(buf: WorkerSharedBuffers,
                            priors: np.ndarray, values: np.ndarray,
                            slot: int = 0):
        """Write gathered priors ``(batch, gather_moves)`` and values into
        the response buffers for *slot*.  ``priors`` must already be at the
        response dtype (the server produces it on-device)."""
        n = len(priors)
        if n > buf.max_batch:
            raise ValueError(
                f"Batch size {n} exceeds shared buffer capacity {buf.max_batch}"
            )
        pri_view = buf.response_priors_np(n, slot)
        np.copyto(pri_view, priors)
        val_view = buf.response_values_np(n, slot)
        np.copyto(val_view, values)

    @staticmethod
    def read_gathered_priors(buf: WorkerSharedBuffers, counts, slot: int = 0):
        """Client side: unpack per-sample prior arrays (variable length).

        ``counts`` is the per-sample count list/array used for the request.
        Returns a list of ``policy_dtype`` float arrays, one per sample.
        """
        gm = buf.gather_moves
        flat = buf.response_priors_np(len(counts), slot)
        out = []
        for k, c in enumerate(counts):
            c = int(c)
            if c:
                out.append(flat[k, :c].copy())
            else:
                out.append(np.zeros(0, dtype=buf.policy_dtype))
        return out