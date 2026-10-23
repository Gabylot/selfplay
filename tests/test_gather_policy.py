"""Correctness tests for the gathered (legal-priors-only) response protocol.

The gathered path changes WHICH numbers cross shared memory: instead of the
full 4672-wide policy vector, only the priors at each sample's legal-move
indices are produced.  A mixup here (wrong rows, wrong packing order, stale
offsets) would silently corrupt priors while remaining numerically plausible,
so the tests verify exact agreement against a from-scratch numpy recompute.

Strategy:
    - MockNetwork: value = state.sum() / 1000.0, policy logits are a
      deterministic per-sample function of the input state (varied, NOT
      uniform -- a gather misalignment would still produce 1/N for a
      uniform-softmax net).
    - Expected gathered priors computed in numpy from the same logits.
"""
import sys
sys.path.insert(0, '.')

import time
import queue as _queue
import multiprocessing as mp
from typing import Dict, List

import numpy as np
import torch

from encoding import NUM_PLANES, NUM_ACTIONS
from gpu_server import GPUInferenceServer
from shared_memory_transport import SharedMemoryTransport, WorkerSharedBuffers
from inference_client import InferenceClient


class VariedLogitsNetwork(torch.nn.Module):
    """Value = state.sum() / 1000; logits depend on SAMPLE POSITION.

    logits[b, a] = sin((b+1)*1.3 + a*0.017) * 2. -- rows differ, so a row
    swap or offset bug in the gather packing produces visibly different
    priors (a state-derived or uniform-softmax net would mask misalignment).
    """

    def __init__(self, network_id=0):
        super().__init__()
        self.network_id = network_id

    def forward(self, x: torch.Tensor):
        sums = x.sum(dim=(1, 2, 3))
        n = len(x)
        b = torch.arange(n, device=x.device, dtype=torch.float32) \
            .unsqueeze(1)
        a = torch.arange(NUM_ACTIONS, device=x.device,
                         dtype=torch.float32).unsqueeze(0)
        logits = torch.sin((b + 1.0) * 1.3 + a * 0.017) * 2.0
        values = (sums / 1000.0).unsqueeze(1)
        return logits, values

    def eval(self):
        return self

    def to(self, device):
        return self


class MockConfig:
    class inference:
        max_batch = 128
        max_wait_ms = 2.0
        prewarm_batch_sizes = [1, 8, 16, 32, 64, 128]

    class network:
        num_residual_blocks = 4
        num_filters = 64
        num_policy_channels = 64
        num_value_channels = 64
        value_fc_size = 128


def make_server(max_batch=128, max_wait_ms=2.0, shared_buffers=None):
    config = MockConfig()
    config.inference.max_batch = max_batch
    config.inference.max_wait_ms = max_wait_ms
    request_queue = mp.Queue()
    response_queues = {i: mp.Queue(maxsize=256) for i in range(10)}
    weight_queue = mp.Queue()
    ready_event = mp.Event()
    shutdown_event = mp.Event()
    server = GPUInferenceServer(
        config=config, request_queue=request_queue,
        response_queues=response_queues, weight_queue=weight_queue,
        ready_event=ready_event, shutdown_event=shutdown_event,
        shared_buffers=shared_buffers,
    )
    return server, request_queue, response_queues


def make_states(batch_size, fill_value):
    return np.full((batch_size, NUM_PLANES, 8, 8), fill_value, dtype=np.float32)


def expected_logits(n_rows):
    """numpy recompute of the network's logits rows for n sample positions."""
    acts = np.arange(NUM_ACTIONS, dtype=np.float64)
    b = (np.arange(n_rows, dtype=np.float64) + 1.0)[:, None]
    rows = np.sin(b * 1.3 + acts * 0.017) * 2.0
    return rows


def expected_softmax_gathered(n_rows, indices):
    """Expected gathered priors per sample (float32), by position."""
    return _gather_cols(expected_softmax_full(n_rows), indices)


def _gather_cols(full, indices):
    return [full[k][np.asarray(idx, dtype=np.int64)].astype(np.float32)
            if len(idx) else np.zeros(0, dtype=np.float32)
            for k, idx in enumerate(indices)]


def expected_packed_raw(n_rows, indices, gather_moves=256):
    """RAW subset-softmax values as written to the shared buffer.

    The gathered columns carry exp(l_j)/Z where Z = sum of exp over ALL 256
    packed columns (padding columns gather logit index 0).  The client
    renormalizes over the per-sample count before delivery.
    """
    full_logits = expected_logits(n_rows)
    out = []
    for k, idx in enumerate(indices):
        if len(idx):
            l = full_logits[k]
            num = np.exp(l[np.asarray(idx, dtype=np.int64)])
            denom = num.sum() + (gather_moves - len(idx)) * np.exp(l[0])
            out.append((num / denom).astype(np.float32))
        else:
            out.append(np.zeros(0, dtype=np.float32))
    return out


def expected_softmax_full(n_rows):
    """Expected full softmax rows (float32), (n_rows, NUM_ACTIONS)."""
    rows = expected_logits(n_rows)
    e = np.exp(rows - rows.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def expected_gathered_renorm(n_rows, indices, rows_offset=0):
    """Expected client-delivered gathered priors.

    The server packs a SUBSET-softmax over the legal-index columns; the
    client renormalizes each sample over its count.  That equals the full
    softmax gathered at the legal indices, renormalized over the legal set
    (the exp(0)=1 padding mass divides out)."""
    full = expected_softmax_full(n_rows)
    out = []
    for k, idx in enumerate(indices):
        if len(idx):
            g = full[rows_offset + k][
                np.asarray(idx, dtype=np.int64)].astype(np.float64)
            s = g.sum()
            if s > 0:
                out.append((g / s).astype(np.float32))
            else:
                out.append(np.full(len(idx), 1.0 / len(idx),
                                   dtype=np.float32))
        else:
            out.append(np.zeros(0, dtype=np.float32))
    return out


def drain_and_process(server, request_queue, nets, device, net_a):
    """Process every queued request as ONE deterministic aggregated group.

    Going through _handle_request would race with mp.Queue's async feeder
    thread (the aggregator polls get_nowait), making the row layout of the
    pass nondeterministic; parsing + processing directly pins the layout.
    """
    pending = []
    while True:
        try:
            req = request_queue.get_nowait()
        except _queue.Empty:
            break
        if req is None:
            continue
        item = server._parse_request(req)
        if item is None:
            continue
        pending.append(item)
    if pending:
        server._process_aggregated(nets, device, pending, net_a)


class TestGatherProtocolSHM:
    """Server-side gathered responses (shared-memory requests, 6-tuples)."""

    def _mk(self, num_workers=2, num_slots=2):
        transport = SharedMemoryTransport(
            num_workers=num_workers, max_batch=128, num_slots=num_slots,
            policy_dtype=np.float16,
        )
        transport.create_buffers()
        shared_buffers = transport.get_all_worker_buffers()
        server, request_queue, response_queues = make_server(
            max_batch=128, shared_buffers=shared_buffers)
        device = torch.device("cpu")
        net_a = VariedLogitsNetwork(0)
        nets = {0: net_a, 1: VariedLogitsNetwork(1)}
        return transport, server, request_queue, response_queues, \
            shared_buffers, device, nets, net_a

    def test_gather_only_pass(self):
        (transport, server, request_queue, response_queues, shared_buffers,
         device, nets, net_a) = self._mk()
        rng = np.random.default_rng(42)
        bs = 4
        fills = [0.11, 0.23, 0.31, 0.47]
        states = np.zeros((bs, NUM_PLANES, 8, 8), dtype=np.float32)
        for i, f in enumerate(fills):
            states[i] = f
        # Variable-size index tables incl. an empty one (count 0 edge).
        legal = [
            rng.choice(NUM_ACTIONS, size=12, replace=False).astype(np.int64),
            rng.choice(218, size=218, replace=False).astype(np.int64),
            np.zeros(0, dtype=np.int64),
            rng.choice(NUM_ACTIONS, size=47, replace=False).astype(np.int64),
        ]

        buf = shared_buffers[0]
        SharedMemoryTransport.write_states(buf, states, slot=0)
        SharedMemoryTransport.write_gather_indices(buf, legal, slot=0)
        counts = [len(x) for x in legal]

        request_queue.put((0, 900, 0, bs, 0, 1))
        server._handle_request(request_queue.get_nowait(),
                               nets, device, net_a)
        drain_and_process(server, request_queue, nets, device, net_a)

        resp = response_queues[0].get(timeout=5.0)
        assert resp == (900, bs), resp

        priors = SharedMemoryTransport.read_gathered_priors(buf, counts, 0)
        values = SharedMemoryTransport.read_values(buf, bs, 0)

        expect = expected_packed_raw(bs, legal, gather_moves=buf.gather_moves)
        for k in range(bs):
            assert len(priors[k]) == len(expect[k])
            if len(expect[k]):
                # f16 quantization of the subset-softmax pack
                assert np.allclose(
                    priors[k].astype(np.float32), expect[k], atol=6e-4), \
                    (f"sample {k}: got {priors[k][:5]} "
                     f"want {expect[k][:5]}")
            else:
                assert len(priors[k]) == 0
        for i, f in enumerate(fills):
            want = NUM_PLANES * 64 * f / 1000.0
            assert abs(values[i] - want) < 5e-3

        transport.close()
        print("  [PASS] test_gather_only_pass")

    def test_mixed_gather_and_full_same_pass(self):
        (transport, server, request_queue, response_queues, shared_buffers,
         device, nets, net_a) = self._mk()
        rng = np.random.default_rng(7)

        # Worker 0: gathered request (bs 3, slot 0)
        bs0, fill0 = 3, 0.15
        states0 = np.full((bs0, NUM_PLANES, 8, 8), fill0, dtype=np.float32)
        legal = [
            rng.choice(NUM_ACTIONS, size=30, replace=False).astype(np.int64),
            rng.choice(NUM_ACTIONS, size=5, replace=False).astype(np.int64),
            rng.choice(NUM_ACTIONS, size=218, replace=False).astype(np.int64),
        ]
        buf0 = shared_buffers[0]
        SharedMemoryTransport.write_states(buf0, states0, slot=0)
        SharedMemoryTransport.write_gather_indices(buf0, legal, slot=0)

        # Worker 1: full-policy request (bs 3, slot 0)
        bs1, fill1 = 3, 0.42
        states1 = np.full((bs1, NUM_PLANES, 8, 8), fill1, dtype=np.float32)
        buf1 = shared_buffers[1]
        SharedMemoryTransport.write_states(buf1, states1, slot=0)

        # Both enter the queue BEFORE the first is processed -> same pass.
        request_queue.put((0, 910, 0, bs0, 0, 1))
        request_queue.put((1, 911, 0, bs1, 0))
        server._handle_request(request_queue.get_nowait(),
                               nets, device, net_a)
        drain_and_process(server, request_queue, nets, device, net_a)

        resp0 = response_queues[0].get(timeout=5.0)
        assert resp0 == (910, bs0), resp0
        priors0 = SharedMemoryTransport.read_gathered_priors(
            buf0, [len(x) for x in legal], 0)
        # Mixed pass: packed values come from full-softmax rows via
        # take_along (server reuses the full download when any item needs
        # full rows).  Unnormalized until the client renormalizes.
        expect0 = _gather_cols(expected_softmax_full(bs0), legal)
        for k in range(bs0):
            assert np.allclose(priors0[k].astype(np.float32), expect0[k],
                               atol=5e-4), \
                f"gathered worker sample {k} mismatch"

        resp1 = response_queues[1].get(timeout=5.0)
        assert resp1 == (911, bs1), resp1
        pol1 = SharedMemoryTransport.read_policies(buf1, bs1, 0)
        val1 = SharedMemoryTransport.read_values(buf1, bs1, 0)
        # Full rows are at absolute pass positions 3..5 (after 3 gathered).
        expect1 = expected_softmax_full(bs0 + bs1)[bs0:bs0 + bs1]
        # Full rows: compare complete rows against expected softmax.
        assert np.allclose(pol1.astype(np.float32), expect1, atol=5e-4)
        for i in range(bs1):
            assert abs(val1[i] - NUM_PLANES * 64 * fill1 / 1000.0) < 5e-3

        transport.close()
        print("  [PASS] test_mixed_gather_and_full_same_pass")

    def test_mixed_pass_absolute_rows(self):
        """Mixed gather+full pass with PER-SAMPLE fills.

        Guards the absolute-row slicing in _distribute: full items must get
        the policy rows at their own pass position (not the first rows of
        the pass).
        """
        (transport, server, request_queue, response_queues, shared_buffers,
         device, nets, net_a) = self._mk()
        rng = np.random.default_rng(11)

        # Gathered worker first (rows 0..2), full worker second (rows 3..5)
        bs0, fill0 = 3, 0.15
        states0 = np.full((bs0, NUM_PLANES, 8, 8), fill0, dtype=np.float32)
        legal = [
            rng.choice(NUM_ACTIONS, size=30, replace=False).astype(np.int64),
            rng.choice(NUM_ACTIONS, size=5, replace=False).astype(np.int64),
            rng.choice(NUM_ACTIONS, size=218, replace=False).astype(np.int64),
        ]
        buf0 = shared_buffers[0]
        SharedMemoryTransport.write_states(buf0, states0, slot=0)
        SharedMemoryTransport.write_gather_indices(buf0, legal, slot=0)

        # Full worker with per-sample fills != gathered worker's fill
        bs1 = 3
        fills1 = [0.61, 0.71, 0.83]
        states1 = np.zeros((bs1, NUM_PLANES, 8, 8), dtype=np.float32)
        for i, f in enumerate(fills1):
            states1[i] = f
        buf1 = shared_buffers[1]
        SharedMemoryTransport.write_states(buf1, states1, slot=0)

        request_queue.put((0, 920, 0, bs0, 0, 1))
        request_queue.put((1, 921, 0, bs1, 0))
        server._handle_request(request_queue.get_nowait(),
                               nets, device, net_a)
        drain_and_process(server, request_queue, nets, device, net_a)

        resp1 = response_queues[1].get(timeout=5.0)
        assert resp1 == (921, bs1), resp1
        pol1 = SharedMemoryTransport.read_policies(buf1, bs1, 0)
        val1 = SharedMemoryTransport.read_values(buf1, bs1, 0)
        for i, f in enumerate(fills1):
            # Sample i of worker 1 sits at absolute pass row bs0 + i.
            e = expected_softmax_full(bs0 + bs1)[bs0 + i]
            assert np.allclose(pol1[i].astype(np.float32), e, atol=5e-4), \
                f"full worker sample {i} got wrong policy row"
            assert abs(val1[i] - NUM_PLANES * 64 * f / 1000.0) < 5e-3, \
                f"full worker sample {i} got wrong value row"

        resp0 = response_queues[0].get(timeout=5.0)
        assert resp0 == (920, bs0), resp0

        transport.close()
        print("  [PASS] test_mixed_pass_absolute_rows")


class TestClientGatherSemantics:
    """Client-side semantics: uniform (priors_list, values) for any path."""

    def test_gather_rows_from_full(self):
        client = InferenceClient.__new__(InferenceClient)
        client._shared_buffers = None
        rows = np.linspace(0.0, 1.0, 3 * NUM_ACTIONS, dtype=np.float32) \
            .reshape(3, NUM_ACTIONS)
        idx = [np.array([0, 5, 4671], dtype=np.int64),
               np.zeros(0, dtype=np.int64),
               np.array([1234], dtype=np.int64)]
        priors, values = client._gather_rows_from_full(rows, np.ones(3), idx)
        assert len(priors) == 3
        assert priors[0].dtype == np.float16
        assert np.array_equal(
            priors[0], rows[0][idx[0]].astype(np.float16))
        assert len(priors[1]) == 0
        assert np.array_equal(
            priors[2], rows[2][idx[2]].astype(np.float16))
        print("  [PASS] test_gather_rows_from_full")

    def test_client_end_to_end_gathered_roundtrip(self):
        """InferenceClient -> server -> wait_result returns gathered priors.

        Also drives two pipelined requests and waits OUT of order so the
        parked-response path with gathered payloads is exercised.
        """
        transport = SharedMemoryTransport(
            num_workers=1, max_batch=128, num_slots=2,
            policy_dtype=np.float16)
        transport.create_buffers()
        shared_buffers = transport.get_all_worker_buffers()
        server, request_queue, response_queues = make_server(
            max_batch=128, shared_buffers=shared_buffers)
        device = torch.device("cpu")
        net_a = VariedLogitsNetwork(0)
        nets = {0: net_a, 1: VariedLogitsNetwork(1)}

        buf = shared_buffers[0]
        client = InferenceClient(
            worker_id=0, request_queue=request_queue,
            response_queue=response_queues[0], network_id=0,
            shared_buffers=buf, concurrency=2)

        rng = np.random.default_rng(3)
        states = make_states(4, 0.2)
        legal = [
            rng.choice(NUM_ACTIONS, size=35, replace=False).astype(np.int64),
            rng.choice(NUM_ACTIONS, size=7, replace=False).astype(np.int64),
            rng.choice(NUM_ACTIONS, size=218, replace=False).astype(np.int64),
            rng.choice(NUM_ACTIONS, size=1, replace=False).astype(np.int64),
        ]
        req_a = client.predict_batch_gathered_async(states, legal)

        # Instead of processing req A first, submit a second request so the
        # first response arrives OUT of submission order at wait time.
        legal2 = [
            rng.choice(NUM_ACTIONS, size=20, replace=False).astype(np.int64),
            rng.choice(NUM_ACTIONS, size=11, replace=False).astype(np.int64),
            rng.choice(NUM_ACTIONS, size=13, replace=False).astype(np.int64),
            rng.choice(NUM_ACTIONS, size=2, replace=False).astype(np.int64),
        ]
        states2 = make_states(4, 0.6)
        req_b = client.predict_batch_gathered_async(states2, legal2)

        # Process BOTH requests (two passes likely, but the client's wait
        # path is order-agnostic).
        drain_and_process(server, request_queue, nets, device, net_a)

        # Wait out of order: req_b first, req_a after.
        priors_b, values_b = client.wait_result(req_b)
        priors_a, values_a = client.wait_result(req_a)

        # Both requests were queued before processing -> one aggregated
        # pass: request A occupies rows 0..3, request B rows 4..7.
        expect_a = expected_gathered_renorm(8, legal)
        expect_b = expected_gathered_renorm(8, legal2, rows_offset=4)
        for k in range(4):
            assert np.allclose(priors_a[k].astype(np.float32),
                               expect_a[k], atol=5e-4), \
                (f"A sample {k}: got {priors_a[k][:4]} want {expect_a[k][:4]}")
            assert np.allclose(priors_b[k].astype(np.float32),
                               expect_b[k], atol=5e-4), \
                (f"B sample {k}: got {priors_b[k][:4]} want {expect_b[k][:4]}")
        for i in range(4):
            assert abs(values_a[i] - NUM_PLANES * 64 * 0.2 / 1000.0) < 5e-3
            assert abs(values_b[i] - NUM_PLANES * 64 * 0.6 / 1000.0) < 5e-3

        # Client bookkeeping must be fully consumed.
        assert not client._gather_by_req, client._gather_by_req

        transport.close()
        print("  [PASS] test_client_end_to_end_gathered_roundtrip")


if __name__ == "__main__":
    print("=" * 60)
    print("  GATHERED-POLICY PROTOCOL TESTS")
    print("=" * 60)
    t = TestGatherProtocolSHM()
    t.test_gather_only_pass()
    t.test_mixed_gather_and_full_same_pass()
    t.test_mixed_pass_absolute_rows()
    t = TestClientGatherSemantics()
    t.test_gather_rows_from_full()
    t.test_client_end_to_end_gathered_roundtrip()
    print("\n" + "=" * 60)
    print("  ALL TESTS PASSED")
    print("=" * 60)
