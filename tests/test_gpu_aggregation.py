"""Correctness tests for GPU server cross-worker batch aggregation.

Tests the highest-risk part of the aggregation change: the offset tracking
in _process_aggregated that splits a mega-batch's results back to individual
workers. An off-by-one here won't crash -- it'll silently hand worker A the
policy/value meant for worker B's state, and MCTS will happily train on it.

Strategy:
    - MockNetwork: forward(states) returns value = state.sum() / 1000.0
      so each sample's output is a deterministic function of its input.
    - Workers send distinguishable synthetic states (different fill values).
    - After aggregation, assert each worker's value matches its own states.

Usage:
    python -m pytest tests/test_gpu_aggregation.py -v
"""

import numpy as np
import torch
import torch.nn.functional as F
import queue
import multiprocessing as mp
import time
from typing import Dict, List

import sys
sys.path.insert(0, '.')

from encoding import NUM_PLANES, NUM_ACTIONS
from gpu_server import GPUInferenceServer
from shared_memory_transport import SharedMemoryTransport


# ─────────────────────────────────────────────────────────────────────────────
# Mock Network
# ─────────────────────────────────────────────────────────────────────────────

class MockNetwork(torch.nn.Module):
    """Network where value = state.sum() / 1000.0.

    Policy logits are all zeros (softmax gives uniform 1/NUM_ACTIONS).
    The value is a deterministic function of the input state, so we can
    verify each worker received the correct slice of the mega-batch results.
    """

    def __init__(self, network_id=0):
        super().__init__()
        self.network_id = network_id

    def forward(self, x: torch.Tensor):
        # x: (batch, NUM_PLANES, 8, 8)
        sums = x.sum(dim=(1, 2, 3))  # (batch,)
        # Policy: all zeros -> softmax gives uniform distribution
        policy_logits = torch.zeros(len(x), NUM_ACTIONS, device=x.device)
        # Value: sum / 1000 (keep in [-1, 1] for typical state sums)
        values = (sums / 1000.0).unsqueeze(1)  # (batch, 1)
        return policy_logits, values

    def eval(self):
        return self

    def to(self, device):
        return self


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

class MockConfig:
    """Minimal config for GPUInferenceServer."""
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
    """Create a GPUInferenceServer with mock queues and config."""
    config = MockConfig()
    config.inference.max_batch = max_batch
    config.inference.max_wait_ms = max_wait_ms

    request_queue = mp.Queue()
    response_queues = {i: mp.Queue(maxsize=256) for i in range(10)}
    weight_queue = mp.Queue()
    ready_event = mp.Event()
    shutdown_event = mp.Event()

    server = GPUInferenceServer(
        config=config,
        request_queue=request_queue,
        response_queues=response_queues,
        weight_queue=weight_queue,
        ready_event=ready_event,
        shutdown_event=shutdown_event,
        shared_buffers=shared_buffers,
    )

    return server, request_queue, response_queues


def make_states(batch_size: int, fill_value: float) -> np.ndarray:
    """Create a batch of synthetic states with a known fill value.

    The per-sample state sum = NUM_PLANES * 8 * 8 * fill_value,
    so the expected per-sample value = state_sum / 1000.0.
    """
    states = np.full((batch_size, NUM_PLANES, 8, 8), fill_value, dtype=np.float32)
    return states


def expected_value(batch_size: int, fill_value: float) -> float:
    """Compute the expected per-sample value for states with the given fill value.

    The mock network returns value = state.sum(dim=(1,2,3)) / 1000.0,
    which is a PER-SAMPLE value (sums over planes/height/width only).
    The batch_size parameter is kept for API compatibility but is not used
    in the calculation.
    """
    state_sum = NUM_PLANES * 8 * 8 * fill_value
    return state_sum / 1000.0


def drain_and_process(server, request_queue, nets, device, net_a):
    """Simulate the main loop: process the first request, then drain any
    remaining requests from the queue (as the real run() loop would)."""
    # The first request was already passed to _handle_request by the caller.
    # Now drain any remaining requests left in the queue.
    while True:
        try:
            req = request_queue.get_nowait()
            if req is None:
                break
            server._handle_request(req, nets, device, net_a)
        except queue.Empty:
            break


# ─────────────────────────────────────────────────────────────────────────────
# Tests — Queue Mode (no shared memory)
# ─────────────────────────────────────────────────────────────────────────────

class TestAggregationQueueMode:
    """Test aggregation logic in queue mode (no shared memory)."""

    def test_basic_aggregation_three_workers(self):
        """3 workers with batch sizes 32, 64, 32 -- all network_id=0.

        Total = 128 samples, fits in one mega-batch.
        Each worker uses a different fill value so we can verify splitting.
        """
        server, request_queue, response_queues = make_server(max_batch=128)
        device = torch.device("cpu")
        net_a = MockNetwork(0)
        nets = {0: net_a, 1: MockNetwork(1)}

        # Worker 0: batch of 32, fill=0.1
        # Worker 1: batch of 64, fill=0.2
        # Worker 2: batch of 32, fill=0.3
        workers = [
            (0, 32, 0.1),
            (1, 64, 0.2),
            (2, 32, 0.3),
        ]

        # Pre-populate queue with workers 1 and 2 (worker 0 is the first request)
        for wid, bs, fill in workers[1:]:
            states = make_states(bs, fill)
            req = (wid, 100 + wid, 0, states)  # (worker_id, request_id, network_id, states)
            request_queue.put(req)

        # First request (worker 0)
        first_states = make_states(workers[0][1], workers[0][2])
        first_req = (workers[0][0], 100, 0, first_states)

        # Run aggregation
        server._handle_request(first_req, nets, device, net_a)
        # Drain any remaining (in case max_batch split the batch)
        drain_and_process(server, request_queue, nets, device, net_a)

        # Verify each worker got correct results
        for wid, bs, fill in workers:
            resp = response_queues[wid].get(timeout=5.0)
            assert resp is not None, f"Worker {wid} got no response"

            # Queue mode: (request_id, policy, value) for single, or (request_id, policies, values) for batch
            if bs == 1:
                req_id, policy, value = resp
                assert req_id == 100 + wid, f"Worker {wid}: expected req_id={100+wid}, got {req_id}"
                expected_val = expected_value(1, fill)
                assert abs(value - expected_val) < 1e-4, \
                    f"Worker {wid} (fill={fill}): expected value={expected_val:.6f}, got {value:.6f}"
            else:
                req_id, policies, values = resp
                assert req_id == 100 + wid, f"Worker {wid}: expected req_id={100+wid}, got {req_id}"
                assert policies.shape == (bs, NUM_ACTIONS), \
                    f"Worker {wid}: expected policy shape ({bs}, {NUM_ACTIONS}), got {policies.shape}"
                assert values.shape == (bs,), \
                    f"Worker {wid}: expected values shape ({bs},), got {values.shape}"
                expected_val = expected_value(bs, fill)
                for i in range(bs):
                    assert abs(values[i] - expected_val) < 1e-4, \
                        f"Worker {wid} sample {i} (fill={fill}): expected value={expected_val:.6f}, got={values[i]:.6f}"

        print("  [PASS] test_basic_aggregation_three_workers")

    def test_single_request_alone(self):
        """Single request with empty queue -- should process alone after short wait."""
        server, request_queue, response_queues = make_server(max_batch=128)
        device = torch.device("cpu")
        net_a = MockNetwork(0)
        nets = {0: net_a, 1: MockNetwork(1)}

        # Single request (ndim==3) from worker 0
        states = make_states(1, 0.5)  # (1, 137, 8, 8) but ndim will be 4
        # Actually for single request, send (137, 8, 8) -- ndim==3
        single_state = states[0]  # (137, 8, 8)
        req = (0, 42, 0, single_state)
        # Queue is empty -- no other requests

        t_start = time.perf_counter()
        server._handle_request(req, nets, device, net_a)
        elapsed = time.perf_counter() - t_start

        # Should process quickly (queue empty -> break after 1ms poll)
        # On Windows, timer resolution is ~15ms so be generous
        assert elapsed < 0.1, f"Single request took too long: {elapsed*1000:.1f}ms"

        resp = response_queues[0].get(timeout=5.0)
        req_id, policy, value = resp
        assert req_id == 42
        expected_val = expected_value(1, 0.5)
        assert abs(value - expected_val) < 1e-4, \
            f"Expected value={expected_val:.6f}, got={value:.6f}"

        print(f"  [PASS] test_single_request_alone (elapsed={elapsed*1000:.1f}ms)")

    def test_mixed_single_and_batch(self):
        """Worker 0 sends single (ndim=3), worker 1 sends batch of 32 (ndim=4)."""
        server, request_queue, response_queues = make_server(max_batch=128)
        device = torch.device("cpu")
        net_a = MockNetwork(0)
        nets = {0: net_a, 1: MockNetwork(1)}

        # Worker 1: batch of 32, fill=0.2
        batch_states = make_states(32, 0.2)
        request_queue.put((1, 201, 0, batch_states))

        # Worker 0: single request, fill=0.1
        single_state = make_states(1, 0.1)[0]  # (137, 8, 8)
        first_req = (0, 200, 0, single_state)

        server._handle_request(first_req, nets, device, net_a)
        drain_and_process(server, request_queue, nets, device, net_a)

        # Worker 0: single response
        resp0 = response_queues[0].get(timeout=5.0)
        req_id0, policy0, value0 = resp0
        assert req_id0 == 200
        expected_val0 = expected_value(1, 0.1)
        assert abs(value0 - expected_val0) < 1e-4, \
            f"Worker 0: expected value={expected_val0:.6f}, got={value0:.6f}"

        # Worker 1: batch response
        resp1 = response_queues[1].get(timeout=5.0)
        req_id1, policies1, values1 = resp1
        assert req_id1 == 201
        expected_val1 = expected_value(32, 0.2)
        for i in range(32):
            assert abs(values1[i] - expected_val1) < 1e-4, \
                f"Worker 1 sample {i}: expected value={expected_val1:.6f}, got={values1[i]:.6f}"

        print("  [PASS] test_mixed_single_and_batch")

    def test_overflow_exceeds_max_batch(self):
        """Total samples exceed max_batch -- should split into multiple forward passes.

        max_batch=64, workers send 32+32+32=96 samples.
        First forward pass: 64 samples (workers 0+1).
        Second forward pass: 32 samples (worker 2).
        """
        server, request_queue, response_queues = make_server(max_batch=64)
        device = torch.device("cpu")
        net_a = MockNetwork(0)
        nets = {0: net_a, 1: MockNetwork(1)}

        # Workers 1 and 2 in queue
        for wid, fill in [(1, 0.2), (2, 0.3)]:
            states = make_states(32, fill)
            request_queue.put((wid, 300 + wid, 0, states))

        # Worker 0 is first request
        first_states = make_states(32, 0.1)
        first_req = (0, 300, 0, first_states)

        server._handle_request(first_req, nets, device, net_a)
        # Worker 2 may still be in the queue -- drain and process
        drain_and_process(server, request_queue, nets, device, net_a)

        # All workers should get correct results
        for wid, fill in [(0, 0.1), (1, 0.2), (2, 0.3)]:
            resp = response_queues[wid].get(timeout=5.0)
            req_id, policies, values = resp
            assert req_id == 300 + wid
            expected_val = expected_value(32, fill)
            for i in range(32):
                assert abs(values[i] - expected_val) < 1e-4, \
                    f"Worker {wid} sample {i} (fill={fill}): expected={expected_val:.6f}, got={values[i]:.6f}"

        print("  [PASS] test_overflow_exceeds_max_batch")

    def test_different_network_ids(self):
        """Workers 0 and 1 use different network_ids -- results must come from correct net."""
        server, request_queue, response_queues = make_server(max_batch=128)
        device = torch.device("cpu")
        net_a = MockNetwork(0)  # network_id=0: value = sum/1000
        net_b = MockNetwork(1)  # network_id=1: value = sum/1000 (same formula, but different net)
        nets = {0: net_a, 1: net_b}

        # Worker 1: network_id=1, batch of 32, fill=0.2
        batch_states = make_states(32, 0.2)
        request_queue.put((1, 401, 1, batch_states))

        # Worker 0: network_id=0, batch of 32, fill=0.1
        first_states = make_states(32, 0.1)
        first_req = (0, 400, 0, first_states)

        server._handle_request(first_req, nets, device, net_a)
        drain_and_process(server, request_queue, nets, device, net_a)

        # Both should get correct values (same formula, but processed separately)
        resp0 = response_queues[0].get(timeout=5.0)
        _, _, values0 = resp0
        expected_val0 = expected_value(32, 0.1)
        assert abs(values0[0] - expected_val0) < 1e-4, \
            f"Worker 0 (net 0): expected={expected_val0:.6f}, got={values0[0]:.6f}"

        resp1 = response_queues[1].get(timeout=5.0)
        _, _, values1 = resp1
        expected_val1 = expected_value(32, 0.2)
        assert abs(values1[0] - expected_val1) < 1e-4, \
            f"Worker 1 (net 1): expected={expected_val1:.6f}, got={values1[0]:.6f}"

        print("  [PASS] test_different_network_ids")

    def test_offset_correctness_distinguishable_states(self):
        """Critical test: each sample in a worker's batch has a unique fill value.

        This catches off-by-one errors that would mix up samples WITHIN a worker's
        batch, not just across workers.
        """
        server, request_queue, response_queues = make_server(max_batch=128)
        device = torch.device("cpu")
        net_a = MockNetwork(0)
        nets = {0: net_a, 1: MockNetwork(1)}

        # Worker 0: batch of 8, each sample has a different fill value
        # Sample i has fill = 0.01 * (i+1)
        batch_size = 8
        states = np.zeros((batch_size, NUM_PLANES, 8, 8), dtype=np.float32)
        expected_values = []
        for i in range(batch_size):
            fill = 0.01 * (i + 1)
            states[i] = fill
            expected_values.append(expected_value(1, fill))

        # Worker 1: batch of 8, fill=0.5
        states1 = make_states(8, 0.5)
        request_queue.put((1, 501, 0, states1))

        # Worker 0 first
        first_req = (0, 500, 0, states)
        server._handle_request(first_req, nets, device, net_a)
        drain_and_process(server, request_queue, nets, device, net_a)

        # Check worker 0 -- each sample must match its own fill value
        resp0 = response_queues[0].get(timeout=5.0)
        req_id, policies, values = resp0
        assert req_id == 500
        for i in range(batch_size):
            assert abs(values[i] - expected_values[i]) < 1e-4, \
                f"Worker 0 sample {i}: expected={expected_values[i]:.6f}, got={values[i]:.6f}"

        # Check worker 1
        resp1 = response_queues[1].get(timeout=5.0)
        _, _, values1 = resp1
        expected_val1 = expected_value(8, 0.5)
        for i in range(8):
            assert abs(values1[i] - expected_val1) < 1e-4, \
                f"Worker 1 sample {i}: expected={expected_val1:.6f}, got={values1[i]:.6f}"

        print("  [PASS] test_offset_correctness_distinguishable_states")


# ─────────────────────────────────────────────────────────────────────────────
# Tests — Shared Memory Mode
# ─────────────────────────────────────────────────────────────────────────────

class TestAggregationSharedMemoryMode:
    """Test aggregation logic with shared-memory transport."""

    def test_basic_aggregation_shm(self):
        """3 workers with shared-memory buffers, batch sizes 32, 64, 32."""
        num_workers = 3
        max_batch = 128
        transport = SharedMemoryTransport(num_workers=num_workers, max_batch=max_batch)
        transport.create_buffers()
        shared_buffers = transport.get_all_worker_buffers()

        server, request_queue, response_queues = make_server(
            max_batch=max_batch, shared_buffers=shared_buffers
        )
        device = torch.device("cpu")
        net_a = MockNetwork(0)
        nets = {0: net_a, 1: MockNetwork(1)}

        workers = [
            (0, 32, 0.1),
            (1, 64, 0.2),
            (2, 32, 0.3),
        ]

        # Write states to shared memory and put metadata in queue
        for wid, bs, fill in workers[1:]:
            states = make_states(bs, fill)
            buf = shared_buffers[wid]
            SharedMemoryTransport.write_states(buf, states)
            request_queue.put((wid, 100 + wid, 0, bs))  # batch_size as int

        # First request (worker 0)
        first_states = make_states(workers[0][1], workers[0][2])
        first_buf = shared_buffers[workers[0][0]]
        SharedMemoryTransport.write_states(first_buf, first_states)
        first_req = (workers[0][0], 100, 0, workers[0][1])  # batch_size as int

        server._handle_request(first_req, nets, device, net_a)
        drain_and_process(server, request_queue, nets, device, net_a)

        # Verify each worker got correct results from shared memory
        for wid, bs, fill in workers:
            resp = response_queues[wid].get(timeout=5.0)
            assert resp is not None, f"Worker {wid} got no response"

            # SHM mode: (request_id, batch_size)
            req_id, batch_size = resp
            assert req_id == 100 + wid, f"Worker {wid}: expected req_id={100+wid}, got {req_id}"
            assert batch_size == bs, f"Worker {wid}: expected batch_size={bs}, got {batch_size}"

            # Read from shared memory
            buf = shared_buffers[wid]
            policies = SharedMemoryTransport.read_policies(buf, bs)
            values = SharedMemoryTransport.read_values(buf, bs)

            assert policies.shape == (bs, NUM_ACTIONS), \
                f"Worker {wid}: expected policy shape ({bs}, {NUM_ACTIONS}), got {policies.shape}"
            assert values.shape == (bs,), \
                f"Worker {wid}: expected values shape ({bs},), got {values.shape}"

            expected_val = expected_value(bs, fill)
            for i in range(bs):
                assert abs(values[i] - expected_val) < 5e-3, \
                    f"Worker {wid} sample {i} (fill={fill}): expected={expected_val:.6f}, got={values[i]:.6f}"

        print("  [PASS] test_basic_aggregation_shm")

    def test_shm_offset_correctness(self):
        """SHM mode: each sample in a worker's batch has a unique fill value."""
        num_workers = 2
        max_batch = 128
        transport = SharedMemoryTransport(num_workers=num_workers, max_batch=max_batch)
        transport.create_buffers()
        shared_buffers = transport.get_all_worker_buffers()

        server, request_queue, response_queues = make_server(
            max_batch=max_batch, shared_buffers=shared_buffers
        )
        device = torch.device("cpu")
        net_a = MockNetwork(0)
        nets = {0: net_a, 1: MockNetwork(1)}

        # Worker 0: batch of 8, each sample has different fill
        batch_size = 8
        states = np.zeros((batch_size, NUM_PLANES, 8, 8), dtype=np.float32)
        expected_values = []
        for i in range(batch_size):
            fill = 0.01 * (i + 1)
            states[i] = fill
            expected_values.append(expected_value(1, fill))

        first_buf = shared_buffers[0]
        SharedMemoryTransport.write_states(first_buf, states)
        first_req = (0, 600, 0, batch_size)

        # Worker 1: batch of 8, fill=0.5
        states1 = make_states(8, 0.5)
        buf1 = shared_buffers[1]
        SharedMemoryTransport.write_states(buf1, states1)
        request_queue.put((1, 601, 0, 8))

        server._handle_request(first_req, nets, device, net_a)
        drain_and_process(server, request_queue, nets, device, net_a)

        # Check worker 0
        resp0 = response_queues[0].get(timeout=5.0)
        req_id0, bs0 = resp0
        assert req_id0 == 600
        assert bs0 == batch_size

        policies0 = SharedMemoryTransport.read_policies(first_buf, batch_size)
        values0 = SharedMemoryTransport.read_values(first_buf, batch_size)
        for i in range(batch_size):
            assert abs(values0[i] - expected_values[i]) < 5e-3, \
                f"Worker 0 sample {i}: expected={expected_values[i]:.6f}, got={values0[i]:.6f}"

        # Check worker 1
        resp1 = response_queues[1].get(timeout=5.0)
        req_id1, bs1 = resp1
        assert req_id1 == 601
        assert bs1 == 8

        values1 = SharedMemoryTransport.read_values(buf1, 8)
        expected_val1 = expected_value(8, 0.5)
        for i in range(8):
            assert abs(values1[i] - expected_val1) < 5e-3, \
                f"Worker 1 sample {i}: expected={expected_val1:.6f}, got={values1[i]:.6f}"

        print("  [PASS] test_shm_offset_correctness")


# ─────────────────────────────────────────────────────────────────────────────
# Tests — Wait Time (short-poll pattern)
# ─────────────────────────────────────────────────────────────────────────────

class TestWaitTime:
    """Verify that the short-poll pattern doesn't add unnecessary latency
    when the queue is empty (single-worker scenario)."""

    def test_single_worker_wait_time_near_zero(self):
        """When only one worker is active, the aggregation wait should be
        near-zero (get_nowait returns immediately on empty queue).

        The old get(timeout=0.001) took ~15ms on Windows due to timer
        resolution.  The new get_nowait() approach should be < 5ms.
        """
        max_wait_ms = 50.0  # 50ms max wait
        server, request_queue, response_queues = make_server(
            max_batch=128, max_wait_ms=max_wait_ms
        )
        device = torch.device("cpu")
        net_a = MockNetwork(0)
        nets = {0: net_a, 1: MockNetwork(1)}

        # Single batch request, queue is empty
        states = make_states(32, 0.1)
        req = (0, 700, 0, states)

        t_start = time.perf_counter()
        server._handle_request(req, nets, device, net_a)
        elapsed = time.perf_counter() - t_start

        # get_nowait() should return immediately on empty queue.
        # Assert < 5ms (generous for CI/slow machines).
        # Old get(timeout=0.001) took ~15ms on Windows.
        assert elapsed < 0.005, \
            f"Single worker wait took {elapsed*1000:.1f}ms " \
            f"(max_wait_ms={max_wait_ms}ms) -- " \
            f"get_nowait() should return immediately on empty queue"

        # Verify result is correct
        resp = response_queues[0].get(timeout=5.0)
        _, _, values = resp
        expected_val = expected_value(32, 0.1)
        assert abs(values[0] - expected_val) < 1e-4

        print(f"  [PASS] test_single_worker_wait_time_near_zero (wait={elapsed*1000:.1f}ms, max_wait={max_wait_ms}ms)")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  GPU AGGREGATION CORRECTNESS TESTS")
    print("=" * 60)

    # Queue mode tests
    print("\n--- Queue Mode Tests ---")
    t = TestAggregationQueueMode()
    t.test_basic_aggregation_three_workers()
    t.test_single_request_alone()
    t.test_mixed_single_and_batch()
    t.test_overflow_exceeds_max_batch()
    t.test_different_network_ids()
    t.test_offset_correctness_distinguishable_states()

    # Shared memory mode tests
    print("\n--- Shared Memory Mode Tests ---")
    t = TestAggregationSharedMemoryMode()
    t.test_basic_aggregation_shm()
    t.test_shm_offset_correctness()

    # Wait time tests
    print("\n--- Wait Time Tests ---")
    t = TestWaitTime()
    t.test_single_worker_wait_time_near_zero()

    print("\n" + "=" * 60)
    print("  ALL TESTS PASSED")
    print("=" * 60)