"""Correctness test for client-side request pipelining (concurrency > 1).

Uses the same MockNetwork (value = sum/1000) and a mock-server pattern so a
client can issue multiple in-flight requests across distinct shared-buffer
slots and receive the right result for each, even with out-of-order drains.

Usage:
    <python> tests/test_client_pipelining.py
"""

import numpy as np
import torch
import torch.nn.functional as F
import multiprocessing as mp

import sys
sys.path.insert(0, '.')

from encoding import NUM_PLANES, NUM_ACTIONS
from gpu_server import GPUInferenceServer
from inference_client import InferenceClient
from shared_memory_transport import SharedMemoryTransport


class MockNetwork(torch.nn.Module):
    def forward(self, x):
        sums = x.sum(dim=(1, 2, 3))
        policy_logits = torch.zeros(len(x), NUM_ACTIONS, device=x.device)
        values = (sums / 1000.0).unsqueeze(1)
        return policy_logits, values
    def eval(self):
        return self
    def to(self, device):
        return self


class MockConfig:
    class inference:
        max_batch = 512
        max_wait_ms = 50.0
        prewarm_batch_sizes = [1, 8, 16, 32, 64, 128]
    class network:
        num_residual_blocks = 4
        num_filters = 64
        num_policy_channels = 64
        num_value_channels = 64
        value_fc_size = 128


def make_states(batch_size, fill):
    return np.full((batch_size, NUM_PLANES, 8, 8), fill, dtype=np.float32)


def expected_value(fill):
    return (NUM_PLANES * 8 * 8 * fill) / 1000.0


def test_pipelined_slots():
    """One client issues concurrency multiple batch requests; server processes;
    client must recover each worker=client result correctly across slots."""
    num_workers = 2
    max_batch = 512
    concurrency = 4

    transport = SharedMemoryTransport(
        num_workers=num_workers, max_batch=max_batch,
        state_dtype=np.float32, num_slots=concurrency,
    )
    transport.create_buffers()

    request_queue = mp.Queue()
    response_queues = {i: mp.Queue(maxsize=16) for i in range(num_workers)}
    server = GPUInferenceServer(
        config=MockConfig(),
        request_queue=request_queue,
        response_queues=response_queues,
        weight_queue=mp.Queue(),
        ready_event=mp.Event(),
        shutdown_event=mp.Event(),
        shared_buffers=transport.get_all_worker_buffers(),
    )
    device = torch.device("cpu")
    net_a = MockNetwork()
    nets = {0: net_a}

    client_a = InferenceClient(
        0, request_queue, response_queues[0], network_id=0,
        shared_buffers=transport.get_worker_buffers(0), concurrency=concurrency,
    )
    client_b = InferenceClient(
        1, request_queue, response_queues[1], network_id=0,
        shared_buffers=transport.get_worker_buffers(1), concurrency=concurrency,
    )

    # Fire `concurrency` requests on each worker; each with a distinct fill.
    ids_a, ids_b = [], []
    fills_a, fills_b = [], []
    for i in range(concurrency):
        bs = 8
        fa = 0.01 * (i + 1)
        fb = 0.05 * (i + 1)
        fills_a.append(fa); fills_b.append(fb)
        ids_a.append(client_a.predict_batch_async(make_states(bs, fa)))
        ids_b.append(client_b.predict_batch_async(make_states(bs, fb)))

    # Now drain them (server aggregates via _handle_request synchronously).
    # Simulate the request queue feeding the server (as run() would).
    # Requests were already put by the clients above via predict_batch_async.
    while True:
        try:
            req = request_queue.get_nowait()
            if req is None:
                break
            server._handle_request(req, nets, device, net_a)
        except Exception as e:
            if isinstance(e, __import__('queue').Empty):
                break
            raise

    # Verify each client's drain returns its own fills in request order
    for i in range(concurrency):
        pol_a, val_a = client_a.wait_result(ids_a[i])
        assert val_a.shape == (8,) and np.allclose(val_a, expected_value(fills_a[i]), atol=1e-4), \
            f"A[{i}] val={val_a} exp={expected_value(fills_a[i])}"
        pol_b, val_b = client_b.wait_result(ids_b[i])
        assert val_b.shape == (8,) and np.allclose(val_b, expected_value(fills_b[i]), atol=1e-4), \
            f"B[{i}] val={val_b} exp={expected_value(fills_b[i])}"

    print(f"  [PASS] pipelined concurrency={concurrency} x {num_workers} workers")


if __name__ == "__main__":
    from queue import Empty
    print("=" * 60)
    print("  CLIENT PIPELINING (SLOTS) TEST")
    print("=" * 60 + "\n")
    test_pipelined_slots()
    print("\n" + "=" * 60)
    print("  ALL TESTS PASSED")
    print("=" * 60)