"""Quick functional test for shared-memory IPC transport."""

import multiprocessing as mp
import numpy as np
import queue

from shared_memory_transport import SharedMemoryTransport, WorkerSharedBuffers
from inference_client import InferenceClient
from encoding import NUM_PLANES, NUM_ACTIONS


def mock_server(request_q, response_qs, shared_bufs, shutdown_event, ready_event):
    """Mock GPU server that reads states from shared memory and writes results back."""
    from shared_memory_transport import SharedMemoryTransport

    ready_event.set()

    while not shutdown_event.is_set():
        try:
            req = request_q.get(timeout=0.5)
        except queue.Empty:
            continue

        if req is None:
            for wq in response_qs.values():
                try: wq.put_nowait(None)
                except: pass
            return

        worker_id, request_id, network_id, fourth = req

        if isinstance(fourth, int):
            batch_size = fourth
            buf = shared_bufs[worker_id]
            states = SharedMemoryTransport.read_states(buf, batch_size)
            n = batch_size
            policies = np.ones((n, NUM_ACTIONS), dtype=np.float32) / NUM_ACTIONS
            values = np.mean(states.reshape(n, -1), axis=1).astype(np.float32)
            SharedMemoryTransport.write_policies_values(buf, policies, values)
            response_qs[worker_id].put_nowait((request_id, n))
        else:
            states = fourth
            if states.ndim == 4:
                n = len(states)
                policies = np.ones((n, NUM_ACTIONS), dtype=np.float32) / NUM_ACTIONS
                values = np.mean(states.reshape(n, -1), axis=1).astype(np.float32)
                response_qs[worker_id].put_nowait((request_id, policies, values))
            else:
                policy = np.ones(NUM_ACTIONS, dtype=np.float32) / NUM_ACTIONS
                value = float(np.mean(states))
                response_qs[worker_id].put_nowait((request_id, policy, value))


def test_shared_memory_roundtrip():
    print("Test 1: Single + batch predict via shared memory...")
    transport = SharedMemoryTransport(num_workers=1, max_batch=16, state_dtype=np.float16)
    transport.create_buffers()
    shared_bufs = transport.get_all_worker_buffers()
    worker_buf = transport.get_worker_buffers(0)

    request_q = mp.Queue()
    response_q = mp.Queue(maxsize=256)
    shutdown_event = mp.Event()
    ready_event = mp.Event()

    server_proc = mp.Process(
        target=mock_server,
        args=(request_q, {0: response_q}, shared_bufs, shutdown_event, ready_event),
        daemon=True,
    )
    server_proc.start()
    ready_event.wait()

    client = InferenceClient(
        worker_id=0, request_queue=request_q, response_queue=response_q,
        network_id=0, shared_buffers=worker_buf,
    )

    state = np.random.rand(NUM_PLANES, 8, 8).astype(np.float32)
    policy, value = client.predict(state)
    assert policy.shape == (NUM_ACTIONS,), f"Expected ({NUM_ACTIONS},), got {policy.shape}"
    assert isinstance(value, float)
    print(f"  predict() OK: policy {policy.shape}, value {value:.4f}")

    states = np.random.rand(8, NUM_PLANES, 8, 8).astype(np.float32)
    policies, values = client.predict_batch(states)
    assert policies.shape == (8, NUM_ACTIONS)
    assert values.shape == (8,)
    print(f"  predict_batch() OK: {policies.shape}, {values.shape}")

    shutdown_event.set()
    request_q.put(None)
    server_proc.join(timeout=5)
    if server_proc.is_alive(): server_proc.kill()
    print("  PASSED\n")


def test_queue_fallback():
    print("Test 2: Queue fallback (no shared buffers)...")
    request_q = mp.Queue()
    response_q = mp.Queue(maxsize=256)
    shutdown_event = mp.Event()
    ready_event = mp.Event()

    server_proc = mp.Process(
        target=mock_server,
        args=(request_q, {0: response_q}, None, shutdown_event, ready_event),
        daemon=True,
    )
    server_proc.start()
    ready_event.wait()

    client = InferenceClient(
        worker_id=0, request_queue=request_q, response_queue=response_q,
        network_id=0, shared_buffers=None,
    )

    state = np.random.rand(NUM_PLANES, 8, 8).astype(np.float32)
    policy, value = client.predict(state)
    assert policy.shape == (NUM_ACTIONS,)
    print(f"  predict() queue fallback OK")

    states = np.random.rand(4, NUM_PLANES, 8, 8).astype(np.float32)
    policies, values = client.predict_batch(states)
    assert policies.shape == (4, NUM_ACTIONS)
    assert values.shape == (4,)
    print(f"  predict_batch() queue fallback OK")

    shutdown_event.set()
    request_q.put(None)
    server_proc.join(timeout=5)
    if server_proc.is_alive(): server_proc.kill()
    print("  PASSED\n")


def test_float16_precision():
    print("Test 3: Float16 precision for binary board data...")
    transport = SharedMemoryTransport(num_workers=1, max_batch=4, state_dtype=np.float16)
    transport.create_buffers()
    buf = transport.get_worker_buffers(0)

    states = np.zeros((4, NUM_PLANES, 8, 8), dtype=np.float32)
    states[0, 0, 3, 4] = 1.0
    states[0, 12, :, :] = 1.0
    states[0, 136, :, :] = 0.15
    states[1, 5, 0, 0] = 1.0
    states[3, 136, :, :] = 0.99

    SharedMemoryTransport.write_states(buf, states)
    recovered = SharedMemoryTransport.read_states(buf, 4)

    assert np.allclose(states[0, 0, 3, 4], recovered[0, 0, 3, 4])
    assert abs(states[0, 136, 0, 0] - recovered[0, 136, 0, 0]) < 0.01
    assert abs(states[3, 136, 0, 0] - recovered[3, 136, 0, 0]) < 0.01
    print(f"  Binary values exact, normalized within float16 precision")
    print(f"  Max diff: {np.max(np.abs(states - recovered)):.6f}")
    print("  PASSED\n")


def test_buffer_sizing():
    print("Test 4: Buffer capacity and overflow...")
    transport = SharedMemoryTransport(num_workers=1, max_batch=8, state_dtype=np.float16)
    transport.create_buffers()
    buf = transport.get_worker_buffers(0)

    states = np.random.rand(8, NUM_PLANES, 8, 8).astype(np.float32)
    SharedMemoryTransport.write_states(buf, states)
    recovered = SharedMemoryTransport.read_states(buf, 8)
    assert recovered.shape == (8, NUM_PLANES, 8, 8)
    print(f"  max_batch=8 OK")

    try:
        overflow = np.random.rand(9, NUM_PLANES, 8, 8).astype(np.float32)
        SharedMemoryTransport.write_states(buf, overflow)
        assert False, "Should have raised"
    except ValueError as e:
        print(f"  Overflow raises ValueError OK")

    print("  PASSED\n")


if __name__ == "__main__":
    print("=" * 60)
    print("  Shared-Memory IPC Transport Tests")
    print("=" * 60 + "\n")

    test_float16_precision()
    test_buffer_sizing()
    test_shared_memory_roundtrip()
    test_queue_fallback()

    print("=" * 60)
    print("  ALL TESTS PASSED")
    print("=" * 60)