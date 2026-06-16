import time
import torch
import torch_directml
from network import AlphaZeroNet

net = AlphaZeroNet(
    num_residual_blocks=6,
    num_filters=128,
    num_policy_channels=32,
    num_value_channels=32,
    value_fc_size=256,
).eval()

dml = torch_directml.device()

# CPU benchmark — network stays on CPU, input stays on CPU
x_cpu = torch.randn(128, 20, 8, 8)
t = time.perf_counter()
for _ in range(20):
    with torch.no_grad():
        net(x_cpu)
print(f"CPU 128-batch: {(time.perf_counter()-t)/20*1000:.1f}ms")

# Move network to GPU
net_gpu = net.to(dml)

# GPU warmup — input must be on same device as network
x_gpu = x_cpu.to(dml)
with torch.no_grad():
    net_gpu(x_gpu)

# GPU benchmark
t = time.perf_counter()
for _ in range(20):
    with torch.no_grad():
        net_gpu(x_gpu)
print(f"GPU 128-batch: {(time.perf_counter()-t)/20*1000:.1f}ms")

# Batch size sweep
for bs in [16, 32, 64, 128, 256, 512]:
    xb = torch.randn(bs, 20, 8, 8).to(dml)
    with torch.no_grad(): net_gpu(xb)  # warmup
    t = time.perf_counter()
    for _ in range(20):
        with torch.no_grad(): net_gpu(xb)
    print(f"GPU batch={bs}: {(time.perf_counter()-t)/20*1000:.1f}ms")

    print("\nRepeatability check:")
for bs in [64, 128, 192, 256]:
    xb = torch.randn(bs, 20, 8, 8).to(dml)
    with torch.no_grad(): net_gpu(xb)  # warmup
    times = []
    for _ in range(50):
        t = time.perf_counter()
        with torch.no_grad(): net_gpu(xb)
        times.append((time.perf_counter()-t)*1000)
    import statistics
    print(f"GPU batch={bs}: mean={statistics.mean(times):.1f}ms "
          f"min={min(times):.1f}ms max={max(times):.1f}ms "
          f"stdev={statistics.stdev(times):.1f}ms")