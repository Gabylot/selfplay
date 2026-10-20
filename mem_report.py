"""Where is the RAM going?  Run while training:  python mem_report.py

Groups the python/tensorboard processes by role and prints private memory
(USS: memory that would be freed if the process exited -- shared DLL pages are
not double counted) so you can see main / GPU server / workers / TensorBoard.

    pip install psutil
"""
import sys
import psutil


def role(p, parent_pids):
    try:
        cmd = " ".join(p.info["cmdline"] or [])
    except Exception:
        cmd = ""
    if "tensorboard" in cmd.lower():
        return "tensorboard"
    if "multiprocessing" in cmd or "spawn_main" in cmd:
        return "mp-child"          # self-play workers + the GPU inference server
    if p.pid in parent_pids:
        return "main"
    return "other-python"


def main():
    procs = []
    for p in psutil.process_iter(["pid", "name", "cmdline", "ppid"]):
        n = (p.info["name"] or "").lower()
        cmd = " ".join(p.info["cmdline"] or []).lower()
        if "python" in n or "tensorboard" in cmd:
            procs.append(p)
    parent_pids = {p.info["ppid"] for p in procs}

    rows = []
    for p in procs:
        try:
            mi = p.memory_full_info()
            uss = getattr(mi, "uss", mi.rss)
            rows.append((uss, mi.rss, p.pid, role(p, parent_pids)))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    rows.sort(reverse=True)

    gb = 1024 ** 3
    print(f"{'USS (GB)':>9} {'RSS (GB)':>9} {'PID':>7}  role")
    for uss, rss, pid, r in rows[:25]:
        print(f"{uss / gb:9.2f} {rss / gb:9.2f} {pid:7d}  {r}")

    by = {}
    for uss, rss, pid, r in rows:
        by.setdefault(r, []).append(uss)
    print("\nby role:")
    for r, v in sorted(by.items(), key=lambda kv: -sum(kv[1])):
        extra = f"  ({len(v)} procs, avg {sum(v) / len(v) / gb:.2f} GB, biggest {max(v) / gb:.2f} GB)" if len(v) > 1 else ""
        print(f"  {r:13s} {sum(v) / gb:6.2f} GB{extra}")
    print("  (for mp-child: the single biggest one is usually the GPU server, the rest are workers)")

    vm = psutil.virtual_memory()
    print(f"\nsystem: {vm.used / gb:.1f} / {vm.total / gb:.1f} GB used ({vm.percent:.0f}%), "
          f"{vm.available / gb:.1f} GB available")


if __name__ == "__main__":
    sys.exit(main())