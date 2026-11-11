"""Host speed probe (run in the SAME allocation / affinity as the benchmark, right before and right after it).
Prints one JSON line: fixed CPU microbenchmarks + the CPU clocks and affinity seen by this process."""
import json, os, socket, sys, time
import numpy as np

t0 = time.perf_counter()
s = 0
for i in range(20_000_000):          # pure-python loop: single-thread interpreter speed (the kit's host-side launch path)
    s += i
loop_s = time.perf_counter() - t0

def matmul(threads):
    os.environ["OMP_NUM_THREADS"] = str(threads)
    a = np.random.default_rng(0).standard_normal((4096, 4096), dtype=np.float32)
    t = time.perf_counter(); a @ a; return time.perf_counter() - t

mm_all = min(matmul(os.cpu_count() or 1) for _ in range(2))    # BLAS threads as the process finds them
x = np.ones(512 * 1024 * 1024 // 4 * 1, dtype=np.float32)       # 2 GiB source
y = np.empty_like(x)
t = time.perf_counter(); np.copyto(y, x); copy_s = time.perf_counter() - t
aff = sorted(os.sched_getaffinity(0))
mhz = []
try:
    for line in open("/proc/cpuinfo"):
        if line.startswith("cpu MHz"):
            mhz.append(float(line.split(":")[1]))
except OSError:
    pass
print(json.dumps({"host": socket.gethostname(), "when": sys.argv[1] if len(sys.argv) > 1 else "", "ts": time.strftime("%H:%M:%S"),
                  "loop_2e7_s": round(loop_s, 3), "matmul4096_f32_s": round(mm_all, 4),
                  "copy_2GiB_s": round(copy_s, 3), "copy_GBps": round(2.147 / copy_s, 2),
                  "n_cpus_allowed": len(aff), "cpu_mhz_mean_allowed": round(sum(mhz[i] for i in aff if i < len(mhz)) / max(1, len([i for i in aff if i < len(mhz)])), 0) if mhz else None,
                  "load1": os.getloadavg()[0]}))
