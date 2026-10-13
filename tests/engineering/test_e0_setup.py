"""Gate E0, environment verification.

Defined in docs/handoff-setup-config-and-api-rev2.md, section 1, "Gate zero". Everything downstream assumes
a working GPU pipeline, and JAX falls back to CPU silently, so this runs before build step one and again
after any change to the driver, the CUDA stack, or the JAX pin.

Run as a script for the full report:

    python tests/engineering/test_e0_setup.py

Run under pytest for the pass or fail:

    pytest tests/engineering/test_e0_setup.py -q

The preallocation environment variable is set at import time, before the first `import jax`, because JAX
claims 75 percent of device memory on its first operation and setting the variable afterwards has no
effect. This module must therefore never be imported after JAX. That ordering is the reason the assignment
sits at module top level rather than inside a fixture.
"""

import os
import subprocess  # noqa: E402
import time  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

# The handoff's gate-zero matmul. 8192 squared float32 is about 268 MB per operand, so the operation
# allocates roughly 800 MB across the two inputs and the output. If this OOMs, preallocation or the display
# is the cause rather than the code.
MATMUL_N = 8192

# The GPU must beat the CPU by enough that a silent CPU fallback is impossible to mistake for success. A
# fallback would give a ratio near 1.0, so any threshold materially above 1 separates the two cases; 3.0 is
# far below the expected ratio for this operation and well above the noise.
MIN_SPEEDUP = 3.0


def _nvidia_smi():
    """Driver version and memory, or None if nvidia-smi is absent."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,memory.used,memory.free",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    name, driver, total, used, free = (f.strip() for f in out.stdout.strip().split(","))
    return {"name": name, "driver": driver, "total": total, "used": used, "free": free}


def _time_matmul(x, n_repeat=3):
    """Median wall-time of a matmul, excluding compile and transfer.

    The first call is discarded because it pays compilation, and CLAUDE.md section 11 requires compile time
    to be measured separately or the first call poisons the measurement. `block_until_ready` is required
    because JAX dispatch is asynchronous and without it this would time the dispatch rather than the work.
    """
    (x @ x).block_until_ready()
    times = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        (x @ x).block_until_ready()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def gpu_devices():
    return [d for d in jax.devices() if d.platform == "gpu"]


# ---------------------------------------------------------------------------
# The checks, one per gate-zero line.
# ---------------------------------------------------------------------------


def test_preallocation_disabled():
    assert os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE") == "false", (
        "XLA_PYTHON_CLIENT_PREALLOCATE must be false before the first jax import. With preallocation on, "
        "measurement M2 reports the same number for every configuration and is meaningless."
    )


def test_gpu_device_present():
    devs = jax.devices()
    assert gpu_devices(), (
        f"jax.devices() reports {devs}, with no CudaDevice. JAX has fallen back to CPU. "
        "Everything downstream assumes a GPU pipeline."
    )


@pytest.mark.slow  # 6.67s measured. Over the five-second rule, CLAUDE.md section 10
def test_matmul_runs_and_beats_cpu():
    assert gpu_devices(), "no GPU, so the comparison is not meaningful"

    key = jax.random.key(0)
    x_gpu = jax.random.normal(key, (MATMUL_N, MATMUL_N), dtype=jnp.float32)
    gpu_s = _time_matmul(x_gpu)

    cpu = jax.devices("cpu")[0]
    x_cpu = jax.device_put(x_gpu, cpu)
    cpu_s = _time_matmul(x_cpu, n_repeat=1)

    speedup = cpu_s / gpu_s
    assert speedup >= MIN_SPEEDUP, (
        f"GPU matmul only {speedup:.1f}x faster than CPU ({gpu_s * 1e3:.1f} ms vs {cpu_s * 1e3:.1f} ms). "
        "A ratio near 1 means the GPU device is a CPU fallback in disguise."
    )


def test_repository_not_on_windows_mount():
    """The handoff requires the repository to live in the WSL2 filesystem, not under /mnt/c.

    The Windows-mounted path is roughly three to five times slower for file operations. This is a
    measurement-integrity check, not a correctness one: it affects checkpoint writes, the sampled-trajectory
    dump, and the NumPy reference runs rather than the GPU-resident rollout.
    """
    root = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
    assert not root.startswith("/mnt/"), (
        f"repository is at {root}, on a Windows mount. See DEV-003 and Q-005 in the record files."
    )


def test_device_memory_statistics_available():
    """M2 must be measurable without nvidia-smi, which is useless while preallocation is on.

    Recording which method was used is part of gate zero, because the two are not comparable.
    """
    assert gpu_devices(), "no GPU, so device memory statistics do not apply"
    stats = gpu_devices()[0].memory_stats()
    assert stats is not None and "bytes_in_use" in stats, (
        "device.memory_stats() did not report bytes_in_use, so M2 has no measurement method other than "
        "nvidia-smi, which is constant under preallocation and therefore meaningless."
    )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def main():
    print("=" * 72)
    print("GATE E0, environment verification")
    print("=" * 72)

    print(f"\nXLA_PYTHON_CLIENT_PREALLOCATE   {os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE')!r}")
    print(f"jax version                     {jax.__version__}")
    print(f"jax.devices()                   {jax.devices()}")
    print(f"default backend                 {jax.default_backend()}")

    smi = _nvidia_smi()
    if smi:
        print(f"\ngpu                             {smi['name']}")
        print(f"driver                          {smi['driver']}")
        print(f"vram total                      {smi['total']}")
        print(f"vram used, desktop running      {smi['used']}")
        print(f"vram FREE, desktop running      {smi['free']}   <- what you actually have")
    else:
        print("\nnvidia-smi                      not available")

    root = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
    print(f"\nrepository path                 {root}")
    print(f"  on windows mount              {'YES, gate line fails' if root.startswith('/mnt/') else 'no'}")

    if gpu_devices():
        dev = gpu_devices()[0]
        key = jax.random.key(0)
        x = jax.random.normal(key, (MATMUL_N, MATMUL_N), dtype=jnp.float32)

        t0 = time.perf_counter()
        (x @ x).block_until_ready()
        compile_s = time.perf_counter() - t0

        gpu_s = _time_matmul(x)
        cpu_s = _time_matmul(jax.device_put(x, jax.devices("cpu")[0]), n_repeat=1)
        flops = 2 * MATMUL_N**3

        print(f"\nmatmul {MATMUL_N}^2 float32")
        print(f"  first call, incl. compile    {compile_s * 1e3:8.1f} ms")
        print(f"  gpu, median of 3             {gpu_s * 1e3:8.1f} ms   {flops / gpu_s / 1e12:.2f} TFLOP/s")
        print(f"  cpu                          {cpu_s * 1e3:8.1f} ms   {flops / cpu_s / 1e12:.2f} TFLOP/s")
        print(f"  speedup                      {cpu_s / gpu_s:8.1f}x")

        stats = dev.memory_stats() or {}
        print("\nmemory measurement method       jax device.memory_stats(), preallocation disabled")
        print(f"  bytes_in_use                 {stats.get('bytes_in_use', 0) / 1e6:8.1f} MB")
        print(f"  peak_bytes_in_use            {stats.get('peak_bytes_in_use', 0) / 1e6:8.1f} MB")
        print(f"  bytes_limit                  {stats.get('bytes_limit', 0) / 1e6:8.1f} MB")
    else:
        print("\nNO GPU DEVICE. Everything below this line is unmeasurable.")

    print("\n" + "=" * 72)


if __name__ == "__main__":
    main()
