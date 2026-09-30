"""M0 environment check. Numbers printed here go into the reproducibility
section of the final report.

Usage: python scripts/check_env.py
Prints Python / torch / CUDA / device / compute capability, times a 1024x1024
matmul on the GPU, and imports numpy, gymnasium and pygame. Exit code 1 if
any item fails (in particular if torch cannot see a CUDA device).
"""

from __future__ import annotations

import os
import platform
import sys
import time

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

MATMUL_N = 1024
MATMUL_ITERS = 200


def main() -> int:
    failures: list[str] = []
    print(f"python      {platform.python_version()}  {sys.executable}")
    print(f"platform    {platform.platform()}")

    for name in ("numpy", "gymnasium", "pygame"):
        try:
            mod = __import__(name)
            print(f"{name:<11} {mod.__version__}")
        except Exception as exc:  # noqa: BLE001 - report every failure
            failures.append(f"import {name}: {exc!r}")
            print(f"{name:<11} FAILED ({exc!r})")

    try:
        import torch

        print(f"torch       {torch.__version__}")
        cuda = torch.cuda.is_available()
        print(f"cuda        available={cuda}  torch.version.cuda={torch.version.cuda}")
        if not cuda:
            failures.append("torch.cuda.is_available() is False (install the cu128 wheel, see README)")
        else:
            cap = torch.cuda.get_device_capability(0)
            print(f"device      {torch.cuda.get_device_name(0)}  capability={cap}  sm_{cap[0]}{cap[1]}")
            print(f"arch list   {torch.cuda.get_arch_list()}")
            x = torch.randn(MATMUL_N, MATMUL_N, device="cuda")
            for _ in range(10):  # warm-up (kernel selection, allocator)
                y = x @ x
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(MATMUL_ITERS):
                y = x @ x
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / MATMUL_ITERS
            tflops = 2 * MATMUL_N**3 / dt / 1e12
            print(f"matmul      {MATMUL_N}x{MATMUL_N} fp32: {dt * 1e3:.3f} ms/op  ({tflops:.1f} TFLOP/s)  checksum={y.sum().item():.3e}")
    except Exception as exc:  # noqa: BLE001
        failures.append(f"torch: {exc!r}")
        print(f"torch       FAILED ({exc!r})")

    if failures:
        print("FAIL")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
