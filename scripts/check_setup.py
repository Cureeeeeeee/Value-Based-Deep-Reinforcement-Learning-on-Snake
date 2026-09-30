"""M0 acceptance check: print library versions and verify CUDA on the GPU.

Usage: python scripts/check_setup.py
Exit code 1 if torch cannot see a CUDA device (the throughput plan assumes it).
"""

from __future__ import annotations

import platform
import sys


def main() -> int:
    import gymnasium
    import numpy
    import torch

    print(f"python     {platform.python_version()} ({sys.executable})")
    print(f"numpy      {numpy.__version__}")
    print(f"gymnasium  {gymnasium.__version__}")
    print(f"torch      {torch.__version__}")
    ok = torch.cuda.is_available()
    print(f"cuda       available={ok}")
    if not ok:
        print("FAIL: torch.cuda.is_available() is False; install the cu128 wheel (see README).")
        return 1
    dev = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"device     {dev} sm_{cap[0]}{cap[1]}")
    print(f"arch list  {torch.cuda.get_arch_list()}")
    x = torch.randn(2048, 2048, device="cuda")
    y = (x @ x).sum().item()
    print(f"matmul     ok ({y:.3e})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
