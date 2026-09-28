"""Confirm PyTorch sees the AMD GPU through ROCm and can run a kernel.

Run with ``uv run --env-file .env python scripts/check_gpu.py``. Exits 1
when no GPU is usable so it can gate a container or k8s Job.
"""

import os
import sys

import torch  # pyright: ignore[reportMissingImports] - gpu group, absent in CI

# Big enough to run a real GPU kernel, small enough for any card's memory.
MATRIX_SIZE = 2048


def main() -> int:
    """Print the ROCm/PyTorch view of the GPUs and run a matmul on device 0.

    Returns:
        Process exit code: 0 when a GPU ran the kernel, 1 otherwise.
    """
    print(f"torch {torch.__version__}, HIP {torch.version.hip}")
    print(f"HIP_VISIBLE_DEVICES={os.environ.get('HIP_VISIBLE_DEVICES', '<unset>')}")

    if torch.version.hip is None:
        print("This torch build has no ROCm support; check the pytorch-rocm index.")
        return 1
    if not torch.cuda.is_available():  # the cuda API is backed by HIP on ROCm
        print("No GPU visible. Check /dev/kfd, /dev/dri and the video/render groups.")
        return 1

    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        print(f"  [{index}] {props.name} ({props.gcnArchName})")
    if torch.cuda.device_count() > 1:
        print("More than one GPU visible; set HIP_VISIBLE_DEVICES in .env.")

    a = torch.randn(MATRIX_SIZE, MATRIX_SIZE, device="cuda")
    checksum = (a @ a).sum().item()
    torch.cuda.synchronize()
    print(f"matmul OK on {torch.cuda.get_device_name(0)} (checksum {checksum:.1f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
