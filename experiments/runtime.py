"""Process-local resource limits and explicit single-GPU selection."""

import importlib.metadata
import os
import platform
import subprocess
import sys

import torch


def configure(device, threads=2, memory_fraction=0.25):
    if threads < 1 or not 0 < memory_fraction <= 1:
        raise ValueError("threads >= 1 and 0 < memory_fraction <= 1 are required")
    torch.set_num_threads(threads)
    # Do not enable reduced-precision GEMM for the reference baseline.
    torch.set_float32_matmul_precision("highest")
    if device == "cpu":
        return torch.device("cpu")
    selected = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not selected or "," in selected or selected.strip() == "-1":
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES to exactly ONE allocated GPU (index or UUID) before starting")
    if platform.system() != "Linux":
        raise RuntimeError("Run the Triton experiments on the Ubuntu server")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly one visible CUDA device")
    import triton  # noqa: F401
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(memory_fraction, device=0)
    return torch.device("cuda:0")


def metadata(device):
    info = {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
            "torch_cuda": torch.version.cuda, "device": str(device),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "threads": torch.get_num_threads()}
    info["packages"] = {}
    for package in ("triton", "geomloss", "pykeops", "ott-jax", "jax", "jaxlib"):
        try:
            info["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            info["packages"][package] = None
    try:
        info["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        info["git_dirty"] = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL).strip())
    except (OSError, subprocess.CalledProcessError):
        info["git_commit"] = None
    info["triton"] = info["packages"]["triton"]
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        free, total = torch.cuda.mem_get_info(device)
        info.update(gpu=props.name, capability=list(torch.cuda.get_device_capability(device)),
                    gpu_total_bytes=total, gpu_free_bytes_at_start=free)
    return info
