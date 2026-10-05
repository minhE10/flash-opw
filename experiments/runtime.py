"""Process-local resource limits and explicit single-GPU selection."""

import importlib.metadata
import hashlib
import os
import platform
import subprocess
import sys
from pathlib import Path

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
    info["kernel_controls"] = {
        "autotune": os.environ.get("FLASHOPW_AUTOTUNE", "0"),
        "vector_kernel": os.environ.get("FLASHOPW_VECTOR_KERNEL", "1"),
        "gradient_kernel": os.environ.get("FLASHOPW_GRADIENT_KERNEL", "1"),
    }
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
    root = Path(__file__).resolve().parents[1]
    info["source_sha256"] = {}
    for relative in ("flashopw/triton_kernels.py", "flashopw/kernel_tuning.py",
                     "flashopw/transport.py", "flashopw/differentiation.py",
                     "experiments/paper_benchmarks.py", "experiments/ott_hessian.py"):
        path = root / relative
        if path.is_file():
            content = path.read_bytes().replace(b"\r\n", b"\n")
            info["source_sha256"][relative] = hashlib.sha256(content).hexdigest()
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        free, total = torch.cuda.mem_get_info(device)
        info.update(gpu=props.name, capability=list(torch.cuda.get_device_capability(device)),
                    gpu_total_bytes=total, gpu_free_bytes_at_start=free)
        try:
            info["nvidia_smi_at_start"] = subprocess.check_output(
                ["nvidia-smi", "-i", os.environ["CUDA_VISIBLE_DEVICES"],
                 "--query-gpu=index,uuid,name,driver_version,temperature.gpu,clocks.sm,clocks.mem,power.draw,memory.total,memory.used",
                 "--format=csv"], text=True, stderr=subprocess.DEVNULL, timeout=5).strip()
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            info["nvidia_smi_at_start"] = None
    return info
