import importlib.util
import os

import pytest
import torch


def pytest_addoption(parser):
    parser.addoption("--require-gpu", action="store_true", help="Fail instead of skipping if CUDA/Triton is unavailable")


def pytest_configure(config):
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    selected = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    explicit = bool(selected) and "," not in selected and selected != "-1"
    ready = explicit and importlib.util.find_spec("triton") is not None and torch.cuda.is_available()
    config._flash_gpu_ready = ready
    if config.getoption("--require-gpu") and not ready:
        raise pytest.UsageError("GPU checks require CUDA_VISIBLE_DEVICES=<one allocated GPU>, CUDA PyTorch and Triton")
    if ready:
        torch.cuda.set_per_process_memory_fraction(0.25)


def pytest_collection_modifyitems(config, items):
    if not config._flash_gpu_ready:
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(pytest.mark.skip(reason="Requires an explicitly allocated CUDA GPU and Triton"))
