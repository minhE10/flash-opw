"""Resource gating can be checked without a CUDA driver or Triton installation."""

import sys
from types import ModuleType, SimpleNamespace

import torch

from flashopw import kernel_tuning


def test_autotuner_does_not_launch_overbudget_candidates(monkeypatch):
    triton = ModuleType("triton")
    triton.__path__ = []
    testing = ModuleType("triton.testing")
    testing.do_bench = lambda fn, **kwargs: fn()
    triton.testing = testing
    compiler = ModuleType("triton.compiler")
    compiler.__path__ = []
    compiler_errors = ModuleType("triton.compiler.errors")
    compiler_errors.CompilationError = type("CompilationError", (Exception,), {})
    runtime = ModuleType("triton.runtime")
    runtime.__path__ = []
    runtime_errors = ModuleType("triton.runtime.errors")
    runtime_errors.OutOfResources = type("OutOfResources", (Exception,), {})
    for name, module in (("triton", triton), ("triton.testing", testing),
                         ("triton.compiler", compiler), ("triton.compiler.errors", compiler_errors),
                         ("triton.runtime", runtime), ("triton.runtime.errors", runtime_errors)):
        monkeypatch.setitem(sys.modules, name, module)
    props = SimpleNamespace(name="resource-test", major=12, minor=0,
                            shared_memory_per_block=49152, shared_memory_per_block_optin=65536)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: props)
    monkeypatch.setattr(kernel_tuning, "_CACHE", {})
    monkeypatch.setattr(kernel_tuning, "_RECORDS", [])
    monkeypatch.setenv("FLASHOPW_AUTOTUNE", "1")
    launches = []

    class Kernel:
        __name__ = "resource_test"

        def warmup(self, *args, **kwargs):
            shared = 16384 if (kwargs["BM"], kwargs["BN"]) == (16, 32) else 65537
            return SimpleNamespace(metadata=SimpleNamespace(shared=shared))

        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                assert (kwargs["BM"], kwargs["BN"]) == (16, 32), "unsafe candidate launched"
                launches.append(grid)
                return 1.0
            return launch

    kernel = Kernel()
    args = (torch.empty(1),)
    config = (16, 32, 1, 4)
    grid = lambda c: (c[0], c[1])
    kernel_tuning.launch(kernel, args, dict(N=37, M=79), grid, config, 32, 64)
    record = kernel_tuning.tuning_records()[0]
    assert record["selected"]["shared_bytes"] == 16384
    assert any(t["status"] == "shared_memory_limit" for t in record["candidates"])
    before = len(launches)
    kernel_tuning.launch(kernel, args, dict(N=37, M=79), grid, config, 32, 64)
    assert len(launches) == before + 1, "cached selection must not retune"
