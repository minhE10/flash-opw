"""Observe the external solver without changing its math, CG budget, or tests."""
from functools import wraps
import json
import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def record_external_hvp(request, monkeypatch):
    if request.node.path.name != "test_hvp_parity.py":
        yield
        return
    import torch_sinkhorn_hessian as baseline
    expected = Path(os.environ["AUTHOR_OTT_HESSIAN_ROOT"]).resolve()
    assert Path(baseline.__file__).resolve().parent == expected, "Unexpected baseline import"
    original = baseline.TorchSinkhornHessian.hessian_vector_product
    records = []
    @wraps(original)
    def observed(self, *args, **kwargs):
        record = {"test":request.node.nodeid, "module":baseline.__file__,
                  "requested_keops":bool(self.use_keops), "parameters":kwargs}
        records.append(record)
        try:
            output = original(self, *args, **kwargs)
            record["cg"] = output[1] if kwargs.get("return_info") else self.last_cg_info
            return output
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            record["used_keops"] = bool(self.use_keops)
    monkeypatch.setattr(baseline.TorchSinkhornHessian, "hessian_vector_product", observed)
    yield
    path = Path(os.environ["AUTHOR_OTT_HESSIAN_REPORT"])
    prior = json.loads(path.read_text()) if path.exists() else []
    path.write_text(json.dumps(prior+records, indent=2)+"\n", encoding="utf-8")
