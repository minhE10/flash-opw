"""Deterministic CPU-generated datasets, copied to the requested device."""

import math
import torch

DATASETS = ("gaussian", "mixture", "rings")


def make_dataset(name, n, m=None, d=2, *, seed=42, weighted=False,
                 dtype=torch.float32, device="cpu"):
    if name not in DATASETS or n < 1 or (m is not None and m < 1) or d < 1:
        raise ValueError("Unknown dataset or invalid dimensions")
    if name == "rings" and d < 2:
        raise ValueError("rings requires at least two dimensions")
    m = n if m is None else m
    rng = torch.Generator(device="cpu").manual_seed(seed)

    def normal(*shape):
        return torch.randn(*shape, generator=rng, dtype=dtype)

    if name == "gaussian":
        x, y = normal(n, d) * 0.3 / math.sqrt(d), normal(m, d) * 0.3 / math.sqrt(d)
        y[:, 0] += 0.5
    elif name == "mixture":
        centers = normal(3, d) * 0.55 / math.sqrt(d)
        x = centers[torch.randint(3, (n,), generator=rng)] + 0.1 * normal(n, d) / math.sqrt(d)
        y = centers[torch.randint(3, (m,), generator=rng)] + 0.1 * normal(m, d) / math.sqrt(d)
        y[:, 0] += 0.25
    else:
        def ring(size, radius):
            theta = torch.rand(size, generator=rng, dtype=dtype) * (2 * math.pi)
            z = 0.02 * normal(size, d)
            z[:, 0] += radius * theta.cos()
            z[:, 1] += radius * theta.sin()
            return z
        x, y = ring(n, 0.55), ring(m, 0.9)
        y[:, 0] += 0.2

    def weights(size):
        w = torch.rand(size, generator=rng, dtype=dtype) + 0.2 if weighted else torch.ones(size, dtype=dtype)
        return w / w.sum()

    return tuple(t.to(device) for t in (x, y, weights(n), weights(m)))
