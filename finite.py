import torch


def finite_check(name, t):
    if not torch.isfinite(t).all():
        raise RuntimeError(f"{name} has NaN/Inf")

