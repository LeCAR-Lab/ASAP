"""Device selection helpers: CUDA first, then Intel XPU, then CPU."""

import torch


def is_available(backend):
    """Return True if the given torch backend ("xpu" or "cuda") has a usable device."""
    module = getattr(torch, backend, None)
    return module is not None and module.is_available()


def default_device_type():
    """Return "cuda" if available, else "xpu" if an Intel XPU is present, else "cpu"."""
    for backend in ("cuda", "xpu"):
        if is_available(backend):
            return backend
    return "cpu"


def resolve_device(device=None):
    """Return a device string.

    An explicit device is honoured as given, except that a bare "xpu" or
    "cuda" becomes index 0. With no device, the best available one is used.
    """
    if device is None:
        device = default_device_type()
    device = str(device)
    if device in ("xpu", "cuda"):
        device = device + ":0"
    return device
