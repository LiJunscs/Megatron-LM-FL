# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Unified backend selection for DSv4 sparse-attention kernels.

Optional dependencies are isolated below ``backends/`` for fused DSA and
``cp/backends/`` for context-parallel operations. They are imported only after
this module has probed their runtime requirements. Model code imports this
package instead of importing Triton, CuTe DSL, cuDNN Frontend, or FlashMLA
directly.
"""

import importlib
import importlib.util
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

_BACKENDS = ("torch", "triton", "cuda")
_BACKEND_MODULES = {
    "torch": "megatron.plugin.dsa_kernel.backends.pytorch",
    "triton": "megatron.plugin.dsa_kernel.backends.triton",
    "cuda": "megatron.plugin.dsa_kernel.backends.cudnn_flashmla",
}

_CP_BACKEND_MODULES = {
    "torch": "megatron.core.transformer.experimental_attention_variant.csa_utils.cp_layout",
    "triton": "megatron.plugin.dsa_kernel.context_parallel.backends.triton",
    "cuda": "megatron.plugin.dsa_kernel.context_parallel.backends.cute",
}

_CP_OPERATIONS = {
    "build_attention_indices",
    "compact_compressor_input",
}


class DSAv4BackendError(RuntimeError):
    """Raised for invalid backend selection or unavailable dependencies."""


def _import(module: str):
    return importlib.import_module(module)


def _has_module(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, AttributeError, ValueError):
        return False


def _has_accelerator() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _torch_backend_available() -> bool:
    try:
        _import(_BACKEND_MODULES["torch"])
        return True
    except Exception:
        return False


def _triton_backend_available() -> bool:
    if not _has_accelerator() or not _has_module("triton"):
        return False
    try:
        _import(_BACKEND_MODULES["triton"])
        return True
    except Exception:
        return False


def _cuda_backend_available() -> bool:
    """Probe the complete cuDNN Frontend + FlashMLA dependency set."""
    if (
        not _has_accelerator()
        or not _has_module("cudnn")
        or not _has_module("flash_mla")
    ):
        return False
    try:
        cudnn = _import("cudnn")
        if not hasattr(cudnn, "DSA"):
            return False
        _import(_BACKEND_MODULES["cuda"])
        return True
    except Exception:
        return False


def _backend_available(backend: str) -> bool:
    """Run the current probe, keeping probes replaceable in tests."""
    return globals()[f"_{backend}_backend_available"]()


def _device_preference() -> Optional[str]:
    """Choose the accelerator family appropriate for the active CUDA device."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        major, _minor = torch.cuda.get_device_capability()
        if major >= 10:
            return "cuda"
        if major >= 9:
            return "triton"
        return None
    except Exception:
        return None


def requested_backend(config: Optional[object] = None) -> str:
    """Return the configured backend name before automatic resolution."""
    value = getattr(config, "dsv4_kernel_backend", None) if config is not None else None
    if value is None:
        value = os.environ.get("DSV4_KERNEL_BACKEND", "auto")
    value = str(value).lower()
    if value not in _BACKENDS + ("auto",):
        raise DSAv4BackendError(
            f"Unknown dsv4_kernel_backend={value!r}; expected 'auto' or one of {_BACKENDS}."
        )
    return value


_resolved_backend: Optional[str] = None
_fallback_warned: set = set()


def available_backend(config: Optional[object] = None) -> str:
    """Resolve one process-wide backend, with PyTorch as the final fallback."""
    global _resolved_backend
    if _resolved_backend is not None:
        return _resolved_backend

    requested = requested_backend(config)
    if requested != "auto":
        if not _backend_available(requested):
            raise DSAv4BackendError(
                f"Requested dsv4_kernel_backend={requested!r}, but its dependencies "
                "or required accelerator are unavailable."
            )
        _resolved_backend = requested
        return requested

    preferred = _device_preference()
    if preferred is None:
        if _backend_available("torch"):
            _resolved_backend = "torch"
            return _resolved_backend
        raise DSAv4BackendError(
            "The DSv4 PyTorch reference backend could not be imported."
        )

    order = [preferred] + [
        backend
        for backend in ("triton", "cuda", "torch")
        if backend != preferred
    ]
    for backend in order:
        if _backend_available(backend):
            _resolved_backend = backend
            logger.info("DSv4 kernel backend auto-resolved to %r.", backend)
            return backend

    raise DSAv4BackendError("The DSv4 PyTorch reference backend could not be imported.")


# Operations actually consumed by the current model. CP operations use the
# core PyTorch oracle or an accelerator under ``cp.backends``; fused DSA
# operations resolve through the top-level backend tree.
_OP_BACKENDS = {
    "build_attention_indices": ("cuda", "triton", "torch"),
    "build_flat_topk_idxs": ("cuda", "triton", "torch"),
    "compact_compressor_input": ("cuda", "triton", "torch"),
    "fused_indexer_sparse_attn": ("cuda", "triton", "torch"),
    "indexer_sparse_attn": ("cuda", "triton", "torch"),
    "indexer_topk": ("cuda", "triton", "torch"),
}


def supports(
    operation: str,
    device: Optional[str] = None,
    dtype: Optional[str] = None,
    layout: Optional[str] = None,
    features: Optional[set] = None,
    config: Optional[object] = None,
) -> bool:
    """Return whether the selected backend owns an implementation."""
    if operation not in _OP_BACKENDS:
        return False
    active = available_backend(config)
    return _backend_supports_operation(operation, active)


def _backend_supports_operation(operation: str, backend: str) -> bool:
    """Probe one operation without changing the process-wide backend choice."""
    if backend not in _OP_BACKENDS[operation]:
        return False
    if backend != "torch" and not _backend_available(backend):
        return False
    module = _import(_operation_module_name(operation, backend))
    backend_supports = getattr(module, "supports", None)
    return backend_supports(operation) if backend_supports is not None else True


def _torch_fused_indexer_unavailable(*args, **kwargs):
    raise DSAv4BackendError(
        "The PyTorch path implements indexer loss and sparse attention as "
        "separate operations; set apply_dsa_kernel_fusion=False."
    )


def _operation_module_name(operation: str, backend: str) -> str:
    """Return the implementation module for one operation/backend pair."""
    modules = _CP_BACKEND_MODULES if operation in _CP_OPERATIONS else _BACKEND_MODULES
    return modules[backend]


def _op_callable(operation: str, backend: str):
    module = _import(_operation_module_name(operation, backend))
    if backend == "torch" and operation == "fused_indexer_sparse_attn":
        return _torch_fused_indexer_unavailable
    try:
        return getattr(module, operation)
    except AttributeError as error:
        raise DSAv4BackendError(
            f"Backend {backend!r} does not export DSv4 operation {operation!r}."
        ) from error


def resolve(operation: str, config: Optional[object] = None):
    """Resolve one operation and fall back to its PyTorch reference."""
    if operation not in _OP_BACKENDS:
        raise DSAv4BackendError(f"Unknown DSv4 backend operation {operation!r}.")

    active = available_backend(config)
    candidates = (active,) + tuple(
        backend for backend in _OP_BACKENDS[operation] if backend != active
    )
    for backend in candidates:
        if not _backend_supports_operation(operation, backend):
            continue
        if backend != active:
            fallback_key = (operation, active, backend)
            if fallback_key not in _fallback_warned:
                _fallback_warned.add(fallback_key)
                logger.warning(
                    "DSv4 operation %r is unavailable on backend %r; using %r.",
                    operation,
                    active,
                    backend,
                )
        return _op_callable(operation, backend)

    raise DSAv4BackendError(
        f"No available backend implements DSv4 operation {operation!r}."
    )


def sparse_attention(*args, config: Optional[object] = None, **kwargs):
    return resolve("indexer_sparse_attn", config=config)(*args, **kwargs)


def fused_indexer_sparse_attn(*args, config: Optional[object] = None, **kwargs):
    return resolve("fused_indexer_sparse_attn", config=config)(*args, **kwargs)


def indexer_topk(*args, config: Optional[object] = None, **kwargs):
    return resolve("indexer_topk", config=config)(*args, **kwargs)


def build_flat_topk_idxs(*args, config: Optional[object] = None, **kwargs):
    return resolve("build_flat_topk_idxs", config=config)(*args, **kwargs)


def compact_compressor_input(*args, config: Optional[object] = None, **kwargs):
    return resolve("compact_compressor_input", config=config)(*args, **kwargs)


def build_attention_indices(*args, config: Optional[object] = None, **kwargs):
    return resolve("build_attention_indices", config=config)(*args, **kwargs)


__all__ = [
    "DSAv4BackendError",
    "available_backend",
    "build_attention_indices",
    "build_flat_topk_idxs",
    "compact_compressor_input",
    "fused_indexer_sparse_attn",
    "indexer_topk",
    "requested_backend",
    "resolve",
    "sparse_attention",
    "supports",
]
