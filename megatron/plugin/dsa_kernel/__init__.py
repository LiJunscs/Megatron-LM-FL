# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Unified backend selection for DSv4 sparse-attention kernels.

Optional dependencies are isolated below ``backends/`` for fused DSA and
``context_parallel/backends/`` for context-parallel operations. They are
imported only after this module has probed their runtime requirements. Model
code imports this package instead of importing Triton, CuTe DSL, cuDNN
Frontend, or FlashMLA directly.
"""

import importlib
import importlib.util
import logging
import os
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

_BACKENDS = ("triton", "cuda")
_BACKEND_MODULES = {
    "triton": "megatron.plugin.dsa_kernel.backends.triton",
    "cuda": "megatron.plugin.dsa_kernel.backends.cudnn_flashmla",
}

_CP_BACKEND_MODULES = {
    "torch": "megatron.core.transformer.experimental_attention_variant.csa_utils.cp_layout",
    "triton": "megatron.plugin.dsa_kernel.context_parallel.backends.triton",
    "cute": "megatron.plugin.dsa_kernel.context_parallel.backends.cute",
}

_CP_OPERATIONS = {
    "build_attention_indices",
    "compact_compressor_input",
}

_NON_CP_FUSED_OPERATIONS = {
    "build_flat_topk_idxs",
    "fused_indexer_sparse_attn",
    "indexer_sparse_attn",
    "indexer_topk",
}


class DSAv4BackendError(RuntimeError):
    """Raised for invalid backend selection or unavailable dependencies."""


@dataclass(frozen=True)
class DSAv4KernelBundle:
    """One atomically selected provider family for a complete capability domain."""

    backend: str
    operations: dict

    def __getitem__(self, operation: str):
        return self.operations[operation]


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


def _triton_backend_available() -> bool:
    if not _has_accelerator() or not _has_module("triton"):
        return False
    try:
        import torch

        if torch.cuda.get_device_capability()[0] < 9:
            return False
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
        import torch

        if torch.cuda.get_device_capability()[0] < 10:
            return False
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


def _backend_allowed_for_config(backend: str, config: Optional[object]) -> bool:
    """Apply model-level restrictions that dependency probes cannot express."""
    return not (
        backend == "cuda"
        and config is not None
        and getattr(config, "tensor_model_parallel_size", 1) > 1
    )


def requested_backend(config: Optional[object] = None) -> str:
    """Return the configured backend name before automatic resolution."""
    value = getattr(config, "dsv4_kernel_backend", None) if config is not None else None
    if value is None:
        value = os.environ.get("DSV4_KERNEL_BACKEND", "auto")
    value = str(value).lower()
    # ``torch`` remains a recognized configuration value only so fused=True
    # can reject it with an actionable error. It is not a registered non-CP
    # plugin backend; the Triton provider owns any internal PyTorch routing.
    if value not in _BACKENDS + ("auto", "torch"):
        raise DSAv4BackendError(
            f"Unknown dsv4_kernel_backend={value!r}; expected 'auto', 'torch', "
            f"or one of {_BACKENDS}."
        )
    return value


# Last-resolution diagnostics only. Bundle selection is deliberately not
# process-wide cached: one process may construct models with different configs
# or devices for parity testing.
_resolved_backend: Optional[str] = None


def _bundle_domain(operations):
    operations = set(operations)
    if not operations:
        raise DSAv4BackendError("A DSv4 kernel bundle requires at least one operation.")
    unknown = operations - set(_OP_BACKENDS)
    if unknown:
        raise DSAv4BackendError(f"Unknown DSv4 backend operations: {sorted(unknown)}.")
    cp_ops = frozenset(operations & _CP_OPERATIONS)
    non_cp_ops = frozenset(operations & _NON_CP_FUSED_OPERATIONS)
    if cp_ops and non_cp_ops:
        raise DSAv4BackendError(
            "DSv4 CP and non-CP kernels are independent capability domains; "
            "resolve each complete bundle separately."
        )
    if cp_ops:
        if cp_ops != frozenset(_CP_OPERATIONS):
            raise DSAv4BackendError(
                f"DSv4 CP fused selection requires the complete operation set "
                f"{sorted(_CP_OPERATIONS)}, got {sorted(cp_ops)}."
            )
        return True, cp_ops
    if non_cp_ops:
        if non_cp_ops != frozenset(_NON_CP_FUSED_OPERATIONS):
            raise DSAv4BackendError(
                f"DSv4 non-CP fused selection requires the complete operation set "
                f"{sorted(_NON_CP_FUSED_OPERATIONS)}, got {sorted(non_cp_ops)}."
            )
        return False, non_cp_ops
    raise DSAv4BackendError("A DSv4 kernel bundle contains no supported operations.")


def _required_operations(is_cp: bool):
    return _CP_OPERATIONS if is_cp else _NON_CP_FUSED_OPERATIONS


def _backend_available_for_domain(backend: str, *, is_cp: bool) -> bool:
    """Probe the implementation tree that owns the requested capability domain."""
    if backend == "torch":
        if not is_cp:
            return False
        try:
            module = _import(_CP_BACKEND_MODULES[backend])
            return all(hasattr(module, operation) for operation in _CP_OPERATIONS)
        except Exception:
            return False
    if backend not in ({"cute", "triton"} if is_cp else {"cuda", "triton"}):
        return False
    if not is_cp:
        return _backend_available(backend)
    if not _has_accelerator():
        return False
    try:
        import torch

        major, _minor = torch.cuda.get_device_capability()
        if backend == "triton" and (major < 9 or not _has_module("triton")):
            return False
        module = _import(_CP_BACKEND_MODULES[backend])
        return all(hasattr(module, operation) for operation in _CP_OPERATIONS)
    except Exception:
        return False


def _backend_allowed_for_domain(
    backend: str, config: Optional[object], *, is_cp: bool
) -> bool:
    # The CUDA non-CP aggregate does not yet own the TP reduction contract.
    # CuTe CP operations do not share that restriction.
    return is_cp or _backend_allowed_for_config(backend, config)


def _bundle_candidate_order(requested: str):
    if requested == "torch":
        raise DSAv4BackendError(
            "dsv4_kernel_backend='torch' is incompatible with "
            "apply_dsa_kernel_fusion=True; disable DSA kernel fusion to use "
            "the PyTorch native/reference path."
        )
    if requested == "auto":
        # Probe complete fused families, never individual operations. On SM90
        # the CUDA probe simply fails and Triton is selected.
        return ("cuda", "triton")
    return (requested,)


def _cp_bundle_candidate_order():
    """CP is an independent plugin with one fixed, atomic fallback chain."""
    return ("cute", "triton", "torch")


def _provider_supports_domain(
    backend: str,
    operations,
    *,
    is_cp: bool,
    device=None,
    dtype=None,
    layout=None,
    features=None,
):
    if not _backend_available_for_domain(backend, is_cp=is_cp):
        return False, "provider dependencies or accelerator are unavailable"
    module_name = _CP_BACKEND_MODULES[backend] if is_cp else _BACKEND_MODULES[backend]
    module = _import(module_name)
    provider_supports = getattr(module, "supports", None)
    for operation in operations:
        if not hasattr(module, operation):
            return False, f"operation {operation!r} is not exported"
        if provider_supports is not None and not provider_supports(
            operation,
            device=device,
            dtype=dtype,
            layout=layout,
            features=features,
        ):
            return False, f"operation {operation!r} does not support the execution contract"
    return True, None


def resolve_bundle(
    operations,
    config: Optional[object] = None,
    *,
    device=None,
    dtype=None,
    layout=None,
    features=None,
) -> DSAv4KernelBundle:
    """Resolve a complete CP or non-CP fused bundle atomically.

    Non-CP fused kernels obey ``dsv4_kernel_backend``. CP ignores that setting
    and always probes complete CuTe, Triton, then PyTorch-native bundles. In
    both domains, the framework never fills missing operations from a second
    provider.
    """
    global _resolved_backend

    operations = frozenset(operations)
    is_cp, domain_operations = _bundle_domain(operations)

    if (
        not is_cp
        and config is not None
        and not getattr(config, "apply_dsa_kernel_fusion", True)
    ):
        raise DSAv4BackendError(
            "Fused DSv4 bundles cannot be resolved when apply_dsa_kernel_fusion=False."
        )

    requested = "auto" if is_cp else requested_backend(config)
    candidates = (
        _cp_bundle_candidate_order()
        if is_cp
        else _bundle_candidate_order(requested)
    )
    failures = []
    for backend in candidates:
        if not _backend_allowed_for_domain(backend, config, is_cp=is_cp):
            tp_size = getattr(config, "tensor_model_parallel_size", 1)
            reason = (
                "the cuDNN+FlashMLA aggregate does not yet own the TP reduction contract"
                if backend == "cuda" and not is_cp and tp_size > 1
                else "incompatible with the model-parallel contract"
            )
            failures.append(f"{backend}: {reason}")
            logger.info(
                "DSv4 complete %s bundle skipped backend %r: %s "
                "(tensor_model_parallel_size=%s).",
                "CP" if is_cp else "non-CP fused",
                backend,
                reason,
                tp_size,
            )
            continue
        supported, reason = _provider_supports_domain(
            backend,
            domain_operations,
            is_cp=is_cp,
            device=device,
            dtype=dtype,
            layout=layout,
            features=features,
        )
        if supported:
            module_name = (
                _CP_BACKEND_MODULES[backend] if is_cp else _BACKEND_MODULES[backend]
            )
            module = _import(module_name)
            bundle = DSAv4KernelBundle(
                backend=backend,
                operations={
                    operation: getattr(module, operation)
                    for operation in domain_operations
                },
            )
            _resolved_backend = backend
            logger.info(
                "DSv4 complete %s bundle resolved to backend %r.",
                "CP" if is_cp else "non-CP fused",
                backend,
            )
            return bundle
        failures.append(f"{backend}: {reason}")
    mode = "CP auto" if is_cp else ("explicit" if requested != "auto" else "auto")
    recovery = (
        "The CP PyTorch-native bundle is expected to be universally available."
        if is_cp
        else "Disable apply_dsa_kernel_fusion to use the PyTorch native/reference path."
    )
    raise DSAv4BackendError(
        f"DSv4 {mode} fused bundle selection failed for the complete "
        f"execution contract. Tried: {'; '.join(failures)}. "
        f"{recovery}"
    )


def available_backend(config: Optional[object] = None) -> str:
    """Return the atomically selected non-CP fused backend family."""
    return resolve_bundle(_NON_CP_FUSED_OPERATIONS, config=config).backend


def resolve_fused_bundle(config: Optional[object] = None, **contract) -> DSAv4KernelBundle:
    """Resolve the complete non-CP fused execution bundle."""
    return resolve_bundle(_NON_CP_FUSED_OPERATIONS, config=config, **contract)


def resolve_cp_bundle(config: Optional[object] = None, **contract) -> DSAv4KernelBundle:
    """Auto-resolve the complete CP bundle: CuTe, Triton, then PyTorch."""
    return resolve_bundle(_CP_OPERATIONS, config=config, **contract)


# Operations actually consumed by the current model. CP operations use the
# core PyTorch oracle or an accelerator under ``cp.backends``; fused DSA
# operations resolve through the top-level backend tree.
_OP_BACKENDS = {
    "build_attention_indices": ("cute", "triton", "torch"),
    "build_flat_topk_idxs": ("cuda", "triton"),
    "compact_compressor_input": ("cute", "triton", "torch"),
    "fused_indexer_sparse_attn": ("cuda", "triton"),
    "indexer_sparse_attn": ("cuda", "triton"),
    "indexer_topk": ("cuda", "triton"),
}


def supports(
    operation: str,
    device: Optional[str] = None,
    dtype: Optional[str] = None,
    layout: Optional[str] = None,
    features: Optional[set] = None,
    config: Optional[object] = None,
) -> bool:
    """Return whether the complete selected bundle owns ``operation``."""
    if operation not in _OP_BACKENDS:
        return False
    is_cp = operation in _CP_OPERATIONS
    try:
        bundle = resolve_bundle(
            _required_operations(is_cp),
            config=config,
            device=device,
            dtype=dtype,
            layout=layout,
            features=features,
        )
    except DSAv4BackendError:
        return False
    return operation in bundle.operations


def _operation_module_name(operation: str, backend: str) -> str:
    """Return the implementation module for one operation/backend pair."""
    modules = _CP_BACKEND_MODULES if operation in _CP_OPERATIONS else _BACKEND_MODULES
    return modules[backend]


def resolve(operation: str, config: Optional[object] = None):
    """Resolve an operation from its atomically selected complete bundle."""
    if operation not in _OP_BACKENDS:
        raise DSAv4BackendError(f"Unknown DSv4 backend operation {operation!r}.")
    bundle = resolve_bundle(
        _required_operations(operation in _CP_OPERATIONS), config=config
    )
    return bundle[operation]


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
    "DSAv4KernelBundle",
    "available_backend",
    "build_attention_indices",
    "build_flat_topk_idxs",
    "compact_compressor_input",
    "fused_indexer_sparse_attn",
    "indexer_topk",
    "requested_backend",
    "resolve",
    "resolve_bundle",
    "resolve_cp_bundle",
    "resolve_fused_bundle",
    "sparse_attention",
    "supports",
]
