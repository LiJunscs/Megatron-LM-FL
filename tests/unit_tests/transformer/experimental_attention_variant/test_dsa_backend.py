# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for the DSv4 backend-neutral dispatcher (PR3 / 阶段2).

These tests run on CPU: importing the dispatcher must never load CuTeDSL,
FlashMLA, cuDNN or Triton, and the ``torch`` (PyTorch reference) backend must
be selected / resolved without any CUDA dependency.
"""

import pytest
import torch

from megatron.plugin import dsa_kernel as dsa_backend


@pytest.fixture(autouse=True)
def reset_backend_cache():
    """Reset dispatcher diagnostics before each test."""
    dsa_backend._resolved_backend = None
    dsa_backend._fallback_warned = set()
    yield
    dsa_backend._resolved_backend = None
    dsa_backend._fallback_warned = set()


class TestBackendEnumeration:
    def test_known_backends(self):
        assert set(("torch", "triton", "cuda")) == set(dsa_backend._BACKENDS)

    def test_requested_backend_valid(self):
        assert dsa_backend.requested_backend() in ("auto", "torch", "triton", "cuda")

    def test_unknown_backend_raises(self, monkeypatch):
        monkeypatch.setenv("DSV4_KERNEL_BACKEND", "bogus")
        with pytest.raises(dsa_backend.DSAv4BackendError):
            dsa_backend.requested_backend()

    def test_torch_backend_available_on_cpu(self):
        assert dsa_backend._torch_backend_available() is True

    def test_every_cuda_operation_has_a_triton_replacement(self):
        cuda_operations = {
            operation
            for operation, backends in dsa_backend._OP_BACKENDS.items()
            if "cuda" in backends
        }
        assert cuda_operations
        assert all(
            "triton" in dsa_backend._OP_BACKENDS[operation]
            for operation in cuda_operations
        )

    @pytest.mark.parametrize("backend", ("torch", "triton", "cuda"))
    def test_cp_operations_resolve_from_the_cp_backend_tree(self, backend):
        for operation in dsa_backend._CP_OPERATIONS:
            module = dsa_backend._operation_module_name(operation, backend)
            assert ".dsa_kernel.context_parallel.backends." in module

    def test_non_cp_backends_do_not_export_cp_operations(self):
        from megatron.plugin.dsa_kernel.backends import pytorch as pytorch_backend
        from megatron.plugin.dsa_kernel.backends import triton as triton_backend

        for backend in (pytorch_backend, triton_backend):
            assert not hasattr(backend, "build_attention_indices")
            assert not hasattr(backend, "compact_compressor_input")


class TestBackendResolution:
    def test_auto_always_resolves_to_torch_when_no_accelerators(self, monkeypatch):
        monkeypatch.setattr(dsa_backend, "_triton_backend_available", lambda: False)
        monkeypatch.setattr(dsa_backend, "_cuda_backend_available", lambda: False)
        # Even if a CUDA device hints at an accelerated backend, its kernels
        # are not importable, so auto must fall back to torch.
        monkeypatch.setattr(dsa_backend, "_device_preference", lambda: "cuda")
        assert dsa_backend.available_backend() == "torch"

    def test_explicit_torch_backend(self, monkeypatch):
        monkeypatch.setenv("DSV4_KERNEL_BACKEND", "torch")
        assert dsa_backend.available_backend() == "torch"

    def test_explicit_unavailable_backend_raises(self, monkeypatch):
        monkeypatch.setenv("DSV4_KERNEL_BACKEND", "triton")
        monkeypatch.setattr(dsa_backend, "_triton_backend_available", lambda: False)
        with pytest.raises(dsa_backend.DSAv4BackendError):
            dsa_backend.available_backend()


class TestSupportAndResolve:
    def test_supports_known_ops_on_torch(self, monkeypatch):
        monkeypatch.setenv("DSV4_KERNEL_BACKEND", "torch")
        for op in dsa_backend._OP_BACKENDS.keys() - {"fused_indexer_sparse_attn"}:
            assert dsa_backend.supports(op) is True
        assert dsa_backend.supports("fused_indexer_sparse_attn") is False

    def test_supports_unknown_op(self, monkeypatch):
        monkeypatch.setenv("DSV4_KERNEL_BACKEND", "torch")
        assert dsa_backend.supports("does_not_exist") is False

    def test_resolve_torch_index_helpers(self, monkeypatch):
        monkeypatch.setenv("DSV4_KERNEL_BACKEND", "torch")
        fn = dsa_backend.resolve("build_flat_topk_idxs")
        assert callable(fn)

    def test_resolve_torch_compaction(self, monkeypatch):
        monkeypatch.setenv("DSV4_KERNEL_BACKEND", "torch")
        fn = dsa_backend.resolve("compact_compressor_input")
        assert callable(fn)

    def test_resolve_torch_build_attention_indices(self, monkeypatch):
        monkeypatch.setenv("DSV4_KERNEL_BACKEND", "torch")
        fn = dsa_backend.resolve("build_attention_indices")
        assert callable(fn)

    def test_resolve_torch_indexer_topk(self, monkeypatch):
        monkeypatch.setenv("DSV4_KERNEL_BACKEND", "torch")
        fn = dsa_backend.resolve("indexer_topk")
        assert callable(fn)

    def test_resolve_torch_sparse_attn(self, monkeypatch):
        monkeypatch.setenv("DSV4_KERNEL_BACKEND", "torch")
        fn = dsa_backend.resolve("indexer_sparse_attn")
        assert callable(fn)

    def test_resolve_fused_indexer_raises_on_torch(self, monkeypatch):
        # The PyTorch reference exposes the fused indexer as separate ops
        # (Stage 6); a single callable must not silently mis-dispatch.
        monkeypatch.setenv("DSV4_KERNEL_BACKEND", "torch")
        with pytest.raises(dsa_backend.DSAv4BackendError):
            dsa_backend.resolve("fused_indexer_sparse_attn")

    @pytest.mark.parametrize(
        "operation", ("compact_compressor_input", "build_attention_indices")
    )
    def test_explicit_cuda_cp_layout_fails_without_cute(self, monkeypatch, operation):
        """Explicit backend selection must not hide a missing CuTe operation."""
        monkeypatch.setenv("DSV4_KERNEL_BACKEND", "cuda")
        monkeypatch.setattr(dsa_backend, "_cuda_backend_available", lambda: True)
        from megatron.plugin.dsa_kernel.context_parallel.backends import (
            cute as cuda_cp_backend,
        )
        monkeypatch.setattr(cuda_cp_backend, "supports", lambda op, **kwargs: False)
        with pytest.raises(dsa_backend.DSAv4BackendError):
            dsa_backend.resolve(operation)

    @pytest.mark.parametrize(
        "operation", ("compact_compressor_input", "build_attention_indices")
    )
    def test_cuda_cp_layout_resolves_to_restored_cute(self, monkeypatch, operation):
        monkeypatch.setenv("DSV4_KERNEL_BACKEND", "cuda")
        monkeypatch.setattr(dsa_backend, "_cuda_backend_available", lambda: True)
        from megatron.plugin.dsa_kernel.context_parallel.backends import (
            cute as cuda_cp_backend,
        )

        monkeypatch.setattr(cuda_cp_backend, "supports", lambda op, **kwargs: True)
        assert dsa_backend.resolve(operation) is getattr(cuda_cp_backend, operation)

    def test_cuda_cp_layout_prefers_triton_before_torch(self, monkeypatch):
        monkeypatch.setenv("DSV4_KERNEL_BACKEND", "auto")
        monkeypatch.setattr(dsa_backend, "_device_preference", lambda: "cuda")
        monkeypatch.setattr(dsa_backend, "_cuda_backend_available", lambda: True)
        monkeypatch.setattr(dsa_backend, "_triton_backend_available", lambda: True)
        from megatron.plugin.dsa_kernel.context_parallel.backends import (
            cute as cuda_cp_backend,
            triton as triton_cp_backend,
        )

        monkeypatch.setattr(cuda_cp_backend, "supports", lambda op, **kwargs: False)
        assert dsa_backend.resolve("build_attention_indices") is (
            triton_cp_backend.build_attention_indices
        )

    def test_backend_resolution_is_not_process_wide_cached(self, monkeypatch):
        monkeypatch.setenv("DSV4_KERNEL_BACKEND", "torch")
        assert dsa_backend.available_backend() == "torch"

        monkeypatch.setenv("DSV4_KERNEL_BACKEND", "auto")
        monkeypatch.setattr(dsa_backend, "_device_preference", lambda: "triton")
        monkeypatch.setattr(dsa_backend, "_triton_backend_available", lambda: True)
        assert dsa_backend.available_backend() == "triton"

    def test_auto_sm100_tp_prefers_triton(self, monkeypatch):
        monkeypatch.setenv("DSV4_KERNEL_BACKEND", "auto")
        monkeypatch.setattr(dsa_backend, "_device_preference", lambda: "cuda")
        monkeypatch.setattr(dsa_backend, "_cuda_backend_available", lambda: True)
        monkeypatch.setattr(dsa_backend, "_triton_backend_available", lambda: True)
        config = type("Config", (), {"tensor_model_parallel_size": 2})()
        assert dsa_backend.available_backend(config) == "triton"

    def test_auto_sm100_tp_never_falls_back_to_cuda(self, monkeypatch):
        monkeypatch.setenv("DSV4_KERNEL_BACKEND", "auto")
        monkeypatch.setattr(dsa_backend, "_device_preference", lambda: "cuda")
        monkeypatch.setattr(dsa_backend, "_cuda_backend_available", lambda: True)
        monkeypatch.setattr(dsa_backend, "_triton_backend_available", lambda: False)
        monkeypatch.setattr(dsa_backend, "_torch_backend_available", lambda: True)
        config = type("Config", (), {"tensor_model_parallel_size": 2})()
        assert dsa_backend.available_backend(config) == "torch"

    def test_explicit_cuda_rejects_tensor_parallel_config(self, monkeypatch):
        monkeypatch.setenv("DSV4_KERNEL_BACKEND", "cuda")
        monkeypatch.setattr(dsa_backend, "_cuda_backend_available", lambda: True)
        config = type("Config", (), {"tensor_model_parallel_size": 2})()
        with pytest.raises(dsa_backend.DSAv4BackendError):
            dsa_backend.available_backend(config)


class TestBackendDelegationIntegration:
    def test_torch_sparse_attention_decodes_sequence_major_flat_rows(self, monkeypatch):
        from megatron.plugin.dsa_kernel.backends import pytorch as pyt

        captured = {}

        def fake_unfused(query, kv, sink, local_topk, scale):
            captured["topk"] = local_topk
            return query.new_empty(
                (query.shape[0], query.shape[1], query.shape[2] * query.shape[3])
            )

        monkeypatch.setattr(pyt, "unfused_sparse_attn", fake_unfused)
        # Flat query rows are ordered (s0,b0), (s0,b1), (s1,b0), (s1,b1).
        local_bsk = torch.tensor([[[1], [2]], [[3], [4]]], dtype=torch.int32)
        flat, _ = pyt.build_flat_topk_idxs(
            local_bsk, batch_size=2, seqlen_kv=5, compact=False
        )
        pyt.indexer_sparse_attn(
            torch.empty(2, 2, 1, 4),
            torch.empty(5, 2, 4),
            torch.empty(1),
            flat,
            1.0,
        )
        assert torch.equal(captured["topk"], local_bsk.long())
