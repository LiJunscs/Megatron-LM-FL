# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""CPU unit tests for atomic DSv4 fused-backend bundle selection."""

from types import SimpleNamespace

import pytest
import torch

from megatron.plugin import dsa_kernel as dsa_backend


@pytest.fixture(autouse=True)
def reset_backend_cache():
    """Reset dispatcher diagnostics before each test."""
    dsa_backend._resolved_backend = None
    yield
    dsa_backend._resolved_backend = None


def _config(backend="auto", *, fusion=True, tp=1):
    return SimpleNamespace(
        dsv4_kernel_backend=backend,
        apply_dsa_kernel_fusion=fusion,
        tensor_model_parallel_size=tp,
    )


def _provider(operations, *, supported=True, tag="provider"):
    module = SimpleNamespace()
    for operation in operations:
        setattr(module, operation, (lambda op=operation: (tag, op)))
    module.supports = lambda operation, **kwargs: supported
    return module


def _install_fake_providers(monkeypatch, modules):
    original_import = dsa_backend._import

    def fake_import(name):
        if name in modules:
            return modules[name]
        return original_import(name)

    monkeypatch.setattr(dsa_backend, "_import", fake_import)
    monkeypatch.setattr(
        dsa_backend,
        "_backend_available_for_domain",
        lambda backend, *, is_cp: (
            dsa_backend._CP_BACKEND_MODULES[backend]
            if is_cp
            else dsa_backend._BACKEND_MODULES[backend]
        )
        in modules,
    )


class TestBackendEnumeration:
    def test_known_backends(self):
        assert set(("triton", "cuda")) == set(dsa_backend._BACKENDS)
        assert "torch" not in dsa_backend._BACKEND_MODULES

    def test_requested_backend_valid(self):
        assert dsa_backend.requested_backend() in ("auto", "torch", "triton", "cuda")

    def test_unknown_backend_raises(self, monkeypatch):
        monkeypatch.setenv("DSV4_KERNEL_BACKEND", "bogus")
        with pytest.raises(dsa_backend.DSAv4BackendError):
            dsa_backend.requested_backend()

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

    @pytest.mark.parametrize("backend", ("triton", "cute"))
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

    def test_cp_torch_bundle_reuses_core_native_layout(self):
        for operation in dsa_backend._CP_OPERATIONS:
            assert dsa_backend._operation_module_name(operation, "torch").endswith(
                "csa_utils.cp_layout"
            )


class TestAtomicBundleResolution:
    def test_fusion_false_rejects_fused_bundle(self):
        with pytest.raises(dsa_backend.DSAv4BackendError, match="fusion=False"):
            dsa_backend.resolve_bundle(
                dsa_backend._NON_CP_FUSED_OPERATIONS,
                config=_config(fusion=False),
            )

    def test_fusion_true_rejects_torch_backend(self):
        with pytest.raises(dsa_backend.DSAv4BackendError, match="incompatible"):
            dsa_backend.resolve_bundle(
                dsa_backend._NON_CP_FUSED_OPERATIONS,
                config=_config("torch"),
            )

    def test_auto_selects_complete_cuda_bundle(self, monkeypatch):
        cuda_name = dsa_backend._BACKEND_MODULES["cuda"]
        triton_name = dsa_backend._BACKEND_MODULES["triton"]
        modules = {
            cuda_name: _provider(dsa_backend._NON_CP_FUSED_OPERATIONS, tag="cuda"),
            triton_name: _provider(dsa_backend._NON_CP_FUSED_OPERATIONS, tag="triton"),
        }
        _install_fake_providers(monkeypatch, modules)
        bundle = dsa_backend.resolve_bundle(
            dsa_backend._NON_CP_FUSED_OPERATIONS, config=_config("auto")
        )
        assert bundle.backend == "cuda"
        assert all(fn()[0] == "cuda" for fn in bundle.operations.values())

    def test_auto_discards_incomplete_cuda_and_selects_whole_triton(self, monkeypatch):
        cuda_ops = dsa_backend._NON_CP_FUSED_OPERATIONS - {"indexer_topk"}
        modules = {
            dsa_backend._BACKEND_MODULES["cuda"]: _provider(cuda_ops, tag="cuda"),
            dsa_backend._BACKEND_MODULES["triton"]: _provider(
                dsa_backend._NON_CP_FUSED_OPERATIONS, tag="triton"
            ),
        }
        _install_fake_providers(monkeypatch, modules)
        bundle = dsa_backend.resolve_bundle(
            dsa_backend._NON_CP_FUSED_OPERATIONS, config=_config("auto")
        )
        assert bundle.backend == "triton"
        assert all(fn()[0] == "triton" for fn in bundle.operations.values())

    def test_explicit_incomplete_backend_fails_without_exploration(self, monkeypatch):
        modules = {
            dsa_backend._BACKEND_MODULES["cuda"]: _provider(
                dsa_backend._NON_CP_FUSED_OPERATIONS - {"indexer_topk"}, tag="cuda"
            ),
            dsa_backend._BACKEND_MODULES["triton"]: _provider(
                dsa_backend._NON_CP_FUSED_OPERATIONS, tag="triton"
            ),
        }
        _install_fake_providers(monkeypatch, modules)
        with pytest.raises(dsa_backend.DSAv4BackendError, match="explicit"):
            dsa_backend.resolve_bundle(
                dsa_backend._NON_CP_FUSED_OPERATIONS, config=_config("cuda")
            )

    def test_explicit_triton_does_not_probe_complete_cuda(self, monkeypatch):
        modules = {
            dsa_backend._BACKEND_MODULES["cuda"]: _provider(
                dsa_backend._NON_CP_FUSED_OPERATIONS, tag="cuda"
            ),
            dsa_backend._BACKEND_MODULES["triton"]: _provider(
                dsa_backend._NON_CP_FUSED_OPERATIONS, tag="triton"
            ),
        }
        _install_fake_providers(monkeypatch, modules)
        bundle = dsa_backend.resolve_bundle(
            dsa_backend._NON_CP_FUSED_OPERATIONS, config=_config("triton")
        )
        assert bundle.backend == "triton"
        assert all(fn()[0] == "triton" for fn in bundle.operations.values())

    def test_auto_has_no_framework_torch_fallback(self, monkeypatch):
        _install_fake_providers(monkeypatch, {})
        with pytest.raises(dsa_backend.DSAv4BackendError, match="Disable"):
            dsa_backend.resolve_bundle(
                dsa_backend._NON_CP_FUSED_OPERATIONS, config=_config("auto")
            )

    def test_partial_operation_set_is_rejected(self):
        with pytest.raises(dsa_backend.DSAv4BackendError, match="complete operation set"):
            dsa_backend.resolve_bundle(
                {"indexer_topk"}, config=_config("auto")
            )

    def test_cp_auto_discards_incomplete_cute_for_complete_triton(self, monkeypatch):
        modules = {
            dsa_backend._CP_BACKEND_MODULES["cute"]: _provider(
                {"build_attention_indices"}, tag="cute"
            ),
            dsa_backend._CP_BACKEND_MODULES["triton"]: _provider(
                dsa_backend._CP_OPERATIONS, tag="triton"
            ),
        }
        _install_fake_providers(monkeypatch, modules)
        bundle = dsa_backend.resolve_cp_bundle(config=_config("cuda"), layout="thd")
        assert bundle.backend == "triton"
        assert all(fn()[0] == "triton" for fn in bundle.operations.values())

    def test_main_cuda_and_cp_triton_are_independent_bundles(self, monkeypatch):
        modules = {
            dsa_backend._BACKEND_MODULES["cuda"]: _provider(
                dsa_backend._NON_CP_FUSED_OPERATIONS, tag="cuda"
            ),
            # An incomplete CuTe CP bundle must not borrow its missing op.
            dsa_backend._CP_BACKEND_MODULES["cute"]: _provider(
                {"build_attention_indices"}, tag="cute"
            ),
            dsa_backend._BACKEND_MODULES["triton"]: _provider(
                dsa_backend._NON_CP_FUSED_OPERATIONS, tag="triton"
            ),
            dsa_backend._CP_BACKEND_MODULES["triton"]: _provider(
                dsa_backend._CP_OPERATIONS, tag="triton"
            ),
        }
        _install_fake_providers(monkeypatch, modules)
        main_bundle = dsa_backend.resolve_fused_bundle(config=_config("cuda"))
        cp_bundle = dsa_backend.resolve_cp_bundle(config=_config("cuda"))
        assert main_bundle.backend == "cuda"
        assert cp_bundle.backend == "triton"
        assert all(fn()[0] == "cuda" for fn in main_bundle.operations.values())
        assert all(fn()[0] == "triton" for fn in cp_bundle.operations.values())

    def test_cp_selection_ignores_explicit_main_backend(self, monkeypatch):
        modules = {
            dsa_backend._BACKEND_MODULES["cuda"]: _provider(
                dsa_backend._NON_CP_FUSED_OPERATIONS, tag="cuda"
            ),
            dsa_backend._CP_BACKEND_MODULES["cute"]: _provider(
                {"build_attention_indices"}, tag="cute"
            ),
            dsa_backend._BACKEND_MODULES["triton"]: _provider(
                dsa_backend._NON_CP_FUSED_OPERATIONS, tag="triton"
            ),
            dsa_backend._CP_BACKEND_MODULES["triton"]: _provider(
                dsa_backend._CP_OPERATIONS, tag="triton"
            ),
        }
        _install_fake_providers(monkeypatch, modules)
        bundle = dsa_backend.resolve_cp_bundle(config=_config("torch"))
        assert bundle.backend == "triton"
        assert all(fn()[0] == "triton" for fn in bundle.operations.values())

    def test_cp_auto_falls_back_as_a_whole_to_torch_native(self, monkeypatch):
        modules = {
            dsa_backend._CP_BACKEND_MODULES["torch"]: _provider(
                dsa_backend._CP_OPERATIONS, tag="torch"
            )
        }
        _install_fake_providers(monkeypatch, modules)
        bundle = dsa_backend.resolve_cp_bundle(config=_config("triton"), layout="thd")
        assert bundle.backend == "torch"
        assert all(fn()[0] == "torch" for fn in bundle.operations.values())

    def test_cp_selection_is_independent_of_main_fusion_switch(self, monkeypatch):
        modules = {
            dsa_backend._CP_BACKEND_MODULES["triton"]: _provider(
                dsa_backend._CP_OPERATIONS, tag="triton"
            ),
            dsa_backend._CP_BACKEND_MODULES["torch"]: _provider(
                dsa_backend._CP_OPERATIONS, tag="torch"
            ),
        }
        _install_fake_providers(monkeypatch, modules)
        bundle = dsa_backend.resolve_cp_bundle(config=_config(fusion=False))
        assert bundle.backend == "triton"

    def test_cp_and_non_cp_must_be_resolved_as_separate_domains(self):
        with pytest.raises(dsa_backend.DSAv4BackendError, match="independent"):
            dsa_backend.resolve_bundle(
                dsa_backend._CP_OPERATIONS | dsa_backend._NON_CP_FUSED_OPERATIONS,
                config=_config("auto"),
            )

    def test_cp_rejects_cross_provider_operation_union(self, monkeypatch):
        modules = {
            dsa_backend._CP_BACKEND_MODULES["cute"]: _provider(
                {"compact_compressor_input"}, tag="cute"
            ),
            dsa_backend._CP_BACKEND_MODULES["triton"]: _provider(
                {"build_attention_indices"}, tag="triton"
            ),
            dsa_backend._CP_BACKEND_MODULES["torch"]: _provider(
                dsa_backend._CP_OPERATIONS, tag="torch"
            ),
        }
        _install_fake_providers(monkeypatch, modules)
        bundle = dsa_backend.resolve_cp_bundle(config=_config("auto"))
        assert bundle.backend == "torch"
        assert all(fn()[0] == "torch" for fn in bundle.operations.values())

    def test_non_cp_tp_skips_cuda_bundle_as_a_whole(self, monkeypatch):
        modules = {
            dsa_backend._BACKEND_MODULES["cuda"]: _provider(
                dsa_backend._NON_CP_FUSED_OPERATIONS, tag="cuda"
            ),
            dsa_backend._BACKEND_MODULES["triton"]: _provider(
                dsa_backend._NON_CP_FUSED_OPERATIONS, tag="triton"
            ),
        }
        _install_fake_providers(monkeypatch, modules)
        bundle = dsa_backend.resolve_bundle(
            dsa_backend._NON_CP_FUSED_OPERATIONS,
            config=_config("auto", tp=2),
        )
        assert bundle.backend == "triton"

    def test_supports_unknown_op(self):
        assert dsa_backend.supports("does_not_exist", config=_config()) is False


class TestBackendDelegationIntegration:
    def test_torch_sparse_attention_decodes_sequence_major_flat_rows(self, monkeypatch):
        from megatron.plugin.dsa_kernel.backends import pytorch as pyt
        from megatron.plugin.dsa_kernel.backends.pytorch import reference_ops

        captured = {}

        def fake_unfused(query, kv, sink, local_topk, scale):
            captured["topk"] = local_topk
            return query.new_empty(
                (query.shape[0], query.shape[1], query.shape[2] * query.shape[3])
            )

        monkeypatch.setattr(reference_ops, "unfused_sparse_attn", fake_unfused)
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
