# Copyright (c) 2026, FlagOS / Megatron-LM-FL. All rights reserved.

"""Unit tests for the Triton CP-layout kernels module (Stage 4 / M3).

The import-boundary test runs on CPU without ``triton``.  Operator parity
tests require CUDA and are skipped as a group on CPU-only workers, preventing
an attempted CUDA allocation from masquerading as a fallback test.

Beyond fallback parity, they exercise the host-side *semantic* layer that
feeds the Triton kernels (per-row sequence metadata and the source/backward
mappings).  On a triton-enabled GPU machine the kernels themselves can then be
validated against the exact same reference.

Case tables are frozen in ``dsv4_parity_gate`` (plan Phase 0): CP tail
padding / zero-visible compressed rows / non-power-of-two widths are part of
the gate for both the PyTorch oracle and the Triton operators.
"""

import torch

import pytest

from megatron.core.transformer.experimental_attention_variant.csa_utils import (
    cp_layout as csa_utils,
)
from megatron.plugin.dsa_kernel.context_parallel.backends.triton import (
    cp_kernel as triton_cp_kernel,
)
from tests.unit_tests.transformer.experimental_attention_variant import dsv4_parity_gate

# Mode 1 (all visible compressed rows) can also process rows that fall
# outside every sequence boundary (fallback rows) after the oracle fix, so
# all three modes share the full frozen table.
ATTENTION_INDEX_CASES = dsv4_parity_gate.ATTENTION_INDEX_CASES
COMPACT_CASES = dsv4_parity_gate.COMPACT_CASES
_PARITY_SEED = dsv4_parity_gate.PARITY_SEED

# Real-Triton gate: kernels run only on a CUDA box with triton importable.
_HAS_REAL_TRITON = triton_cp_kernel._TRITON_AVAILABLE and torch.cuda.is_available()


@pytest.fixture(autouse=True)
def _require_cuda_for_operator_parity(request):
    """Keep the import-boundary check on CPU; all operator tests use CUDA."""
    if (
        request.node.name != "test_module_importable_without_triton"
        and not torch.cuda.is_available()
    ):
        pytest.skip("Triton CP-layout operator parity requires CUDA")


# ===========================================================================
# Module contract (importable without Triton)
# ===========================================================================


def test_module_importable_without_triton():
    import importlib

    mod = importlib.import_module(
        "megatron.plugin.dsa_kernel.context_parallel.backends.triton.cp_kernel"
    )
    assert hasattr(mod, "compress_compressor_input")
    assert hasattr(mod, "build_attention_indices")
    # On this CPU-only box Triton is not installed; the module must still work.
    if not mod._TRITON_AVAILABLE:
        assert callable(mod.compress_compressor_input)
        assert callable(mod.build_attention_indices)


# ===========================================================================
# Compaction: Triton-entry-point parity with the torch reference
# ===========================================================================


@pytest.mark.parametrize(
    "cu_seqlens, global_start, l_local, d_comp, ratio, c_cap", COMPACT_CASES
)
def test_compress_compressor_input_fallback_matches_reference(
    cu_seqlens, global_start, l_local, d_comp, ratio, c_cap
):
    torch.manual_seed(_PARITY_SEED)
    cu = torch.tensor(cu_seqlens, dtype=torch.int32, device="cuda")
    hidden = torch.randn(l_local, 5, device="cuda")
    boundary = torch.randn(d_comp, 5, device="cuda")

    h_tri, ids_tri = triton_cp_kernel.compress_compressor_input(
        hidden, boundary, cu, global_start, ratio, d_comp, c_cap
    )
    h_ref, ids_ref = csa_utils.compressor_input_compact(
        hidden, boundary, cu, global_start, ratio, d_comp, c_cap
    )
    assert torch.equal(h_tri, h_ref)
    assert torch.equal(ids_tri, ids_ref)


@pytest.mark.parametrize(
    "cu_seqlens, global_start, l_local, d_comp, ratio, c_cap", COMPACT_CASES
)
def test_compress_compressor_input_fallback_backward_matches_reference(
    cu_seqlens, global_start, l_local, d_comp, ratio, c_cap
):
    torch.manual_seed(_PARITY_SEED + 1)
    cu = torch.tensor(cu_seqlens, dtype=torch.int32, device="cuda")
    hidden = torch.randn(l_local, 5, requires_grad=True, device="cuda")
    boundary = torch.randn(d_comp, 5, requires_grad=True, device="cuda")

    h_tri, _ = triton_cp_kernel.compress_compressor_input(
        hidden, boundary, cu, global_start, ratio, d_comp, c_cap
    )
    h_tri.sum().backward(retain_graph=True)
    g_tri_h, g_tri_b = hidden.grad.clone(), boundary.grad.clone()

    hidden.grad = None
    boundary.grad = None
    h_ref, _ = csa_utils.compressor_input_compact(
        hidden, boundary, cu, global_start, ratio, d_comp, c_cap
    )
    h_ref.sum().backward()
    assert torch.equal(h_tri, h_ref)
    assert torch.equal(g_tri_h, hidden.grad)
    assert torch.equal(g_tri_b, boundary.grad)


# ===========================================================================
# build_attention_indices parity (modes 0 / 1 / 2)
# ===========================================================================

@pytest.mark.parametrize(
    "cu_seqlens, cu_seqlens_compressed, global_start, l_local, d_window, "
    "window_size, ratio, compressed_width",
    ATTENTION_INDEX_CASES,
)
def test_build_attention_indices_mode1_parity(
    cu_seqlens,
    cu_seqlens_compressed,
    global_start,
    l_local,
    d_window,
    window_size,
    ratio,
    compressed_width,
):
    cu = torch.tensor(cu_seqlens, dtype=torch.int32, device="cuda")
    cu_compressed = (
        None
        if cu_seqlens_compressed is None
        else torch.tensor(cu_seqlens_compressed, dtype=torch.int32, device="cuda")
    )
    seq_major_rows = (l_local * 2) // ratio
    strr = torch.tensor(list(range(seq_major_rows)) + [-1], dtype=torch.int32, device="cuda")
    a = triton_cp_kernel.build_attention_indices(
        cu, global_start, l_local, d_window, window_size, ratio, compressed_width,
        compressed_topk=None, cu_seqlens_compressed=cu_compressed, seq_to_rank_row=strr,
    )
    b = csa_utils.build_attention_indices(
        cu, global_start, l_local, d_window, window_size, ratio, compressed_width,
        compressed_topk=None, cu_seqlens_compressed=cu_compressed, seq_to_rank_row=strr,
    )
    for i, (x, y) in enumerate(zip(a, b)):
        if x is None:
            assert y is None
        else:
            assert torch.equal(x, y), f"Mismatch at index {i}: {x} vs {y}"


@pytest.mark.parametrize(
    "cu_seqlens, cu_seqlens_compressed, global_start, l_local, d_window, "
    "window_size, ratio, compressed_width",
    ATTENTION_INDEX_CASES,
)
def test_build_attention_indices_mode0_parity(
    cu_seqlens,
    cu_seqlens_compressed,
    global_start,
    l_local,
    d_window,
    window_size,
    ratio,
    compressed_width,
):
    torch.manual_seed(_PARITY_SEED)
    cu = torch.tensor(cu_seqlens, dtype=torch.int32, device="cuda")
    cu_compressed = (
        None
        if cu_seqlens_compressed is None
        else torch.tensor(cu_seqlens_compressed, dtype=torch.int32, device="cuda")
    )
    seq_major_rows = (l_local * 2) // ratio
    strr = torch.tensor(list(range(seq_major_rows)) + [-1], dtype=torch.int32, device="cuda")
    topk = torch.randint(-1, 3, (l_local, compressed_width), dtype=torch.int32, device="cuda")
    a = triton_cp_kernel.build_attention_indices(
        cu, global_start, l_local, d_window, window_size, ratio, compressed_width,
        compressed_topk=topk, cu_seqlens_compressed=cu_compressed, seq_to_rank_row=strr,
    )
    b = csa_utils.build_attention_indices(
        cu, global_start, l_local, d_window, window_size, ratio, compressed_width,
        compressed_topk=topk, cu_seqlens_compressed=cu_compressed, seq_to_rank_row=strr,
    )
    for x, y in zip(a, b):
        if x is None:
            assert y is None
        else:
            assert torch.equal(x, y)


@pytest.mark.parametrize(
    "cu_seqlens, cu_seqlens_compressed, global_start, l_local, d_window, "
    "window_size, ratio, compressed_width",
    ATTENTION_INDEX_CASES,
)
def test_build_attention_indices_mode2_parity(
    cu_seqlens,
    cu_seqlens_compressed,
    global_start,
    l_local,
    d_window,
    window_size,
    ratio,
    compressed_width,
):
    torch.manual_seed(_PARITY_SEED + 1)
    cu = torch.tensor(cu_seqlens, dtype=torch.int32, device="cuda")
    cu_compressed = (
        None
        if cu_seqlens_compressed is None
        else torch.tensor(cu_seqlens_compressed, dtype=torch.int32, device="cuda")
    )
    seq_major_rows = (l_local * 2) // ratio
    strr = torch.tensor(list(range(seq_major_rows)) + [-1], dtype=torch.int32, device="cuda")
    topk = torch.randint(-1, 4, (l_local, compressed_width), dtype=torch.int32, device="cuda")
    a = triton_cp_kernel.build_attention_indices(
        cu, global_start, l_local, d_window, window_size, ratio, compressed_width,
        compressed_topk=topk, cu_seqlens_compressed=cu_compressed, seq_to_rank_row=strr,
        for_indexer_loss=True,
    )
    b = csa_utils.build_attention_indices(
        cu, global_start, l_local, d_window, window_size, ratio, compressed_width,
        compressed_topk=topk, cu_seqlens_compressed=cu_compressed, seq_to_rank_row=strr,
        for_indexer_loss=True,
    )
    for x, y in zip(a, b):
        if x is None:
            assert y is None
        else:
            assert torch.equal(x, y)


# ===========================================================================
# Host-side semantic layer feeding the Triton kernels
# ===========================================================================


def test_row_metadata_torch_matches_naive_per_row():
    cu = torch.tensor([0, 5, 21, 40], dtype=torch.int32, device="cuda")
    cu_comp = torch.tensor([0, 1, 4, 8], dtype=torch.int32, device="cuda")
    gs, ll, ws = 11, 18, 4
    seq_start, seq_comp_start, seq_comp_len, window_start, window_count = (
        triton_cp_kernel._row_metadata_torch(cu, cu_comp, gs, ll, ws)
    )
    assert seq_start.shape == (ll,)
    assert seq_comp_start.shape == (ll,)
    assert seq_comp_len.shape == (ll,)
    assert window_start.shape == (ll,)
    assert window_count.shape == (ll,)

    for r in range(ll):
        qg = gs + r
        seq_id = next(
            (s for s in range(3) if int(cu[s]) <= qg < int(cu[s + 1])), None
        )
        if seq_id is None:
            assert int(seq_start[r]) == -1
            assert int(window_count[r]) == 0
            continue
        ss = int(cu[seq_id])
        assert int(seq_start[r]) == ss
        assert int(seq_comp_start[r]) == int(cu_comp[seq_id])
        assert int(seq_comp_len[r]) == int(cu_comp[seq_id + 1]) - int(cu_comp[seq_id])
        ws_ = max(qg - ws + 1, ss)
        assert int(window_start[r]) == ws_
        assert int(window_count[r]) == qg - ws_ + 1


def test_compaction_host_mapping_is_one_to_one_and_matches_reference():
    """The flat source mapping (fwd) and output-row mapping (bwd) used to feed
    the Triton kernels must be one-to-one and reproduce the torch reference."""
    for cu_seqlens, gs, ll, d_comp, ratio, c_cap in COMPACT_CASES:
        cu = torch.tensor(cu_seqlens, dtype=torch.int32, device="cuda")
        d_window = d_comp
        hidden = torch.randn(ll, 5, device="cuda")
        boundary = torch.randn(d_window, 5, device="cuda")

        src_global, comp_ids, valid = csa_utils._compact_row_to_source(
            cu, gs, ll, ratio, d_comp, c_cap
        )
        flat = csa_utils._flat_source_index(src_global, valid, gs, d_window)

        # Forward: manual gather must equal the reference compaction output.
        compact_len = c_cap * ratio
        src_cat = torch.cat((boundary, hidden, torch.zeros(1, 5)), dim=0)
        gathered = torch.index_select(src_cat, 0, flat)
        h_manual = torch.where(valid.unsqueeze(-1), gathered, torch.zeros(1, 5))
        h_ref, _ = csa_utils.compressor_input_compact(
            hidden, boundary, cu, gs, ratio, d_comp, c_cap
        )
        assert torch.equal(h_manual.reshape(compact_len, 5), h_ref.reshape(compact_len, 5))
        assert torch.equal(comp_ids, (csa_utils.compressor_input_compact(
            hidden, boundary, cu, gs, ratio, d_comp, c_cap
        )[1]))

        # Backward: each valid compact row maps to a unique output row
        # (boundary rows first, then local rows), i.e. the Triton scatter can
        # be a plain (non-atomic) store.
        srcs = src_global[valid]
        compact_idx = torch.nonzero(valid).squeeze(-1)
        out_row = torch.where(
            srcs < gs,
            srcs - (gs - d_window),
            d_window + (srcs - gs),
        )
        assert out_row.numel() == torch.unique(out_row).numel()
        if out_row.numel() > 0:
            assert int(out_row.min()) >= 0
            assert int(out_row.max()) < ll + d_window


def test_compaction_backward_repeated_source_uses_index_add_fallback(monkeypatch):
    """A duplicated compact source row must accumulate both gradients.

    The real compaction mapping is always one-to-one (each source token
    belongs to exactly one visible group), so the duplicate path is forced by
    patching the shared source mapping.  Both the Triton entry point and the
    reference read ``dsa_layout._compact_row_to_source`` through the same
    module object, so the patched mapping produces a meaningful parity
    comparison that also exercises the defensive ``index_add`` fallback.
    """
    cu_seqlens, gs, ll, d_comp, ratio, c_cap = ([0, 32], 0, 16, 4, 4, 8)
    cu = torch.tensor(cu_seqlens, dtype=torch.int32, device="cuda")

    def duplicated_sources(*args, **kwargs):
        src_global, comp_ids, valid = csa_utils._compact_row_to_source(*args, **kwargs)
        src_global = src_global.clone()
        # Compact rows 5 and 6 both reference global row 5 (both local rows).
        src_global[6] = src_global[5]
        return src_global, comp_ids, valid

    monkeypatch.setattr(
        triton_cp_kernel.dsa_layout, "_compact_row_to_source", duplicated_sources
    )

    torch.manual_seed(dsv4_parity_gate.PARITY_SEED + 2)
    hidden = torch.randn(ll, 5, requires_grad=True, device="cuda")
    boundary = torch.randn(d_comp, 5, requires_grad=True, device="cuda")

    h_tri, _ = triton_cp_kernel.compress_compressor_input(
        hidden, boundary, cu, gs, ratio, d_comp, c_cap
    )
    h_tri.sum().backward(retain_graph=True)
    g_tri_h, g_tri_b = hidden.grad.clone(), boundary.grad.clone()

    # The reference uses the same patched (non-unique) mapping.
    hidden.grad = None
    boundary.grad = None
    h_ref, _ = csa_utils.compressor_input_compact(hidden, boundary, cu, gs, ratio, d_comp, c_cap)
    h_ref.sum().backward()

    assert torch.equal(h_tri, h_ref)
    assert torch.equal(g_tri_h, hidden.grad)
    assert torch.equal(g_tri_b, boundary.grad)
    # Duplicated row 5 must accumulate both compact-row contributions.  With
    # the patched mapping, grad_hidden[5] carries 2.0 (rows 5 and 6 of the
    # compact buffer both copy global row 5 of the local hidden block).
    assert torch.equal(g_tri_h[5], torch.full_like(g_tri_h[5], 2.0))


@pytest.mark.skipif(not _HAS_REAL_TRITON, reason="real Triton kernels require CUDA")
def test_triton_kernel_launch_smoke():
    """Compile and launch the real Triton kernels (Phase-1 gate).

    Guards against the operator passing through the PyTorch fallback on a
    Triton-capable box: the kernels must compile and produce the same output
    as the reference on a compact case.
    """
    assert triton_cp_kernel._TRITON_AVAILABLE
    cu_seqlens, gs, ll, d_comp, ratio, c_cap = ([0, 5, 21, 40], 11, 18, 8, 4, 8)
    torch.manual_seed(dsv4_parity_gate.PARITY_SEED + 3)
    cu = torch.tensor(cu_seqlens, dtype=torch.int32, device="cuda")
    hidden = torch.randn(ll, 5, requires_grad=True, device="cuda")
    boundary = torch.randn(d_comp, 5, requires_grad=True, device="cuda")

    h_tri, ids_tri = triton_cp_kernel.compress_compressor_input(
        hidden, boundary, cu, gs, ratio, d_comp, c_cap
    )
    h_ref, ids_ref = csa_utils.compressor_input_compact(
        hidden, boundary, cu, gs, ratio, d_comp, c_cap
    )
    assert torch.equal(h_tri, h_ref)
    assert torch.equal(ids_tri, ids_ref)

    h_tri.sum().backward()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    assert boundary.grad is not None and torch.isfinite(boundary.grad).all()


def test_build_attention_indices_dispatcher_triton_module():
    """cp_utils must route the CP-layout ops through the Triton module name."""
    assert (
        triton_cp_kernel.__name__
        == "megatron.plugin.dsa_kernel.context_parallel.backends.triton.cp_kernel"
    )
    cu = torch.tensor([0, 32], dtype=torch.int32, device="cuda")
    topk_idxs, topk_length, _ = triton_cp_kernel.build_attention_indices(
        cu, 0, 16, 4, 4, 4, 2, compressed_topk=None
    )
    assert topk_idxs.shape == (16, 6)
    assert topk_length.shape == (16,)
