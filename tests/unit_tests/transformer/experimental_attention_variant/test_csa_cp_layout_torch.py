# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for the pure-PyTorch DSv4 THD-CP layout reference.

These tests exercise the portable correctness reference in
``csa_utils/cp_layout.py`` and the backend-neutral
dispatchers in ``csa_utils/cp_utils.py``. They run on CPU (no
CuTe/FlashMLA/cuDNN needed), which is exactly the cross-platform contract M1
establishes.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from megatron.core.transformer.experimental_attention_variant.csa import (
    _apply_rope,
    unfused_compressed_sparse_attn,
)
from megatron.core.transformer.experimental_attention_variant.csa_utils import (
    cp_layout as csa_utils,
)
from megatron.core.transformer.experimental_attention_variant.csa_utils import cp_utils
from megatron.plugin.platform import get_platform
from tests.unit_tests.transformer.experimental_attention_variant import dsv4_parity_gate

# Frozen Phase-0 oracle gate: canonical case tables, seed and diagnostic.
ATTENTION_INDEX_CASES = dsv4_parity_gate.ATTENTION_INDEX_CASES
COMPACT_CASES = dsv4_parity_gate.COMPACT_CASES


# ===========================================================================
# CompressorInputCompact (compaction) reference
# ===========================================================================


def _visible_groups_naive(cu_seqlens, global_start, l_local, ratio, d_comp):
    """Serial Python re-implementation of the CuTe fwd loop (for parity)."""
    range_start = global_start
    range_end = global_start + l_local
    first_range_group_start = range_start - d_comp
    running = 0
    out = {}
    for seq in range(len(cu_seqlens) - 1):
        seq_start = int(cu_seqlens[seq])
        seq_end = int(cu_seqlens[seq + 1])
        local_seq_end = min(seq_end, range_end)
        if seq_start < local_seq_end and range_start < local_seq_end:
            first_num = max(first_range_group_start - seq_start, 0)
            first_group = (first_num + ratio - 1) // ratio
            stop_group = (local_seq_end - seq_start) // ratio
            count = max(stop_group - first_group, 0)
            for g in range(first_group, first_group + count):
                for t in range(ratio):
                    out[running + (g - first_group) * ratio + t] = (
                        seq_start + g * ratio + t
                    )
            running += count * ratio
    return out, running


def _compact_to_source_naive(cu_seqlens, global_start, l_local, ratio, d_comp, c_cap):
    mapping, total = _visible_groups_naive(cu_seqlens, global_start, l_local, ratio, d_comp)
    compact_len = c_cap * ratio
    src = []
    for row in range(compact_len):
        src.append(mapping.get(row, -1))
    return src, total


@pytest.mark.parametrize(
    "cu_seqlens, global_start, l_local, d_comp, ratio, c_cap",
    COMPACT_CASES,
)
def test_compressor_input_compact_forward_matches_naive(
    cu_seqlens, global_start, l_local, d_comp, ratio, c_cap
):
    cu = torch.tensor(cu_seqlens, dtype=torch.int32)
    src_expected, total = _compact_to_source_naive(
        cu, global_start, l_local, ratio, d_comp, c_cap
    )

    hidden_local = torch.randn(l_local, 3, dtype=torch.float64)
    boundary_hidden = torch.randn(d_comp, 3, dtype=torch.float64)
    compact, comp_ids = csa_utils.compressor_input_compact(
        hidden_local, boundary_hidden, cu, global_start, ratio, d_comp, c_cap
    )

    assert compact.shape == (c_cap * ratio, 3)
    for row, src_global in enumerate(src_expected):
        if src_global < 0:
            assert torch.equal(compact[row], torch.zeros(3, dtype=torch.float64))
        elif src_global < global_start:
            boundary_row = src_global - (global_start - d_comp)
            assert torch.equal(compact[row], boundary_hidden[boundary_row])
        else:
            local_row = src_global - global_start
            assert torch.equal(compact[row], hidden_local[local_row])

    # comp_ids = per-sequence compressed id of each group (or -1).
    assert comp_ids.shape == (c_cap,)
    mapping, _ = _visible_groups_naive(cu, global_start, l_local, ratio, d_comp)
    for g in range(c_cap):
        # Group g leads at compact row g*ratio; its comp id is the per-sequence
        # compressed-group id of that group's first source token (or -1).
        row_lead = g * ratio
        if row_lead in mapping:
            assert int(comp_ids[g]) == _comp_id_of(mapping[row_lead], cu)
        else:
            assert int(comp_ids[g]) == -1


def _comp_id_of(src_global, cu_seqlens):
    for seq in range(len(cu_seqlens) - 1):
        if int(cu_seqlens[seq]) <= int(src_global) < int(cu_seqlens[seq + 1]):
            return (int(src_global) - int(cu_seqlens[seq])) // 4
    return -1


@pytest.mark.parametrize(
    "cu_seqlens, global_start, l_local, d_comp",
    [([0, 40], 8, 16, 4), ([0, 5, 21, 40], 11, 18, 8)],
)
def test_compressor_input_compact_backward_matches_naive(
    cu_seqlens, global_start, l_local, d_comp
):
    ratio = 4
    c_cap = 8
    cu = torch.tensor(cu_seqlens, dtype=torch.int32)
    src_expected, _ = _compact_to_source_naive(cu, global_start, l_local, ratio, d_comp, c_cap)

    hidden_local = torch.randn(l_local, 3, dtype=torch.float64, requires_grad=True)
    boundary_hidden = torch.randn(d_comp, 3, dtype=torch.float64, requires_grad=True)
    compact, _ = csa_utils.compressor_input_compact(
        hidden_local, boundary_hidden, cu, global_start, ratio, d_comp, c_cap
    )
    compact.sum().backward()

    g_hidden = hidden_local.grad
    g_boundary = boundary_hidden.grad
    assert g_hidden.shape == hidden_local.shape
    assert g_boundary.shape == boundary_hidden.shape
    # Row r of compact maps to a unique source row; backward scatters 1.0 to it.
    for row, src_global in enumerate(src_expected):
        if src_global < 0:
            continue
        if src_global < global_start:
            assert torch.all(g_boundary[src_global - (global_start - d_comp)] == 1.0)
        else:
            assert torch.all(g_hidden[src_global - global_start] == 1.0)
    # No gradient leaves compaction boundaries (dtype float64, exact 0/1).
    assert g_hidden.abs().sum().item() == (
        hidden_local.shape[1] * sum(1 for s in src_expected if s >= global_start)
    )


def test_compressor_input_compact_repeated_sources_accumulate():
    """A source row referenced twice must accumulate both gradients."""
    # Force overlap: two sequences are too short to produce groups, so build a
    # case where the compaction references the same global row in two groups.
    cu = torch.tensor([0, 32], dtype=torch.int32)
    global_start = 0
    l_local = 32
    ratio = 4
    d_comp = 4
    c_cap = 8
    hidden = torch.randn(l_local, 1, dtype=torch.float64, requires_grad=True)
    compact, _ = csa_utils.compressor_input_compact(
        hidden, torch.zeros(d_comp, 1, dtype=torch.float64), cu, global_start, ratio, d_comp, c_cap
    )
    # Perturb all compact rows and expect gradients to be ones (each compact row
    # copies a distinct source in the non-overlapping interior).
    compact.sum().backward()
    assert hidden.grad is not None
    assert hidden.grad[:32].eq(1.0).all()


def test_compressor_input_compact_preserves_sbhd_batch_axis():
    """The validity mask must broadcast over both B and hidden dimensions."""
    cu = torch.tensor([0, 32], dtype=torch.int32)
    hidden = torch.randn(16, 3, 5, dtype=torch.float64, requires_grad=True)
    boundary = torch.randn(4, 3, 5, dtype=torch.float64, requires_grad=True)
    compact, _ = csa_utils.compressor_input_compact(
        hidden, boundary, cu, global_start=0, ratio=4, d_comp=4, c_cap=8
    )
    assert compact.shape == (32, 3, 5)
    assert torch.equal(compact[:16], hidden)
    assert compact[16:].eq(0).all()
    compact.sum().backward()
    assert hidden.grad is not None and hidden.grad.eq(1).all()


# ===========================================================================
# build_attention_indices reference
# ===========================================================================


def _naive_build_indices(
    cu_seqlens,
    global_start,
    l_local,
    d_window,
    window_size,
    ratio,
    compressed_width,
    compressed_topk=None,
    cu_seqlens_compressed=None,
    seq_to_rank_row=None,
):
    """Serial Python re-implementation of the CuTe build_attention_indices."""
    if cu_seqlens_compressed is None:
        cu_seqlens_compressed = cu_seqlens
    if seq_to_rank_row is None:
        seq_to_rank_row = [-1]
    n_seq = len(cu_seqlens) - 1
    total_width = window_size + compressed_width
    topk_idxs = [[-1] * total_width for _ in range(l_local)]
    topk_length = [0] * l_local

    def seq_of(qg):
        for s in range(n_seq):
            if cu_seqlens[s] <= qg < cu_seqlens[s + 1]:
                return s
        return None

    for row in range(l_local):
        qg = global_start + row
        seq = seq_of(qg)
        if seq is None:
            topk_idxs[row][0] = 0
            topk_length[row] = 1 if total_width > 0 else 0
            continue
        seq_start = cu_seqlens[seq]
        ws = max(qg - window_size + 1, seq_start)
        wcount = qg - ws + 1
        write_col = 0
        for wcol in range(wcount):
            pos = ws + wcol
            topk_idxs[row][write_col] = (
                pos - (global_start - d_window) if pos < global_start else d_window + pos - global_start
            )
            write_col += 1
        if compressed_width > 0 and ratio > 1:
            seq_comp_start = cu_seqlens_compressed[seq]
            seq_comp_len = cu_seqlens_compressed[seq + 1] - seq_comp_start
            if compressed_topk is not None:
                for c in range(compressed_width):
                    cid = compressed_topk[row][c]
                    if 0 <= cid < seq_comp_len:
                        seq_major = seq_comp_start + cid
                        if seq_major < len(seq_to_rank_row):
                            rmr = seq_to_rank_row[seq_major]
                            if rmr >= 0:
                                topk_idxs[row][write_col] = d_window + l_local + rmr
                                write_col += 1
            else:
                comp_count = min((qg - seq_start + 1) // ratio, compressed_width, seq_comp_len)
                for c in range(comp_count):
                    seq_major = seq_comp_start + c
                    if seq_major < len(seq_to_rank_row):
                        rmr = seq_to_rank_row[seq_major]
                        if rmr >= 0:
                            topk_idxs[row][write_col] = d_window + l_local + rmr
                            write_col += 1
        # CuTe semantics: mode 1 (all visible compressed rows) reports
        # topk_length = window_count + comp_count even when an individual
        # compressed row cannot be lowered to a valid rank row (it is left -1
        # and skipped downstream).  Mode 0 (selected top-k) uses write_col,
        # which only counts entries actually placed.
        if compressed_topk is None and ratio > 1 and compressed_width > 0:
            topk_length[row] = wcount + comp_count
        else:
            topk_length[row] = write_col
    return topk_idxs, topk_length


@pytest.mark.parametrize(
    "cu_seqlens, cu_seqlens_compressed, global_start, l_local, "
    "d_window, window_size, ratio, compressed_width",
    ATTENTION_INDEX_CASES,
)
def test_build_attention_indices_mode1_matches_naive(
    cu_seqlens,
    cu_seqlens_compressed,
    global_start,
    l_local,
    d_window,
    window_size,
    ratio,
    compressed_width,
):
    cu = torch.tensor(cu_seqlens, dtype=torch.int32)
    if cu_seqlens_compressed is not None:
        cu_compressed = torch.tensor(cu_seqlens_compressed, dtype=torch.int32)
    else:
        cu_compressed = None
    seq_major_rows = (l_local * 2) // ratio
    seq_to_rank_row = torch.arange(seq_major_rows, dtype=torch.int32).tolist() + [-1]

    exp_idxs, exp_tlen = _naive_build_indices(
        cu.tolist(),
        global_start,
        l_local,
        d_window,
        window_size,
        ratio,
        compressed_width,
        compressed_topk=None,
        cu_seqlens_compressed=cu_compressed.tolist() if cu_compressed is not None else None,
        seq_to_rank_row=seq_to_rank_row,
    )
    topk_idxs, topk_length, _ = csa_utils.build_attention_indices(
        cu,
        global_start,
        l_local,
        d_window,
        window_size,
        ratio,
        compressed_width,
        compressed_topk=None,
        cu_seqlens_compressed=cu_compressed,
        seq_to_rank_row=torch.tensor(seq_to_rank_row, dtype=torch.int32),
    )
    assert topk_idxs.shape == (l_local, window_size + compressed_width)
    for row in range(l_local):
        assert topk_length[row].item() == exp_tlen[row]
        for col in range(window_size + compressed_width):
            assert topk_idxs[row, col].item() == exp_idxs[row][col]


@pytest.mark.parametrize(
    "cu_seqlens, cu_seqlens_compressed, global_start, l_local, "
    "d_window, window_size, ratio, compressed_width",
    ATTENTION_INDEX_CASES,
)
def test_build_attention_indices_mode0_matches_naive(
    cu_seqlens,
    cu_seqlens_compressed,
    global_start,
    l_local,
    d_window,
    window_size,
    ratio,
    compressed_width,
):
    cu = torch.tensor(cu_seqlens, dtype=torch.int32)
    if cu_seqlens_compressed is not None:
        cu_compressed = torch.tensor(cu_seqlens_compressed, dtype=torch.int32)
    else:
        cu_compressed = None
    seq_major_rows = (l_local * 2) // ratio
    seq_to_rank_row = torch.arange(seq_major_rows, dtype=torch.int32).tolist() + [-1]
    torch.manual_seed(dsv4_parity_gate.PARITY_SEED)
    compressed_topk = torch.randint(-1, 3, (l_local, compressed_width), dtype=torch.int32)

    exp_idxs, exp_tlen = _naive_build_indices(
        cu.tolist(),
        global_start,
        l_local,
        d_window,
        window_size,
        ratio,
        compressed_width,
        compressed_topk=compressed_topk.tolist(),
        cu_seqlens_compressed=cu_compressed.tolist() if cu_compressed is not None else None,
        seq_to_rank_row=seq_to_rank_row,
    )
    topk_idxs, topk_length, _ = csa_utils.build_attention_indices(
        cu,
        global_start,
        l_local,
        d_window,
        window_size,
        ratio,
        compressed_width,
        compressed_topk=compressed_topk,
        seq_to_rank_row=torch.tensor(seq_to_rank_row, dtype=torch.int32),
    )
    for row in range(l_local):
        assert topk_length[row].item() == exp_tlen[row]
        for col in range(window_size + compressed_width):
            assert topk_idxs[row, col].item() == exp_idxs[row][col]


def _naive_build_indices_mode2(
    cu_seqlens,
    global_start,
    l_local,
    d_window,
    window_size,
    ratio,
    compressed_width,
    compressed_topk,
    cu_seqlens_compressed=None,
    seq_to_rank_row=None,
):
    """Serial Python re-implementation of the CuTe mode-2 (indexer loss)
    lowering: compressed ids first, then window ids; rank-major compressed
    rows are returned separately for the indexer-loss gather."""
    if cu_seqlens_compressed is None:
        cu_seqlens_compressed = cu_seqlens
    if seq_to_rank_row is None:
        seq_to_rank_row = [-1]
    base = d_window + l_local
    total_width = window_size + compressed_width
    topk_idxs = [[-1] * total_width for _ in range(l_local)]
    indexer_rank_major = [[-1] * compressed_width for _ in range(l_local)]

    def seq_of(qg):
        for s in range(len(cu_seqlens) - 1):
            if cu_seqlens[s] <= qg < cu_seqlens[s + 1]:
                return s
        return None

    for row in range(l_local):
        qg = global_start + row
        seq = seq_of(qg)
        if seq is None:
            # Fallback (tail-padding / truncated) rows: mode 2 emits -1
            # everywhere and never fabricates a window entry.
            continue
        seq_start = cu_seqlens[seq]
        seq_comp_start = cu_seqlens_compressed[seq]
        seq_comp_len = cu_seqlens_compressed[seq + 1] - seq_comp_start

        # Compressed ids first: only caller-selected ids inside the
        # sequence's compressed extent that also lower to a rank-major row.
        for c in range(compressed_width):
            cid = compressed_topk[row][c]
            if 0 <= cid < seq_comp_len:
                seq_major = seq_comp_start + cid
                if seq_major < len(seq_to_rank_row):
                    rmr = seq_to_rank_row[seq_major]
                    if rmr >= 0:
                        topk_idxs[row][c] = base + rmr
                        indexer_rank_major[row][c] = rmr

        # Window ids after the compressed block, compacted at the front.
        ws = max(qg - window_size + 1, seq_start)
        wcount = qg - ws + 1
        for wcol in range(wcount):
            pos = ws + wcol
            topk_idxs[row][compressed_width + wcol] = (
                pos - (global_start - d_window)
                if pos < global_start
                else d_window + pos - global_start
            )
    return topk_idxs, indexer_rank_major


@pytest.mark.parametrize(
    "cu_seqlens, cu_seqlens_compressed, global_start, l_local, "
    "d_window, window_size, ratio, compressed_width",
    ATTENTION_INDEX_CASES,
)
def test_build_attention_indices_mode2_matches_naive(
    cu_seqlens,
    cu_seqlens_compressed,
    global_start,
    l_local,
    d_window,
    window_size,
    ratio,
    compressed_width,
):
    """Mode 2 (indexer loss) lowering must match the serial CuTe reference."""
    cu = torch.tensor(cu_seqlens, dtype=torch.int32)
    if cu_seqlens_compressed is not None:
        cu_compressed = torch.tensor(cu_seqlens_compressed, dtype=torch.int32)
    else:
        cu_compressed = None
    seq_major_rows = (l_local * 2) // ratio
    seq_to_rank_row = torch.arange(seq_major_rows, dtype=torch.int32).tolist() + [-1]
    torch.manual_seed(dsv4_parity_gate.PARITY_SEED)
    compressed_topk = torch.randint(-1, 4, (l_local, compressed_width), dtype=torch.int32)

    exp_idxs, exp_rank_major = _naive_build_indices_mode2(
        cu.tolist(),
        global_start,
        l_local,
        d_window,
        window_size,
        ratio,
        compressed_width,
        compressed_topk.tolist(),
        cu_seqlens_compressed=cu_compressed.tolist() if cu_compressed is not None else None,
        seq_to_rank_row=seq_to_rank_row,
    )

    topk_idxs, topk_length, indexer_rank_major = csa_utils.build_attention_indices(
        cu,
        global_start,
        l_local,
        d_window,
        window_size,
        ratio,
        compressed_width,
        compressed_topk=compressed_topk,
        cu_seqlens_compressed=cu_compressed,
        seq_to_rank_row=torch.tensor(seq_to_rank_row, dtype=torch.int32),
        for_indexer_loss=True,
    )
    assert topk_length is None, "mode 2 does not produce a topk_length"
    assert topk_idxs.shape == (l_local, window_size + compressed_width)
    dsv4_parity_gate.assert_index_parity(
        topk_idxs,
        torch.tensor(exp_idxs, dtype=torch.int32),
        topology=f"cpu-oracle:cu_seqlens={cu_seqlens}",
        layer_mode="mode2-indexer-loss",
        shape=(l_local, window_size + compressed_width),
    )
    dsv4_parity_gate.assert_index_parity(
        indexer_rank_major,
        torch.tensor(exp_rank_major, dtype=torch.int32),
        topology=f"cpu-oracle:cu_seqlens={cu_seqlens}",
        layer_mode="mode2-indexer-loss",
        shape=(l_local, compressed_width),
    )


def test_build_attention_indices_mode2_unlowerable_rank_rows():
    """Mode 2 must skip compressed rows that cannot lower to a rank-major
    row while still placing later valid rows and the window block."""
    cu = torch.tensor([0, 5, 21, 40], dtype=torch.int32)
    cu_compressed = torch.tensor([0, 1, 4, 8], dtype=torch.int32)
    global_start, l_local, d_window, window_size, ratio, compressed_width = (
        11,
        18,
        4,
        4,
        4,
        3,
    )
    # Rows 1, 3 and 6 of the sequence-major compressed buffer are unavailable
    # on this rank (owned elsewhere / padding).
    seq_to_rank_row = [0, -1, 2, -1, 3, 4, -1, 5]
    torch.manual_seed(dsv4_parity_gate.PARITY_SEED)
    compressed_topk = torch.randint(0, 5, (l_local, compressed_width), dtype=torch.int32)

    exp_idxs, exp_rank_major = _naive_build_indices_mode2(
        cu.tolist(),
        global_start,
        l_local,
        d_window,
        window_size,
        ratio,
        compressed_width,
        compressed_topk.tolist(),
        cu_seqlens_compressed=cu_compressed.tolist(),
        seq_to_rank_row=seq_to_rank_row,
    )
    topk_idxs, _, indexer_rank_major = csa_utils.build_attention_indices(
        cu,
        global_start,
        l_local,
        d_window,
        window_size,
        ratio,
        compressed_width,
        compressed_topk=compressed_topk,
        cu_seqlens_compressed=cu_compressed,
        seq_to_rank_row=torch.tensor(seq_to_rank_row, dtype=torch.int32),
        for_indexer_loss=True,
    )
    dsv4_parity_gate.assert_index_parity(
        topk_idxs,
        torch.tensor(exp_idxs, dtype=torch.int32),
        topology="cpu-oracle:unlowerable-rank-rows",
        layer_mode="mode2-indexer-loss",
        shape=(l_local, window_size + compressed_width),
    )
    dsv4_parity_gate.assert_index_parity(
        indexer_rank_major,
        torch.tensor(exp_rank_major, dtype=torch.int32),
        topology="cpu-oracle:unlowerable-rank-rows",
        layer_mode="mode2-indexer-loss",
        shape=(l_local, compressed_width),
    )


# ===========================================================================
# build_flat_topk_idxs / local_to_global_flat reference (flat index packing)
# ===========================================================================


def _naive_global_sbhd(idxs_combined, batch_size):
    """Serial SBHD-flat conversion: global row = local * B + b."""
    b, sq, topk = idxs_combined.shape
    out = [[-1] * topk for _ in range(sq * b)]
    for i in range(b):
        for s in range(sq):
            for k in range(topk):
                v = int(idxs_combined[i, s, k])
                out[s * b + i][k] = v * b + i if v >= 0 else -1
    return out


def _naive_global_thd(idxs_combined, cu_seqlens_q, cu_seqlens_kv):
    """Serial THD-flat conversion: global row = cu_seqlens_kv[batch(q)] + local.

    Query rows beyond ``cu_seqlens_q[-1]`` (padded capacity) clamp to the
    last segment, mirroring ``batch_of_row``.
    """
    total_q, topk = idxs_combined.shape
    n_seq = len(cu_seqlens_q) - 1

    def batch_of(q):
        seq = n_seq - 1
        for s in range(n_seq):
            if q < cu_seqlens_q[s + 1]:
                seq = s
                break
        return seq

    out = [[-1] * topk for _ in range(total_q)]
    for q in range(total_q):
        bq = batch_of(q)
        offset = cu_seqlens_kv[bq]
        for k in range(topk):
            v = int(idxs_combined[q, k])
            out[q][k] = v + offset if v >= 0 else -1
    return out


@pytest.mark.parametrize("b, sq, topk_1, topk_2", dsv4_parity_gate.FLAT_INDEX_SBHD_CASES)
def test_build_flat_topk_idxs_sbhd_matches_naive(b, sq, topk_1, topk_2):
    """SBHD flat packing must match the serial reference, preserving -1."""
    torch.manual_seed(dsv4_parity_gate.PARITY_SEED)
    group_1 = torch.randint(-1, 3, (b, sq, topk_1), dtype=torch.int32)
    group_2 = torch.randint(-1, 5, (b, sq, topk_2), dtype=torch.int32)
    combined = torch.cat((group_1, group_2), dim=-1)

    flat, length = csa_utils.build_flat_topk_idxs(
        group_1, group_2, batch_size=b, compact=False
    )
    assert length is None
    assert flat.shape == (sq * b, topk_1 + topk_2)
    expected = torch.tensor(_naive_global_sbhd(combined, b), dtype=torch.int32)
    dsv4_parity_gate.assert_index_parity(
        flat,
        expected,
        topology=f"cpu-oracle:sbhd-flat:b={b}:sq={sq}",
        layer_mode="flat-index-packing",
        shape=(sq * b, topk_1 + topk_2),
    )


@pytest.mark.parametrize(
    "cu_seqlens, total_q, topk", dsv4_parity_gate.FLAT_INDEX_THD_CASES
)
def test_build_flat_topk_idxs_thd_matches_naive(cu_seqlens, total_q, topk):
    """THD flat packing (per-sequence KV offsets, orphan-row clamping)."""
    torch.manual_seed(dsv4_parity_gate.PARITY_SEED)
    cu_q = torch.tensor(cu_seqlens, dtype=torch.int32)
    idxs = torch.randint(-1, 7, (total_q, topk), dtype=torch.int32)

    flat, length = csa_utils.build_flat_topk_idxs(
        idxs,
        batch_size=0,
        compact=False,
        cu_seqlens_q=cu_q,
        cu_seqlens_kv=cu_q,
    )
    assert length is None
    assert flat.shape == (total_q, topk)
    expected = torch.tensor(
        _naive_global_thd(idxs.tolist(), cu_seqlens, cu_seqlens), dtype=torch.int32
    )
    dsv4_parity_gate.assert_index_parity(
        flat,
        expected,
        topology=f"cpu-oracle:thd-flat:cu_seqlens={cu_seqlens}:total_q={total_q}",
        layer_mode="flat-index-packing",
        shape=(total_q, topk),
    )


def test_build_flat_topk_idxs_thd_uses_kv_boundaries():
    """THD packing offsets by cu_seqlens_kv, which may differ from Q."""
    torch.manual_seed(dsv4_parity_gate.PARITY_SEED)
    cu_q = torch.tensor([0, 4, 12], dtype=torch.int32)
    cu_kv = torch.tensor([0, 6, 18], dtype=torch.int32)
    idxs = torch.randint(-1, 5, (12, 4), dtype=torch.int32)

    flat, _ = csa_utils.build_flat_topk_idxs(
        idxs,
        batch_size=0,
        compact=False,
        cu_seqlens_q=cu_q,
        cu_seqlens_kv=cu_kv,
    )
    expected = torch.tensor(
        _naive_global_thd(idxs.tolist(), [0, 4, 12], [0, 6, 18]), dtype=torch.int32
    )
    dsv4_parity_gate.assert_index_parity(
        flat,
        expected,
        topology="cpu-oracle:thd-flat:distinct-kv-boundaries",
        layer_mode="flat-index-packing",
        shape=(12, 4),
    )


def test_build_flat_topk_idxs_compact_prefix_contract():
    """compact=True must move valid entries to a row prefix and report the
    exact prefix length (no power-of-two padding lanes)."""
    torch.manual_seed(dsv4_parity_gate.PARITY_SEED)
    b, sq, topk_1, topk_2 = 2, 8, 4, 3
    group_1 = torch.randint(-1, 3, (b, sq, topk_1), dtype=torch.int32)
    group_2 = torch.randint(-1, 5, (b, sq, topk_2), dtype=torch.int32)

    flat, length = csa_utils.build_flat_topk_idxs(
        group_1, group_2, batch_size=b, compact=True
    )
    expected = torch.tensor(_naive_global_sbhd(torch.cat((group_1, group_2), dim=-1), b))
    assert flat.shape == expected.shape
    assert length.shape == (sq * b,)
    for row in range(sq * b):
        valid = expected[row] >= 0
        count = int(valid.sum().item())
        assert length[row].item() == count
        # Valid entries keep their original relative order at the prefix
        # (compaction moves them to the front, it never reorders them).
        assert torch.equal(flat[row, :count], expected[row][valid])
        # The suffix is exactly -1 and never carries a padding lane.
        suffix = torch.full((flat.shape[1] - count,), -1, dtype=flat.dtype)
        assert torch.equal(flat[row, count:], suffix)


def test_build_flat_topk_idxs_invalid_row_all_minus_one():
    """A fully invalid row must produce length 0 and all -1 entries."""
    torch.manual_seed(dsv4_parity_gate.PARITY_SEED)
    b, sq, topk = 1, 4, 5
    idxs = torch.full((b, sq, topk), -1, dtype=torch.int32)
    flat, length = csa_utils.build_flat_topk_idxs(idxs, batch_size=b, compact=True)
    assert torch.equal(flat, torch.full((sq, topk), -1, dtype=torch.int32))
    assert torch.equal(length, torch.zeros(sq, dtype=torch.int32))


# ===========================================================================
# cp_utils dispatcher wiring
# ===========================================================================


def test_get_thd_cp_position_ids_public_name():
    """The public name used by deepseek_v4_hybrid_attention must resolve."""
    assert hasattr(cp_utils, "get_thd_cp_position_ids")
    cu = torch.tensor([0, 4, 12], dtype=torch.int32)
    out = cp_utils.get_thd_cp_position_ids(cu, global_start=2, local_rows=4)
    assert torch.equal(out, torch.tensor([2, 3, 0, 1], dtype=torch.int32))


def test_get_sbhd_contiguous_cp_position_ids():
    expected = torch.arange(8, 16, dtype=torch.long)
    out = cp_utils.get_cp_position_ids(
        local_rows=8,
        device=expected.device,
        global_start=8,
    )
    assert torch.equal(out, expected)


def test_get_thd_contiguous_cp_position_ids():
    cu = torch.tensor([0, 5, 12], dtype=torch.int32)
    out = cp_utils.get_cp_position_ids(
        local_rows=4,
        device=cu.device,
        global_start=3,
        cu_seqlens_padded=cu,
    )
    assert torch.equal(out, torch.tensor([3, 4, 0, 1], dtype=torch.long))


def test_unfused_cp_rope_contiguous_positions_and_inverse():
    torch.manual_seed(5)
    x = torch.randn(4, 2, 1, 6)
    # A valid non-interleaved RoPE table repeats each pair's angle in the
    # second half: [theta_0, theta_1, theta_0, theta_1]. Independent random
    # values in all four columns do not describe rotations, so negating their
    # sine terms would not be the inverse of the forward transform.
    half_freqs = torch.randn(16, 1, 1, 2)
    freqs = torch.cat((half_freqs, half_freqs), dim=-1)
    config = SimpleNamespace(rotary_interleaved=False)
    rotated = cp_utils.apply_cp_local_rope_unfused(
        x, freqs, 2, 4, None, 4, config
    )
    restored = cp_utils.apply_cp_local_rope_unfused(
        rotated,
        freqs,
        2,
        4,
        None,
        4,
        config,
        inverse=True,
    )
    torch.testing.assert_close(restored, x, rtol=1e-5, atol=1e-6)


def test_csa_internal_positions_request_unsharded_rope_table():
    class RecordingRotaryEmbedding:
        def __init__(self):
            self.packed_seq_calls = []

        def __call__(self, seq_len, packed_seq=False):
            self.packed_seq_calls.append(packed_seq)
            half = torch.arange(seq_len, dtype=torch.float32).view(seq_len, 1, 1, 1)
            return torch.cat((half, half), dim=-1)

    module = RecordingRotaryEmbedding()
    config = SimpleNamespace(apply_rope_fusion=False, rotary_interleaved=False)
    x = torch.randn(4, 2, 1, 4)

    _apply_rope(
        x,
        nope_dim=2,
        pos_dim=2,
        rotary_pos_emb_module=module,
        config=config,
        rotary_seq_len=8,
        position_ids=torch.arange(4, 8, dtype=torch.long),
    )

    assert module.packed_seq_calls == [True]


def test_prepare_cp_compressor_input_uses_core_torch_reference(monkeypatch):
    called = False
    reference = csa_utils.compressor_input_compact

    def wrapped(*args, **kwargs):
        nonlocal called
        called = True
        return reference(*args, **kwargs)

    monkeypatch.setattr(cp_utils.csa_utils, "compressor_input_compact", wrapped)
    hidden = torch.randn(16, 2, 3)
    boundary = torch.zeros(8, 2, 3)
    cu = torch.tensor([0, 32], dtype=torch.int32)
    cu_compressed = torch.tensor([0, 8], dtype=torch.int32)
    compact, group_ids, seq_to_rank = cp_utils.prepare_cp_compressor_input(
        hidden,
        boundary,
        cu,
        cu_compressed,
        global_start=0,
        cp_size=2,
        ratio=4,
    )
    assert called
    assert compact.ndim == 3 and compact.shape[1:] == (2, 3)
    assert group_ids.ndim == 1
    assert seq_to_rank.shape == (8,)


def test_unfused_sparse_attention_preserves_sbhd_batch_selection_and_backward():
    torch.manual_seed(11)
    query = torch.randn(4, 2, 3, 6, requires_grad=True)
    kv = torch.randn(7, 2, 6, requires_grad=True)
    sink = torch.randn(3, requires_grad=True)
    topk = torch.tensor(
        [
            [[0, -1, -1], [0, 1, -1], [1, 2, 0], [3, 2, 1]],
            [[4, -1, -1], [4, 5, -1], [5, 6, 4], [6, 5, 4]],
        ],
        dtype=torch.int32,
    )
    output = unfused_compressed_sparse_attn(query, kv, sink, topk, softmax_scale=0.5)
    assert output.shape == (4, 2, 18)
    assert not torch.equal(output[:, 0], output[:, 1])
    output.square().mean().backward()
    assert query.grad is not None and torch.isfinite(query.grad).all()
    assert kv.grad is not None and torch.isfinite(kv.grad).all()
    assert sink.grad is not None and torch.isfinite(sink.grad).all()


def _mean_compressor(compact, ratio):
    return compact.reshape(-1, ratio, *compact.shape[1:]).mean(dim=1)


def test_virtual_cp2_sbhd_b2_matches_full_unfused_reference_forward_backward():
    """Exercise the complete portable layout/attention chain without collectives."""
    torch.manual_seed(19)
    seqlen, batch, heads, dim = 32, 2, 2, 6
    cp_size, local_rows, ratio, window = 2, 16, 8, 4
    d_window = 8
    cu = torch.tensor([0, seqlen], dtype=torch.int32)
    cu_compressed = torch.tensor([0, seqlen // ratio], dtype=torch.int32)

    query = torch.randn(seqlen, batch, heads, dim, requires_grad=True)
    kv = torch.randn(seqlen, batch, dim, requires_grad=True)
    hidden = torch.randn(seqlen, batch, dim, requires_grad=True)
    sink = torch.randn(heads, requires_grad=True)

    rank_compressed = []
    mappings = []
    for rank in range(cp_size):
        start = rank * local_rows
        boundary_hidden = (
            torch.zeros(d_window, batch, dim)
            if rank == 0
            else hidden[start - d_window : start]
        )
        compact, _, mapping = cp_utils.prepare_cp_compressor_input(
            hidden[start : start + local_rows],
            boundary_hidden,
            cu,
            cu_compressed,
            global_start=start,
            cp_size=cp_size,
            ratio=ratio,
        )
        rank_compressed.append(_mean_compressor(compact, ratio))
        mappings.append(mapping)
    assert torch.equal(mappings[0], mappings[1])
    rank_major_compressed = torch.cat(rank_compressed, dim=0)
    seq_to_rank = mappings[0]

    cp_outputs = []
    for rank in range(cp_size):
        start = rank * local_rows
        boundary_kv = (
            torch.zeros(d_window, batch, dim)
            if rank == 0
            else kv[start - d_window : start]
        )
        kv_full = torch.cat(
            (boundary_kv, kv[start : start + local_rows], rank_major_compressed), dim=0
        )
        indices, _, _ = csa_utils.build_attention_indices(
            cu,
            start,
            local_rows,
            d_window,
            window,
            ratio,
            seqlen // ratio,
            cu_seqlens_compressed=cu_compressed,
            seq_to_rank_row=seq_to_rank,
        )
        topk = indices.unsqueeze(0).expand(batch, -1, -1).contiguous()
        cp_outputs.append(
            unfused_compressed_sparse_attn(
                query[start : start + local_rows], kv_full, sink, topk, softmax_scale=0.4
            )
        )
    cp_output = torch.cat(cp_outputs, dim=0)

    compressed_reference = hidden.reshape(
        seqlen // ratio, ratio, batch, dim
    ).mean(dim=1)
    reference_kv = torch.cat((kv, compressed_reference), dim=0)
    reference_indices = torch.full(
        (batch, seqlen, window + seqlen // ratio), -1, dtype=torch.int32
    )
    for q_row in range(seqlen):
        window_rows = list(range(max(0, q_row - window + 1), q_row + 1))
        compressed_rows = [
            seqlen + group for group in range((q_row + 1) // ratio)
        ]
        rows = window_rows + compressed_rows
        reference_indices[:, q_row, : len(rows)] = torch.tensor(rows, dtype=torch.int32)
    reference_output = unfused_compressed_sparse_attn(
        query, reference_kv, sink, reference_indices, softmax_scale=0.4
    )
    torch.testing.assert_close(cp_output, reference_output, rtol=1e-5, atol=1e-6)

    cp_grads = torch.autograd.grad(cp_output.square().sum(), (query, kv, hidden, sink), retain_graph=True)
    ref_grads = torch.autograd.grad(reference_output.square().sum(), (query, kv, hidden, sink))
    for cp_grad, ref_grad in zip(cp_grads, ref_grads):
        torch.testing.assert_close(cp_grad, ref_grad, rtol=2e-5, atol=2e-6)


def test_virtual_cp2_thd_ragged_matches_full_unfused_reference_forward_backward():
    torch.manual_seed(23)
    seqlen, heads, dim = 32, 2, 6
    cp_size, local_rows, ratio, window, d_window = 2, 16, 8, 4, 8
    cu = torch.tensor([0, 8, 32], dtype=torch.int32)
    cu_compressed = torch.tensor([0, 1, 4], dtype=torch.int32)
    query = torch.randn(seqlen, 1, heads, dim, requires_grad=True)
    kv = torch.randn(seqlen, 1, dim, requires_grad=True)
    hidden = torch.randn(seqlen, 1, dim, requires_grad=True)
    sink = torch.randn(heads, requires_grad=True)

    compressed_by_rank = []
    mappings = []
    for rank in range(cp_size):
        start = rank * local_rows
        boundary = torch.zeros(d_window, 1, dim) if rank == 0 else hidden[start - d_window : start]
        compact, _, mapping = cp_utils.prepare_cp_compressor_input(
            hidden[start : start + local_rows],
            boundary,
            cu,
            cu_compressed,
            start,
            cp_size,
            ratio,
        )
        compressed_by_rank.append(_mean_compressor(compact, ratio))
        mappings.append(mapping)
    rank_major = torch.cat(compressed_by_rank, dim=0)
    assert torch.equal(mappings[0], mappings[1])

    cp_outputs = []
    for rank in range(cp_size):
        start = rank * local_rows
        boundary = torch.zeros(d_window, 1, dim) if rank == 0 else kv[start - d_window : start]
        kv_full = torch.cat((boundary, kv[start : start + local_rows], rank_major), dim=0)
        indices, _, _ = csa_utils.build_attention_indices(
            cu,
            start,
            local_rows,
            d_window,
            window,
            ratio,
            3,
            cu_seqlens_compressed=cu_compressed,
            seq_to_rank_row=mappings[0],
        )
        cp_outputs.append(
            unfused_compressed_sparse_attn(
                query[start : start + local_rows],
                kv_full,
                sink,
                indices.unsqueeze(0),
                softmax_scale=0.4,
            )
        )
    cp_output = torch.cat(cp_outputs, dim=0)

    compressed_reference = torch.cat(
        tuple(
            hidden[int(cu[i]) : int(cu[i + 1])].reshape(-1, ratio, 1, dim).mean(dim=1)
            for i in range(cu.numel() - 1)
        ),
        dim=0,
    )
    reference_kv = torch.cat((kv, compressed_reference), dim=0)
    reference_indices = torch.full((1, seqlen, window + 3), -1, dtype=torch.int32)
    for seq in range(cu.numel() - 1):
        seq_start, seq_end = int(cu[seq]), int(cu[seq + 1])
        comp_start = int(cu_compressed[seq])
        for q_row in range(seq_start, seq_end):
            window_rows = list(range(max(seq_start, q_row - window + 1), q_row + 1))
            compressed_rows = [
                seqlen + comp_start + group
                for group in range((q_row - seq_start + 1) // ratio)
            ]
            rows = window_rows + compressed_rows
            reference_indices[0, q_row, : len(rows)] = torch.tensor(rows, dtype=torch.int32)
    reference_output = unfused_compressed_sparse_attn(
        query, reference_kv, sink, reference_indices, softmax_scale=0.4
    )
    torch.testing.assert_close(cp_output, reference_output, rtol=1e-5, atol=1e-6)
    cp_grads = torch.autograd.grad(cp_output.square().sum(), (query, kv, hidden, sink), retain_graph=True)
    ref_grads = torch.autograd.grad(reference_output.square().sum(), (query, kv, hidden, sink))
    for cp_grad, ref_grad in zip(cp_grads, ref_grads):
        torch.testing.assert_close(cp_grad, ref_grad, rtol=2e-5, atol=2e-6)


def test_cp_boundary_exchange_forward_and_backward_under_torchrun():
    """Run with torchrun --nproc-per-node>=2 to validate the P2P autograd edge."""
    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() < 2:
        pytest.skip("requires a torchrun-initialized process group with at least two ranks")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local = torch.full(
        (4, 2),
        float(rank),
        device=torch.device(get_platform().current_device_name()),
        requires_grad=True,
    )
    boundary = cp_utils._LeftBoundaryExchange.apply(local, 2, dist.group.WORLD)
    expected = torch.zeros_like(boundary) if rank == 0 else torch.full_like(boundary, rank - 1)
    assert torch.equal(boundary, expected)
    boundary.sum().backward()
    expected_grad = torch.zeros_like(local)
    if rank + 1 < world:
        expected_grad[-2:] = 1
    assert torch.equal(local.grad, expected_grad)
