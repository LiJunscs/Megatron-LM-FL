# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Frozen DSv4 oracle-parity gate (Phase 0 of the Triton replacement plan).

Single source of truth for the fixed seed, canonical case shapes, dtype,
tolerance contract and the canonical parity-failure diagnostic required by
``docs/developer/dsv4_triton_fused_kernel_replacement_plan.md`` (Phase 0 and
Section 6).

Conventions:

* Every test that generates randoms for an oracle/fused comparison seeds
  ``torch.manual_seed(PARITY_SEED)`` (plus an optional per-case offset).
* Inputs are BF16 (``PARITY_DTYPE``) and reductions are always validated in
  FP32; the tensor-parity gate asserts in the input dtype with a fixed
  rtol/atol pair and additionally reports FP32 metrics.
* Failure diagnostics print topology, layer mode, shape, max absolute error,
  relative-L2 and cosine similarity with a stable key=value format, so a
  parity regression can be bisected without re-running the experiment.

This module must not import ``megatron``: it is the CPU-runnable half of the
gate that can run in environments without CUDA.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# Frozen input contract
# ---------------------------------------------------------------------------

# Fixed seed shared by all oracle/fused input generators in Phase 0 tests.
PARITY_SEED = 42

# Frozen dtype: BF16 inputs; all critical reductions are validated in FP32.
PARITY_DTYPE = torch.bfloat16

# Frozen tolerance contract for the PyTorch oracle vs fused/native parity.
# Integer index tensors and metadata must match exactly (INDEX_EXACT); only
# floating-point activations use rtol/atol.
PARITY_RTOL = 1e-5
PARITY_ATOL = 1e-6
INDEX_PARITY = "exact"

# ---------------------------------------------------------------------------
# Canonical case shapes (frozen once; extend only with the plan's approval)
# ---------------------------------------------------------------------------

# compressor_input_compact:
#   (cu_seqlens, global_start, l_local, d_comp, ratio, c_cap)
CompactionCase = Tuple[Sequence[int], int, int, int, int, int]
COMPACT_CASES: Tuple[CompactionCase, ...] = (
    # single sequence starting at 0 (no boundary).
    ([0, 32], 0, 16, 4, 4, 8),
    # single sequence spanning multiple CP ranks.
    ([0, 40], 8, 16, 4, 4, 8),
    # two sequences; rank starts inside the first.
    ([0, 14, 40], 8, 16, 4, 4, 8),
    # ragged multiple sequences.
    ([0, 5, 21, 40], 11, 18, 4, 4, 8),
    # ratio-4 overlap needs d_comp=8.
    ([0, 32], 8, 16, 8, 4, 8),
    # ragged sequences with d_comp=8.
    ([0, 7, 9, 40], 14, 10, 8, 4, 6),
    # CP tail padding: zero visible compressed rows.
    ([0, 32], 32, 16, 4, 4, 8),
    # ragged sequence with CP tail padding (only the final group's tail
    # overlaps the rank interval).
    ([0, 5, 21, 40], 34, 12, 4, 4, 8),
    # non-power-of-two compressed capacity.
    ([0, 30], 6, 12, 4, 4, 4),
    # ratio 128 (no indexer); capacity 2.
    ([0, 40], 8, 16, 4, 128, 2),
)

# build_attention_indices case tuple:
#   (cu_seqlens, cu_seqlens_compressed, global_start, l_local, d_window,
#    window_size, ratio, compressed_width)
AttentionIndexCase = Tuple[Sequence[int], Optional[Sequence[int]], int, int, int, int, int, int]

# ``cu_seqlens_compressed`` is None when compressed boundaries equal the
# token boundaries (single layout).
#
# Cases whose rank interval extends beyond ``cu_seqlens[-1]`` (CP tail
# padding / truncated tail) produce ``in_seq == False`` (fallback) rows;
# all three modes are exercised on every case below.  The unfused oracle
# previously raised an out-of-bounds scatter in mode 1 for such rows
# (``compact_cols = window_count + c`` unclamped); it is clamped to a
# padding column now (see ``csa_utils.cp_layout.build_attention_indices``), so
# the full table is valid for all modes.
ATTENTION_INDEX_CASES: Tuple[AttentionIndexCase, ...] = (
    # single sequence starting at 0 (no boundary).
    ([0, 32], None, 0, 16, 4, 4, 4, 2),
    # single sequence spanning multiple CP ranks.
    ([0, 40], None, 8, 16, 4, 4, 4, 2),
    # ragged multiple sequences.
    ([0, 5, 21, 40], None, 11, 18, 4, 4, 4, 2),
    # ragged sequences with non-power-of-two compressed width.
    ([0, 5, 21, 40], None, 11, 18, 4, 4, 4, 3),
    # ragged compressed boundaries distinct from token boundaries.
    ([0, 5, 21, 40], [0, 1, 4, 8], 11, 18, 4, 4, 4, 2),
    # non-power-of-two compressed width.
    ([0, 32], None, 8, 16, 8, 8, 4, 3),
    # ratio 128 (no indexer).
    ([0, 40], None, 8, 16, 4, 4, 128, 1),
    # non-power-of-two window size.
    ([0, 50], None, 20, 14, 4, 6, 4, 2),
    # CP tail padding with non-power-of-two window (fallback rows).
    ([0, 24], None, 16, 16, 4, 5, 4, 3),
    # ragged sequences with CP tail padding and non-power-of-two window.
    ([0, 5, 21, 40], None, 30, 16, 8, 6, 4, 3),
    # rank interval extends beyond every sequence (fallback rows only).
    ([0, 40], None, 0, 48, 4, 4, 4, 2),
)

# build_flat_topk_idxs / local_to_global_flat SBHD shapes:
#   (batch, sq, topk_1, topk_2) -- two index groups are always combined.
FLAT_INDEX_SBHD_CASES: Tuple[Tuple[int, int, int, int], ...] = (
    (1, 4, 4, 3),
    (2, 8, 4, 3),
    (3, 6, 2, 2),
)

# build_flat_topk_idxs / local_to_global_flat THD layouts:
#   (cu_seqlens_q, total_q, topk) -- orphan rows (total_q > cu_seqlens[-1])
#   exercise the padded-capacity clamping of batch_of_row.
FLAT_INDEX_THD_CASES: Tuple[Tuple[Sequence[int], int, int], ...] = (
    ([0, 4, 12], 12, 5),
    ([0, 4, 12], 16, 5),
    ([0, 6], 6, 3),
)


# ---------------------------------------------------------------------------
# Parity metrics and canonical failure diagnostic
# ---------------------------------------------------------------------------


def parity_metrics(
    actual: Tensor, expected: Tensor
) -> Tuple[float, float, float, float, float, float]:
    """Return FP32 metrics ``(cosine, norm_ratio, relative_l2, max_abs,
    reference_max_abs, minimum_atol_eq_rtol)`` between two tensors.

    All metrics are computed in FP32 after flattening, mirroring the
    diagnostic vocabulary used by the DSv4 hybrid attention suite.
    """
    actual_flat = actual.float().reshape(-1)
    expected_flat = expected.float().reshape(-1)
    expected_norm = torch.linalg.vector_norm(expected_flat).clamp_min(1e-12)
    error = actual_flat - expected_flat
    if expected_flat.numel() == 0:
        cosine = 1.0 if torch.equal(actual_flat, expected_flat) else 0.0
    else:
        cosine = torch.nn.functional.cosine_similarity(
            actual_flat.unsqueeze(0), expected_flat.unsqueeze(0)
        ).item()
    actual_norm = torch.linalg.vector_norm(actual_flat)
    norm_ratio = (actual_norm / expected_norm).item()
    relative_l2 = (torch.linalg.vector_norm(error) / expected_norm).item()
    max_abs = error.abs().max().item() if error.numel() else 0.0
    reference_max_abs = expected_flat.abs().max().item() if expected_flat.numel() else 0.0
    minimum_atol_eq_rtol = (
        error.abs() / (1.0 + expected_flat.abs())
    ).max().item() if error.numel() else 0.0
    return (
        cosine,
        norm_ratio,
        relative_l2,
        max_abs,
        reference_max_abs,
        minimum_atol_eq_rtol,
    )


def format_parity_report(
    *,
    topology: str,
    layer_mode: str,
    shape: Tuple[int, ...],
    actual: Tensor,
    expected: Tensor,
    rtol: Optional[float] = None,
    atol: Optional[float] = None,
    extra: Optional[str] = None,
) -> str:
    """Canonical parity-failure diagnostic (plan Section 6).

    Prints topology, layer mode, shape, dtype, the FP32 metric set (maximum
    absolute error, relative-L2, cosine) and the tolerance contract that was
    violated.
    """
    cosine, norm_ratio, relative_l2, max_abs, ref_max_abs, min_tol = parity_metrics(
        actual, expected
    )
    parts = [
        f"DSV4 parity mismatch",
        f"topology={topology}",
        f"layer_mode={layer_mode}",
        f"shape={tuple(shape)}",
        f"dtype={expected.dtype}",
        f"max_abs={max_abs:.9e}",
        f"relative_l2={relative_l2:.9e}",
        f"cosine={cosine:.9f}",
        f"norm_ratio={norm_ratio:.9f}",
        f"reference_max_abs={ref_max_abs:.9e}",
        f"minimum_atol_eq_rtol={min_tol:.9e}",
    ]
    if rtol is not None:
        parts.append(f"rtol={rtol:.0e}")
    if atol is not None:
        parts.append(f"atol={atol:.0e}")
    if extra:
        parts.append(extra)
    return "; ".join(parts)


def assert_tensor_parity(
    actual: Tensor,
    expected: Tensor,
    *,
    topology: str,
    layer_mode: str,
    shape: Optional[Iterable[int]] = None,
    rtol: float = PARITY_RTOL,
    atol: float = PARITY_ATOL,
    extra: Optional[str] = None,
) -> None:
    """Gate a floating-point tensor against the frozen oracle contract.

    Raises with the canonical diagnostic from :func:`format_parity_report`
    when finite-ness, shape or the rtol/atol contract is violated.
    """
    expected_shape = tuple(expected.shape) if shape is None else tuple(shape)
    if tuple(actual.shape) != expected_shape:
        raise AssertionError(
            format_parity_report(
                topology=topology,
                layer_mode=layer_mode,
                shape=tuple(actual.shape),
                actual=actual,
                expected=expected,
                rtol=rtol,
                atol=atol,
                extra=f"expected_shape={expected_shape}",
            )
        )
    if not torch.isfinite(actual).all():
        raise AssertionError(
            format_parity_report(
                topology=topology,
                layer_mode=layer_mode,
                shape=expected_shape,
                actual=actual,
                expected=expected,
                rtol=rtol,
                atol=atol,
                extra="actual contains NaN or Inf",
            )
        )
    if not torch.isfinite(expected).all():
        raise AssertionError(
            format_parity_report(
                topology=topology,
                layer_mode=layer_mode,
                shape=expected_shape,
                actual=actual,
                expected=expected,
                rtol=rtol,
                atol=atol,
                extra="expected contains NaN or Inf",
            )
        )
    if torch.equal(actual, expected):
        return
    try:
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    except AssertionError:
        raise AssertionError(
            format_parity_report(
                topology=topology,
                layer_mode=layer_mode,
                shape=expected_shape,
                actual=actual,
                expected=expected,
                rtol=rtol,
                atol=atol,
            )
        ) from None


def assert_index_parity(
    actual: Tensor,
    expected: Tensor,
    *,
    topology: str,
    layer_mode: str,
    shape: Optional[Iterable[int]] = None,
    extra: Optional[str] = None,
) -> None:
    """Gate an integer index/metadata tensor: must match the oracle exactly.

    Index values are a semantic contract (they name KV rows), so a single
    mismatched entry is a hard failure, reported with the canonical
    diagnostic plus the count and location of the first mismatch.
    """
    expected_shape = tuple(expected.shape) if shape is None else tuple(shape)
    if tuple(actual.shape) != expected_shape:
        raise AssertionError(
            f"DSV4 index parity mismatch; topology={topology}; layer_mode={layer_mode}; "
            f"shape={tuple(actual.shape)}; expected_shape={expected_shape}"
        )
    if torch.equal(actual, expected):
        return
    mismatch = (actual != expected)
    flat_index = mismatch.reshape(-1).nonzero().flatten()
    first = int(flat_index[0].item()) if flat_index.numel() else -1
    raise AssertionError(
        f"DSV4 index parity mismatch; topology={topology}; layer_mode={layer_mode}; "
        f"shape={expected_shape}; mismatched={int(mismatch.sum().item())}/{mismatch.numel()}; "
        f"first_flat_index={first}"
    )
