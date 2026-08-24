# Copyright (c) 2026, FlagOS / Megatron-LM-FL. All rights reserved.

"""Semantic parity gate for the DSv4 indexer primitives (plan Phase 2A).

Frozen contracts validated here (all against the PyTorch oracle in
``megatron.core.transformer.experimental_attention_variant.dsa`` and the
CP caller conversion in ``csa.py``):

1. **Q/K score formula** — ``_compute_index_scores`` semantics: FP32 einsum of
   BF16 inputs, ReLU, per-head weight multiply, head sum.  The softmax scale
   belongs to the *weights* and the caller pre-scales them in FP32
   (``weights.float() * scale``), never inside the score op.
2. **causal / ratio mask** — ``valid(q, k) == (k < floor((q + 1) / ratio))``;
   for ratio=1 this equals the standard upper-triangular causal mask.
3. **stable top-k with invalid fill** — indices, per-row ``topk_length`` and
   the -1 / padding lanes match ``torch.topk`` on the oracle-masked scores
   after the CP caller's invalid conversion; ``topk_length`` counts the
   effective prefix (no power-of-two padding lanes).
4. **tie / NaN / Inf behavior** — the fused path must pick the exact same
   rows as the oracle for identical inputs.
5. **loss reduction contract** — local mean vs ``calculate_per_token_loss``
   raw sum must scale loss and gradients by the same divisor.

These checks freeze the PyTorch-composite semantics currently hosted by the
Triton backend package.  They do not by themselves prove that a Triton JIT
kernel ran; GPU compile/launch parity belongs to Phase 2B.

The module/topology gates (TP1, TP, CP, TP-SP-CP loss and indexer parameter
gradients) live in ``test_attention_variant_dsa.py``,
``test_fused_dsa_tp.py`` and ``test_dsv4_hybrid_attention_cp.py``.
"""

import pytest
import torch

from megatron.core.transformer.experimental_attention_variant.dsa import (
    _compute_index_scores,
)
from tests.unit_tests.transformer.experimental_attention_variant import dsv4_parity_gate

try:
    from megatron.plugin.dsa_kernel.backends.triton import indexer as tri_indexer
    from megatron.plugin.dsa_kernel.backends.triton import fused_ops as tri_fused_ops
    from megatron.plugin.dsa_kernel.backends.triton import utils as tri_utils

    _HAS_TRITON = True
except Exception:  # pragma: no cover - triton-less environments
    tri_indexer = None
    tri_fused_ops = None
    tri_utils = None
    _HAS_TRITON = False

pytestmark = pytest.mark.skipif(
    not _HAS_TRITON, reason="DSv4 indexer primitive parity requires the triton backend"
)

_PARITY_SEED = dsv4_parity_gate.PARITY_SEED

# Score tolerance: the oracle and fused paths both contract to FP32 einsum,
# but different cuBLAS layouts may accumulate in a different order, so the
# gate is a tight relative/absolute pair instead of bitwise equality.
_SCORE_RTOL = 1e-4
_SCORE_ATOL = 1e-4

_INDEXER_SHAPES = [
    (16, 2, 4, 64, 32),   # (sq, b, idx_nh, idx_hd, sk)
    (32, 1, 2, 128, 24),
    (8, 3, 4, 32, 20),
]


def _make_indexer_inputs(sq, b, idx_nh, idx_hd, sk, seed=_PARITY_SEED):
    """Random BF16 indexer inputs (SBHD) + the oracle pre-scaled weights."""
    torch.manual_seed(seed)
    q = torch.randn(sq, b, idx_nh, idx_hd, dtype=torch.bfloat16)
    k = torch.randn(sk, b, idx_hd, dtype=torch.bfloat16)
    w = torch.randn(sq, b, idx_nh, dtype=torch.bfloat16)
    scale = idx_hd ** -0.5
    w_scaled_fp32 = w.float() * scale  # oracle pre-scaling (csa.py contract)
    return q, k, w, scale, w_scaled_fp32


# ---------------------------------------------------------------------------
# 1. Q/K score formula
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sq, b, idx_nh, idx_hd, sk", _INDEXER_SHAPES)
def test_indexer_scores_match_oracle_formula(sq, b, idx_nh, idx_hd, sk):
    """fused full_scores must equal ``_compute_index_scores`` (FP32)."""
    q, k, w, scale, w_scaled_fp32 = _make_indexer_inputs(sq, b, idx_nh, idx_hd, sk)

    oracle_scores = _compute_index_scores(q, w_scaled_fp32, k)  # (b, sq, sk) fp32

    q_bshd, k_bsd, _, w_bsh_scaled = tri_fused_ops._sbhd_to_bshd_indexer_inputs(
        q, k, w, scale
    )
    fused = tri_indexer.indexer_topk_selection(
        q_bshd, k_bsd, w_bsh_scaled, topk=min(8, sk), ratio=4
    )
    fused_scores = fused["full_scores"]

    assert fused_scores.shape == oracle_scores.shape == (b, sq, sk)
    assert fused_scores.dtype == torch.float32
    dsv4_parity_gate.assert_tensor_parity(
        fused_scores,
        oracle_scores,
        topology=f"indexer-primitive:score:sq={sq}:b={b}:nh={idx_nh}:hd={idx_hd}:sk={sk}",
        layer_mode="indexer-score",
        shape=(b, sq, sk),
        rtol=_SCORE_RTOL,
        atol=_SCORE_ATOL,
    )


def test_indexer_scores_are_fp32_accumulated_from_bf16_inputs():
    """The score op must accumulate in FP32 and never round in BF16."""
    torch.manual_seed(_PARITY_SEED + 9)
    probe = torch.randn(1, 16, 4, 8, dtype=torch.bfloat16)
    k_probe = torch.randn(1, 32, 8, dtype=torch.bfloat16)
    fused = tri_indexer.indexer_topk_selection(
        probe, k_probe, torch.ones(1, 16, 4, dtype=torch.float32), topk=8, ratio=4
    )
    assert fused["full_scores"].dtype == torch.float32
    assert fused["full_scores"].float().isfinite().all()


def test_sbhd_to_bshd_weight_scaling_is_fp32():
    """Regression: weights must be pre-scaled in FP32 like the oracle
    (``weights_indexer_cp.float() * indexer.softmax_scale``); scaling in BF16
    applies roundings that shift near-tie top-K boundaries."""
    torch.manual_seed(_PARITY_SEED + 1)
    q = torch.randn(8, 2, 2, 16, dtype=torch.bfloat16)
    k = torch.randn(12, 2, 16, dtype=torch.bfloat16)
    w = torch.randn(8, 2, 2, dtype=torch.bfloat16)
    scale = 1.0 / 3.0  # not representable in bf16 → two bf16 roundings if scaled there

    _, _, _, w_bsh_scaled = tri_fused_ops._sbhd_to_bshd_indexer_inputs(q, k, w, scale)
    assert w_bsh_scaled.dtype == torch.float32
    assert torch.equal(w_bsh_scaled, w.float() * scale)


# ---------------------------------------------------------------------------
# 2. causal / ratio mask
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ratio", [1, 2, 4, 128])
def test_ratio_causal_mask_matches_formula(ratio):
    sq, sk = 33, 20
    mask = tri_utils.compute_ratio_causal_mask(sq, sk, ratio, torch.device("cpu"))
    assert mask.shape == (sq, sk)
    for q in range(sq):
        expected_valid = min(max((q + 1) // ratio, 0), sk)
        for kk in range(sk):
            expected = 0.0 if kk < expected_valid else float("-inf")
            assert float(mask[q, kk]) == expected, f"q={q} k={kk}"


def test_ratio_causal_mask_ratio1_equals_triu():
    sq, sk = 24, 16
    mask = tri_utils.compute_ratio_causal_mask(sq, sk, 1, torch.device("cpu"))
    triu = torch.triu(torch.full((sq, sk), float("-inf")), diagonal=1)
    assert torch.equal(mask, triu)


# ---------------------------------------------------------------------------
# 3. top-k with invalid fill / padding
# ---------------------------------------------------------------------------


def _oracle_topk_with_caller_conversion(scores, topk, ratio):
    """Oracle ``fused_qk_topk_naive`` + the CP caller's invalid conversion."""
    sq, sk = scores.shape[1:]
    query_rows = torch.arange(sq, device=scores.device, dtype=torch.int64)
    key_rows = torch.arange(sk, device=scores.device, dtype=torch.int64)
    valid = key_rows.unsqueeze(0) < ((query_rows + 1) // ratio).unsqueeze(1)
    masked_scores = scores.masked_fill(~valid.unsqueeze(0), float("-inf"))
    gathered, indices = masked_scores.topk(min(topk, scores.shape[2]), dim=-1)
    invalid = torch.isinf(gathered) & (gathered < 0)
    indices = torch.where(invalid, torch.full_like(indices, -1), indices)
    topk_length = (~invalid).sum(dim=-1).to(torch.int32)
    return masked_scores, indices, topk_length


@pytest.mark.parametrize("b, sq, sk", [(2, 16, 32), (1, 33, 20)])
@pytest.mark.parametrize("ratio", [4, 128])
@pytest.mark.parametrize("topk", [4, 8, 6])
def test_topk_with_causal_mask_matches_oracle_conversion(b, sq, sk, ratio, topk):
    """Indices, order and -1 fill must match oracle top-k + caller conversion."""
    torch.manual_seed(_PARITY_SEED + 2)
    scores = torch.randn(b, sq, sk, dtype=torch.float32)
    _, oracle_idxs, oracle_length = _oracle_topk_with_caller_conversion(
        scores, topk, ratio
    )

    fused_idxs, fused_length = tri_utils.topk_with_causal_mask(scores, topk, ratio)

    effective_k = min(topk, sk)
    assert fused_idxs.shape == (b, sq, topk)
    # The first effective_k columns order-match the oracle conversion.
    assert torch.equal(fused_idxs[..., :effective_k], oracle_idxs)
    # Padding columns (when topk > sk) are -1.
    if topk > sk:
        assert torch.equal(
            fused_idxs[..., effective_k:], torch.full_like(fused_idxs[..., effective_k:], -1)
        )
    # topk_length == valid count in the materialized top-k prefix.
    assert torch.equal(fused_length, oracle_length)


def test_topk_length_is_effective_prefix_no_pow2_padding():
    """Non-power-of-two topk must report the true valid prefix."""
    b, sq, sk, ratio = 2, 16, 24, 4
    torch.manual_seed(_PARITY_SEED + 3)
    q_bshd = torch.randn(b, sq, 4, 64, dtype=torch.bfloat16)
    k_bsd = torch.randn(b, sk, 64, dtype=torch.bfloat16)
    w_bsh = torch.randn(b, sq, 4, dtype=torch.bfloat16)
    topk = 6  # non-power-of-two

    idxs, length, _ = tri_fused_ops._indexer_topk_bshd(q_bshd, k_bsd, w_bsh, topk, ratio)
    assert idxs.shape == (b, sq, topk)
    assert (length <= topk).all()
    # Effective prefix: every reported-valid lane is a real index, and every
    # column ≥ length is -1.
    for kk in range(topk):
        assert torch.equal(
            idxs[:, :, kk] >= 0,
            (length > kk),
        ), f"column {kk}: length must describe the valid prefix exactly"


def test_topk_tie_behavior_identical_to_oracle():
    """Tied rows must be resolved by the same torch.topk as the oracle."""
    b, sq, sk, ratio, topk = 2, 4, 12, 4, 5
    torch.manual_seed(_PARITY_SEED + 4)
    scores = torch.randn(b, sq, sk, dtype=torch.float32)
    # Force exact ties in one column pair for every row and overlap some
    # columns across rows to produce many ties on the selection boundary.
    scores[..., 0] = scores[..., 1] = 1.2345
    scores[..., 3] = scores[..., 4] = -0.75

    _, oracle_idxs, oracle_length = _oracle_topk_with_caller_conversion(
        scores, topk, ratio
    )
    fused_idxs, fused_length = tri_utils.topk_with_causal_mask(scores, topk, ratio)

    assert torch.equal(fused_idxs[..., :min(topk, sk)], oracle_idxs)
    assert torch.equal(fused_length, oracle_length)


def test_topk_nan_inf_behavior_matches_oracle():
    """NaN rows propagate exactly like oracle top-k; -inf rows become -1."""
    b, sq, sk, ratio, topk = 1, 4, 16, 4, 5
    torch.manual_seed(_PARITY_SEED + 5)
    scores = torch.randn(b, sq, sk, dtype=torch.float32)
    scores[0, 1, 7] = float("nan")  # NaN behaves as "largest" in both paths
    scores[0, 2, :] = float("-inf")  # fully masked row

    _, oracle_idxs, oracle_length = _oracle_topk_with_caller_conversion(
        scores, topk, ratio
    )
    fused_idxs, fused_length = tri_utils.topk_with_causal_mask(scores, topk, ratio)

    assert torch.equal(fused_idxs, oracle_idxs)
    assert torch.equal(fused_length, oracle_length)
    # The NaN position must be selected (not silently dropped) and the fully
    # masked row must be all -1 with length 0.  NaN is treated as larger than
    # every number by ``torch.topk`` (the shared primitive of both paths), so
    # the NaN column leads the row's selection and is never classified as an
    # invalid -1 lane.
    assert int((fused_idxs[0, 1] == 7).sum()) == 1
    assert int(fused_idxs[0, 1][0]) == 7
    assert torch.equal(fused_idxs[0, 2], torch.full((topk,), -1, dtype=torch.int32))
    assert int(fused_length[0, 2]) == 0


# ---------------------------------------------------------------------------
# 4. loss reduction contract (mean vs calculate_per_token_loss)
# ---------------------------------------------------------------------------


def _make_loss_inputs(sq, b, np_, d, n_comp, topk, ratio=4, seed=_PARITY_SEED + 6):
    torch.manual_seed(seed)
    attn_hn = 64
    q_attn_bshd = torch.randn(b, sq, np_, attn_hn, dtype=torch.bfloat16)
    k_attn_bsd = torch.randn(b, sq + n_comp, attn_hn, dtype=torch.bfloat16)
    lse_bsh = torch.randn(b, sq, np_, dtype=torch.float32) + 5.0
    q_idx_bshd = torch.randn(b, sq, 4, d, dtype=torch.bfloat16)
    k_idx_bsd = torch.randn(b, n_comp, d, dtype=torch.bfloat16)
    w_bsh = (torch.randn(b, sq, 4, dtype=torch.bfloat16) * 0.1).abs()
    topk_indices = torch.randint(0, n_comp, (b, sq, topk), dtype=torch.int32)
    return {
        "q_idx_bshd": q_idx_bshd,
        "k_idx_bsd": k_idx_bsd,
        "w_bsh": w_bsh,
        "topk_indices_cmp": topk_indices,
        "q_attn_bshd": q_attn_bshd,
        "k_attn_bsd": k_attn_bsd,
        "lse_bsh": lse_bsh,
    }


@pytest.mark.parametrize("sparse", [True, False], ids=["sparse", "dense"])
def test_loss_mean_vs_per_token_share_same_reduction(sparse):
    """Local-mean and per-token-sum must scale loss and grads by 1/(B*S_q)
    and 1 respectively — a mismatch would double-count or average twice."""
    inputs = _make_loss_inputs(24, 2, 8, 64, 32, 8)
    loss_coeff = 0.1
    token_count = 24 * 2

    if sparse:
        kwargs = dict(
            indexer_softmax_scale=64 ** -0.5,
            softmax_scale=64 ** -0.5,
            loss_coeff=loss_coeff,
            idx_nh=4,
            kv_offset=24,
        )
        loss_mean, gq_mean, gk_mean, gw_mean = tri_indexer.fused_sparse_indexer_loss_and_backward(
            inputs["q_idx_bshd"], inputs["k_idx_bsd"], inputs["w_bsh"],
            inputs["topk_indices_cmp"],
            inputs["q_attn_bshd"], inputs["k_attn_bsd"], inputs["lse_bsh"],
            calculate_per_token_loss=False, **kwargs,
        )
        loss_sum, gq_sum, gk_sum, gw_sum = tri_indexer.fused_sparse_indexer_loss_and_backward(
            inputs["q_idx_bshd"], inputs["k_idx_bsd"], inputs["w_bsh"],
            inputs["topk_indices_cmp"],
            inputs["q_attn_bshd"], inputs["k_attn_bsd"], inputs["lse_bsh"],
            calculate_per_token_loss=True, **kwargs,
        )
    else:
        kwargs = dict(
            softmax_scale=64 ** -0.5,
            loss_coeff=loss_coeff,
            ratio=4,
            idx_nh=4,
        )
        # Dense path indexes k_attn_bsd over the compressed range only.
        inputs_dense = dict(inputs)
        inputs_dense["k_attn_bsd"] = inputs["k_attn_bsd"][:, : inputs["k_idx_bsd"].shape[1]]
        loss_mean, gq_mean, gk_mean, gw_mean = (
            tri_indexer.fused_dense_indexer_loss_and_backward(
                inputs_dense["q_idx_bshd"], inputs_dense["k_idx_bsd"],
                inputs_dense["w_bsh"], inputs_dense["topk_indices_cmp"],
                inputs_dense["q_attn_bshd"], inputs_dense["k_attn_bsd"],
                inputs_dense["lse_bsh"],
                calculate_per_token_loss=False, **kwargs,
            )
        )
        loss_sum, gq_sum, gk_sum, gw_sum = (
            tri_indexer.fused_dense_indexer_loss_and_backward(
                inputs_dense["q_idx_bshd"], inputs_dense["k_idx_bsd"],
                inputs_dense["w_bsh"], inputs_dense["topk_indices_cmp"],
                inputs_dense["q_attn_bshd"], inputs_dense["k_attn_bsd"],
                inputs_dense["lse_bsh"],
                calculate_per_token_loss=True, **kwargs,
            )
        )

    dsv4_parity_gate.assert_tensor_parity(
        loss_sum,
        loss_mean * token_count,
        topology=f"indexer-primitive:loss-reduction:{'sparse' if sparse else 'dense'}",
        layer_mode="indexer-loss",
        shape=(),
        rtol=1e-3,
        atol=1e-3,
    )
    for name, grad_sum, grad_mean in (
        ("grad_q", gq_sum, gq_mean),
        ("grad_k", gk_sum, gk_mean),
        ("grad_w", gw_sum, gw_mean),
    ):
        # Compare in FP32: the fused functions return BF16 grads, and a BF16
        # multiply by token_count would add a third rounding to the scale.
        dsv4_parity_gate.assert_tensor_parity(
            grad_sum.float(),
            grad_mean.float() * token_count,
            topology=f"indexer-primitive:loss-reduction:{'sparse' if sparse else 'dense'}",
            layer_mode=f"indexer-loss:{name}",
            shape=tuple(grad_mean.shape),
            rtol=2e-3,
            atol=2e-3,
        )


def test_loss_all_invalid_rows_contribute_zero():
    """Fully invalid rows (all top-k -1) must add zero loss and zero grads."""
    inputs = _make_loss_inputs(16, 2, 8, 64, 16, 6)
    inputs["topk_indices_cmp"][:] = -1
    loss, gq, gk, gw = tri_indexer.fused_sparse_indexer_loss_and_backward(
        inputs["q_idx_bshd"], inputs["k_idx_bsd"], inputs["w_bsh"],
        inputs["topk_indices_cmp"],
        inputs["q_attn_bshd"], inputs["k_attn_bsd"], inputs["lse_bsh"],
        indexer_softmax_scale=64 ** -0.5,
        softmax_scale=64 ** -0.5,
        loss_coeff=0.5,
        idx_nh=4,
        kv_offset=16,
    )
    assert float(loss) == 0.0
    assert torch.equal(gq, torch.zeros_like(gq))
    assert torch.equal(gk, torch.zeros_like(gk))
    assert torch.equal(gw, torch.zeros_like(gw))
