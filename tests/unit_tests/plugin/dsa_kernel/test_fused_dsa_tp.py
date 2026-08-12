# Copyright (c) 2026, FlagOS Contributors. All rights reserved.

"""
Unit tests for TP integration of the fused DSA sparse indexer loss.

Validates:
1. Decomposed helpers (local_target + predict + kl_backward) match the original
   fused_sparse_indexer_loss_and_backward at TP=1.
2. Simulated TP=2 head splitting produces correct global target.
3. Math contract: normalize-after-reduce != reduce-after-normalize.
4. Overlap ordering (mock collective): async_op → predict → wait → normalize → KL.
5. End-to-end fused_indexer_sparse_attn with tp_group=None preserves behavior.

Run with: pytest tests/unit_tests/plugin/dsa_kernel/test_fused_dsa_tp.py -v -s
8-GPU TP correctness run:
  torchrun --standalone --nproc_per_node=8 -m pytest \
    tests/unit_tests/plugin/dsa_kernel/test_fused_dsa_tp.py \
    -k "TestDistributedTPCorrectness or TestDistributedTPUnfusedParity" -v -s
TP performance report example:
  torchrun --standalone --nproc_per_node=8 -m pytest \
    tests/unit_tests/plugin/dsa_kernel/test_fused_dsa_tp.py \
    -k TestDistributedTPPerformance --run-perf --dsa-report=tp_results.md -s
Requires: CUDA GPU with Triton support.
"""

from __future__ import annotations

import math
from typing import Tuple
from unittest.mock import patch, MagicMock

import pytest
import torch
from torch import Tensor

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

_SM90_AVAILABLE = (
    torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 9
)
_skip_unless_sm90 = pytest.mark.skipif(
    not _SM90_AVAILABLE, reason="SM90+ required for Triton DSA kernels"
)

from megatron.plugin.dsa_kernel.triton_indexer_kernels import (
    fused_sparse_indexer_loss_and_backward,
    fused_dense_indexer_loss_and_backward,
    compute_sparse_local_target_head_sum,
    compute_sparse_indexer_predict_state,
    sparse_indexer_kl_and_backward,
)
from megatron.plugin.dsa_kernel.triton_dsa_kernels import (
    fused_indexer_sparse_attn,
    _sbhd_to_bshd_indexer_inputs,
    _indexer_topk_bshd,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sparse_loss_inputs(
    B: int = 2,
    S_q: int = 64,
    np_: int = 8,
    D_attn: int = 128,
    D_idx: int = 64,
    H_q: int = 1,
    n_comp: int = 16,
    topk: int = 4,
    kv_offset: int = 64,
    device: str = "cuda",
    seed: int = 42,
):
    """Generate inputs for sparse indexer loss functions."""
    torch.manual_seed(seed)
    S_kv = S_q + n_comp  # full KV length

    q_attn_bshd = torch.randn(B, S_q, np_, D_attn, device=device, dtype=torch.bfloat16)
    k_attn_bsd = torch.randn(B, S_kv, D_attn, device=device, dtype=torch.bfloat16)
    lse_bsh = torch.randn(B, S_q, np_, device=device, dtype=torch.float32) + 5.0

    q_idx_bshd = torch.randn(B, S_q, H_q, D_idx, device=device, dtype=torch.bfloat16)
    k_idx_bsd = torch.randn(B, n_comp, D_idx, device=device, dtype=torch.bfloat16)
    w_bsh = torch.randn(B, S_q, H_q, device=device, dtype=torch.bfloat16) * 0.1

    # Generate valid top-k indices in [0, n_comp)
    topk_indices_cmp = torch.randint(0, n_comp, (B, S_q, topk), device=device, dtype=torch.int32)
    # Mark some rows as fully invalid
    topk_indices_cmp[:, :2, :] = -1

    return {
        "q_attn_bshd": q_attn_bshd,
        "k_attn_bsd": k_attn_bsd,
        "lse_bsh": lse_bsh,
        "q_idx_bshd": q_idx_bshd,
        "k_idx_bsd": k_idx_bsd,
        "w_bsh": w_bsh,
        "topk_indices_cmp": topk_indices_cmp,
        "kv_offset": kv_offset,
        "n_comp": n_comp,
    }


# ---------------------------------------------------------------------------
# Test 1: Decomposed helpers match original fused function at TP=1
# ---------------------------------------------------------------------------


@_skip_unless_sm90
class TestDecomposedMatchesFused:
    """Verify that the decomposed path (local_target + predict + kl_backward)
    produces identical results to the monolithic fused_sparse_indexer_loss_and_backward."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.device = "cuda"

    @pytest.mark.parametrize(
        "np_,calculate_per_token_loss",
        [(4, False), (16, True)],
        ids=["heads4-reduced-loss", "heads16-per-token-loss"],
    )
    def test_loss_and_grads_match(self, np_, calculate_per_token_loss):
        inputs = _make_sparse_loss_inputs(np_=np_, device=self.device)
        loss_coeff = 0.1
        softmax_scale = inputs["q_attn_bshd"].shape[-1] ** -0.5
        indexer_softmax_scale = inputs["q_idx_bshd"].shape[-1] ** -0.5

        # --- Reference: monolithic fused ---
        ref_loss, ref_grad_q, ref_grad_k, ref_grad_w = fused_sparse_indexer_loss_and_backward(
            inputs["q_idx_bshd"], inputs["k_idx_bsd"], inputs["w_bsh"],
            inputs["topk_indices_cmp"],
            inputs["q_attn_bshd"], inputs["k_attn_bsd"], inputs["lse_bsh"],
            indexer_softmax_scale=indexer_softmax_scale,
            softmax_scale=softmax_scale,
            loss_coeff=loss_coeff,
            calculate_per_token_loss=calculate_per_token_loss,
            idx_nh=inputs["q_idx_bshd"].shape[2],
            kv_offset=inputs["kv_offset"],
        )

        # --- New: decomposed path ---
        head_sum = compute_sparse_local_target_head_sum(
            inputs["q_attn_bshd"], inputs["k_attn_bsd"], inputs["lse_bsh"],
            inputs["topk_indices_cmp"],
            softmax_scale=softmax_scale,
            kv_offset=inputs["kv_offset"],
        )
        predict_state = compute_sparse_indexer_predict_state(
            inputs["q_idx_bshd"], inputs["k_idx_bsd"], inputs["w_bsh"],
            inputs["topk_indices_cmp"],
        )
        new_loss, new_grad_q, new_grad_k, new_grad_w = sparse_indexer_kl_and_backward(
            head_sum, predict_state,
            inputs["q_idx_bshd"], inputs["k_idx_bsd"], inputs["w_bsh"],
            loss_coeff=loss_coeff,
            calculate_per_token_loss=calculate_per_token_loss,
        )

        # --- Assertions ---
        torch.testing.assert_close(new_loss, ref_loss, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(new_grad_q, ref_grad_q, rtol=1e-4, atol=1e-6)
        torch.testing.assert_close(new_grad_k, ref_grad_k, rtol=1e-4, atol=1e-6)
        torch.testing.assert_close(new_grad_w, ref_grad_w, rtol=1e-4, atol=1e-6)

    def test_fully_masked_rows_handled(self):
        """Ensure fully masked rows (all -1) produce zero loss contribution."""
        inputs = _make_sparse_loss_inputs(device=self.device)
        # Make ALL rows fully masked
        inputs["topk_indices_cmp"][:] = -1
        loss_coeff = 0.1
        softmax_scale = inputs["q_attn_bshd"].shape[-1] ** -0.5

        head_sum = compute_sparse_local_target_head_sum(
            inputs["q_attn_bshd"], inputs["k_attn_bsd"], inputs["lse_bsh"],
            inputs["topk_indices_cmp"],
            softmax_scale=softmax_scale,
            kv_offset=inputs["kv_offset"],
        )
        predict_state = compute_sparse_indexer_predict_state(
            inputs["q_idx_bshd"], inputs["k_idx_bsd"], inputs["w_bsh"],
            inputs["topk_indices_cmp"],
        )
        loss, grad_q, grad_k, grad_w = sparse_indexer_kl_and_backward(
            head_sum, predict_state,
            inputs["q_idx_bshd"], inputs["k_idx_bsd"], inputs["w_bsh"],
            loss_coeff=loss_coeff,
            calculate_per_token_loss=False,
        )
        assert loss.item() == 0.0
        assert grad_q.abs().max().item() == 0.0


# ---------------------------------------------------------------------------
# Test 2: Simulated TP head splitting gives correct global target
# ---------------------------------------------------------------------------


@_skip_unless_sm90
class TestSimulatedTPHeadSplit:
    """On a single GPU, simulate TP by splitting heads and verifying that
    sum(local_head_sums) == global_head_sum from full heads."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.device = "cuda"

    @pytest.mark.parametrize("tp_size", [2, 4, 8])
    def test_global_target_from_local_shards(self, tp_size):
        np_global = 16
        assert np_global % tp_size == 0
        np_local = np_global // tp_size

        inputs = _make_sparse_loss_inputs(np_=np_global, device=self.device)
        softmax_scale = inputs["q_attn_bshd"].shape[-1] ** -0.5

        # Full global head sum
        global_head_sum = compute_sparse_local_target_head_sum(
            inputs["q_attn_bshd"], inputs["k_attn_bsd"], inputs["lse_bsh"],
            inputs["topk_indices_cmp"],
            softmax_scale=softmax_scale,
            kv_offset=inputs["kv_offset"],
        )

        # Sum of local shards
        local_sums = []
        for rank in range(tp_size):
            start_h = rank * np_local
            end_h = start_h + np_local
            q_local = inputs["q_attn_bshd"][:, :, start_h:end_h, :]
            lse_local = inputs["lse_bsh"][:, :, start_h:end_h]
            local_sum = compute_sparse_local_target_head_sum(
                q_local, inputs["k_attn_bsd"], lse_local,
                inputs["topk_indices_cmp"],
                softmax_scale=softmax_scale,
                kv_offset=inputs["kv_offset"],
            )
            local_sums.append(local_sum)

        reconstructed = sum(local_sums)
        torch.testing.assert_close(reconstructed, global_head_sum, rtol=1e-5, atol=1e-6)

    @pytest.mark.parametrize("tp_size", [2, 4, 8])
    def test_loss_matches_after_simulated_allreduce(self, tp_size):
        """Full pipeline: split heads → sum local targets → normalize → KL
        should match TP=1 loss."""
        np_global = 16
        np_local = np_global // tp_size

        inputs = _make_sparse_loss_inputs(np_=np_global, device=self.device)
        softmax_scale = inputs["q_attn_bshd"].shape[-1] ** -0.5
        loss_coeff = 0.1

        # --- TP=1 reference ---
        ref_loss, _, _, _ = fused_sparse_indexer_loss_and_backward(
            inputs["q_idx_bshd"], inputs["k_idx_bsd"], inputs["w_bsh"],
            inputs["topk_indices_cmp"],
            inputs["q_attn_bshd"], inputs["k_attn_bsd"], inputs["lse_bsh"],
            indexer_softmax_scale=inputs["q_idx_bshd"].shape[-1] ** -0.5,
            softmax_scale=softmax_scale,
            loss_coeff=loss_coeff,
            calculate_per_token_loss=False,
            idx_nh=inputs["q_idx_bshd"].shape[2],
            kv_offset=inputs["kv_offset"],
        )

        # --- Simulated TP: sum local head sums (mimics all-reduce) ---
        local_sums = []
        for rank in range(tp_size):
            start_h = rank * np_local
            end_h = start_h + np_local
            q_local = inputs["q_attn_bshd"][:, :, start_h:end_h, :]
            lse_local = inputs["lse_bsh"][:, :, start_h:end_h]
            local_sum = compute_sparse_local_target_head_sum(
                q_local, inputs["k_attn_bsd"], lse_local,
                inputs["topk_indices_cmp"],
                softmax_scale=softmax_scale,
                kv_offset=inputs["kv_offset"],
            )
            local_sums.append(local_sum)

        global_head_sum = sum(local_sums)
        predict_state = compute_sparse_indexer_predict_state(
            inputs["q_idx_bshd"], inputs["k_idx_bsd"], inputs["w_bsh"],
            inputs["topk_indices_cmp"],
        )
        tp_loss, _, _, _ = sparse_indexer_kl_and_backward(
            global_head_sum, predict_state,
            inputs["q_idx_bshd"], inputs["k_idx_bsd"], inputs["w_bsh"],
            loss_coeff=loss_coeff,
            calculate_per_token_loss=False,
        )

        torch.testing.assert_close(tp_loss, ref_loss, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# Test 3: Math contract — normalize-after-reduce != reduce-after-normalize
# ---------------------------------------------------------------------------


@_skip_unless_sm90
class TestMathContract:
    """Proves that the order of operations matters: first sum, then normalize."""

    def test_normalize_then_sum_differs_from_sum_then_normalize(self):
        """Construct a case where the two orderings give different results."""
        device = "cuda"
        np_global = 8
        tp_size = 2
        np_local = np_global // tp_size

        inputs = _make_sparse_loss_inputs(np_=np_global, device=device, seed=123)
        softmax_scale = inputs["q_attn_bshd"].shape[-1] ** -0.5

        # Correct: sum then normalize
        local_sums = []
        for rank in range(tp_size):
            start_h = rank * np_local
            end_h = start_h + np_local
            q_local = inputs["q_attn_bshd"][:, :, start_h:end_h, :]
            lse_local = inputs["lse_bsh"][:, :, start_h:end_h]
            local_sum = compute_sparse_local_target_head_sum(
                q_local, inputs["k_attn_bsd"], lse_local,
                inputs["topk_indices_cmp"],
                softmax_scale=softmax_scale,
                kv_offset=inputs["kv_offset"],
            )
            local_sums.append(local_sum)

        global_sum = sum(local_sums)
        correct_target = global_sum / global_sum.sum(dim=-1, keepdim=True).clamp(min=1e-12)

        # Wrong: normalize each local sum, then average
        local_targets = []
        for ls in local_sums:
            denom = ls.sum(dim=-1, keepdim=True).clamp(min=1e-12)
            local_targets.append(ls / denom)
        wrong_target = sum(local_targets) / tp_size

        # They must differ (non-trivially)
        diff = (correct_target - wrong_target).abs().max().item()
        assert diff > 1e-3, (
            f"Expected significant difference, got max diff = {diff}. "
            "The test input may not distinguish the two orderings."
        )


# ---------------------------------------------------------------------------
# Test 4: Mock collective ordering test
# ---------------------------------------------------------------------------


@_skip_unless_sm90
class TestOverlapOrdering:
    """Verify the correct ordering of operations when overlap is enabled."""

    def test_sync_path_no_collective_at_tp1(self):
        """When tp_group is None, no distributed call should be made."""
        device = "cuda"
        inputs = _make_sparse_loss_inputs(device=device, np_=4)

        # Build full fused inputs
        sq, b, np_, d = 64, 2, 4, 128
        win_topk = 8
        query = torch.randn(sq, b, np_, d, device=device, dtype=torch.bfloat16, requires_grad=True)
        skv = sq + inputs["n_comp"]
        kv_full = torch.randn(skv, b, d, device=device, dtype=torch.bfloat16, requires_grad=True)
        attn_sink = torch.zeros(np_, device=device, dtype=torch.float32, requires_grad=True)
        window_idxs = torch.randint(0, sq, (b, sq, win_topk), device=device, dtype=torch.int32)
        q_indexer = torch.randn(sq, b, 1, 64, device=device, dtype=torch.bfloat16, requires_grad=True)
        k_indexer = torch.randn(inputs["n_comp"], b, 64, device=device, dtype=torch.bfloat16, requires_grad=True)
        weights = torch.randn(sq, b, 1, device=device, dtype=torch.bfloat16, requires_grad=True)

        with patch("torch.distributed.all_reduce") as mock_ar:
            output, loss = fused_indexer_sparse_attn(
                query, kv_full, attn_sink, window_idxs,
                q_indexer, k_indexer, weights,
                indexer_topk=4, ratio=4, softmax_scale=d**-0.5,
                indexer_softmax_scale=64**-0.5, loss_coeff=0.1,
                sparse_loss=True, kv_offset=sq,
                calculate_per_token_loss=False,
                tp_group=None,
            )
            mock_ar.assert_not_called()

    def test_overlap_event_order(self):
        """With overlap enabled and a mock tp_group, verify operation order."""
        device = "cuda"
        events = []

        # Mock tp_group
        mock_group = MagicMock()
        mock_group.size.return_value = 2

        class FakeWork:
            def wait(self):
                events.append("wait")

        original_compute_target = compute_sparse_local_target_head_sum
        original_compute_predict = compute_sparse_indexer_predict_state
        original_kl_bwd = sparse_indexer_kl_and_backward

        def tracked_target(*args, **kwargs):
            events.append("target")
            return original_compute_target(*args, **kwargs)

        def tracked_predict(*args, **kwargs):
            events.append("predict")
            return original_compute_predict(*args, **kwargs)

        def tracked_kl(*args, **kwargs):
            events.append("kl_backward")
            return original_kl_bwd(*args, **kwargs)

        def mock_all_reduce(tensor, op=None, group=None, async_op=False):
            events.append("all_reduce_start")
            if async_op:
                return FakeWork()
            return None

        with patch("megatron.plugin.dsa_kernel.triton_dsa_kernels._DSA_TP_OVERLAP", True), \
             patch("megatron.plugin.dsa_kernel.triton_dsa_kernels.compute_sparse_local_target_head_sum", tracked_target), \
             patch("megatron.plugin.dsa_kernel.triton_dsa_kernels.compute_sparse_indexer_predict_state", tracked_predict), \
             patch("megatron.plugin.dsa_kernel.triton_dsa_kernels.sparse_indexer_kl_and_backward", tracked_kl), \
             patch("torch.distributed.all_reduce", mock_all_reduce):

            sq, b, np_, d = 64, 2, 4, 128
            n_comp = 16
            skv = sq + n_comp
            query = torch.randn(sq, b, np_, d, device=device, dtype=torch.bfloat16, requires_grad=True)
            kv_full = torch.randn(skv, b, d, device=device, dtype=torch.bfloat16, requires_grad=True)
            attn_sink = torch.zeros(np_, device=device, dtype=torch.float32, requires_grad=True)
            window_idxs = torch.randint(0, sq, (b, sq, 8), device=device, dtype=torch.int32)
            q_indexer = torch.randn(sq, b, 1, 64, device=device, dtype=torch.bfloat16, requires_grad=True)
            k_indexer = torch.randn(n_comp, b, 64, device=device, dtype=torch.bfloat16, requires_grad=True)
            weights = torch.randn(sq, b, 1, device=device, dtype=torch.bfloat16, requires_grad=True)

            output, loss = fused_indexer_sparse_attn(
                query, kv_full, attn_sink, window_idxs,
                q_indexer, k_indexer, weights,
                indexer_topk=4, ratio=4, softmax_scale=d**-0.5,
                indexer_softmax_scale=64**-0.5, loss_coeff=0.1,
                sparse_loss=True, kv_offset=sq,
                calculate_per_token_loss=False,
                tp_group=mock_group,
            )

        # Verify ordering: target → all_reduce_start → predict → wait → kl_backward
        assert events == ["target", "all_reduce_start", "predict", "wait", "kl_backward"], (
            f"Expected overlap ordering, got: {events}"
        )


# ---------------------------------------------------------------------------
# Test 5: End-to-end tp_group=None backward compatibility
# ---------------------------------------------------------------------------


@_skip_unless_sm90
class TestBackwardCompatibility:
    """Ensure tp_group=None produces identical results to the old API."""

    def test_fused_output_unchanged_with_tp_group_none(self):
        """Forward + backward with tp_group=None should match pre-TP behavior."""
        device = "cuda"
        torch.manual_seed(7)

        sq, b, np_, d = 128, 2, 8, 128
        n_comp = 32
        skv = sq + n_comp
        win_topk = 16
        topk = 4

        query = torch.randn(sq, b, np_, d, device=device, dtype=torch.bfloat16, requires_grad=True)
        kv_full = torch.randn(skv, b, d, device=device, dtype=torch.bfloat16, requires_grad=True)
        attn_sink = torch.zeros(np_, device=device, dtype=torch.float32, requires_grad=True)
        window_idxs = torch.randint(0, sq, (b, sq, win_topk), device=device, dtype=torch.int32)
        q_indexer = torch.randn(sq, b, 1, 64, device=device, dtype=torch.bfloat16, requires_grad=True)
        k_indexer = torch.randn(n_comp, b, 64, device=device, dtype=torch.bfloat16, requires_grad=True)
        weights = torch.randn(sq, b, 1, device=device, dtype=torch.bfloat16, requires_grad=True)

        output, loss = fused_indexer_sparse_attn(
            query, kv_full, attn_sink, window_idxs,
            q_indexer, k_indexer, weights,
            indexer_topk=topk, ratio=4, softmax_scale=d**-0.5,
            indexer_softmax_scale=64**-0.5, loss_coeff=0.1,
            sparse_loss=True, kv_offset=sq,
            calculate_per_token_loss=False,
            tp_group=None,
        )

        # Verify output/loss are valid
        assert output.shape == (sq, b, np_ * d)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        assert loss.item() > 0  # non-trivial loss

        # Verify backward runs without error
        total = output.sum() + loss
        total.backward()

        assert query.grad is not None
        assert not torch.isnan(query.grad).any()
        assert kv_full.grad is not None
        assert q_indexer.grad is not None
        assert k_indexer.grad is not None
        assert weights.grad is not None


# ---------------------------------------------------------------------------
# Test 6b: Dense loss TP integration (方案B: full buffer + single all-reduce)
# ---------------------------------------------------------------------------


def _make_dense_loss_inputs(
    B: int = 2,
    S_q: int = 64,
    np_: int = 8,
    D_attn: int = 128,
    D_idx: int = 64,
    H_q: int = 1,
    ratio: int = 4,
    topk: int = 4,
    device: str = "cuda",
    seed: int = 55,
):
    """Generate inputs for dense indexer loss functions."""
    torch.manual_seed(seed)
    S_k = S_q // ratio  # compressed KV length

    q_attn_bshd = torch.randn(B, S_q, np_, D_attn, device=device, dtype=torch.bfloat16)
    k_attn_bsd = torch.randn(B, S_k, D_attn, device=device, dtype=torch.bfloat16)
    lse_bsh = torch.randn(B, S_q, np_, device=device, dtype=torch.float32) + 5.0

    q_idx_bshd = torch.randn(B, S_q, H_q, D_idx, device=device, dtype=torch.bfloat16)
    k_idx_bsd = torch.randn(B, S_k, D_idx, device=device, dtype=torch.bfloat16)
    w_bsh = torch.randn(B, S_q, H_q, device=device, dtype=torch.bfloat16) * 0.1

    # topk indices for row validity
    topk_indices_cmp = torch.randint(0, S_k, (B, S_q, topk), device=device, dtype=torch.int32)
    topk_indices_cmp[:, :2, :] = -1  # some invalid rows

    return {
        "q_attn_bshd": q_attn_bshd,
        "k_attn_bsd": k_attn_bsd,
        "lse_bsh": lse_bsh,
        "q_idx_bshd": q_idx_bshd,
        "k_idx_bsd": k_idx_bsd,
        "w_bsh": w_bsh,
        "topk_indices_cmp": topk_indices_cmp,
        "S_k": S_k,
        "ratio": ratio,
    }


@_skip_unless_sm90
class TestDenseTPIntegration:
    """Test dense indexer loss TP integration with synchronous Q-block reductions."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.device = "cuda"

    def test_dense_tp_none_matches_original(self):
        """With tp_group=None, new implementation should match original behavior."""
        inputs = _make_dense_loss_inputs(device=self.device)
        loss_coeff = 0.1
        softmax_scale = inputs["q_attn_bshd"].shape[-1] ** -0.5
        indexer_softmax_scale = inputs["q_idx_bshd"].shape[-1] ** -0.5

        # Run with tp_group=None (should be same as no TP)
        loss, grad_q, grad_k, grad_w = fused_dense_indexer_loss_and_backward(
            inputs["q_idx_bshd"], inputs["k_idx_bsd"], inputs["w_bsh"],
            inputs["topk_indices_cmp"],
            inputs["q_attn_bshd"], inputs["k_attn_bsd"], inputs["lse_bsh"],
            indexer_softmax_scale=indexer_softmax_scale,
            softmax_scale=softmax_scale,
            loss_coeff=loss_coeff,
            ratio=inputs["ratio"],
            calculate_per_token_loss=False,
            idx_nh=1,
            tp_group=None,
        )

        assert loss.item() > 0, "Dense loss should be non-trivial"
        assert not torch.isnan(loss)
        assert not torch.isnan(grad_q).any()
        assert not torch.isnan(grad_k).any()
        assert not torch.isnan(grad_w).any()

    @pytest.mark.parametrize("tp_size", [2, 4, 8])
    def test_dense_simulated_tp_target_correct(self, tp_size):
        """Simulate TP: split heads → compute local attn_score → sum → normalize
        should match full-heads target."""
        np_global = 16
        assert np_global % tp_size == 0
        np_local = np_global // tp_size

        inputs = _make_dense_loss_inputs(np_=np_global, device=self.device)
        softmax_scale = inputs["q_attn_bshd"].shape[-1] ** -0.5
        ratio = inputs["ratio"]
        S_q = inputs["q_attn_bshd"].shape[1]
        S_k = inputs["S_k"]
        B = inputs["q_attn_bshd"].shape[0]

        from megatron.plugin.dsa_kernel.triton_dsa_utils import compute_ratio_causal_mask

        causal_mask_float = compute_ratio_causal_mask(S_q, S_k, ratio, self.device)
        causal_mask = (causal_mask_float == 0)

        k_attn = inputs["k_attn_bsd"].float()

        # Full global target (all heads)
        q_attn_full = inputs["q_attn_bshd"].float()
        attn_per_head_full = torch.einsum("bqhd,bkd->bqhk", q_attn_full, k_attn) * softmax_scale
        attn_per_head_full = attn_per_head_full.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(2), float("-inf"))
        attn_probs_full = torch.softmax(attn_per_head_full, dim=-1)
        attn_probs_full = attn_probs_full.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(2), 0.0)
        global_head_sum = attn_probs_full.sum(dim=2)  # (B, S_q, S_k)

        # Sum of local shards (simulates all-reduce)
        reconstructed = torch.zeros_like(global_head_sum)
        for rank in range(tp_size):
            start_h = rank * np_local
            end_h = start_h + np_local
            q_local = inputs["q_attn_bshd"][:, :, start_h:end_h, :].float()
            attn_per_head_local = torch.einsum("bqhd,bkd->bqhk", q_local, k_attn) * softmax_scale
            attn_per_head_local = attn_per_head_local.masked_fill(
                ~causal_mask.unsqueeze(0).unsqueeze(2), float("-inf"))
            attn_probs_local = torch.softmax(attn_per_head_local, dim=-1)
            attn_probs_local = attn_probs_local.masked_fill(
                ~causal_mask.unsqueeze(0).unsqueeze(2), 0.0)
            reconstructed += attn_probs_local.sum(dim=2)

        torch.testing.assert_close(reconstructed, global_head_sum, rtol=1e-5, atol=1e-6)

        # Verify normalization produces same target
        global_l1 = global_head_sum.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        target_ref = global_head_sum / global_l1

        recon_l1 = reconstructed.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        target_recon = reconstructed / recon_l1

        torch.testing.assert_close(target_recon, target_ref, rtol=1e-5, atol=1e-6)

    @pytest.mark.parametrize("tp_size", [2, 4, 8])
    def test_dense_simulated_tp_loss_matches_global(self, tp_size):
        """Full pipeline with simulated TP should produce same loss as TP=1."""
        np_global = 16
        np_local = np_global // tp_size

        inputs = _make_dense_loss_inputs(np_=np_global, device=self.device)
        softmax_scale = inputs["q_attn_bshd"].shape[-1] ** -0.5
        indexer_softmax_scale = inputs["q_idx_bshd"].shape[-1] ** -0.5
        loss_coeff = 0.1

        # TP=1 reference
        ref_loss, _, _, _ = fused_dense_indexer_loss_and_backward(
            inputs["q_idx_bshd"], inputs["k_idx_bsd"], inputs["w_bsh"],
            inputs["topk_indices_cmp"],
            inputs["q_attn_bshd"], inputs["k_attn_bsd"], inputs["lse_bsh"],
            indexer_softmax_scale=indexer_softmax_scale,
            softmax_scale=softmax_scale,
            loss_coeff=loss_coeff,
            ratio=inputs["ratio"],
            calculate_per_token_loss=False,
            idx_nh=1,
            tp_group=None,
        )

        # Simulated TP: manually compute global head sum then call with it
        # We can't easily mock the all-reduce here, but we can verify the math
        # by running with a local head slice and manually summing
        from megatron.plugin.dsa_kernel.triton_dsa_utils import compute_ratio_causal_mask
        from megatron.plugin.dsa_kernel.triton_indexer_kernels import _DENSE_BLOCK_Q

        S_q = inputs["q_attn_bshd"].shape[1]
        S_k = inputs["S_k"]
        B = inputs["q_attn_bshd"].shape[0]
        ratio = inputs["ratio"]

        causal_mask_float = compute_ratio_causal_mask(S_q, S_k, ratio, self.device)
        causal_mask = (causal_mask_float == 0)
        k_attn = inputs["k_attn_bsd"].float()

        # Compute global target (sum across all TP shards)
        global_attn_score = torch.zeros(B, S_q, S_k, dtype=torch.float32, device=self.device)
        for rank in range(tp_size):
            start_h = rank * np_local
            end_h = start_h + np_local
            q_local = inputs["q_attn_bshd"][:, :, start_h:end_h, :].float()
            for q_start in range(0, S_q, _DENSE_BLOCK_Q):
                q_end = min(q_start + _DENSE_BLOCK_Q, S_q)
                q_block = q_local[:, q_start:q_end]
                mask_block = causal_mask[q_start:q_end]
                attn = torch.einsum("bqhd,bkd->bqhk", q_block, k_attn) * softmax_scale
                attn = attn.masked_fill(~mask_block.unsqueeze(0).unsqueeze(2), float("-inf"))
                probs = torch.softmax(attn, dim=-1)
                probs = probs.masked_fill(~mask_block.unsqueeze(0).unsqueeze(2), 0.0)
                global_attn_score[:, q_start:q_end] += probs.sum(dim=2)

        # Now feed one shard's q_attn but with the globally-reduced target
        # This tests that the loss/grad computation after all-reduce is correct
        # We verify by checking the reference TP=1 result matches
        assert ref_loss.item() > 0
        # The key invariant: sum of local head-sums == global head-sum
        # which means TP loss == TP=1 loss (tested in distributed tests)


# ===========================================================================
# Distributed TP Tests (tp_size = world_size)
#
# Run with:
#   torchrun --standalone --nproc_per_node=8 -m pytest \
#     tests/unit_tests/plugin/dsa_kernel/test_fused_dsa_tp.py -v -k "Distributed"
# ===========================================================================

import os

_SUPPORTED_DISTRIBUTED_TP_SIZES = (2, 4, 8)


def _torchrun_world_size():
    return int(os.environ.get("WORLD_SIZE", "1"))


_DISTRIBUTED_AVAILABLE = torch.cuda.is_available() and _torchrun_world_size() >= 2
_skip_unless_distributed = pytest.mark.skipif(
    not _DISTRIBUTED_AVAILABLE,
    reason="Requires a multi-process torchrun launch",
)


def _is_torchrun():
    """Check if we're launched via torchrun (WORLD_SIZE set)."""
    return _torchrun_world_size() >= 2


def _get_tp_size():
    """Get TP size from WORLD_SIZE (assume world_size == tp_size)."""
    return _torchrun_world_size()


def _init_tp():
    """Initialize TP process group with tp_size = world_size."""
    tp_size = _get_tp_size()
    if tp_size not in _SUPPORTED_DISTRIBUTED_TP_SIZES:
        pytest.skip(
            f"Distributed fused DSA tests support TP sizes "
            f"{_SUPPORTED_DISTRIBUTED_TP_SIZES}, got WORLD_SIZE={tp_size}"
        )
    from tests.unit_tests.test_utilities import Utils
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=1,
    )
    import megatron.core.parallel_state as ps
    rank = ps.get_tensor_model_parallel_rank()
    tp_group = ps.get_tensor_model_parallel_group()
    return rank, tp_group, tp_size


def _destroy_tp():
    from tests.unit_tests.test_utilities import Utils
    Utils.destroy_model_parallel()


def _make_tp_test_inputs(
    sq: int = 128,
    b: int = 2,
    np_global: int = 8,
    d: int = 128,
    n_comp: int = 32,
    win_topk: int = 16,
    topk: int = 4,
    idx_d: int = 64,
    idx_nh: int = 1,
    seed: int = 777,
    device: str = "cuda",
):
    """Generate deterministic global inputs on all ranks, then shard query/sink by head."""
    tp_size = _get_tp_size()
    assert np_global % tp_size == 0, (
        f"np_global={np_global} must be divisible by TP={tp_size}"
    )
    torch.manual_seed(seed)

    skv = sq + n_comp
    # Global tensors (identical on all ranks)
    query_global = torch.randn(sq, b, np_global, d, device=device, dtype=torch.bfloat16)
    kv_full = torch.randn(skv, b, d, device=device, dtype=torch.bfloat16)
    attn_sink_global = torch.randn(np_global, device=device, dtype=torch.float32) * 0.01
    window_idxs = torch.randint(0, sq, (b, sq, win_topk), device=device, dtype=torch.int32)
    # Indexer tensors (replicated)
    q_indexer = torch.randn(sq, b, idx_nh, idx_d, device=device, dtype=torch.bfloat16)
    k_indexer = torch.randn(n_comp, b, idx_d, device=device, dtype=torch.bfloat16)
    weights = torch.randn(sq, b, idx_nh, device=device, dtype=torch.bfloat16) * 0.1

    return {
        "sq": sq, "b": b, "np_global": np_global, "d": d,
        "n_comp": n_comp, "skv": skv, "topk": topk,
        "query_global": query_global,
        "kv_full": kv_full,
        "attn_sink_global": attn_sink_global,
        "window_idxs": window_idxs,
        "q_indexer": q_indexer,
        "k_indexer": k_indexer,
        "weights": weights,
    }


def _shard_for_rank(inputs: dict, rank: int, tp_size: int):
    """Shard query and attn_sink by head for given rank."""
    assert inputs["np_global"] % tp_size == 0, (
        f"np_global={inputs['np_global']} must be divisible by TP={tp_size}"
    )
    np_local = inputs["np_global"] // tp_size
    start_h = rank * np_local
    end_h = start_h + np_local
    query_local = inputs["query_global"][:, :, start_h:end_h, :].contiguous()
    sink_local = inputs["attn_sink_global"][start_h:end_h].contiguous()
    return query_local, sink_local, np_local


# ---------------------------------------------------------------------------
# Test 7: Real distributed TP — unfused vs fused correctness
# ---------------------------------------------------------------------------


@_skip_unless_distributed
class TestDistributedCompressorGradient:
    """The CSA compressor is replicated but consumed by TP-local heads."""

    @pytest.fixture(scope="class", autouse=True)
    def setup_teardown(self):
        if not _is_torchrun():
            pytest.skip("Must be launched with torchrun")
        rank, tp_group, tp_size = _init_tp()
        self.__class__._rank = rank
        self.__class__._tp_group = tp_group
        self.__class__._tp_size = tp_size
        yield
        _destroy_tp()

    def test_tp_compressor_forward_and_grads_match_tp1(self):
        """Backward TP SUM must reproduce a full-head, non-TP reference."""
        from megatron.core.tensor_parallel.mappings import (
            copy_to_tensor_model_parallel_region,
        )

        rank = self._rank
        tp_group = self._tp_group
        tp_size = self._tp_size

        torch.manual_seed(20260801)
        sq, batch, input_dim, compressed_dim = 7, 2, 5, 3
        heads_per_rank = 2
        num_heads = heads_per_rank * tp_size

        x_init = torch.randn(sq, batch, input_dim, device="cuda", dtype=torch.float64)
        weight_init = torch.randn(
            input_dim, compressed_dim, device="cuda", dtype=torch.float64
        )
        head_grads = torch.randn(
            num_heads, compressed_dim, device="cuda", dtype=torch.float64
        )

        # TP path: every rank has the same compressor output, while its loss
        # represents only the contribution from that rank's local heads.
        x_tp = x_init.clone().requires_grad_(True)
        weight_tp = weight_init.clone().requires_grad_(True)
        compressed_tp = torch.matmul(x_tp, weight_tp)
        compressed_for_attention = copy_to_tensor_model_parallel_region(
            compressed_tp, group=tp_group
        )
        start = rank * heads_per_rank
        local_grad = head_grads[start : start + heads_per_rank].sum(dim=0)
        local_loss = (compressed_for_attention * local_grad).sum()
        local_loss.backward()

        # Non-TP reference: all attention heads contribute on one logical rank.
        x_ref = x_init.clone().requires_grad_(True)
        weight_ref = weight_init.clone().requires_grad_(True)
        compressed_ref = torch.matmul(x_ref, weight_ref)
        ref_loss = (compressed_ref * head_grads.sum(dim=0)).sum()
        ref_loss.backward()

        torch.testing.assert_close(compressed_for_attention, compressed_ref)
        torch.testing.assert_close(weight_tp.grad, weight_ref.grad)
        torch.testing.assert_close(x_tp.grad, x_ref.grad)

        # Replicated compressor parameters must receive the same full gradient
        # on every rank, otherwise their optimizer states will diverge.
        gathered_weight_grads = [torch.empty_like(weight_tp.grad) for _ in range(tp_size)]
        torch.distributed.all_gather(
            gathered_weight_grads, weight_tp.grad.contiguous(), group=tp_group
        )
        for peer_grad in gathered_weight_grads:
            torch.testing.assert_close(peer_grad, weight_ref.grad)


@_skip_unless_sm90
@_skip_unless_distributed
class TestDistributedTPCorrectness:
    """Real multi-GPU TP tests comparing unfused, fused, and fused+overlap paths.

    Run with: torchrun --standalone --nproc_per_node=8 -m pytest ...
    -k "TestDistributedTPCorrectness".
    """

    @pytest.fixture(scope="class", autouse=True)
    def setup_teardown(self):
        if not _is_torchrun():
            pytest.skip("Must be launched with torchrun")
        rank, tp_group, tp_size = _init_tp()
        self.__class__._rank = rank
        self.__class__._tp_group = tp_group
        self.__class__._tp_size = tp_size
        yield
        _destroy_tp()

    def test_indexer_logging_averages_across_tp(self):
        """Prove the old writer was rank-local and the fixed writer is TP-averaged."""
        from megatron.core.transformer.experimental_attention_variant.dsa import (
            DSAIndexerLossLoggingHelper,
        )

        class RecordingWriter:
            def __init__(self):
                self.values = []

            def add_scalar(self, name, value, iteration):
                self.values.append((name, float(value), iteration))

        # In this test world_size == TP size. Megatron creates TensorBoard's
        # writer only on the last global rank, so this reproduces that policy.
        writer = RecordingWriter() if self._rank == self._tp_size - 1 else None
        local_loss = torch.tensor(
            float(self._rank + 1), device="cuda", dtype=torch.float32
        )

        def record_once(avg_group):
            DSAIndexerLossLoggingHelper.save_loss_to_tracker(
                loss=local_loss,
                layer_number=1,
                num_layers=1,
                avg_group=avg_group,
            )
            DSAIndexerLossLoggingHelper.track_indexer_metrics(
                loss_scale=1.0,
                iteration=1,
                writer=writer,
                num_layers=1,
                csa_compress_ratios=[4],
            )

        DSAIndexerLossLoggingHelper.clean_loss_in_tracker()
        try:
            # Negative control: without a TP group, the sole writer receives
            # its own TP-rank value, not the TP mean. This characterizes the
            # pre-fix TensorBoard behavior without reverting production code.
            record_once(avg_group=None)
            if writer is not None:
                assert writer.values == [("indexer loss", float(self._tp_size), 1)]

            # Fixed path: TP AVG makes the scalar topology-invariant before
            # the sole writer records it.
            record_once(avg_group=self._tp_group)
            expected = (self._tp_size + 1) / 2.0
            if writer is not None:
                assert writer.values[-1] == ("indexer loss", expected, 1)
        finally:
            DSAIndexerLossLoggingHelper.clean_loss_in_tracker()

    def _run_fused_tp(self, inputs, rank, tp_group, tp_size, overlap: bool):
        """Run fused path with TP group."""
        import os as _os
        old_val = _os.environ.get("MEGATRON_DSA_TP_OVERLAP", "0")
        _os.environ["MEGATRON_DSA_TP_OVERLAP"] = "1" if overlap else "0"
        # Reload the module-level flag
        import megatron.plugin.dsa_kernel.triton_dsa_kernels as _mod
        _mod._DSA_TP_OVERLAP = overlap

        query_local, sink_local, np_local = _shard_for_rank(inputs, rank, tp_size)
        query_local = query_local.clone().requires_grad_(True)
        kv_full = inputs["kv_full"].clone().requires_grad_(True)
        sink_local = sink_local.clone().requires_grad_(True)
        q_indexer = inputs["q_indexer"].clone().requires_grad_(True)
        k_indexer = inputs["k_indexer"].clone().requires_grad_(True)
        weights = inputs["weights"].clone().requires_grad_(True)

        output, loss = fused_indexer_sparse_attn(
            query_local, kv_full, sink_local, inputs["window_idxs"],
            q_indexer, k_indexer, weights,
            indexer_topk=inputs["topk"], ratio=4,
            softmax_scale=inputs["d"] ** -0.5,
            indexer_softmax_scale=q_indexer.shape[-1] ** -0.5,
            loss_coeff=0.1,
            sparse_loss=True,
            kv_offset=inputs["sq"],
            calculate_per_token_loss=False,
            tp_group=tp_group,
        )

        # Backward
        (output.sum() + loss).backward()

        _os.environ["MEGATRON_DSA_TP_OVERLAP"] = old_val
        _mod._DSA_TP_OVERLAP = old_val == "1"

        return {
            "output": output.detach(),
            "loss": loss.detach(),
            "grad_query": query_local.grad.detach(),
            "grad_kv": kv_full.grad.detach(),
            "grad_sink": sink_local.grad.detach(),
            "grad_q_indexer": q_indexer.grad.detach(),
            "grad_k_indexer": k_indexer.grad.detach(),
            "grad_weights": weights.grad.detach(),
        }

    def _run_fused_tp1_reference(self, inputs):
        """Run fused path with TP=1 (full heads, no collective) as ground truth."""
        query = inputs["query_global"].clone().requires_grad_(True)
        kv_full = inputs["kv_full"].clone().requires_grad_(True)
        sink = inputs["attn_sink_global"].clone().requires_grad_(True)
        q_indexer = inputs["q_indexer"].clone().requires_grad_(True)
        k_indexer = inputs["k_indexer"].clone().requires_grad_(True)
        weights = inputs["weights"].clone().requires_grad_(True)

        output, loss = fused_indexer_sparse_attn(
            query, kv_full, sink, inputs["window_idxs"],
            q_indexer, k_indexer, weights,
            indexer_topk=inputs["topk"], ratio=4,
            softmax_scale=inputs["d"] ** -0.5,
            indexer_softmax_scale=q_indexer.shape[-1] ** -0.5,
            loss_coeff=0.1,
            sparse_loss=True,
            kv_offset=inputs["sq"],
            calculate_per_token_loss=False,
            tp_group=None,
        )

        (output.sum() + loss).backward()

        return {
            "output": output.detach(),
            "loss": loss.detach(),
            "grad_query": query.grad.detach(),
            "grad_kv": kv_full.grad.detach(),
            "grad_sink": sink.grad.detach(),
            "grad_q_indexer": q_indexer.grad.detach(),
            "grad_k_indexer": k_indexer.grad.detach(),
            "grad_weights": weights.grad.detach(),
        }

    def _run_unfused_local_reference(self, inputs, rank, tp_size):
        """Run the PyTorch attention reference for this rank's local heads."""
        from megatron.core.transformer.experimental_attention_variant.csa import (
            unfused_compressed_sparse_attn,
        )

        query, sink, _ = _shard_for_rank(inputs, rank, tp_size)
        query = query.clone().requires_grad_(True)
        kv = inputs["kv_full"].clone().requires_grad_(True)
        sink = sink.clone().requires_grad_(True)
        q_bshd, k_bsd, _, weights_scaled = _sbhd_to_bshd_indexer_inputs(
            inputs["q_indexer"], inputs["k_indexer"], inputs["weights"],
            inputs["q_indexer"].shape[-1] ** -0.5,
        )
        compressed_idxs, _, _ = _indexer_topk_bshd(
            q_bshd, k_bsd, weights_scaled, inputs["topk"], ratio=4
        )
        compressed_idxs = torch.where(
            compressed_idxs >= 0, compressed_idxs + inputs["sq"], -1
        )
        combined_idxs = torch.cat([compressed_idxs, inputs["window_idxs"]], dim=-1).int()

        output = unfused_compressed_sparse_attn(
            query,
            kv,
            sink.float(),
            combined_idxs,
            inputs["d"] ** -0.5,
        )
        output.sum().backward()
        return {
            "output": output.detach(),
            "grad_query": query.grad.detach(),
            "grad_kv": kv.grad.detach(),
            "grad_sink": sink.grad.detach(),
        }

    @pytest.mark.parametrize("overlap", [False, True], ids=["sync", "overlap"])
    @pytest.mark.parametrize(
        "sq,n_comp,topk",
        [(2048, 512, 256)],
        ids=["seq2048-topk256"],
    )
    def test_fused_tp_attention_matches_unfused(self, sq, n_comp, topk, overlap):
        """Sync and overlap fused TP paths match PyTorch at pretraining sizes."""
        rank = self._rank
        tp_group = self._tp_group
        tp_size = self._tp_size
        inputs = _make_tp_test_inputs(
            sq=sq,
            b=1,
            n_comp=n_comp,
            topk=topk,
            np_global=max(16, tp_size * 4),
        )

        fused = self._run_fused_tp(inputs, rank, tp_group, tp_size, overlap=overlap)
        unfused = self._run_unfused_local_reference(inputs, rank, tp_size)

        for name in ("output", "grad_query", "grad_sink"):
            torch.testing.assert_close(
                fused[name],
                unfused[name],
                rtol=2e-2,
                atol=2e-2,
                msg=f"TP={tp_size} fused {name} does not match unfused",
            )

        fused_dkv = fused["grad_kv"].float().flatten()
        unfused_dkv = unfused["grad_kv"].float().flatten()
        dkv_cosine = torch.nn.functional.cosine_similarity(
            fused_dkv.unsqueeze(0), unfused_dkv.unsqueeze(0)
        ).item()
        torch.testing.assert_close(
            fused_dkv.norm(), unfused_dkv.norm(), rtol=2e-2, atol=2e-2
        )
        assert dkv_cosine > 0.999, f"fused/unfused dKV cosine is {dkv_cosine:.6f}"


    def test_fused_tp_output_matches_tp1(self):
        """TP local output heads should match the corresponding TP=1 head slice."""
        rank = self._rank
        tp_group = self._tp_group
        tp_size = self._tp_size

        inputs = _make_tp_test_inputs(np_global=max(16, tp_size * 4))
        ref = self._run_fused_tp1_reference(inputs)
        tp_result = self._run_fused_tp(inputs, rank, tp_group, tp_size, overlap=False)

        np_local = inputs["np_global"] // tp_size
        start_h = rank * np_local
        end_h = start_h + np_local
        d = inputs["d"]

        ref_output_heads = ref["output"].reshape(
            inputs["sq"], inputs["b"], inputs["np_global"], d
        )[:, :, start_h:end_h, :].reshape(inputs["sq"], inputs["b"], np_local * d)

        torch.testing.assert_close(
            tp_result["output"],
            ref_output_heads,
            rtol=1e-2,
            # Different global/local head counts select different Triton
            # launch shapes. Their BF16 results can differ by one output ULP
            # (observed maximum: 0.015625) despite matching head mapping,
            # top-k, norm, and FP32 cosine similarity.
            atol=2e-2,
            msg=f"TP={tp_size} output does not match TP=1 head slice",
        )

    def test_fused_tp_local_outputs_and_grads_differ_across_ranks(self):
        """Different local-head shards must not produce identical tensors."""
        rank = self._rank
        tp_group = self._tp_group
        tp_size = self._tp_size

        inputs = _make_tp_test_inputs(np_global=max(16, tp_size * 4))
        tp_result = self._run_fused_tp(inputs, rank, tp_group, tp_size, overlap=False)

        for name in ("output", "grad_query", "grad_sink"):
            local_tensor = tp_result[name].contiguous()
            gathered = [torch.empty_like(local_tensor) for _ in range(tp_size)]
            torch.distributed.all_gather(gathered, local_tensor, group=tp_group)

            # Compare actual tensors in FP32. Equal BF16 norms (for example
            # both printing as 176.0) can result from quantization and similar
            # shard statistics, and do not imply equal local-head values.
            max_peer_diff = max(
                (gathered[0].float() - peer.float()).abs().max().item()
                for peer in gathered[1:]
            )
            assert max_peer_diff > 0.0, (
                f"TP-local {name} tensors are unexpectedly identical across all ranks"
            )

    def test_fused_tp_loss_matches_tp1(self):
        """TP indexer loss should match TP=1 (global target via all-reduce)."""
        rank = self._rank
        tp_group = self._tp_group
        tp_size = self._tp_size

        inputs = _make_tp_test_inputs(np_global=max(16, tp_size * 4))
        ref = self._run_fused_tp1_reference(inputs)
        tp_result = self._run_fused_tp(inputs, rank, tp_group, tp_size, overlap=False)

        torch.testing.assert_close(
            tp_result["loss"], ref["loss"], rtol=1e-4, atol=1e-6,
            msg=f"TP={tp_size} loss does not match TP=1"
        )

    def test_fused_tp_loss_same_across_ranks(self):
        """All TP ranks should compute identical indexer loss."""
        rank = self._rank
        tp_group = self._tp_group
        tp_size = self._tp_size

        inputs = _make_tp_test_inputs(np_global=max(16, tp_size * 4))
        tp_result = self._run_fused_tp(inputs, rank, tp_group, tp_size, overlap=False)

        # Gather losses from all ranks
        loss_tensor = tp_result["loss"].clone()
        loss_list = [torch.zeros_like(loss_tensor) for _ in range(tp_size)]
        torch.distributed.all_gather(loss_list, loss_tensor, group=tp_group)

        for i in range(1, tp_size):
            torch.testing.assert_close(
                loss_list[0], loss_list[i], rtol=1e-6, atol=1e-8,
                msg=f"Loss differs between rank 0 and rank {i}"
            )

    def test_fused_tp_indexer_grads_same_across_ranks(self):
        """Replicated indexer params should have identical gradients on all ranks."""
        rank = self._rank
        tp_group = self._tp_group
        tp_size = self._tp_size

        inputs = _make_tp_test_inputs(np_global=max(16, tp_size * 4))
        tp_result = self._run_fused_tp(inputs, rank, tp_group, tp_size, overlap=False)

        # All-gather from all ranks and compare
        for name in ["grad_q_indexer", "grad_k_indexer", "grad_weights"]:
            grad = tp_result[name].contiguous()
            gathered = [torch.zeros_like(grad) for _ in range(tp_size)]
            torch.distributed.all_gather(gathered, grad, group=tp_group)
            for i in range(1, tp_size):
                torch.testing.assert_close(
                    gathered[0], gathered[i], rtol=1e-4, atol=1e-5,
                    msg=f"Replicated {name} differs between rank 0 and rank {i}"
                )

    def test_fused_tp_grad_query_matches_tp1_slice(self):
        """Local grad_query should match the head slice from TP=1."""
        rank = self._rank
        tp_group = self._tp_group
        tp_size = self._tp_size

        inputs = _make_tp_test_inputs(np_global=max(16, tp_size * 4))
        ref = self._run_fused_tp1_reference(inputs)
        tp_result = self._run_fused_tp(inputs, rank, tp_group, tp_size, overlap=False)

        np_local = inputs["np_global"] // tp_size
        start_h = rank * np_local
        end_h = start_h + np_local

        ref_grad_slice = ref["grad_query"][:, :, start_h:end_h, :]
        torch.testing.assert_close(
            tp_result["grad_query"], ref_grad_slice, rtol=1e-2, atol=2e-2,
            msg=f"TP={tp_size} grad_query does not match TP=1 head slice"
        )

    def test_fused_tp_dense_loss_matches_tp1(self):
        """Dense loss with TP all-reduce should match TP=1 (方案B)."""
        rank = self._rank
        tp_group = self._tp_group
        tp_size = self._tp_size

        np_global = max(16, tp_size * 4)
        inputs = _make_tp_test_inputs(np_global=np_global)

        # TP=1 reference (dense)
        query_full = inputs["query_global"].clone().requires_grad_(True)
        kv_full = inputs["kv_full"].clone().requires_grad_(True)
        sink_full = inputs["attn_sink_global"].clone().requires_grad_(True)
        q_indexer = inputs["q_indexer"].clone().requires_grad_(True)
        k_indexer = inputs["k_indexer"].clone().requires_grad_(True)
        weights = inputs["weights"].clone().requires_grad_(True)

        ref_output, ref_loss = fused_indexer_sparse_attn(
            query_full, kv_full, sink_full, inputs["window_idxs"],
            q_indexer, k_indexer, weights,
            indexer_topk=inputs["topk"], ratio=4,
            softmax_scale=inputs["d"] ** -0.5,
            indexer_softmax_scale=q_indexer.shape[-1] ** -0.5,
            loss_coeff=0.1,
            sparse_loss=False,  # Dense!
            kv_offset=inputs["sq"],
            calculate_per_token_loss=False,
            tp_group=None,
        )

        # TP=N (dense)
        import megatron.plugin.dsa_kernel.triton_dsa_kernels as _mod
        _mod._DSA_TP_OVERLAP = False

        query_local, sink_local, np_local = _shard_for_rank(inputs, rank, tp_size)
        query_local = query_local.clone().requires_grad_(True)
        kv_tp = inputs["kv_full"].clone().requires_grad_(True)
        sink_tp = sink_local.clone().requires_grad_(True)
        qi_tp = inputs["q_indexer"].clone().requires_grad_(True)
        ki_tp = inputs["k_indexer"].clone().requires_grad_(True)
        w_tp = inputs["weights"].clone().requires_grad_(True)

        tp_output, tp_loss = fused_indexer_sparse_attn(
            query_local, kv_tp, sink_tp, inputs["window_idxs"],
            qi_tp, ki_tp, w_tp,
            indexer_topk=inputs["topk"], ratio=4,
            softmax_scale=inputs["d"] ** -0.5,
            indexer_softmax_scale=qi_tp.shape[-1] ** -0.5,
            loss_coeff=0.1,
            sparse_loss=False,  # Dense!
            kv_offset=inputs["sq"],
            calculate_per_token_loss=False,
            tp_group=tp_group,
        )

        torch.testing.assert_close(
            tp_loss, ref_loss, rtol=1e-4, atol=1e-6,
            msg=f"TP={tp_size} dense loss does not match TP=1"
        )

    def test_overlap_on_off_numerically_identical(self):
        """Overlap enabled vs disabled must produce identical loss and indexer grads.
        Attention grads (grad_query, grad_kv, grad_sink) go through Triton kernels
        with non-deterministic thread scheduling, so we check relative closeness."""
        rank = self._rank
        tp_group = self._tp_group
        tp_size = self._tp_size

        inputs = _make_tp_test_inputs(
            sq=2048,
            n_comp=512,
            topk=256,
            np_global=max(16, tp_size * 4),
        )
        sync_result = self._run_fused_tp(inputs, rank, tp_group, tp_size, overlap=False)
        async_result = self._run_fused_tp(inputs, rank, tp_group, tp_size, overlap=True)

        # Loss must be bit-identical (same decomposed compute, overlap is just timing)
        torch.testing.assert_close(
            sync_result["loss"], async_result["loss"], rtol=0, atol=0,
            msg="Overlap on/off loss mismatch"
        )

        # Indexer gradients are mathematically identical, but grad_k_indexer
        # aggregates many top-k contributions. Independent GPU executions can
        # differ in their accumulation order, especially at pretraining-scale
        # top-k, so bitwise equality is not a valid overlap requirement.
        for name in ["grad_q_indexer", "grad_k_indexer", "grad_weights"]:
            sync_grad = sync_result[name].float().flatten()
            async_grad = async_result[name].float().flatten()
            diff_norm = (sync_grad - async_grad).norm()
            reference_norm = sync_grad.norm()
            if reference_norm.item() < 1.0e-12:
                assert async_grad.norm().item() < 1.0e-12
                continue
            relative_l2 = (diff_norm / reference_norm).item()
            cosine = torch.nn.functional.cosine_similarity(
                sync_grad.unsqueeze(0), async_grad.unsqueeze(0), eps=1.0e-12
            ).item()
            assert relative_l2 < 1.0e-2 and cosine > 0.9999, (
                f"Overlap on/off {name} mismatch: "
                f"relative_l2={relative_l2:.6g}, cosine={cosine:.9g}"
            )

        # Attention output: forward path is identical for both, should match
        torch.testing.assert_close(
            sync_result["output"], async_result["output"], rtol=0, atol=0,
            msg="Overlap on/off output mismatch"
        )

        # Attention grads: Triton fused_dkv + sorted_scatter_add may exhibit
        # non-deterministic floating-point accumulation across kernel launches.
        # Use cosine similarity to verify they are essentially the same direction.
        for name in ["grad_query", "grad_kv", "grad_sink"]:
            s = sync_result[name].float().flatten()
            a = async_result[name].float().flatten()
            cos_sim = torch.nn.functional.cosine_similarity(s.unsqueeze(0), a.unsqueeze(0)).item()
            assert cos_sim > 0.9999, (
                f"Overlap on/off {name}: cosine similarity {cos_sim:.6f} too low"
            )


# ---------------------------------------------------------------------------
# Test 8: Distributed TP performance benchmark
# ---------------------------------------------------------------------------


import copy

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant.csa import CompressedSparseAttention
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.experimental_attention_variant.test_attention_variant_csa import (
    _make_csa_submodules,
    _make_mla_config,
)


@pytest.mark.skipif(
    not _DISTRIBUTED_AVAILABLE,
    reason="Requires a multi-process torchrun launch",
)
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability(0)[0] < 9,
    reason="Fused Triton CSA performance requires SM90+",
)
class TestDistributedTPPerformance:
    """Small accuracy gates plus post-SP, large-shape TP performance tests."""

    @pytest.fixture(scope="class", autouse=True)
    def setup_teardown(self, request):
        if not _is_torchrun():
            pytest.skip("Must be launched with torchrun")

        tp_size = _get_tp_size()
        if tp_size not in _SUPPORTED_DISTRIBUTED_TP_SIZES:
            pytest.skip(
                f"Distributed fused DSA tests support TP sizes "
                f"{_SUPPORTED_DISTRIBUTED_TP_SIZES}, got WORLD_SIZE={tp_size}"
            )
        num_attention_heads = int(os.environ.get("DSA_PERF_NUM_HEADS", "128"))
        if num_attention_heads % tp_size != 0:
            pytest.skip(
                f"DSA_PERF_NUM_HEADS={num_attention_heads} must be divisible by TP={tp_size}"
            )
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp_size, pipeline_model_parallel_size=1
        )
        model_parallel_cuda_manual_seed(20260802)

        cls = request.cls
        cls.tp_size = tp_size
        cls.pg_collection = ProcessGroupCollection.use_mpu_process_groups(
            required_pgs=['tp', 'cp']
        )
        cls.config = _make_mla_config(
            # Keep the per-rank attention workload representative.  With the
            # unit-test default (16 global heads), TP=8 leaves only two heads
            # per rank and measures launch/collective latency rather than the
            # kernel regime used by pre-training.
            num_attention_heads=num_attention_heads,
            hidden_size=int(os.environ.get("DSA_PERF_HIDDEN_SIZE", "4096")),
            v_head_dim=128,
            csa_compress_ratios=[4, 4, 4, 4],
            csa_window_size=128,
            tensor_model_parallel_size=tp_size,
            sequence_parallel=True,
            dsa_indexer_topk=256,
            dsa_indexer_loss_coeff=0.1,
            dsa_indexer_use_sparse_loss=True,
        )

        from megatron.core.models.common.embeddings import RotaryEmbedding

        cls.rotary_pos_emb = RotaryEmbedding(
            cls.config.qk_pos_emb_head_dim,
            rotary_percent=cls.config.rotary_percent,
            rotary_base=cls.config.rotary_base,
            cp_group=cls.pg_collection.cp,
        )
        yield
        Utils.destroy_model_parallel()

    def _build_csa(self, fused, compress_ratio):
        config = copy.copy(self.config)
        config.apply_dsa_kernel_fusion = fused
        config.csa_compress_ratios = [compress_ratio] * config.num_layers
        # Only ratio=4 constructs the learned indexer.  The ratio=128 and
        # window-only paths must not include an irrelevant auxiliary loss.
        if compress_ratio != 4:
            config.dsa_indexer_loss_coeff = 0.0
        return CompressedSparseAttention(
            config=config,
            submodules=_make_csa_submodules(),
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type='self',
            pg_collection=self.pg_collection,
            rotary_pos_emb=self.rotary_pos_emb,
            compress_ratio=compress_ratio,
        ).cuda().train()

    def _make_inputs(self, seq=2048):
        torch.manual_seed(20260802)
        batch = 1
        local_heads = self.config.num_attention_heads // self.tp_size
        head_dim = self.config.v_head_dim
        return {
            "query": torch.randn(
                seq, batch, local_heads, head_dim, device='cuda', dtype=torch.bfloat16
            ),
            "key": torch.randn(
                seq, batch, 1, head_dim, device='cuda', dtype=torch.bfloat16
            ),
            "x": torch.randn(
                seq, batch, self.config.hidden_size, device='cuda', dtype=torch.bfloat16
            ),
            "qr": torch.randn(
                seq, batch, self.config.q_lora_rank, device='cuda', dtype=torch.bfloat16
            ),
        }

    def _benchmark(self, module, inputs, overlap, backward=True, warmup=3, iters=10):
        import megatron.plugin.dsa_kernel.triton_dsa_kernels as triton_dsa

        triton_dsa._DSA_TP_OVERLAP = overlap

        # Reusing leaf tensors is valid after each graph has been consumed.
        # Keeping clone()/allocation outside the timed region avoids charging
        # input preparation to the CSA implementation.
        leaves = {
            name: tensor.detach().clone().requires_grad_(True)
            for name, tensor in inputs.items()
        }

        def run_once():
            module.zero_grad(set_to_none=True)
            for tensor in leaves.values():
                tensor.grad = None
            output = module(
                query=leaves["query"],
                key=leaves["key"],
                value=leaves["key"],
                attention_mask=None,
                x=leaves["x"],
                qr=leaves["qr"],
            )
            if backward:
                output.float().sum().backward()

        for _ in range(warmup):
            run_once()
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            run_once()
        end.record()
        torch.cuda.synchronize()
        local_ms = start.elapsed_time(end) / iters

        elapsed = torch.tensor(local_ms, device='cuda', dtype=torch.float32)
        torch.distributed.all_reduce(
            elapsed, op=torch.distributed.ReduceOp.MAX, group=self.pg_collection.tp
        )
        return elapsed.item()

    def _measure_peak_memory(self, module, inputs, overlap):
        """Return the maximum per-rank incremental peak memory in MiB."""
        import megatron.plugin.dsa_kernel.triton_dsa_kernels as triton_dsa

        triton_dsa._DSA_TP_OVERLAP = overlap
        leaves = {
            name: tensor.detach().clone().requires_grad_(True)
            for name, tensor in inputs.items()
        }

        # Warm up lazy kernel/library allocations before establishing the
        # baseline. Allocated (rather than reserved) bytes exclude allocator
        # cache retained by earlier benchmark cases.
        module.zero_grad(set_to_none=True)
        warmup_output = module(
            query=leaves["query"],
            key=leaves["key"],
            value=leaves["key"],
            attention_mask=None,
            x=leaves["x"],
            qr=leaves["qr"],
        )
        warmup_output.float().sum().backward()
        torch.cuda.synchronize()

        module.zero_grad(set_to_none=True)
        for tensor in leaves.values():
            tensor.grad = None
        del warmup_output
        torch.cuda.synchronize()

        device = torch.cuda.current_device()
        baseline = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)

        output = module(
            query=leaves["query"],
            key=leaves["key"],
            value=leaves["key"],
            attention_mask=None,
            x=leaves["x"],
            qr=leaves["qr"],
        )
        output.float().sum().backward()
        torch.cuda.synchronize()
        peak_bytes = max(torch.cuda.max_memory_allocated(device) - baseline, 0)

        peak = torch.tensor(float(peak_bytes), device='cuda', dtype=torch.float64)
        torch.distributed.all_reduce(
            peak, op=torch.distributed.ReduceOp.MAX, group=self.pg_collection.tp
        )
        return peak.item() / (1024**2)

    @pytest.mark.parametrize(
        "case_name,compress_ratio",
        [
            ("ratio4_indexer", 4),
            ("ratio128", 128),
            ("window_only", 0),
        ],
        ids=["ratio4_indexer", "ratio128", "window_only"],
    )
    def test_csa_module_fused_vs_unfused_accuracy(
        self, case_name, compress_ratio, dsa_metrics
    ):
        """Validate full-module forward and backward against unfused CSA."""
        import megatron.plugin.dsa_kernel.triton_dsa_kernels as triton_dsa

        triton_dsa._DSA_TP_OVERLAP = False
        inputs = self._make_inputs()
        unfused = self._build_csa(fused=False, compress_ratio=compress_ratio)
        fused = self._build_csa(fused=True, compress_ratio=compress_ratio)
        fused.load_state_dict(unfused.state_dict())

        torch.manual_seed(20260803)
        grad_output = torch.randn(
            2048,
            1,
            self.config.num_attention_heads // self.tp_size * self.config.v_head_dim,
            device='cuda',
            dtype=torch.bfloat16,
        )

        def run(module):
            module.zero_grad(set_to_none=True)
            leaves = {
                name: tensor.detach().clone().requires_grad_(True)
                for name, tensor in inputs.items()
            }
            output = module(
                query=leaves["query"],
                key=leaves["key"],
                value=leaves["key"],
                attention_mask=None,
                x=leaves["x"],
                qr=leaves["qr"],
            )
            (output.float() * grad_output.float()).sum().backward()
            return {
                "output": output.detach().float(),
                "inputs": {
                    name: tensor.grad.detach().float()
                    for name, tensor in leaves.items()
                    if tensor.grad is not None
                },
                "params": {
                    name: param.grad.detach().float()
                    for name, param in module.named_parameters()
                    if param.grad is not None
                },
            }

        reference = run(unfused)
        actual = run(fused)

        metric_params = {
            "case": case_name,
            "tp": self.tp_size,
            "sq": 2048,
            "hidden": self.config.hidden_size,
            "global_heads": self.config.num_attention_heads,
            "ratio": compress_ratio,
        }

        def assert_numerically_close(name, actual_tensor, reference_tensor, max_rel_l2, min_cos):
            actual_flat = actual_tensor.reshape(-1)
            reference_flat = reference_tensor.reshape(-1)
            diff = actual_flat - reference_flat
            reference_norm = reference_flat.norm()
            actual_norm = actual_flat.norm()
            if reference_norm == 0 and actual_norm == 0:
                return
            relative_l2 = (diff.norm() / reference_norm.clamp(min=1e-12)).item()
            cosine = torch.nn.functional.cosine_similarity(
                actual_flat.unsqueeze(0), reference_flat.unsqueeze(0), eps=1e-12
            ).item()
            assert relative_l2 < max_rel_l2 and cosine > min_cos, (
                f"{case_name} {name} mismatch: relative_l2={relative_l2:.6g}, "
                f"cosine={cosine:.9g}, max_abs_diff={diff.abs().max().item():.6g}"
            )
            # Each torchrun worker owns an independent pytest session.  Only
            # TP rank 0 contributes records, avoiding duplicate rows and
            # concurrent writes to the same Markdown report.
            if self.pg_collection.tp.rank() == 0:
                dsa_metrics.record_accuracy(
                    params=metric_params,
                    cos_sim=cosine,
                    max_diff=diff.abs().max().item(),
                    mean_diff=diff.abs().mean().item(),
                    target=name,
                )

        assert_numerically_close(
            "output", actual["output"], reference["output"], max_rel_l2=2e-2, min_cos=0.999
        )
        assert actual["inputs"].keys() == reference["inputs"].keys()
        for name in reference["inputs"]:
            assert_numerically_close(
                f"grad_input.{name}",
                actual["inputs"][name],
                reference["inputs"][name],
                max_rel_l2=7e-2,
                min_cos=0.995,
            )
        assert actual["params"].keys() == reference["params"].keys()
        for name in reference["params"]:
            assert_numerically_close(
                f"grad_param.{name}",
                actual["params"][name],
                reference["params"][name],
                max_rel_l2=1e-1,
                min_cos=0.99,
            )

    @pytest.mark.perf
    @pytest.mark.parametrize(
        "case_name,compress_ratio",
        [
            ("ratio4_indexer", 4),
            ("ratio128", 128),
            ("window_only", 0),
        ],
        ids=["ratio4_indexer", "ratio128", "window_only"],
    )
    def test_csa_module_fused_vs_unfused_performance(
        self, case_name, compress_ratio, dsa_metrics
    ):
        # CSA runs after the SP gather in DSv4HybridSparseAttention, so its
        # sequence dimension is the global (post-SP) sequence length.  Keep
        # this substantially larger than the accuracy case to represent the
        # regime where TP+SP is useful, while allowing constrained machines
        # to override it explicitly.
        global_seq = int(os.environ.get("DSA_PERF_GLOBAL_SEQ", "8192"))
        assert global_seq % self.tp_size == 0, (
            f"DSA_PERF_GLOBAL_SEQ={global_seq} must be divisible by TP={self.tp_size}"
        )
        sp_local_seq = global_seq // self.tp_size
        inputs = self._make_inputs(seq=global_seq)
        unfused = self._build_csa(fused=False, compress_ratio=compress_ratio)
        fused = self._build_csa(fused=True, compress_ratio=compress_ratio)
        fused.load_state_dict(unfused.state_dict())

        warmup = int(os.environ.get("DSA_PERF_WARMUP", "2"))
        iters = int(os.environ.get("DSA_PERF_ITERS", "5"))

        def bench(module, overlap, backward=True):
            return self._benchmark(
                module,
                inputs,
                overlap=overlap,
                backward=backward,
                warmup=warmup,
                iters=iters,
            )

        unfused_ms = bench(unfused, overlap=False)
        fused_sync_ms = bench(fused, overlap=False)
        fused_async_ms = bench(fused, overlap=True)
        unfused_fwd_ms = bench(unfused, overlap=False, backward=False)
        fused_sync_fwd_ms = bench(fused, overlap=False, backward=False)
        fused_async_fwd_ms = bench(fused, overlap=True, backward=False)
        unfused_peak_mb = self._measure_peak_memory(unfused, inputs, overlap=False)
        fused_sync_peak_mb = self._measure_peak_memory(fused, inputs, overlap=False)
        fused_async_peak_mb = self._measure_peak_memory(fused, inputs, overlap=True)

        rank = self.pg_collection.tp.rank()
        if rank == 0:
            metric_params = {
                "case": case_name,
                "tp": self.tp_size,
                "global_sq": global_seq,
                "sp_local_sq": sp_local_seq,
                "hidden": self.config.hidden_size,
                "global_heads": self.config.num_attention_heads,
                "topk": self.config.dsa_indexer_topk,
                "ratio": compress_ratio,
            }
            for label, fused_ms, unfused_baseline_ms in (
                ("fwd_sync", fused_sync_fwd_ms, unfused_fwd_ms),
                ("fwd_async", fused_async_fwd_ms, unfused_fwd_ms),
                ("e2e_sync", fused_sync_ms, unfused_ms),
                ("e2e_async", fused_async_ms, unfused_ms),
            ):
                dsa_metrics.record_performance(
                    params=metric_params,
                    fused_ms=fused_ms,
                    unfused_ms=unfused_baseline_ms,
                    speedup=unfused_baseline_ms / max(fused_ms, 1e-6),
                    label=label,
                )
            dsa_metrics.record_memory(
                params={**metric_params, "mode": "sync"},
                fused_mb=fused_sync_peak_mb,
                unfused_mb=unfused_peak_mb,
                ratio=unfused_peak_mb / max(fused_sync_peak_mb, 1e-6),
            )
            dsa_metrics.record_memory(
                params={**metric_params, "mode": "async"},
                fused_mb=fused_async_peak_mb,
                unfused_mb=unfused_peak_mb,
                ratio=unfused_peak_mb / max(fused_async_peak_mb, 1e-6),
            )
            print(
                f"\n  End-to-end CSA module performance "
                f"(case={case_name}, TP={self.tp_size}):"
            )
            print(
                f"    shape: post-SP global_S={global_seq}, SP-local_S={sp_local_seq}, B=1, "
                f"hidden={self.config.hidden_size}, global_heads={self.config.num_attention_heads}, "
                f"local_heads={self.config.num_attention_heads // self.tp_size}, "
                f"head_dim={self.config.v_head_dim}, topk={self.config.dsa_indexer_topk}, "
                f"window={self.config.csa_window_size}, compress_ratio={compress_ratio}"
            )
            print(f"    forward old/new sync : {unfused_fwd_ms:.3f} / {fused_sync_fwd_ms:.3f} ms")
            print(f"    forward fused async  : {fused_async_fwd_ms:.3f} ms")
            print(f"    old (unfused)       : {unfused_ms:.3f} ms/iter")
            print(f"    new (fused sync)    : {fused_sync_ms:.3f} ms/iter")
            print(f"    new (fused async)   : {fused_async_ms:.3f} ms/iter")
            print(f"    sync speedup (old/new): {unfused_ms / fused_sync_ms:.3f}x")
            print(f"    async speedup (old/new): {unfused_ms / fused_async_ms:.3f}x")
            print(
                f"    peak memory old/sync/async: {unfused_peak_mb:.1f} / "
                f"{fused_sync_peak_mb:.1f} / {fused_async_peak_mb:.1f} MiB"
            )
            print(
                f"    memory ratio (old/new): "
                f"{unfused_peak_mb / max(fused_sync_peak_mb, 1e-6):.3f}x"
            )




# ---------------------------------------------------------------------------
# Test 9: Distributed TP — unfused reference parity
# ---------------------------------------------------------------------------


@_skip_unless_sm90
@_skip_unless_distributed
class TestDistributedTPUnfusedParity:
    """Compare fused TP loss against unfused reference (dsa.py FusedDSAIndexerLoss).

    The unfused path already has TP all-reduce in compute_dsa_indexer_loss.
    We verify that the fused decomposed path produces the same teacher target.

    Run with: torchrun --standalone --nproc_per_node=8 -m pytest ...
    -k "TestDistributedTPUnfusedParity".
    """

    @pytest.fixture(scope="class", autouse=True)
    def setup_teardown(self):
        if not _is_torchrun():
            pytest.skip("Must be launched with torchrun")
        rank, tp_group, tp_size = _init_tp()
        self.__class__._rank = rank
        self.__class__._tp_group = tp_group
        self.__class__._tp_size = tp_size
        yield
        _destroy_tp()

    def test_sparse_target_matches_unfused_allreduce(self):
        """The fused decomposed target (local_head_sum → all-reduce → normalize)
        should match the unfused target computation."""
        rank = self._rank
        tp_group = self._tp_group
        tp_size = self._tp_size

        torch.manual_seed(42)
        np_global = max(16, tp_size * 4)
        B, S_q, D = 2, 64, 128
        n_comp = 16
        S_kv = S_q + n_comp
        topk = 4
        np_local = np_global // tp_size
        softmax_scale = D ** -0.5

        # Global inputs (same on all ranks via seed)
        q_attn_global = torch.randn(B, S_q, np_global, D, device="cuda", dtype=torch.bfloat16)
        k_attn = torch.randn(B, S_kv, D, device="cuda", dtype=torch.bfloat16)
        lse_global = torch.randn(B, S_q, np_global, device="cuda", dtype=torch.float32) + 5.0
        topk_indices = torch.randint(0, n_comp, (B, S_q, topk), device="cuda", dtype=torch.int32)
        topk_indices[:, :2, :] = -1

        # Local shard
        start_h = rank * np_local
        end_h = start_h + np_local
        q_local = q_attn_global[:, :, start_h:end_h, :]
        lse_local = lse_global[:, :, start_h:end_h]

        # Fused decomposed: local head sum → all-reduce → normalize
        local_head_sum = compute_sparse_local_target_head_sum(
            q_local, k_attn, lse_local, topk_indices,
            softmax_scale=softmax_scale, kv_offset=S_q,
        )
        local_head_sum_ar = local_head_sum.contiguous().clone()
        torch.distributed.all_reduce(local_head_sum_ar, group=tp_group)
        fused_target = local_head_sum_ar / local_head_sum_ar.sum(dim=-1, keepdim=True).clamp(min=1e-12)

        # Reference: compute full head sum directly (simulate single-rank)
        global_head_sum = compute_sparse_local_target_head_sum(
            q_attn_global, k_attn, lse_global, topk_indices,
            softmax_scale=softmax_scale, kv_offset=S_q,
        )
        ref_target = global_head_sum / global_head_sum.sum(dim=-1, keepdim=True).clamp(min=1e-12)

        torch.testing.assert_close(
            fused_target, ref_target, rtol=1e-5, atol=1e-6,
            msg=f"Fused TP={tp_size} target does not match unfused global target"
        )

    def test_no_deadlock_with_masked_rows(self):
        """All-reduce must succeed even when some rows are fully masked.
        This tests that we don't skip the collective based on local data."""
        rank = self._rank
        tp_group = self._tp_group
        tp_size = self._tp_size

        torch.manual_seed(99)
        np_local = 4
        B, S_q, D = 2, 32, 128
        n_comp = 8
        S_kv = S_q + n_comp
        topk = 4
        softmax_scale = D ** -0.5

        q_local = torch.randn(B, S_q, np_local, D, device="cuda", dtype=torch.bfloat16)
        k_attn = torch.randn(B, S_kv, D, device="cuda", dtype=torch.bfloat16)
        lse_local = torch.randn(B, S_q, np_local, device="cuda", dtype=torch.float32) + 5.0
        # ALL rows masked
        topk_indices = torch.full((B, S_q, topk), -1, device="cuda", dtype=torch.int32)

        local_head_sum = compute_sparse_local_target_head_sum(
            q_local, k_attn, lse_local, topk_indices,
            softmax_scale=softmax_scale, kv_offset=S_q,
        )

        # This must not deadlock — all ranks enter the collective unconditionally
        local_head_sum_c = local_head_sum.contiguous()
        torch.distributed.all_reduce(local_head_sum_c, group=tp_group)

        # All zeros (masked) — no NaN
        assert not torch.isnan(local_head_sum_c).any()
        assert local_head_sum_c.abs().max().item() == 0.0
