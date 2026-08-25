# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("triton")

from megatron.plugin.dsa_kernel import triton_dsa_kernels as tdk  # noqa: E402
from megatron.plugin.dsa_kernel.legacy import pytorch_sparse_attn as legacy_attn  # noqa: E402


@dataclass(frozen=True)
class _AccuracyThresholds:
    """Independent gates; ``None`` records a metric without enforcing it."""

    rtol: float
    atol: float
    min_cosine: float | None
    max_relative_l2: float | None
    max_norm_relative_error: float | None


# The repository's 1e-4 allclose gate is for FP32 scatter-vs-scatter tests.
# End-to-end sparse attention crosses BF16 WGMMA, online softmax, and BF16
# output quantization. Match the existing fused-attention parity conventions:
# output relative L2 < 5e-3, and gradients primarily by direction and scale.
_OUTPUT_THRESHOLDS = _AccuracyThresholds(
    rtol=1e-2,
    atol=1e-3,  # Accommodates one BF16 ULP for the observed output range.
    min_cosine=0.999,
    max_relative_l2=5e-3,
    max_norm_relative_error=5e-3,
)

_LOSS_THRESHOLDS = _AccuracyThresholds(
    rtol=1e-2,
    atol=2e-4,
    min_cosine=None,  # Cosine is not discriminative for a non-negative scalar.
    max_relative_l2=1e-2,
    max_norm_relative_error=None,  # Duplicates relative error for a scalar.
)

_ATTN_GRAD_THRESHOLDS = _AccuracyThresholds(
    rtol=5e-2,
    atol=5e-2,
    min_cosine=0.995,
    max_relative_l2=7e-2,
    max_norm_relative_error=5e-2,
)

_SINK_GRAD_THRESHOLDS = _AccuracyThresholds(
    rtol=5e-2,
    atol=5e-2,
    min_cosine=0.95,
    max_relative_l2=2e-1,
    max_norm_relative_error=2e-1,
)

_INDEXER_GRAD_THRESHOLDS = _AccuracyThresholds(
    rtol=3e-2,
    atol=2e-2,
    min_cosine=0.99,
    max_relative_l2=1e-1,
    max_norm_relative_error=5e-2,
)


def _assert_accuracy(name, actual, reference, thresholds):
    """Require local closeness as well as matching global direction and norm."""
    actual_f = actual.detach().float().reshape(-1)
    reference_f = reference.detach().float().reshape(-1)
    assert actual_f.shape == reference_f.shape, (
        f"{name}: shape mismatch {tuple(actual.shape)} != {tuple(reference.shape)}"
    )
    assert torch.isfinite(actual_f).all(), f"{name}: actual contains NaN or Inf"
    assert torch.isfinite(reference_f).all(), f"{name}: reference contains NaN or Inf"

    diff = actual_f - reference_f
    actual_norm = actual_f.norm()
    reference_norm = reference_f.norm()
    scale = reference_norm.clamp_min(1e-12)
    relative_l2 = (diff.norm() / scale).item()
    norm_relative_error = (torch.abs(actual_norm - reference_norm) / scale).item()
    if actual_norm.item() == 0.0 and reference_norm.item() == 0.0:
        cosine = 1.0
    elif actual_norm.item() == 0.0 or reference_norm.item() == 0.0:
        cosine = 0.0
    else:
        cosine = F.cosine_similarity(
            actual_f.unsqueeze(0), reference_f.unsqueeze(0), eps=1e-12
        ).item()
    allclose = torch.allclose(
        actual_f, reference_f, rtol=thresholds.rtol, atol=thresholds.atol
    )
    max_abs = diff.abs().max().item() if diff.numel() else 0.0
    mean_abs = diff.abs().mean().item() if diff.numel() else 0.0

    failures = []
    if not allclose:
        failures.append(f"allclose(rtol={thresholds.rtol}, atol={thresholds.atol})")
    if thresholds.min_cosine is not None and cosine < thresholds.min_cosine:
        failures.append(f"cosine>={thresholds.min_cosine}")
    if (
        thresholds.max_relative_l2 is not None
        and relative_l2 > thresholds.max_relative_l2
    ):
        failures.append(f"relative_l2<={thresholds.max_relative_l2}")
    if (
        thresholds.max_norm_relative_error is not None
        and norm_relative_error > thresholds.max_norm_relative_error
    ):
        failures.append(f"norm_relative_error<={thresholds.max_norm_relative_error}")
    assert not failures, (
        f"{name} failed {', '.join(failures)}: cosine={cosine:.9g}, "
        f"relative_l2={relative_l2:.6g}, norm_relative_error={norm_relative_error:.6g}, "
        f"max_abs={max_abs:.6g}, mean_abs={mean_abs:.6g}"
    )


def _cu(lengths):
    return torch.tensor([0] + list(torch.tensor(lengths).cumsum(0).tolist()), dtype=torch.int32)


class TestTritonDSAUnifiedLayouts:
    """Shape-contract tests for the shared SBHD/THD Triton wrappers."""

    def test_thd_local_indices_are_globalized_per_sequence(self):
        cu_q = _cu([2, 3])
        cu_kv = _cu([4, 5])
        local = torch.tensor(
            [[0, 3], [1, -1], [0, 4], [2, 1], [-1, 3]], dtype=torch.int32
        )

        actual = tdk.local_to_global_flat(
            local, -1, cu_seqlens_q=cu_q, cu_seqlens_kv=cu_kv
        )

        expected = torch.tensor(
            [[0, 3], [1, -1], [4, 8], [6, 5], [-1, 7]], dtype=torch.int32
        )
        torch.testing.assert_close(actual, expected)

    def test_thd_indexer_topk_applies_cp_causal_offsets(self):
        cu_q = _cu([2, 2])
        cu_k = _cu([3, 3])
        q = torch.randn(4, 2, 4)
        k = torch.randn(6, 4)
        weights = torch.randn(4, 2)
        scores = torch.tensor(
            [
                [[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]],
                [[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]],
            ]
        )

        with patch.object(
            tdk,
            "_indexer_topk_bshd",
            return_value=(torch.empty(0), torch.empty(0), scores),
        ):
            topk, lengths = tdk.indexer_topk(
                q,
                k,
                weights,
                topk=1,
                ratio=2,
                cu_seqlens_q=cu_q,
                cu_seqlens_kv=cu_k,
                max_seqlen_q=2,
                max_seqlen_kv=3,
                q_causal_offsets=torch.tensor([0, 4], dtype=torch.int32),
            )

        torch.testing.assert_close(
            topk, torch.tensor([[-1], [0], [1], [2]], dtype=torch.int32)
        )
        torch.testing.assert_close(lengths, torch.tensor([0, 1, 1, 1], dtype=torch.int32))

    def test_sparse_attention_thd_reuses_flat_kernel_contract(self):
        query = torch.randn(5, 2, 4)
        kv = torch.randn(7, 4)
        topk = torch.zeros(5, 3, dtype=torch.int32)
        sink = torch.zeros(2)
        flat_out = torch.randn(5, 2, 4)

        with patch.object(
            tdk,
            "_dsa_sparse_attn_flat",
            return_value=(flat_out, torch.empty(5, 2), None),
        ) as flat_kernel:
            output = tdk.dsa_sparse_attn(
                query, kv, sink, topk, softmax_scale=0.5, is_thd=True
            )

        assert output.shape == (5, 8)
        assert flat_kernel.call_args.args[0] is query
        assert flat_kernel.call_args.args[1] is kv

    def test_fused_thd_packs_once_and_restores_natural_row_order(self):
        cu_q = _cu([3, 2])
        cu_kv = _cu([3, 2])
        cu_comp = _cu([2, 1])
        cu_full = _cu([5, 3])
        query = torch.randn(5, 2, 4)
        kv_full = torch.arange(8 * 4, dtype=torch.float32).reshape(8, 4)
        q_indexer = torch.randn(5, 2, 3)
        k_indexer = torch.randn(3, 3)
        weights = torch.randn(5, 2)
        window = torch.zeros(5, 2, dtype=torch.int32)

        padded_output = torch.arange(6, dtype=torch.float32).reshape(3, 2, 1).expand(-1, -1, 8)

        with patch.object(
            tdk.FusedIndexerSparseAttnFunc,
            "apply",
            return_value=(padded_output, torch.tensor(1.0)),
        ) as fused_apply:
            output, loss = tdk.fused_indexer_sparse_attn(
                query,
                kv_full,
                torch.zeros(2),
                window,
                q_indexer,
                k_indexer,
                weights,
                indexer_topk=2,
                ratio=4,
                softmax_scale=0.5,
                cu_seqlens_q=cu_q,
                cu_seqlens_kv=cu_kv,
                cu_seqlens_kv_full=cu_full,
                cu_seqlens_compressed_idx=cu_comp,
                max_seqlen_q=3,
                max_seqlen_compressed_idx=2,
                compressed_kv=torch.empty(3, 4),
            )

        args = fused_apply.call_args.args
        assert args[0].shape == (3, 2, 2, 4)  # padded query SBHD
        assert args[1].shape == (5, 2, 4)  # [max Q + max compressed, B, D]
        torch.testing.assert_close(
            args[16], torch.tensor([[True, True], [True, False]])
        )
        torch.testing.assert_close(
            args[17], torch.tensor([[True, True, True], [True, True, False]])
        )
        assert output.shape == (5, 8)
        torch.testing.assert_close(output[:, 0], torch.tensor([0.0, 2.0, 4.0, 1.0, 3.0]))
        torch.testing.assert_close(loss, torch.tensor(1.0))

    def test_cp_from_topk_sparse_loss_routes_all_gradient_slots(self):
        query = torch.randn(2, 2, 4, requires_grad=True)
        kv_full = torch.randn(5, 4, requires_grad=True)
        sink = torch.zeros(2, requires_grad=True)
        q_indexer = torch.randn(2, 1, 3, requires_grad=True)
        k_indexer = torch.randn(2, 3, requires_grad=True)
        weights = torch.randn(2, 1, requires_grad=True)
        compressed_kv = torch.randn(2, 4, requires_grad=True)
        topk = torch.tensor([[0, 2], [1, 3]], dtype=torch.int32)
        indexer_topk = torch.tensor([[0], [1]], dtype=torch.int32)
        out = torch.randn(2, 2, 4)
        lse = torch.randn(2, 2)

        with (
            patch.object(
                tdk,
                "triton_sparse_attn_forward",
                return_value=(out, lse, lse),
            ),
            patch.object(
                tdk,
                "compute_sparse_local_target_head_sum",
                return_value=torch.ones(1, 2, 1),
            ),
            patch.object(tdk, "compute_sparse_indexer_predict_state", return_value={}),
            patch.object(
                tdk,
                "sparse_indexer_kl_and_backward",
                return_value=(
                    torch.tensor(2.0),
                    torch.ones(1, 2, 1, 3),
                    torch.full((1, 2, 3), 2.0),
                    torch.full((1, 2, 1), 3.0),
                ),
            ),
            patch.object(
                tdk,
                "triton_sparse_attn_backward",
                return_value={
                    "dq": torch.ones_like(query),
                    "dkv": torch.ones_like(kv_full),
                    "d_sink": torch.ones_like(sink),
                },
            ),
        ):
            output, loss = tdk.FusedIndexerSparseAttnFromTopkFunc.apply(
                query,
                kv_full,
                sink,
                topk,
                q_indexer,
                k_indexer,
                weights,
                indexer_topk,
                compressed_kv,
                0.5,
                0.25,
                1.0,
                2.0,
                True,
                4,
                8,
                (),
                None,
                None,
            )
            (output.sum() + loss).backward()

        torch.testing.assert_close(query.grad, torch.ones_like(query))
        torch.testing.assert_close(kv_full.grad, torch.ones_like(kv_full))
        torch.testing.assert_close(sink.grad, torch.ones_like(sink))
        torch.testing.assert_close(q_indexer.grad, torch.ones_like(q_indexer))
        torch.testing.assert_close(k_indexer.grad, torch.full_like(k_indexer, 2.0))
        torch.testing.assert_close(weights.grad, torch.full_like(weights, 0.75))
        assert compressed_kv.grad is None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
    def test_sm90_from_topk_sparse_loss_matches_pytorch_reference(self):
        """Exercise the Triton kernel directly, without importing CP/CuTe code."""
        if torch.cuda.get_device_capability()[0] < 9:
            pytest.skip("the head-parallel Triton kernel requires SM90 or newer")

        torch.manual_seed(1234)
        device = torch.device("cuda")
        total_q, heads, dim = 16, 16, 64
        original_rows, compressed_rows = 32, 16
        indexer_heads, indexer_dim = 4, 32
        indexer_topk, window_topk = 8, 8
        softmax_scale = dim**-0.5
        indexer_softmax_scale = indexer_dim**-0.5
        loss_coeff, loss_divisor = 0.7, total_q

        query_ref = (
            torch.randn(total_q, heads, dim, device=device, dtype=torch.bfloat16) * 0.2
        )
        kv_ref = (
            torch.randn(
                original_rows + compressed_rows,
                dim,
                device=device,
                dtype=torch.bfloat16,
            )
            * 0.2
        )
        sink_ref = torch.linspace(-0.2, 0.2, heads, device=device, dtype=torch.float32)
        q_indexer_ref = (
            torch.randn(
                total_q,
                indexer_heads,
                indexer_dim,
                device=device,
                dtype=torch.bfloat16,
            )
            * 0.2
        )
        k_indexer_ref = (
            torch.randn(
                compressed_rows, indexer_dim, device=device, dtype=torch.bfloat16
            )
            * 0.2
        )
        weights_ref = torch.sigmoid(
            torch.randn(
                total_q, indexer_heads, device=device, dtype=torch.float32
            )
        ).to(torch.bfloat16)

        rows = torch.arange(total_q, device=device)[:, None]
        slots = torch.arange(indexer_topk, device=device)[None, :]
        indexer_topk_idxs = ((rows + slots) % compressed_rows).to(torch.int32)
        compressed_global_idxs = indexer_topk_idxs + original_rows
        window_slots = torch.arange(window_topk, device=device)[None, :]
        window_idxs = ((rows * 3 + window_slots) % original_rows).to(torch.int32)
        topk_idxs = torch.cat((compressed_global_idxs, window_idxs), dim=-1).contiguous()
        shared_topk = topk_idxs.unsqueeze(1).expand(-1, heads, -1)
        grad_output = torch.randn(
            total_q, heads, dim, device=device, dtype=torch.bfloat16
        )

        reference_output, reference_lse, reference_indexer_lse = (
            legacy_attn.pytorch_sparse_attn_fwd(
                query_ref,
                kv_ref,
                shared_topk,
                softmax_scale,
                dim,
                sink_ref,
                indexer_topk,
            )
        )
        reference_dq, reference_dkv, reference_dsink = (
            legacy_attn.pytorch_sparse_attn_bwd(
                grad_output,
                query_ref,
                kv_ref,
                shared_topk,
                reference_output,
                reference_lse,
                sink_ref,
                softmax_scale,
                dim,
            )
        )
        selected_attn_k = kv_ref[original_rows:][indexer_topk_idxs.long()].float()
        reference_attn_scores = torch.einsum(
            "qhd,qtd->qht", query_ref.float(), selected_attn_k
        ) * softmax_scale
        reference_head_sum = torch.exp(
            reference_attn_scores - reference_indexer_lse.unsqueeze(-1)
        ).sum(dim=1)

        # Keep the sparse-loss oracle independent from the manual backward
        # helper used by FusedIndexerSparseAttnFromTopkFunc.
        q_indexer_oracle = q_indexer_ref.detach().clone().requires_grad_(True)
        k_indexer_oracle = k_indexer_ref.detach().clone().requires_grad_(True)
        weights_oracle = weights_ref.detach().clone().requires_grad_(True)
        selected_k_indexer = k_indexer_oracle[indexer_topk_idxs.long()].float()
        per_head_scores = torch.einsum(
            "qhd,qtd->qht", q_indexer_oracle.float(), selected_k_indexer
        )
        logits = (
            torch.relu(per_head_scores)
            * (weights_oracle.float() * indexer_softmax_scale).unsqueeze(-1)
        ).sum(dim=1)
        reference_predict = torch.softmax(logits, dim=-1)
        reference_target = reference_head_sum / reference_head_sum.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-12)
        reference_loss = (
            reference_target
            * (
                torch.log(reference_target + 1e-10)
                - torch.log(reference_predict + 1e-10)
            )
        ).sum() * (loss_coeff / loss_divisor)
        (
            reference_dq_indexer,
            reference_dk_indexer,
            reference_dweights,
        ) = torch.autograd.grad(
            reference_loss, (q_indexer_oracle, k_indexer_oracle, weights_oracle)
        )

        query = query_ref.detach().clone().requires_grad_(True)
        kv = kv_ref.detach().clone().requires_grad_(True)
        sink = sink_ref.detach().clone().requires_grad_(True)
        q_indexer = q_indexer_ref.detach().clone().requires_grad_(True)
        k_indexer = k_indexer_ref.detach().clone().requires_grad_(True)
        weights = weights_ref.detach().clone().requires_grad_(True)

        fallback_error = AssertionError("eligible SM90 test unexpectedly used legacy fallback")
        with (
            patch.object(legacy_attn, "pytorch_sparse_attn_fwd", side_effect=fallback_error),
            patch.object(legacy_attn, "pytorch_sparse_attn_bwd", side_effect=fallback_error),
        ):
            output, loss = tdk.FusedIndexerSparseAttnFromTopkFunc.apply(
                query,
                kv,
                sink,
                topk_idxs,
                q_indexer,
                k_indexer,
                weights,
                indexer_topk_idxs,
                kv[original_rows:].detach(),
                softmax_scale,
                indexer_softmax_scale,
                loss_coeff,
                loss_divisor,
                True,
                4,
                total_q,
                (),
                None,
                None,
            )
            ((output.reshape_as(grad_output) * grad_output).sum() + loss).backward()

        _assert_accuracy(
            "output", output.reshape_as(reference_output), reference_output, _OUTPUT_THRESHOLDS
        )
        _assert_accuracy("loss", loss, reference_loss, _LOSS_THRESHOLDS)
        _assert_accuracy("grad.query", query.grad, reference_dq, _ATTN_GRAD_THRESHOLDS)
        _assert_accuracy("grad.kv", kv.grad, reference_dkv, _ATTN_GRAD_THRESHOLDS)
        _assert_accuracy("grad.sink", sink.grad, reference_dsink, _SINK_GRAD_THRESHOLDS)
        _assert_accuracy(
            "grad.q_indexer",
            q_indexer.grad,
            reference_dq_indexer,
            _INDEXER_GRAD_THRESHOLDS,
        )
        _assert_accuracy(
            "grad.k_indexer",
            k_indexer.grad,
            reference_dk_indexer,
            _INDEXER_GRAD_THRESHOLDS,
        )
        _assert_accuracy(
            "grad.weights", weights.grad, reference_dweights, _INDEXER_GRAD_THRESHOLDS
        )
