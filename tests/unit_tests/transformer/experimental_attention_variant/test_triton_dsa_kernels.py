# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from unittest.mock import patch

import pytest
import torch

pytest.importorskip("triton")

from megatron.plugin.dsa_kernel import triton_dsa_kernels as tdk  # noqa: E402


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
