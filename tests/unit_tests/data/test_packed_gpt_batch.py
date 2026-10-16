# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Synthetic packed GPT batches across real TP/PP/CP groups."""

import pytest
import torch

from megatron.core import parallel_state
from pretrain_gpt import get_batch
from tests.unit_tests.data.test_get_batch import initialize_test_environment
from tests.unit_tests.test_utilities import Utils


@pytest.mark.parametrize(
    "tp,pp,cp,dsv4", [(2, 1, 1, False), (1, 2, 1, True), (1, 1, 2, True), (1, 2, 2, True)]
)
def test_packed_gpt_batch_keeps_real_and_physical_metadata(tp, pp, cp, dsv4):
    if Utils.world_size % (tp * pp * cp):
        pytest.skip("launch size does not support this TP/PP/CP topology")
    args = initialize_test_environment(tp, pp, cp, 64, 2, global_batch_size=4, sft=True)
    try:
        if dsv4:
            args.experimental_attention_variant = 'dsv4_hybrid'
            args.multi_latent_attention = True
            args.csa_compress_ratios = [0] * args.num_layers
            args.v_head_dim = 512
            args.qk_pos_emb_head_dim = 64
            args.q_lora_rank = 64
            args.dsa_kernel_backend = 'cudnn'
        real = torch.tensor([[0, 30, 60], [0, 30, 60]], dtype=torch.int32, device='cuda')
        physical = torch.tensor([[0, 32, 64], [0, 32, 64]], dtype=torch.int32, device='cuda')
        tokens = torch.arange(128, device='cuda').view(2, 64)
        sample = dict(
            tokens=tokens,
            labels=tokens.clone(),
            loss_mask=torch.ones_like(tokens).float(),
            position_ids=torch.arange(64, device='cuda').repeat(2, 1),
            cu_seqlens=real,
            cu_seqlens_padded=physical,
            max_seqlen=torch.tensor([32, 32], device='cuda', dtype=torch.int32),
        )
        batch = get_batch(iter([sample]))
        assert batch[1].tolist() == [[0, 30, 60, 90, 120]]
        assert batch[2].tolist() == [[0, 32, 64, 96, 128]]
        assert batch[7].tolist() == [32]
        local_count = 128 // cp
        if parallel_state.is_pipeline_first_stage():
            assert batch[9].shape == (1, local_count)
            expected = tokens.flatten().view(1, -1)
            if dsv4:
                rank = parallel_state.get_context_parallel_rank()
                torch.testing.assert_close(
                    batch[9], expected[:, rank * local_count : (rank + 1) * local_count]
                )
        if parallel_state.is_pipeline_last_stage():
            assert batch[4].shape == (1, local_count)
    finally:
        Utils.destroy_model_parallel()
