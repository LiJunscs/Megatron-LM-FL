# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Actual PP2 forward/backward with flattened packed DSv4 microbatches."""

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_transformer_block_with_experimental_attention_variant_spec,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.experimental_attention_variant.test_dsv4_hybrid_attention import (
    _make_config,
)


@pytest.mark.experimental
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_packed_dsv4_pp2_real_forward_backward():
    if Utils.world_size < 2 or Utils.world_size % 2:
        pytest.skip("requires a multiple of two ranks")
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=2)
    try:
        model_parallel_cuda_manual_seed(240)
        pg = ProcessGroupCollection.use_mpu_process_groups()
        config = _make_config(
            num_layers=2,
            csa_compress_ratios=[0, 4],
            hidden_size=256,
            num_attention_heads=64,
            v_head_dim=512,
            qk_pos_emb_head_dim=64,
            q_lora_rank=128,
            csa_window_size=16,
            dsa_indexer_n_heads=64,
            dsa_indexer_head_dim=128,
            dsa_indexer_topk=128,
            dsa_indexer_loss_coeff=0.2,
            dsa_indexer_use_sparse_loss=True,
            dsa_kernel_backend="cudnn",
            qk_layernorm=True,
            apply_rope_fusion=True,
            pipeline_model_parallel_size=2,
            pipeline_dtype=torch.bfloat16,
            variable_seq_lengths=True,
            gradient_accumulation_fusion=False,
            hidden_dropout=0.0,
            attention_dropout=0.0,
        )
        model = (
            GPTModel(
                config=config,
                transformer_layer_spec=get_transformer_block_with_experimental_attention_variant_spec(
                    config
                ),
                vocab_size=128,
                max_sequence_length=128,
                pre_process=parallel_state.is_pipeline_first_stage(),
                post_process=parallel_state.is_pipeline_last_stage(),
                pg_collection=pg,
                position_embedding_type="rope",
                share_embeddings_and_output_weights=False,
            )
            .cuda()
            .to(torch.bfloat16)
        )
        cu = torch.tensor([0, 32, 64, 96, 128], device="cuda", dtype=torch.int32)
        real = torch.tensor([0, 31, 62, 93, 124], device="cuda", dtype=torch.int32)
        packed = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=real,
            cu_seqlens_kv=real,
            cu_seqlens_q_padded=cu,
            cu_seqlens_kv_padded=cu,
            max_seqlen_q=32,
            max_seqlen_kv=32,
        )
        positions = torch.arange(32, device="cuda").repeat(4).view(1, -1)
        ids = (torch.arange(128, device="cuda") % 127).view(1, -1)
        mask = positions.eq(31)

        def loss_func(output):
            loss = output.float().masked_fill(mask, 0).sum()
            count = (~mask).sum()
            return loss, count, {"lm loss": torch.stack((loss.detach(), count.float()))}

        def forward_step(iterator, module):
            output = module(
                ids if module.pre_process else None,
                positions,
                None,
                labels=ids if module.post_process else None,
                packed_seq_params=packed,
                padding_mask=mask,
            )
            return output, loss_func

        losses = get_forward_backward_func()(
            forward_step_func=forward_step,
            data_iterator=None,
            model=[model],
            num_microbatches=2,
            seq_length=64,
            micro_batch_size=2,
            forward_only=False,
        )
        gradients = [p.grad for p in model.parameters() if p.requires_grad]
        assert gradients and all(g is not None and torch.isfinite(g).all() for g in gradients)
        assert any(torch.count_nonzero(g) for g in gradients)
        if model.post_process:
            assert len(losses) == 2
            assert all(torch.isfinite(item['lm loss']).all() for item in losses)
    finally:
        Utils.destroy_model_parallel()
