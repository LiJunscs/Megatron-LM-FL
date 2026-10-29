# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Exercise synthetic DSv4 packed CP training and a local checkpoint round trip.

Run from the repository root with PYTHONPATH=. and torchrun --nproc-per-node=8.
Each CP group trains an identical tiny model independently. This checks ordinary
GPT scheduling, cuDNN attention, SGD updates and state_dict restore; it does not
exercise the business dataset, distributed optimizer or distributed checkpoint.
"""

import argparse
import io
import json
import os
from copy import deepcopy
from datetime import timedelta

import torch
import torch.distributed as dist

from megatron.core import config as core_config
from megatron.core import parallel_state
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_transformer_block_with_experimental_attention_variant_spec,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import MLATransformerConfig


def make_batch(step: int, cp_group: dist.ProcessGroup) -> tuple:
    """Make changing document boundaries, including an empty document and padding."""
    physical, real = (
        ((0, 133, 133, 389, 512), (0, 129, 129, 381, 500)),
        ((0, 256, 512), (0, 251, 502)),
        ((0, 128, 384, 512), (0, 124, 376, 500)),
    )[step % 3]
    cu = torch.tensor(physical, device="cuda", dtype=torch.int32)
    real_cu = torch.tensor(real, device="cuda", dtype=torch.int32)
    max_length = max(end - start for start, end in zip(physical, physical[1:]))
    packed = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=real_cu,
        cu_seqlens_kv=real_cu,
        cu_seqlens_q_padded=cu,
        cu_seqlens_kv_padded=cu,
        max_seqlen_q=max_length,
        max_seqlen_kv=max_length,
        total_tokens=512,
        tokens_per_sample=512,
    )
    tokens = (torch.arange(512, device="cuda") % 127).view(1, -1)
    positions = torch.cat(
        [torch.arange(end - start, device="cuda") for start, end in zip(physical, physical[1:])]
    ).view(1, -1)
    padding = torch.zeros_like(tokens, dtype=torch.bool)
    for i, (start, end) in enumerate(zip(physical, physical[1:])):
        padding[:, start + real[i + 1] - real[i] : end] = True
    assert padding.sum().item() == 512 - real[-1]
    local_rows = 512 // cp_group.size()
    rows = slice(cp_group.rank() * local_rows, (cp_group.rank() + 1) * local_rows)
    return packed, tokens[:, rows], positions[:, rows], padding[:, rows]


def run_smoke(cp_size: int, enable_hyper_connections: bool) -> None:
    """Train three steps and replay the last step after restoring a local checkpoint."""
    parallel_state.initialize_model_parallel(context_parallel_size=cp_size)
    pg = ProcessGroupCollection.use_mpu_process_groups()
    torch.manual_seed(440)
    model_parallel_cuda_manual_seed(440)
    config = MLATransformerConfig(
        num_layers=3,
        hidden_size=256,
        num_attention_heads=64,
        use_cpu_initialization=True,
        bf16=True,
        params_dtype=torch.bfloat16,
        add_bias_linear=False,
        multi_latent_attention=True,
        experimental_attention_variant="dsv4_hybrid",
        q_lora_rank=128,
        kv_lora_rank=448,
        qk_head_dim=448,
        qk_pos_emb_head_dim=64,
        v_head_dim=512,
        o_groups=8,
        o_lora_rank=64,
        rope_type="rope",
        rotary_base=10000,
        rotary_percent=1.0,
        csa_compress_ratios=[0, 4, 128],
        csa_window_size=128,
        dsa_indexer_n_heads=64,
        dsa_indexer_head_dim=128,
        dsa_indexer_topk=512,
        dsa_indexer_loss_coeff=0.2,
        dsa_indexer_use_sparse_loss=True,
        dsa_kernel_backend="cudnn",
        qk_layernorm=True,
        apply_rope_fusion=True,
        context_parallel_size=cp_size,
        attention_cp_layout="contiguous",
        gradient_accumulation_fusion=False,
        calculate_per_token_loss=True,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        enable_hyper_connections=enable_hyper_connections,
    )
    model = (
        GPTModel(
            config=config,
            transformer_layer_spec=get_transformer_block_with_experimental_attention_variant_spec(
                config
            ),
            vocab_size=128,
            max_sequence_length=512,
            pg_collection=pg,
            position_embedding_type="rope",
            share_embeddings_and_output_weights=False,
        )
        .cuda()
        .to(torch.bfloat16)
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    def train_step(step):
        packed, tokens, positions, padding = make_batch(step, pg.cp)
        if dist.get_rank() == 0:
            print(f"Starting packed CP step {step + 1} (CP={cp_size})", flush=True)

        def forward_step(iterator, module):
            output = module(
                tokens,
                positions,
                None,
                labels=tokens,
                packed_seq_params=packed,
                padding_mask=padding,
            )

            def loss_func(output):
                loss = output.float().masked_fill(padding, 0).sum()
                count = (~padding).sum()
                return loss, count, {"lm loss": torch.stack((loss.detach(), count.float()))}

            return output, loss_func

        optimizer.zero_grad(set_to_none=True)
        losses = get_forward_backward_func()(
            forward_step_func=forward_step,
            data_iterator=None,
            model=[model],
            num_microbatches=2,
            seq_length=512,
            micro_batch_size=1,
            forward_only=False,
        )
        gradients = [p.grad for p in model.parameters() if p.requires_grad]
        assert gradients and all(g is not None and torch.isfinite(g).all() for g in gradients)
        assert any(torch.count_nonzero(g) for g in gradients)
        # This synthetic runner has no DDP/finalize_model_grads hook. Normalize
        # the summed gradients by real tokens across both microbatches and CP.
        token_count = (~padding).sum() * 2
        dist.all_reduce(token_count, group=pg.cp)
        for gradient in gradients:
            dist.all_reduce(gradient, group=pg.cp)
            gradient.div_(token_count.clamp_min(1))
        optimizer.step()
        assert all(torch.isfinite(p).all() for p in model.parameters())
        assert losses and all(torch.isfinite(item["lm loss"]).all() for item in losses)
        parameter = next(model.parameters()).detach()
        replica = parameter.clone()
        dist.broadcast(replica, src=dist.get_global_rank(pg.cp, 0), group=pg.cp)
        torch.testing.assert_close(parameter, replica, rtol=0, atol=0)
        metrics = torch.stack([item["lm loss"] for item in losses]).sum(dim=0)
        dist.all_reduce(metrics, group=pg.cp)
        if dist.get_rank() == 0:
            print(f"Step {step + 1}: loss={(metrics[0] / metrics[1]).item():.6f}", flush=True)
        return metrics[0] / metrics[1]

    before = next(model.parameters()).detach().clone()
    train_step(0)
    train_step(1)
    assert not torch.equal(before, next(model.parameters()))
    checkpoint_buffer = io.BytesIO()
    torch.save(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict()}, checkpoint_buffer
    )
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
    expected_loss = train_step(2)
    expected = deepcopy(model.state_dict())
    checkpoint_buffer.seek(0)
    checkpoint = torch.load(checkpoint_buffer, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    for name, value in model.state_dict().items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, checkpoint["model"][name], rtol=0, atol=0)
    torch.set_rng_state(cpu_rng)
    torch.cuda.set_rng_state(cuda_rng)
    replay_loss = train_step(2)
    torch.testing.assert_close(replay_loss, expected_loss, rtol=1e-5, atol=1e-5)
    exact = True
    max_abs_difference = 0.0
    for name, value in model.state_dict().items():
        if isinstance(value, torch.Tensor):
            # cuDNN packed backward does not guarantee bitwise determinism.
            # Restore itself is exact above; compare the subsequent update at
            # PyTorch's standard dtype tolerances (including BF16 rounding).
            torch.testing.assert_close(value, expected[name], msg=lambda msg: f"{name}: {msg}")
            exact = exact and torch.equal(value, expected[name])
            if value.numel():
                max_abs_difference = max(
                    max_abs_difference, (value.float() - expected[name].float()).abs().max().item()
                )
        elif isinstance(value, io.BytesIO):
            assert value.getvalue() == expected[name].getvalue(), name
        else:
            assert value == expected[name], name
    print(
        "PACKED_CP_SMOKE "
        + json.dumps(
            dict(
                rank=dist.get_rank(),
                cp_size=cp_size,
                mhc=enable_hyper_connections,
                steps=3,
                resume_next_step_close=True,
                resume_parameters_exact=exact,
                resume_max_abs_difference=max_abs_difference,
                finite_gradients=True,
            )
        ),
        flush=True,
    )


def main() -> None:
    """Initialize the torchrun world and run one CP topology."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cp-size", type=int, choices=(2, 4), default=2)
    parser.add_argument("--enable-hyper-connections", action="store_true")
    args = parser.parse_args()
    core_config.ENABLE_EXPERIMENTAL = True
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(minutes=10))
    try:
        if dist.get_world_size() % args.cp_size:
            raise ValueError("WORLD_SIZE must be divisible by --cp-size")
        run_smoke(args.cp_size, args.enable_hyper_connections)
        dist.barrier()
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
