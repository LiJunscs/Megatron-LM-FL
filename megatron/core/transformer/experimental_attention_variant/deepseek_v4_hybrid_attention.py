# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import os
from dataclasses import dataclass
from typing import NoReturn, Optional, Union

import torch

from megatron.core import tensor_parallel
from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.tensor_parallel.layers import set_tensor_model_parallel_attributes
from megatron.core.models.common.embeddings import (
    RotaryEmbedding,
    YarnRotaryEmbedding,
    apply_rotary_pos_emb,
)
from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
    FineGrainedActivationOffloadingInterface as off_interface,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.attention import Attention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.torch_norm import LayerNormBuilder
from megatron.core.transformer.transformer_config import MLATransformerConfig
from megatron.core.transformer.utils import ensure_metadata_has_dp_cp_group
from megatron.core.typed_torch import apply_module
from megatron.core.utils import (
    deprecated,
    get_pg_size,
    is_te_min_version,
    make_tp_sharded_tensor_for_checkpoint,
)

try:
    from megatron.core.fusions.fused_mla_yarn_rope_apply import (
        _FusedMLARoPEInplace,
        fused_mla_rope_inplace,
    )
except Exception:
    _FusedMLARoPEInplace = None
    fused_mla_rope_inplace = None


if HAVE_TE:
    from megatron.core.extensions.transformer_engine import TELinear, set_save_original_input
else:
    (TEColumnParallelLinear, TELinear, set_save_original_input) = (None, None, None)


@torch.compile
def _q_rms_norm(q: torch.Tensor, eps: float) -> torch.Tensor:
    """Fused RMS normalization for query tensor (no learnable weight)."""
    return q * torch.rsqrt(q.square().mean(-1, keepdim=True) + eps)


class _DSv4TPRopeGather(torch.autograd.Function):
    """Overlap Q RoPE with a sequence-parallel all-gather."""

    @staticmethod
    def forward(
        ctx,
        q,
        local_tensor,
        cos,
        sin,
        nope_dim,
        emb_dim,
        cu_seqlens_q,
        cp_rank,
        cp_size,
        tp_group,
        async_communication,
        remove_interleaving,
    ):
        world_size = tp_group.size()
        gathered = local_tensor.new_empty(
            local_tensor.size(0) * world_size, *local_tensor.shape[1:]
        )
        work = torch.distributed.all_gather_into_tensor(
            gathered,
            local_tensor.contiguous(),
            group=tp_group,
            async_op=async_communication,
        )

        # Reuse the fused RoPE autograd implementation and its saved context.
        query = _FusedMLARoPEInplace.forward(
            ctx,
            q,
            cos,
            sin,
            nope_dim,
            emb_dim,
            cu_seqlens_q,
            cp_rank,
            cp_size,
            False,
            False,
            remove_interleaving,
        )
        if work is not None:
            work.wait()
        ctx.tp_group = tp_group
        ctx.async_communication = async_communication
        return query, gathered

    @staticmethod
    def backward(ctx, grad_query, grad_gathered):
        local_grad = grad_gathered.new_empty(
            grad_gathered.size(0) // ctx.tp_group.size(), *grad_gathered.shape[1:]
        )
        work = torch.distributed.reduce_scatter_tensor(
            local_grad,
            grad_gathered.contiguous(),
            group=ctx.tp_group,
            async_op=ctx.async_communication,
        )
        grad_q = _FusedMLARoPEInplace.backward(ctx, grad_query)[0]
        if work is not None:
            work.wait()
        return grad_q, local_grad, None, None, None, None, None, None, None, None, None, None


def _dsv4_tp_rope_gather(
    q,
    local_tensor,
    cos,
    sin,
    nope_dim,
    emb_dim,
    cu_seqlens_q,
    cp_rank,
    cp_size,
    tp_group,
    async_communication=True,
    remove_interleaving=True,
):
    if not async_communication:
        return _deprecated_dsv4_tp_rope_gather_non_overlap(
            q,
            local_tensor,
            cos,
            sin,
            nope_dim,
            emb_dim,
            cu_seqlens_q,
            cp_rank,
            cp_size,
            tp_group,
            remove_interleaving,
        )
    return _DSv4TPRopeGather.apply(
        q,
        local_tensor,
        cos,
        sin,
        nope_dim,
        emb_dim,
        cu_seqlens_q,
        cp_rank,
        cp_size,
        tp_group,
        async_communication,
        remove_interleaving,
    )


@deprecated(
    version="0.0.0",
    alternative="_dsv4_tp_rope_gather",
    reason="Temporary non-overlap performance baseline.",
)
def _deprecated_dsv4_tp_rope_gather_non_overlap(
    q,
    local_tensor,
    cos,
    sin,
    nope_dim,
    emb_dim,
    cu_seqlens_q,
    cp_rank,
    cp_size,
    tp_group,
    remove_interleaving=True,
):
    return _DSv4TPRopeGather.apply(
        q,
        local_tensor,
        cos,
        sin,
        nope_dim,
        emb_dim,
        cu_seqlens_q,
        cp_rank,
        cp_size,
        tp_group,
        False,
        remove_interleaving,
    )


@dataclass
class DSv4HybridSelfAttentionSubmodules:
    """Submodules for the DSv4HybridAttention layer."""

    q_layernorm: LayerNormBuilder
    kv_layernorm: LayerNormBuilder

    linear_q_down_proj: Union[ModuleSpec, type] = None
    linear_q_up_proj: Union[ModuleSpec, type] = None
    linear_kv_proj: Union[ModuleSpec, type] = None
    core_attention: Union[ModuleSpec, type] = None
    linear_proj: Union[ModuleSpec, type] = None


class DSv4HybridAttention(Attention):
    """DeepSeek-v4 Hybrid Attention layer."""

    def __init__(
        self,
        config: MLATransformerConfig,
        submodules: DSv4HybridSelfAttentionSubmodules,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        cp_comm_type: Optional[str] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        pp_layer_offset: Optional[int] = None,
        is_mtp_layer: bool = False,
    ) -> None:

        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attention_type=attention_type,
            attn_mask_type=attn_mask_type,
            pg_collection=pg_collection,
            pp_layer_offset=pp_layer_offset,
            is_mtp_layer=is_mtp_layer,
        )
        self.config: MLATransformerConfig

        tp_size = get_pg_size(self.pg_collection.tp)
        assert self.config.num_attention_heads % tp_size == 0, (
            f"num_attention_heads ({self.config.num_attention_heads}) must be divisible by "
            f"tensor parallel size ({tp_size})"
        )
        # DSv4 uses a single replicated MQA KV head.  The generic Attention
        # value has GQA-specific semantics when num_query_groups < TP size, so
        # it cannot describe the column-parallel Q projection on this path.
        self.num_local_q_heads = self.config.num_attention_heads // tp_size
        assert tp_size == 1 or self.config.apply_rope_fusion, (
            "DSv4 Hybrid TP requires apply_rope_fusion=True"
        )

        assert (
            not self.checkpoint_core_attention
        ), "Checkpoint core attention is not supported in DSv4 Hybrid Attention."
        assert (
            not self.offload_qkv_linear
        ), "Offload qkv linear is not supported in DSv4 Hybrid Attention."

        # ColumnParallelLinear constructors take global dimensions and perform
        # the TP division internally.  Keep Megatron's standard global meaning
        # for query_projection_size and track the local width separately.
        self.query_projection_size = self.config.v_head_dim * self.config.num_attention_heads
        self.query_projection_size_per_partition = (
            self.config.v_head_dim * self.num_local_q_heads
        )

        self.q_head_dim = self.config.v_head_dim

        self.key_hidden_size = self.q_head_dim
        self.val_hidden_size = self.config.v_head_dim

        self.recompute_up_proj = (
            self.config.recompute_granularity == 'selective'
            and "mla_up_proj" in self.config.recompute_modules
        )
        self.qkv_up_checkpoint = None

        self.softmax_scale = None

        if is_mtp_layer:
            layer_idx = self.config.num_layers + layer_number - 1
            compress_ratio = self.config.csa_compress_ratios[layer_idx]
        else:
            compress_ratio = self.config.csa_compress_ratios[layer_number - 1]
        use_compressed_yarn = compress_ratio > 1
        rope_base = (
            self.config.csa_compress_rotary_base if use_compressed_yarn else self.config.rotary_base
        )
        self._dsv4_compress_ratio = compress_ratio
        self._dsv4_rope_base = rope_base
        self._dsv4_uses_yarn_rope = use_compressed_yarn
        if not use_compressed_yarn:
            self.rotary_pos_emb = RotaryEmbedding(
                self.config.qk_pos_emb_head_dim,
                rotary_percent=self.config.rotary_percent,
                rotary_base=rope_base,
                cp_group=self.pg_collection.cp,
            )
        else:
            self.rotary_pos_emb = YarnRotaryEmbedding(
                self.config.qk_pos_emb_head_dim,
                rotary_base=rope_base,
                scaling_factor=self.config.rotary_scaling_factor,
                original_max_position_embeddings=self.config.original_max_position_embeddings,
                beta_fast=self.config.beta_fast,
                beta_slow=self.config.beta_slow,
                mscale=self.config.mscale,
                mscale_all_dim=self.config.mscale_all_dim,
                cp_group=self.pg_collection.cp,
            )

        core_attn_extra_kwargs = {
            "rotary_pos_emb": self.rotary_pos_emb,
            "compress_ratio": compress_ratio,
            "is_mtp_layer": is_mtp_layer,
        }
        self.core_attention = build_module(
            submodules.core_attention,
            config=self.config,
            layer_number=self.layer_number,
            attn_mask_type=self.attn_mask_type,
            attention_type=self.attention_type,
            softmax_scale=self.softmax_scale,
            k_channels=self.q_head_dim,
            v_channels=self.config.v_head_dim,
            cp_comm_type=cp_comm_type,
            pg_collection=self.pg_collection,
            **core_attn_extra_kwargs,
        )

        # Output.
        assert self.config.o_groups % tp_size == 0, (
            f"o_groups ({self.config.o_groups}) must be divisible by tensor parallel "
            f"size ({tp_size})"
        )
        self.o_local_groups = self.config.o_groups // tp_size
        assert self.query_projection_size_per_partition % self.o_local_groups == 0, (
            "local_num_attention_heads * v_head_dim must be divisible by local o_groups"
        )
        group_proj_in_size = self.query_projection_size_per_partition // self.o_local_groups
        group_proj_out_size = self.o_local_groups * self.config.o_lora_rank

        _linear_o_group_proj = torch.empty(
            group_proj_out_size,
            group_proj_in_size,
            device=torch.cuda.current_device(),
            dtype=self.config.params_dtype,
        )
        self.config.init_method(_linear_o_group_proj)
        self.linear_o_group_proj = torch.nn.Parameter(_linear_o_group_proj)
        set_tensor_model_parallel_attributes(
            self.linear_o_group_proj, is_parallel=True, dim=0, stride=1
        )

        linear_proj_in_size = self.config.o_groups * self.config.o_lora_rank

        self.linear_proj = build_module(
            submodules.linear_proj,
            linear_proj_in_size,
            self.config.hidden_size,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=self.config.add_bias_linear,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name='proj',
            tp_group=self.pg_collection.tp,
        )

        if (
            HAVE_TE
            and isinstance(self.linear_proj, TELinear)
            and (
                (
                    self.config.fp8
                    and self.config.fp8_recipe != 'delayed'
                    and is_te_min_version("2.6.0dev0")
                )
                or (self.config.fp4 and is_te_min_version("2.7.0.dev0"))
            )
        ):
            # For fp8/fp4 training, the output of the fused core_attn is saved by itself, and
            # linear_proj also saves the quantized tensor of this output. Here we set the
            # linear_proj to save the original input tensors to avoid the extra memory usage of
            # the quantized tensor.
            set_save_original_input(self.linear_proj)

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Shard the grouped output projection along its group/output axis."""
        metadata = ensure_metadata_has_dp_cp_group(metadata)
        sharded_state_dict = super().sharded_state_dict(
            prefix=prefix, sharded_offsets=sharded_offsets, metadata=metadata
        )
        weight_key = f"{prefix}linear_o_group_proj"
        sharded_state_dict[weight_key] = make_tp_sharded_tensor_for_checkpoint(
            self.linear_o_group_proj,
            weight_key,
            tp_axis=0,
            prepend_offsets=sharded_offsets,
            tp_group=self.pg_collection.tp,
            dp_cp_group=metadata["dp_cp_group"],
        )
        return sharded_state_dict

    def forward(
        self,
        hidden_states,
        attention_mask,
        key_value_states=None,
        inference_context=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        rotary_pos_cos_sin=None,
        attention_bias=None,
        packed_seq_params=None,
        position_ids=None,
        sequence_len_offset=None,
        *,
        inference_params=None,
    ):
        """Forward pass for DeepSeek-v4 Hybrid Attention"""
        assert (
            rotary_pos_emb is None
        ), "Rotary position embeddings should not be passed into DSv4HybridAttention."
        assert (
            attention_bias is None
        ), "Attention bias should not be passed into DSv4HybridAttention."
        assert (
            rotary_pos_cos is None and rotary_pos_sin is None
        ), "DSv4HybridAttention does not support Flash Decoding"
        assert (
            not rotary_pos_cos_sin
        ), "Flash-infer rope has not been tested with DSv4HybridAttention."
        assert (
            inference_context is None and inference_params is None
        ), "Inference is not supported for DSv4HybridAttention."

        # =====================
        # Query, Key, and Value
        # =====================
        # Get the query, key and value tensors based on the type of attention -
        # self or cross attn.
        query, key, value, q_compressed, gathered_hidden_states = self.get_query_key_value_tensors(
            hidden_states,
            key_value_states,
            position_ids,
            packed_seq_params,
            inference_context=inference_context,
        )

        # TODO: Currently, TE can only accept contiguous tensors for MLA
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()

        # ==================================
        # core attention computation
        # ==================================
        # Need corresponding TE change
        core_attn_manager = off_interface(
            self.offload_core_attention and self.training, query, "core_attn"
        )
        with core_attn_manager as query:
            core_attn_out = self.core_attention(
                query,
                key,
                value,
                attention_mask,
                packed_seq_params=packed_seq_params,
                # CSA consumes the full sequence, unlike the SP input to this layer.
                x=gathered_hidden_states,
                qr=q_compressed,
            )
        # NOTE: No implement in megatron version core_v0.17.0, group_commit(v0.17.0) -> group_offload(latest)
        if self.offload_core_attention and self.training:
            core_attn_out = core_attn_manager.group_commit(
                core_attn_out, name="core_attn", forced_released_tensors=[query, key, value]
            )

        if packed_seq_params is not None and packed_seq_params.qkv_format == 'thd':
            # reshape to same output shape as unpacked case
            # (t, np, hn) -> (t, b=1, h=np*hn)
            # t is the pack size = sum (sq_i)
            # note that batch is a dummy dimension in the packed case
            core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)

        if self.recompute_up_proj:
            assert self.qkv_up_checkpoint is not None
            self.qkv_up_checkpoint.discard_output_and_register_recompute(core_attn_out)
            self.qkv_up_checkpoint = None

        # inverse RoPE on last qk_pos_emb_head_dim of each head
        seq_len = core_attn_out.size(0)
        n_heads = self.num_local_q_heads
        pos_dim = self.config.qk_pos_emb_head_dim
        nope_dim = self.config.v_head_dim - pos_dim
        core_attn_out = core_attn_out.view(seq_len, core_attn_out.size(1), n_heads, -1)
        packed_seq = packed_seq_params is not None and packed_seq_params.qkv_format == 'thd'
        if packed_seq:
            cu_seqlens_kv = (
                packed_seq_params.cu_seqlens_kv_padded
                if packed_seq_params.cu_seqlens_kv_padded is not None
                else packed_seq_params.cu_seqlens_kv
            )
            rope_seqlen = cu_seqlens_kv
        else:
            cu_seqlens_kv = None
            rope_seqlen = seq_len
        # DSv4 reference (DS-Inf) RoPE is pure rotation (norm-preserving). Yarn's
        # concentration factor (mscale) is NOT part of the DSv4 model contract --
        # the model relies on Q/KV RMS-norm + unit-magnitude rotation. Force 1.0.
        mscale = 1.0
        rotary_pos_cos = None
        rotary_pos_sin = None
        if self.config.apply_rope_fusion:
            # ``mscale=1.0`` strips yarn's concentration factor from the
            # cached cos/sin so the fused kernel matches the unfused
            # path's forced ``mscale=1.0`` (DSv4 "pure rotation").
            rotary_pos_cos, rotary_pos_sin = self.rotary_pos_emb.get_cached_cos_sin(
                rope_seqlen, dtype=hidden_states.dtype, packed_seq=packed_seq, mscale=mscale
            )
            rotary_pos_emb = None
            assert inference_context is None, "Inference with MLA RoPE fusion is not supported"
            assert (
                fused_mla_rope_inplace is not None
            ), "Fused MLA RoPE apply is not imported successfully"
        elif self._dsv4_uses_yarn_rope:
            rotary_pos_emb, _ = self.rotary_pos_emb(rope_seqlen, packed_seq=packed_seq)
        else:
            rotary_pos_emb = self.rotary_pos_emb(rope_seqlen, packed_seq=packed_seq)
        if self.config.apply_rope_fusion:
            core_attn_out = fused_mla_rope_inplace(
                core_attn_out,
                rotary_pos_cos,
                rotary_pos_sin,
                nope_dim,
                pos_dim,
                cu_seqlens_kv,
                self.pg_collection.cp.rank(),
                self.pg_collection.cp.size(),
                inverse=True,
                remove_interleaving=True,
            )
        else:
            content_part, rot_part = torch.split(
                core_attn_out, [core_attn_out.size(-1) - pos_dim, pos_dim], dim=-1
            )
            rot_part = apply_rotary_pos_emb(
                rot_part,
                rotary_pos_emb,
                self.config,
                cu_seqlens=cu_seqlens_kv,
                mscale=mscale,
                cp_group=self.pg_collection.cp,
                mla_rotary_interleaved=True,
                inverse=True,
                mla_output_remove_interleaving=True,
            )
            core_attn_out = torch.cat([content_part, rot_part], dim=-1)
        core_attn_out = core_attn_out.view(seq_len, core_attn_out.size(1), -1)

        # Grouped output
        core_attn_out = core_attn_out.view(
            core_attn_out.size(0), core_attn_out.size(1), self.o_local_groups, -1
        )
        wo_a_weight = self.linear_o_group_proj.view(
            self.o_local_groups, self.config.o_lora_rank, -1
        )
        core_attn_out = torch.einsum("...gd,grd->...gr", core_attn_out, wo_a_weight)
        core_attn_out = core_attn_out.reshape(*core_attn_out.shape[:-2], -1)

        # =================
        # Output. [sq, b, h]
        # =================
        attn_proj_manager = off_interface(self.offload_attn_proj, core_attn_out, "attn_proj")
        with attn_proj_manager as core_attn_out:
            output, bias = self.linear_proj(core_attn_out)
        # NOTE: No implement in megatron version core_v0.17.0, group_commit(v0.17.0) -> group_offload(latest)
        if self.offload_attn_proj:
            output = attn_proj_manager.group_commit(output, name="attn_proj", forced_released_tensors=[core_attn_out])

        return output, bias


class DSv4HybridSelfAttention(DSv4HybridAttention):
    """DSv4Hybrid Self-attention layer class

    Self-attention layer takes input with size [s, b, h]
    and returns output of the same size.
    """

    def __init__(
        self,
        config: MLATransformerConfig,
        submodules: DSv4HybridSelfAttentionSubmodules,
        layer_number: int,
        attn_mask_type=AttnMaskType.padding,
        cp_comm_type: Optional[str] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        pp_layer_offset: Optional[int] = None,
        is_mtp_layer: bool = False,
    ):
        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups()

        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            attention_type="self",
            cp_comm_type=cp_comm_type,
            pg_collection=pg_collection,
            pp_layer_offset=pp_layer_offset,
            is_mtp_layer=is_mtp_layer,
        )

        q_down_proj_kwargs = {}
        if submodules.linear_q_down_proj in [TELinear]:
            q_down_proj_kwargs['parallel_mode'] = 'duplicated'
        else:
            raise ValueError(f"Unsupported linear_q_down_proj: {submodules.linear_q_down_proj}")

        self.linear_q_down_proj = build_module(
            submodules.linear_q_down_proj,
            self.config.hidden_size,
            self.config.q_lora_rank,
            config=self.config,
            init_method=self.config.init_method,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name='q_down_proj',
            skip_weight_param_allocation=False,
            tp_group=None,
            **q_down_proj_kwargs,
        )

        self.linear_q_up_proj = build_module(
            submodules.linear_q_up_proj,
            self.config.q_lora_rank,
            # ColumnParallelLinear takes the global output width and returns
            # query_projection_size (= local Q heads * head dim) on each rank.
            self.query_projection_size,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name='q_up_proj',
            skip_weight_param_allocation=False,
            tp_group=pg_collection.tp,
        )

        kv_proj_kwargs = {}
        if submodules.linear_kv_proj in [TELinear]:
            # The single MQA KV head is intentionally replicated.  Sharding
            # v_head_dim would leave RoPE and CSA with only a partial head.
            kv_proj_kwargs['parallel_mode'] = 'duplicated'
        else:
            raise ValueError(f"Unsupported linear_kv_proj: {submodules.linear_kv_proj}")

        self.linear_kv_proj = build_module(
            submodules.linear_kv_proj,
            self.config.hidden_size,
            self.config.v_head_dim,
            config=self.config,
            init_method=self.config.init_method,
            skip_weight_param_allocation=False,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name='kv_up_proj',
            tp_group=None,
            **kv_proj_kwargs,
        )
        self.kv_layernorm = submodules.kv_layernorm(
            hidden_size=self.config.v_head_dim,
            config=self.config,
            eps=self.config.layernorm_epsilon,
        )

        self.q_layernorm = submodules.q_layernorm(
            hidden_size=self.config.q_lora_rank,
            config=self.config,
            eps=self.config.layernorm_epsilon,
        )

    def get_query_key_value_tensors(
        self,
        hidden_states,
        key_value_states=None,
        position_ids=None,
        packed_seq_params=None,
        inference_context=None,
        *,
        inference_params=None,
    ):
        """
        Derives `query`, `key` and `value` tensors from `hidden_states`.
        """
        # s = sequence length, b = batch size, h = hidden size, n = num attention heads
        # Attention heads [s, b, n*h]
        assert (
            hidden_states.ndim == 3
        ), f"hidden_states should be 3D, [s, b, n*h], got {hidden_states.ndim}D"
        if packed_seq_params is not None:
            assert (
                packed_seq_params.local_cp_size is None
            ), "dynamic_context_parallel is not supported with MLA yet and is planned for future. \
            Please disable dynamic_context_parallel."

        assert (
            inference_context is None and inference_params is None
        ), "Inference is not supported for DSv4HybridSelfAttention."

        # =========================================
        # Prepare RoPE and seqlen related params
        # =========================================
        rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
            inference_context, None, hidden_states, self.config, packed_seq_params
        )

        # rotary_pos_emb:[s, b, 1, 64]
        # DSv4 reference (DS-Inf) RoPE is pure rotation (norm-preserving). Yarn's
        # concentration factor (mscale) is NOT part of the DSv4 model contract --
        # the model relies on Q/KV RMS-norm + unit-magnitude rotation. Force 1.0.
        mscale = 1.0
        rotary_pos_cos = None
        rotary_pos_sin = None
        packed_seq = packed_seq_params is not None and packed_seq_params.qkv_format == 'thd'
        if self.config.apply_rope_fusion:
            # ``mscale=1.0`` strips yarn's concentration factor from the
            # cached cos/sin so the fused kernel matches the unfused
            # path's forced ``mscale=1.0`` (DSv4 "pure rotation").
            rotary_pos_cos, rotary_pos_sin = self.rotary_pos_emb.get_cached_cos_sin(
                rotary_seq_len, dtype=hidden_states.dtype, packed_seq=packed_seq, mscale=mscale
            )
            rotary_pos_emb = None
            assert inference_context is None, "Inference with MLA RoPE fusion is not supported"
            assert (
                fused_mla_rope_inplace is not None
            ), "Fused MLA RoPE apply is not imported successfully"
        elif self._dsv4_uses_yarn_rope:
            rotary_pos_emb, _ = self.rotary_pos_emb(rotary_seq_len, packed_seq=packed_seq)
        else:
            rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len, packed_seq=packed_seq)

        if packed_seq_params is not None and packed_seq_params.qkv_format == 'thd':
            if packed_seq_params.cu_seqlens_q_padded is not None:
                cu_seqlens_q = packed_seq_params.cu_seqlens_q_padded
            else:
                cu_seqlens_q = packed_seq_params.cu_seqlens_q
            if packed_seq_params.cu_seqlens_kv_padded is not None:
                cu_seqlens_kv = packed_seq_params.cu_seqlens_kv_padded
            else:
                cu_seqlens_kv = packed_seq_params.cu_seqlens_kv
        else:
            cu_seqlens_q = cu_seqlens_kv = None

        # =========================================
        # QKV down projection and layernorm
        # =========================================
        # q_compressed: [s, b, q_lora_rank]
        q_compressed, _ = self.linear_q_down_proj(hidden_states)

        k_pos_emb = None

        if packed_seq_params is not None:
            # If sequence packing, TE expect [t, h, d] shaped qkv input.
            # In Megatron-Core, the qkv shape is [t, 1, h, d].
            # So we need to reshape qkv from [t, 1, h, d] to [t, h, d].
            q_compressed = q_compressed.squeeze(1)

        # =========================================
        # Apply norm
        # =========================================

        if self.config.q_lora_rank is not None:
            # q_compressed: [num_tokens, q_lora_rank]
            q_compressed = apply_module(self.q_layernorm)(q_compressed)

        # =========================================
        # QKV up projection and RoPE apply
        # =========================================

        def qkv_up_proj_and_rope_apply(q_compressed, hidden_states, k_pos_emb, rotary_pos_emb):
            """
            Apply the up projection and RoPE to the query and key.
            When sequence packing enabled, the input tensors adopt a packed shape of [t, ...];
            otherwise, they maintain the unpacked shape [s, b, ...]. In subsequent code comments,
            we uniformly use [num_tokens, ...] to denote [s, b, ...] or [t, ...] for two cases.
            """
            # q_compressed: [num_tokens, q_lora_rank]
            # q: [num_tokens, n * (qk_head_dim + qk_pos_emb_head_dim)]
            q, _ = self.linear_q_up_proj(q_compressed)

            # q: [num_tokens, n, q_head_dim]
            assert q.size(-1) == self.query_projection_size_per_partition, (
                f"local Q projection width ({q.size(-1)}) must equal "
                "num_local_q_heads * q_head_dim "
                f"({self.query_projection_size_per_partition})"
            )
            q = q.view(*q.size()[:-1], self.num_local_q_heads, self.q_head_dim)
            q = _q_rms_norm(q, self.config.layernorm_epsilon)

            kv, _ = self.linear_kv_proj(hidden_states)

            # [num_tokens, qk_pos_emb_head_dim] -> [num_tokens, 1, qk_pos_emb_head_dim]
            if k_pos_emb is not None:
                k_pos_emb = torch.unsqueeze(k_pos_emb, -2)

            if self.config.apply_rope_fusion:
                cp_rank = self.pg_collection.cp.rank()
                cp_size = self.pg_collection.cp.size()
                tp_sp = self.config.sequence_parallel and self.pg_collection.tp.size() > 1
                if tp_sp:
                    assert packed_seq_params is None, (
                        "Packed sequence is not supported by the deprecated DSv4 TP overlap path"
                    )
                    # KV projection output is consumed immediately after this gather.
                    # hidden_states and q_compressed are also gathered because CSA uses
                    # the full hidden sequence and its indexer uses the full compressed Q.
                    split_sizes = [hidden_states.size(-1), kv.size(-1), q_compressed.size(-1)]
                    local_tensor = torch.cat([hidden_states, kv, q_compressed], dim=-1)
                    async_communication = os.environ.get("DSV4_TP_ASYNC_COMM", "0") == "1"
                    query, gathered = _dsv4_tp_rope_gather(
                        q,
                        local_tensor,
                        rotary_pos_cos,
                        rotary_pos_sin,
                        self.config.qk_head_dim,
                        self.config.qk_pos_emb_head_dim,
                        cu_seqlens_q,
                        cp_rank,
                        cp_size,
                        self.pg_collection.tp,
                        async_communication=async_communication,
                    )
                    hidden_states, kv, q_compressed = torch.split(
                        gathered, split_sizes, dim=-1
                    )
                else:
                    query = fused_mla_rope_inplace(
                        q,
                        rotary_pos_cos,
                        rotary_pos_sin,
                        self.config.qk_head_dim,
                        self.config.qk_pos_emb_head_dim,
                        cu_seqlens_q,
                        cp_rank,
                        cp_size,
                        remove_interleaving=True,
                    )
                kv = self.kv_layernorm(kv)
                kv = kv.unsqueeze(-2)
                kv = fused_mla_rope_inplace(
                    kv,
                    rotary_pos_cos,
                    rotary_pos_sin,
                    self.config.qk_head_dim,
                    self.config.qk_pos_emb_head_dim,
                    cu_seqlens_q,
                    cp_rank,
                    cp_size,
                    remove_interleaving=True,
                )
                key = kv
                value = kv
            else:
                kv = self.kv_layernorm(kv)
                q_len = q.size()[0]
                if packed_seq_params is None or self.config.context_parallel_size == 1:
                    # Shorten rotary_pos_emb to the sequence length when inference_params
                    # is not provided. This makes sure we can run forward directly with
                    # any sequence length. During training, the sequence length is always
                    # the full rotary_pos_emb length, except for sequence packing + CP.
                    # When sequence packing and context parallel are both enabled, the
                    # position embedding will not split rotary_pos_emb, so it may exceed
                    # the sequence length on this CP rank, but we need the full rotary_pos_emb
                    # to cover the full sequence, so we do not shorten it here.
                    rotary_pos_emb = rotary_pos_emb[0:q_len]

                # q_no_pe: [num_tokens, n, qk_head_dim]
                # q_pos_emb: [num_tokens, n, qk_pos_emb_head_dim]
                q_no_pe, q_pos_emb = torch.split(
                    q, [self.config.qk_head_dim, self.config.qk_pos_emb_head_dim], dim=-1
                )

                # RoPE and query (shared for wkv and latent)
                # q_pos_emb: [num_tokens, n, qk_pos_emb_head_dim]
                q_pos_emb = apply_rotary_pos_emb(
                    q_pos_emb,
                    rotary_pos_emb,
                    config=self.config,
                    cu_seqlens=cu_seqlens_q,
                    mscale=mscale,
                    cp_group=self.pg_collection.cp,
                    mla_rotary_interleaved=True,
                    mla_output_remove_interleaving=True,
                )
                # query: [num_tokens, n, (qk_head_dim + v_head_dim)]
                query = torch.cat([q_no_pe, q_pos_emb], dim=-1)

                pos_dim = self.config.qk_pos_emb_head_dim
                kv_no_pe, k_pos_emb = torch.split(kv, [kv.size(-1) - pos_dim, pos_dim], dim=-1)

                # k_pos_emb:[num_tokens, 1, qk_pos_emb_head_dim]
                k_pos_emb = apply_rotary_pos_emb(
                    k_pos_emb,
                    rotary_pos_emb,
                    config=self.config,
                    cu_seqlens=cu_seqlens_kv,
                    mscale=mscale,
                    cp_group=self.pg_collection.cp,
                    mla_rotary_interleaved=True,
                    mla_output_remove_interleaving=True,
                )

                # Single head: key = value = [num_tokens, 1, v_head_dim]
                kv = torch.cat([kv_no_pe, k_pos_emb], dim=-1).unsqueeze(-2)
                key = kv
                value = kv

            query = query.contiguous()
            key = key.contiguous()
            value = value.contiguous()

            return query, key, value, q_compressed, hidden_states

        if self.recompute_up_proj:
            quantization = self.config.fp8 or self.config.fp4
            self.qkv_up_checkpoint = tensor_parallel.CheckpointWithoutOutput(fp8=quantization)
            query, key, value, q_compressed, hidden_states = self.qkv_up_checkpoint.checkpoint(
                qkv_up_proj_and_rope_apply, q_compressed, hidden_states, k_pos_emb, rotary_pos_emb
            )
        else:
            query, key, value, q_compressed, hidden_states = qkv_up_proj_and_rope_apply(
                q_compressed, hidden_states, k_pos_emb, rotary_pos_emb
            )

        assert q_compressed.size(0) == query.size(0), (
            f"q_compressed sequence length ({q_compressed.size(0)}) must match "
            f"query sequence length ({query.size(0)})"
        )
        return query, key, value, q_compressed, hidden_states

    def backward_dw(self) -> NoReturn:
        """Execute weight gradient computation"""
        self._backward_kv_proj()
        self._backward_q_proj()
        self._backward_output_proj()

    def _backward_kv_proj(self):
        """Computes weight gradients of KV projection layers"""
        self.linear_kv_proj.backward_dw()

    def _backward_q_proj(self):
        """Computes weight gradients of Q projection layers"""
        self.linear_q_down_proj.backward_dw()
        self.linear_q_up_proj.backward_dw()

    def _backward_output_proj(self):
        """Computes weight gradients of output projection layer"""
        self.linear_proj.backward_dw()

    def set_for_recompute_input_layernorm(self):
        """Set the attention layer for recompute input_layernorm. Only needed for fp8/fp4."""
        set_save_original_input(self.linear_q_down_proj)
        set_save_original_input(self.linear_kv_proj)
