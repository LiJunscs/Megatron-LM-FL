# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.


from copy import copy
from dataclasses import dataclass
##### FlagScale Add #####
from enum import Enum
from typing import NoReturn, Optional, Sequence, Union
##### FlagScale End #####

import torch

from megatron.core import tensor_parallel
from megatron.core.extensions.transformer_engine import HAVE_TE
##### FlagScale Add #####
from megatron.core.tensor_parallel.layers import set_tensor_model_parallel_attributes
from megatron.core.tensor_parallel.random import get_cuda_rng_tracker
##### FlagScale End #####
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
from megatron.core.transformer.experimental_attention_variant.csa_utils import cp_utils
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.torch_norm import LayerNormBuilder
from megatron.core.transformer.transformer_config import MLATransformerConfig
from megatron.core.transformer.utils import ensure_metadata_has_dp_cp_group  ##### FlagScale Add #####
from megatron.core.typed_torch import apply_module
##### FlagScale Add #####
from megatron.core.utils import (
    get_pg_size,
    is_te_min_version,
    make_tp_sharded_tensor_for_checkpoint,
)
##### FlagScale End #####

##### FlagScale Add #####
from megatron.core.fusions.fused_mla_yarn_rope_apply import (
    _FusedMLARoPEInplace,
    fused_mla_rope_inplace,
)
##### FlagScale End #####

if HAVE_TE:
    from megatron.core.extensions.transformer_engine import TELinear, set_save_original_input
else:
    (TEColumnParallelLinear, TELinear, set_save_original_input) = (None, None, None)


@torch.compile
def _q_rms_norm(q: torch.Tensor, eps: float) -> torch.Tensor:
    """Fused RMS normalization for query tensor (no learnable weight)."""
    return q * torch.rsqrt(q.square().mean(-1, keepdim=True) + eps)

##### FlagScale Add #####
class _DSv4TPBackwardPolicy(Enum):
    """Backward ownership of a field crossing the DSv4 TP adapter."""

    PASS_THROUGH = "pass_through"
    SCATTER = "scatter"
    REDUCE_SCATTER = "reduce_scatter"
    ALL_REDUCE = "all_reduce"
    NO_GRAD = "no_grad"


@dataclass(frozen=True)
class _DSv4TPField:
    """Named tensor and its backward policy in the DSv4 TP adapter."""

    name: str
    tensor: torch.Tensor
    backward_policy: _DSv4TPBackwardPolicy


class _DSv4TPRopeExchange(torch.autograd.Function):
    """Apply Q RoPE while hiding DSv4's SP and replicated-KV TP communication."""

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
        position_ids,
        cp_rank,
        cp_size,
        tp_group,
        field_specs,
        gather_sequence,
        apply_fused_rope,
        remove_interleaving,
    ):
        work = None
        if gather_sequence:
            gathered = local_tensor.new_empty(
                local_tensor.size(0) * tp_group.size(), *local_tensor.shape[1:]
            )
            work = torch.distributed.all_gather_into_tensor(
                gathered,
                local_tensor.contiguous(),
                group=tp_group,
                async_op=True,
            )
        else:
            gathered = local_tensor

        if apply_fused_rope:
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
                position_ids,
            )
        else:
            query = q
        if work is not None:
            work.wait()
        ctx.tp_group = tp_group
        ctx.field_specs = field_specs
        ctx.gather_sequence = gather_sequence
        ctx.apply_fused_rope = apply_fused_rope
        return query, gathered

    @staticmethod
    def backward(ctx, grad_query, grad_fields):
        tp_size = ctx.tp_group.size()
        field_widths = [width for _, width, _ in ctx.field_specs]
        field_grads = torch.split(grad_fields, field_widths, dim=-1)
        local_field_grads = [None] * len(ctx.field_specs)
        communication_indices = []
        communication_grads = []

        if ctx.gather_sequence:
            assert grad_fields.size(0) % tp_size == 0
            local_rows = grad_fields.size(0) // tp_size
            sequence_start = ctx.tp_group.rank() * local_rows
            communication_policy = _DSv4TPBackwardPolicy.REDUCE_SCATTER
        else:
            local_rows = grad_fields.size(0)
            sequence_start = 0
            communication_policy = _DSv4TPBackwardPolicy.ALL_REDUCE

        for index, ((_, _, policy), field_grad) in enumerate(
            zip(ctx.field_specs, field_grads)
        ):
            if policy is _DSv4TPBackwardPolicy.PASS_THROUGH:
                local_field_grads[index] = field_grad
            elif policy is _DSv4TPBackwardPolicy.SCATTER:
                local_field_grads[index] = field_grad.narrow(
                    0, sequence_start, local_rows
                ).contiguous()
            elif policy is communication_policy:
                communication_indices.append(index)
                communication_grads.append(field_grad)
            elif policy is _DSv4TPBackwardPolicy.NO_GRAD:
                local_field_grads[index] = field_grad.new_zeros(
                    local_rows, *field_grad.shape[1:]
                )
            else:
                raise AssertionError(
                    f"Invalid DSv4 TP policy {policy} for gather_sequence={ctx.gather_sequence}"
                )

        work = None
        local_communicated_grad = None
        if communication_grads:
            communicated_grad = (
                communication_grads[0]
                if len(communication_grads) == 1
                else torch.cat(communication_grads, dim=-1)
            ).contiguous()
            if ctx.gather_sequence:
                local_communicated_grad = communicated_grad.new_empty(
                    local_rows, *communicated_grad.shape[1:]
                )
                work = torch.distributed.reduce_scatter_tensor(
                    local_communicated_grad,
                    communicated_grad,
                    group=ctx.tp_group,
                    async_op=True,
                )
            else:
                local_communicated_grad = communicated_grad
                work = torch.distributed.all_reduce(
                    local_communicated_grad,
                    group=ctx.tp_group,
                    async_op=True,
                )

        grad_q = (
            _FusedMLARoPEInplace.backward(ctx, grad_query)[0]
            if ctx.apply_fused_rope
            else grad_query
        )
        if work is not None:
            work.wait()
        if local_communicated_grad is not None:
            communicated_widths = [field_widths[index] for index in communication_indices]
            for index, field_grad in zip(
                communication_indices,
                torch.split(local_communicated_grad, communicated_widths, dim=-1),
            ):
                local_field_grads[index] = field_grad

        assert all(field_grad is not None for field_grad in local_field_grads)
        return (
            grad_q,
            torch.cat(local_field_grads, dim=-1),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def _dsv4_tp_rope_exchange(
    q,
    fields: Sequence[_DSv4TPField],
    cos,
    sin,
    nope_dim,
    emb_dim,
    cu_seqlens_q,
    cp_rank,
    cp_size,
    tp_group,
    gather_sequence,
    is_thd,
    apply_fused_rope,
    remove_interleaving=True,
):
    """Normalize tensors to a CP-local layout and apply the configured TP backward contract.

    ``PackedSeqParams`` is intentionally not passed through this adapter.  The TP collective
    treats the packed token dimension as an ordinary contiguous leading dimension.  The only
    packed-sequence metadata consumed by fused Q RoPE is ``cu_seqlens_q``; ``max_seqlen_q`` has
    already been used by the caller to construct ``cos`` and ``sin``.  KV sequence metadata is
    consumed later, after the CP boundary exchange, and therefore does not belong here.

    For THD, TE's column-parallel Q up-projection has already gathered its sequence input, so
    ``q`` is CP-local while the tensors in ``fields`` are SP-local when ``gather_sequence`` is
    true.  This adapter gathers only the latter tensors and leaves Q's TP communication to TE.
    """
    assert fields, "DSv4 TP exchange requires at least one field"
    assert tp_group is not None, "DSv4 TP exchange requires an explicit TP process group"
    tp_size = get_pg_size(tp_group)
    assert tp_size >= 1
    assert cp_size >= 1 and 0 <= cp_rank < cp_size, (
        f"Invalid CP coordinates: rank={cp_rank}, size={cp_size}"
    )
    assert not gather_sequence or tp_size > 1, (
        "DSv4 sequence gather is only valid with tensor parallel size greater than one"
    )

    names = [field.name for field in fields]
    assert len(names) == len(set(names)), f"DSv4 TP field names must be unique: {names}"
    leading_shape = fields[0].tensor.shape[:-1]
    for field in fields:
        assert field.tensor.shape[:-1] == leading_shape, (
            f"DSv4 TP field {field.name!r} has leading shape {field.tensor.shape[:-1]}, "
            f"expected {leading_shape}"
        )
        assert field.tensor.size(-1) > 0, field.name

    # Linear/adapter layout contract. Linear layers and TP collectives preserve
    # arbitrary leading dimensions, so both SBHD and THD fields keep their
    # three-dimensional activation layout here. Only Q has been normalized at
    # the RoPE boundary already: THD removes its dummy batch axis before this
    # call, and fused RoPE (when enabled) executes inside the adapter.
    expected_local_tokens = fields[0].tensor.size(0)
    expected_q_tokens = expected_local_tokens * (tp_size if gather_sequence else 1)
    if is_thd:
        assert q.ndim == 3, f"THD Q must have shape [T, H, D], got {tuple(q.shape)}"
        assert all(field.tensor.ndim == 3 and field.tensor.size(1) == 1 for field in fields), (
            "THD TP fields must preserve the linear layout [T,1,D], "
            f"before exchange, got {[tuple(field.tensor.shape) for field in fields]}"
        )
        assert cu_seqlens_q is not None, "THD layout requires cu_seqlens_q"
        assert cu_seqlens_q.ndim == 1, (
            f"THD cu_seqlens_q must be one-dimensional, got {tuple(cu_seqlens_q.shape)}"
        )
        assert cu_seqlens_q.dtype in (torch.int32, torch.int64), (
            f"THD cu_seqlens_q must be integral, got {cu_seqlens_q.dtype}"
        )
        assert cu_seqlens_q.device == q.device, (
            f"THD cu_seqlens_q is on {cu_seqlens_q.device}, but Q is on {q.device}"
        )
    else:
        assert q.ndim == 4, f"SBHD Q must have shape [S, B, H, D], got {tuple(q.shape)}"
        assert all(field.tensor.ndim == 3 for field in fields), (
            "SBHD TP fields must have shape [S, B, D], got "
            f"{[tuple(field.tensor.shape) for field in fields]}"
        )
        assert q.size(1) == fields[0].tensor.size(1), (
            f"Q batch size {q.size(1)} does not match TP field batch size "
            f"{fields[0].tensor.size(1)}"
        )
    assert q.size(0) == expected_q_tokens, (
        "TE column-parallel Q must already be CP-local, while TP fields must be "
        f"{'SP-local' if gather_sequence else 'CP-local'}: Q has {q.size(0)} tokens, "
        f"expected {expected_q_tokens} from field length {expected_local_tokens} and "
        f"TP size {tp_size}"
    )

    local_tensor = (
        fields[0].tensor
        if len(fields) == 1
        else torch.cat([field.tensor for field in fields], dim=-1)
    )
    field_specs = tuple(
        (field.name, field.tensor.size(-1), field.backward_policy) for field in fields
    )
    allowed_policies = (
        {
            _DSv4TPBackwardPolicy.SCATTER,
            _DSv4TPBackwardPolicy.REDUCE_SCATTER,
            _DSv4TPBackwardPolicy.NO_GRAD,
        }
        if gather_sequence
        else {
            _DSv4TPBackwardPolicy.PASS_THROUGH,
            _DSv4TPBackwardPolicy.ALL_REDUCE,
            _DSv4TPBackwardPolicy.NO_GRAD,
        }
    )
    for field in fields:
        assert field.backward_policy in allowed_policies, (
            f"Invalid policy {field.backward_policy} for DSv4 field {field.name!r} "
            f"with gather_sequence={gather_sequence}"
        )

    position_ids = (
        cp_utils.get_cp_position_ids(
            q.size(0),
            cp_rank * q.size(0),
            q.device,
            cu_seqlens_padded=cu_seqlens_q if is_thd else None,
        )
        if cp_size > 1
        else None
    )
    query, exchanged = _DSv4TPRopeExchange.apply(
        q,
        local_tensor,
        cos,
        sin,
        nope_dim,
        emb_dim,
        cu_seqlens_q,
        position_ids,
        cp_rank,
        cp_size,
        tp_group,
        field_specs,
        gather_sequence,
        apply_fused_rope,
        remove_interleaving,
    )
    exchanged_fields = torch.split(
        exchanged, [field.tensor.size(-1) for field in fields], dim=-1
    )
    result = {}
    for field, exchanged_field in zip(fields, exchanged_fields):
        if field.backward_policy is _DSv4TPBackwardPolicy.NO_GRAD:
            exchanged_field = exchanged_field.detach()
        result[field.name] = exchanged_field
    return query, result
##### FlagScale End #####

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

        ##### FlagScale Add #####
        tp_size = get_pg_size(self.pg_collection.tp)
        assert self.config.num_attention_heads % tp_size == 0, (
            f"num_attention_heads ({self.config.num_attention_heads}) must be divisible by "
            f"tensor parallel size ({tp_size})"
        )
        # DSv4 uses a single replicated MQA KV head.  The generic Attention
        # value has GQA-specific semantics when num_query_groups < TP size, so
        # it cannot describe the column-parallel Q projection on this path.
        self.num_local_q_heads = self.config.num_attention_heads // tp_size
        ##### FlagScale End #####

        assert (
            not self.checkpoint_core_attention
        ), "Checkpoint core attention is not supported in DSv4 Hybrid Attention."
        assert (
            not self.offload_qkv_linear
        ), "Offload qkv linear is not supported in DSv4 Hybrid Attention."

        ##### FlagScale Add #####
        # ColumnParallelLinear constructors take global dimensions and perform
        # the TP division internally.  Keep Megatron's standard global meaning
        # for query_projection_size and track the local width separately.
        ##### FlagScale End #####
        self.query_projection_size = self.config.v_head_dim * self.config.num_attention_heads
        ##### FlagScale Add #####
        self.query_projection_size_per_partition = (
            self.config.v_head_dim * self.num_local_q_heads
        )
        ##### FlagScale End #####

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
        ##### FlagScale Add #####
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
        ##### FlagScale End #####

        _linear_o_group_proj = torch.empty(
            group_proj_out_size,
            group_proj_in_size,
            device=torch.cuda.current_device(),
            dtype=self.config.params_dtype,
        )
        ##### FlagScale Add #####
        # This parameter is TP-sharded along its group/output axis.  Initialize
        # it from the model-parallel RNG stream so TP ranks receive distinct
        # local shards.  Forking also restores the default/DP RNG afterwards,
        # preventing this TP-size-dependent tensor from shifting subsequent
        # replicated parameter initialization (notably compressor ``ape``).
        rng_tracker = get_cuda_rng_tracker()
        assert rng_tracker.is_initialized(), (
            "The CUDA RNG tracker must be initialized before constructing "
            "DSv4HybridAttention.linear_o_group_proj"
        )
        with rng_tracker.fork():
            self.config.init_method(_linear_o_group_proj)
        ##### FlagScale End #####
        self.linear_o_group_proj = torch.nn.Parameter(_linear_o_group_proj)
        ##### FlagScale Add #####
        set_tensor_model_parallel_attributes(
            self.linear_o_group_proj, is_parallel=True, dim=0, stride=1
        )
        ##### FlagScale End #####

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

    ##### FlagScale Add #####
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
    ##### FlagScale End #####

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

        # Select this microbatch's dynamic CP group. QKV captures it explicitly
        # for recompute; the rest of this forward reads it from pg_collection.
        # Restore the static group before returning.
        _orig_cp_group = self.pg_collection.cp
        cp_group = _orig_cp_group
        if packed_seq_params is not None and packed_seq_params.local_cp_size is not None:
            assert packed_seq_params.cp_group is not None, "cp_group must be set in dynamic-cp mode"
            cp_group = packed_seq_params.cp_group

        cp_size = cp_group.size()
        use_cp = cp_size > 1
        self.pg_collection.cp = cp_group

        # =====================
        # Query, Key, and Value
        # =====================
        # Get the query, key and value tensors based on the type of attention -
        # self or cross attn.
        ##### FlagScale Add #####
        qkv = (
            self.get_query_key_value_tensors(
                hidden_states,
                key_value_states,
                position_ids,
                packed_seq_params,
                inference_context=inference_context,
            )
        ##### FlagScale End #####
        )
        if use_cp:
            (
                query,
                key,
                value,
                q_compressed,
                gathered_hidden_states,
                boundary_hidden,
                boundary_kv,
            ) = qkv
        else:
            query, key, value, q_compressed, gathered_hidden_states = qkv
            boundary_hidden = None
            boundary_kv = None

        # TODO: Currently, TE can only accept contiguous tensors for MLA
        query = query.contiguous()
        key = key.contiguous()
        ##### FlagScale Add #####
        # DSv4's single MQA tensor is shared by key and value. Preserve that
        # alias instead of materializing the same contiguous tensor twice.
        value = key
        ##### FlagScale End #####

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
                ##### FlagScale Add #####
                # CSA consumes the full sequence, unlike the SP input to this layer.
                x=gathered_hidden_states,
                ##### FlagScale End #####
                qr=q_compressed,
                boundary_hidden=boundary_hidden,
                boundary_kv=boundary_kv,
            )
        # NOTE: No implement in megatron version core_v0.17.0, group_commit(v0.17.0) -> group_offload(latest)
        if self.offload_core_attention and self.training:
            # ``value`` aliases ``key`` for DSv4 MQA. Do not place the same tensor
            # in the offload release list twice: the release path clears its
            # underlying storage, and duplicate releases are unnecessary and
            # fragile if that path later stops being idempotent.
            forced_released_tensors = [query, key]
            if value is not key:
                forced_released_tensors.append(value)
            if boundary_kv is not None:
                forced_released_tensors.append(boundary_kv)
            core_attn_out = core_attn_manager.group_commit(
                core_attn_out, name="core_attn", forced_released_tensors=forced_released_tensors
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
        n_heads = self.num_local_q_heads  ##### FlagScale Add #####
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
            rope_seqlen = packed_seq_params.max_seqlen_kv
        else:
            cu_seqlens_kv = None
            rope_seqlen = seq_len * cp_size if use_cp else seq_len
        # DSv4 reference (DS-Inf) RoPE is pure rotation (norm-preserving). Yarn's
        # concentration factor (mscale) is NOT part of the DSv4 model contract --
        # the model relies on Q/KV RMS-norm + unit-magnitude rotation. Force 1.0.
        mscale = 1.0
        rotary_pos_cos = None
        rotary_pos_sin = None
        full_rope_table = packed_seq or use_cp
        if self.config.apply_rope_fusion:
            # ``mscale=1.0`` strips yarn's concentration factor from the
            # cached cos/sin so the fused kernel matches the unfused
            # path's forced ``mscale=1.0`` (DSv4 "pure rotation").
            rotary_pos_cos, rotary_pos_sin = self.rotary_pos_emb.get_cached_cos_sin(
                rope_seqlen,
                dtype=hidden_states.dtype,
                packed_seq=full_rope_table,
                mscale=mscale,
            )
            rotary_pos_emb = None
            assert inference_context is None, "Inference with MLA RoPE fusion is not supported"
            assert (
                fused_mla_rope_inplace is not None
            ), "Fused MLA RoPE apply is not imported successfully"
        elif self._dsv4_uses_yarn_rope:
            rotary_pos_emb, _ = self.rotary_pos_emb(
                rope_seqlen, packed_seq=full_rope_table
            )
        else:
            rotary_pos_emb = self.rotary_pos_emb(
                rope_seqlen, packed_seq=full_rope_table
            )
        inverse_global_start = (
            self.pg_collection.cp.rank() * core_attn_out.shape[0] if use_cp else 0
        )
        if self.config.apply_rope_fusion:
            if use_cp:
                core_attn_out = cp_utils.apply_cp_local_rope_fused(
                    core_attn_out,
                    rotary_pos_cos,
                    rotary_pos_sin,
                    nope_dim,
                    pos_dim,
                    cu_seqlens_kv,
                    inverse_global_start,
                    inverse=True,
                )
            else:
                if packed_seq:
                    core_attn_out = core_attn_out.squeeze(1)
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
                if packed_seq:
                    core_attn_out = core_attn_out.unsqueeze(1)
        elif use_cp:
            core_attn_out = cp_utils.apply_cp_local_rope_unfused(
                core_attn_out,
                rotary_pos_emb,
                nope_dim,
                pos_dim,
                cu_seqlens_kv,
                inverse_global_start,
                self.config,
                inverse=True,
            )
        else:
            content_part, rot_part = torch.split(
                core_attn_out, [core_attn_out.size(-1) - pos_dim, pos_dim], dim=-1
            )
            # ``_apply_rotary_pos_emb_thd`` documents 3-D ``(total, h, d)`` input
            # and adds its own batch dim internally; drop the dummy ``b=1`` axis
            # for THD before the rope and add it back after.
            if packed_seq:
                rot_part_in = rot_part.squeeze(1)
            else:
                rot_part_in = rot_part
            rot_part_out = apply_rotary_pos_emb(
                rot_part_in,
                rotary_pos_emb,
                self.config,
                cu_seqlens=cu_seqlens_kv,
                mscale=mscale,
                cp_group=self.pg_collection.cp,
                mla_rotary_interleaved=True,
                inverse=True,
                mla_output_remove_interleaving=True,
            )
            if packed_seq:
                rot_part = rot_part_out.unsqueeze(1)
            else:
                rot_part = rot_part_out
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
        self.pg_collection.cp = _orig_cp_group
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
            ##### FlagScale Add #####
            # ColumnParallelLinear takes the global output width and returns
            # query_projection_size (= local Q heads * head dim) on each rank.
            self.query_projection_size,
            ##### FlagScale End #####
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name='q_up_proj',
            skip_weight_param_allocation=False,  ##### FlagScale Add #####
            tp_group=pg_collection.tp,
        )

        ##### FlagScale Add #####
        kv_proj_kwargs = {}
        if submodules.linear_kv_proj in [TELinear]:
            # The single MQA KV head is intentionally replicated.  Sharding
            # v_head_dim would leave RoPE and CSA with only a partial head.
            kv_proj_kwargs['parallel_mode'] = 'duplicated'
        else:
            raise ValueError(f"Unsupported linear_kv_proj: {submodules.linear_kv_proj}")

        ##### FlagScale End #####
        self.linear_kv_proj = build_module(
            submodules.linear_kv_proj,
            self.config.hidden_size,
            self.config.v_head_dim,
            config=self.config,
            init_method=self.config.init_method,
            skip_weight_param_allocation=False,  ##### FlagScale Add #####
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name='kv_up_proj',
            ##### FlagScale Add #####
            tp_group=None,
            **kv_proj_kwargs,
            ##### FlagScale End #####
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

        ``position_ids`` is retained for the common Attention API. DSv4 derives
        RoPE positions from its SBHD/THD layout metadata instead.
        """
        del position_ids
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
        cp_group = self.pg_collection.cp
        use_cp = cp_group.size() > 1
        # Explicit CP positions must index the complete global table.  The RoPE
        # modules use ``packed_seq=True`` as their existing no-CP-sharding mode;
        # this avoids their legacy SBHD zigzag slicing for contiguous CP.
        full_rope_table = packed_seq or use_cp
        if self.config.apply_rope_fusion:
            # ``mscale=1.0`` strips yarn's concentration factor from the
            # cached cos/sin so the fused kernel matches the unfused
            # path's forced ``mscale=1.0`` (DSv4 "pure rotation").
            rotary_pos_cos, rotary_pos_sin = self.rotary_pos_emb.get_cached_cos_sin(
                rotary_seq_len,
                dtype=hidden_states.dtype,
                packed_seq=full_rope_table,
                mscale=mscale,
            )
            rotary_pos_emb = None
            assert inference_context is None, "Inference with MLA RoPE fusion is not supported"
            assert (
                fused_mla_rope_inplace is not None
            ), "Fused MLA RoPE apply is not imported successfully"
        elif self._dsv4_uses_yarn_rope:
            rotary_pos_emb, _ = self.rotary_pos_emb(
                rotary_seq_len, packed_seq=full_rope_table
            )
        else:
            rotary_pos_emb = self.rotary_pos_emb(
                rotary_seq_len, packed_seq=full_rope_table
            )

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

        # Linear layers and TP communication are agnostic to the trailing
        # leading dimensions. Keep SBHD [S,B,D] and THD [T,1,D] unchanged;
        # only RoPE converts Q/KV to TE's packed attention layout.
        hidden_states_for_tp = hidden_states
        k_pos_emb = None

        # =========================================
        # Apply norm
        # =========================================

        if self.config.q_lora_rank is not None:
            # q_compressed: [num_tokens, q_lora_rank]
            q_compressed = apply_module(self.q_layernorm)(q_compressed)

        # =========================================
        # QKV up projection and RoPE apply
        # =========================================

        def qkv_up_proj_and_rope_apply(
                q_compressed,
                hidden_states_for_tp,
                k_pos_emb,
                rotary_pos_emb,
                cp_group,
            ):  ##### FlagScale Add #####
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
            ##### FlagScale Add #####
            assert q.size(-1) == self.query_projection_size_per_partition, (
                f"local Q projection width ({q.size(-1)}) must equal "
                "num_local_q_heads * q_head_dim "
                f"({self.query_projection_size_per_partition})"
            )
            q = q.view(*q.size()[:-1], self.num_local_q_heads, self.q_head_dim)
            ##### FlagScale End #####
            q = _q_rms_norm(q, self.config.layernorm_epsilon)
            # RoPE/attention layout contract: THD Q has no batch axis.
            if packed_seq:
                q = q.squeeze(1)

            kv, _ = self.linear_kv_proj(hidden_states_for_tp)  ##### FlagScale Add #####

            # [num_tokens, qk_pos_emb_head_dim] -> [num_tokens, 1, qk_pos_emb_head_dim]
            if k_pos_emb is not None:
                k_pos_emb = torch.unsqueeze(k_pos_emb, -2)

            cp_size = cp_group.size()
            cp_rank = cp_group.rank()
            # TP-SP Allgather
            tp_size = get_pg_size(self.pg_collection.tp)
            sp_enabled = self.config.sequence_parallel and tp_size > 1
            if sp_enabled:
                fields = (
                    _DSv4TPField(
                        "hidden_states", hidden_states_for_tp, _DSv4TPBackwardPolicy.SCATTER
                    ),
                    _DSv4TPField("kv", kv, _DSv4TPBackwardPolicy.REDUCE_SCATTER),
                    _DSv4TPField(
                        "q_compressed", q_compressed, _DSv4TPBackwardPolicy.NO_GRAD
                    ),
                )
            else:
                kv_policy = (
                    _DSv4TPBackwardPolicy.ALL_REDUCE
                    if tp_size > 1
                    else _DSv4TPBackwardPolicy.PASS_THROUGH
                )
                fields = (_DSv4TPField("kv", kv, kv_policy),)

            # Do not pass PackedSeqParams as an opaque object across this boundary.  At this
            # point its Q-side contract is fully represented by ``packed_seq`` (layout) and
            # ``cu_seqlens_q`` (fused RoPE); the remaining metadata is used by KV/CP below.
            query, exchanged = _dsv4_tp_rope_exchange(
                q=q,
                fields=fields,
                cos=rotary_pos_cos,
                sin=rotary_pos_sin,
                nope_dim=self.config.qk_head_dim,
                emb_dim=self.config.qk_pos_emb_head_dim,
                cu_seqlens_q=cu_seqlens_q,
                cp_rank=cp_rank,
                cp_size=cp_size,
                tp_group=self.pg_collection.tp,
                gather_sequence=sp_enabled,
                is_thd=packed_seq,
                apply_fused_rope=self.config.apply_rope_fusion,
            )
            kv = exchanged["kv"]
            if sp_enabled:
                hidden_states_for_tp = exchanged["hidden_states"]
                q_compressed = exchanged["q_compressed"]

            gathered_hidden_states = hidden_states_for_tp
            gathered_q_compressed = q_compressed
            # DSv4 Context Parallel
            boundary_hidden = None
            boundary_kv = None
            boundary_rows = 0
            if cp_size > 1:
                boundary_hidden = cp_utils.exchange_cp_boundary_hidden(
                    gathered_hidden_states,
                    self._dsv4_compress_ratio,
                    self.config.csa_window_size,
                    cp_group,
                )
                boundary_kv_raw = cp_utils.exchange_cp_boundary_hidden(
                    kv,
                    self._dsv4_compress_ratio,
                    self.config.csa_window_size,
                    cp_group,
                )
                boundary_rows = boundary_kv_raw.shape[0]
                kv = torch.cat((boundary_kv_raw, kv), dim=0)
            kv = self.kv_layernorm(kv)

            # RoPE/attention layout contract: linear/adapter KV is [T,1,D]
            # for THD. Remove only the dummy batch axis here; the following
            # unsqueeze creates the MQA head axis expected by attention.
            if packed_seq:
                assert kv.ndim == 3 and kv.size(1) == 1, (
                    f"THD KV projection must be [T,1,D] before RoPE, got {tuple(kv.shape)}"
                )
                kv = kv.squeeze(1)
            if self.config.apply_rope_fusion:
                if cp_size > 1:
                    # Rank r owns global rows [r * local_rows, (r + 1) * local_rows).
                    global_start = cp_rank * q.shape[0]
                    kv = kv.unsqueeze(-2)
                    kv = cp_utils.apply_cp_local_rope_fused(
                        kv,
                        rotary_pos_cos,
                        rotary_pos_sin,
                        self.config.qk_head_dim,
                        self.config.qk_pos_emb_head_dim,
                        cu_seqlens_q,
                        global_start - boundary_rows,
                    )
                    boundary_kv = kv[:boundary_rows]
                    kv = kv[boundary_rows:]
                else:
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
                if cp_size > 1:
                    global_start = cp_group.rank() * q.shape[0]
                    query = cp_utils.apply_cp_local_rope_unfused(
                        q,
                        rotary_pos_emb,
                        self.config.qk_head_dim,
                        self.config.qk_pos_emb_head_dim,
                        cu_seqlens_q,
                        global_start,
                        self.config,
                    )
                    kv = cp_utils.apply_cp_local_rope_unfused(
                        kv.unsqueeze(-2),
                        rotary_pos_emb,
                        self.config.qk_head_dim,
                        self.config.qk_pos_emb_head_dim,
                        cu_seqlens_kv,
                        global_start - boundary_rows,
                        self.config,
                    )
                    boundary_kv = kv[:boundary_rows]
                    kv = kv[boundary_rows:]
                    key = value = kv
                else:
                    q_len = q.size()[0]
                    # Shorten rotary_pos_emb to the sequence length when inference_params
                    # is not provided so direct forward accepts any sequence length.
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
                    key = value = kv

            query = query.contiguous()
            key = key.contiguous()
            value = key
            if boundary_kv is not None:
                boundary_kv = boundary_kv.contiguous()

            if boundary_kv is None:
                return query, key, value, gathered_q_compressed, gathered_hidden_states
            return (
                query,
                key,
                value,
                gathered_q_compressed,
                gathered_hidden_states,
                boundary_hidden,
                boundary_kv,
            )

        if self.recompute_up_proj:
            quantization = self.config.fp8 or self.config.fp4
            self.qkv_up_checkpoint = tensor_parallel.CheckpointWithoutOutput(fp8=quantization)
            return self.qkv_up_checkpoint.checkpoint(
                qkv_up_proj_and_rope_apply,
                q_compressed,
                hidden_states_for_tp,
                k_pos_emb,
                rotary_pos_emb,
                self.pg_collection.cp,
            )
        return qkv_up_proj_and_rope_apply(
            q_compressed,
            hidden_states_for_tp,
            k_pos_emb,
            rotary_pos_emb,
            self.pg_collection.cp,
        )

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
