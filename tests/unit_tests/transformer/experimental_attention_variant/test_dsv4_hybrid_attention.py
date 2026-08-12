# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from unittest import mock
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

import megatron.core.parallel_state as parallel_state
from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_transformer_block_with_experimental_attention_variant_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_config import MLATransformerConfig
from megatron.training.arguments import parse_args
from megatron.training.checkpointing import load_checkpoint, save_checkpoint
from megatron.training.global_vars import set_args
from megatron.training.training import get_model
from megatron.training.utils import unwrap_model
from tests.unit_tests.dist_checkpointing import (
    TempNamedDir,
    init_basic_mock_args,
    init_checkpointing_mock_args,
)
from tests.unit_tests.test_utilities import Utils

try:
    from fast_hadamard_transform import hadamard_transform as _hadamard_transform

    HAVE_HADAMARD = True
except ImportError:
    HAVE_HADAMARD = False
    _hadamard_transform = None

_SEED = 42


def _mock_hadamard_transform(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    return x * scale


@pytest.fixture(autouse=True)
def patch_hadamard_if_needed():
    """Patch hadamard_transform in dsa/csa modules if the library is not installed."""
    if not HAVE_HADAMARD:
        with (
            patch(
                'megatron.core.transformer.experimental_attention_variant.dsa.hadamard_transform',
                _mock_hadamard_transform,
            ),
            patch(
                'megatron.core.transformer.experimental_attention_variant.csa.rotate_activation',
                lambda x: x * (x.size(-1) ** -0.5),
            ),
        ):
            yield
    else:
        yield


# ---------------------------------------------------------------------------
# Config / spec helpers
# ---------------------------------------------------------------------------


def _make_config(
    num_layers=4,
    hidden_size=256,
    num_attention_heads=16,
    v_head_dim=64,
    qk_pos_emb_head_dim=32,
    q_lora_rank=64,
    o_groups=8,
    o_lora_rank=64,
    csa_compress_ratios=None,
    csa_window_size=8,
    tensor_model_parallel_size=1,
    sequence_parallel=False,
    dsa_indexer_n_heads=8,
    dsa_indexer_head_dim=64,
    dsa_indexer_topk=8,
    dsa_indexer_loss_coeff=0.0,
    **extra_config_kwargs,
):
    """Create an MLATransformerConfig for DSv4 hybrid attention tests."""
    if csa_compress_ratios is None:
        csa_compress_ratios = [0, 4, 128, 4]
    return MLATransformerConfig(
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        use_cpu_initialization=True,
        bf16=True,
        params_dtype=torch.bfloat16,
        add_bias_linear=False,
        tensor_model_parallel_size=tensor_model_parallel_size,
        sequence_parallel=sequence_parallel,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=v_head_dim - qk_pos_emb_head_dim,
        qk_head_dim=v_head_dim - qk_pos_emb_head_dim,
        qk_pos_emb_head_dim=qk_pos_emb_head_dim,
        v_head_dim=v_head_dim,
        o_groups=o_groups,
        o_lora_rank=o_lora_rank,
        rope_type='rope',
        rotary_base=10000,
        rotary_percent=1.0,
        multi_latent_attention=True,
        experimental_attention_variant='dsv4_hybrid',
        csa_compress_ratios=csa_compress_ratios,
        csa_window_size=csa_window_size,
        dsa_indexer_n_heads=dsa_indexer_n_heads,
        dsa_indexer_head_dim=dsa_indexer_head_dim,
        dsa_indexer_topk=dsa_indexer_topk,
        dsa_indexer_loss_coeff=dsa_indexer_loss_coeff,
        **extra_config_kwargs,
    )


def _make_attention_spec(config):
    """Build the full DSv4HybridSelfAttention ModuleSpec using the canonical spec builder."""
    from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
    from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
        get_dsv4_hybrid_module_spec_for_backend,
    )

    return get_dsv4_hybrid_module_spec_for_backend(config=config, backend=TESpecProvider())


def _build_attention(config, layer_number, pg_collection):
    """Instantiate a DSv4HybridSelfAttention from config."""
    from megatron.core.transformer.spec_utils import build_module

    spec = _make_attention_spec(config)
    return build_module(spec, config=config, layer_number=layer_number, pg_collection=pg_collection)


# ===========================================================================
# Constructor tests
# ===========================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not HAVE_TE, reason="transformer_engine not available")
class TestDSv4HybridAttentionConstructor:
    """Test construction of DSv4HybridSelfAttention across TP sizes."""

    @pytest.fixture(scope='class', autouse=True)
    def setup_method(self):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        yield
        Utils.destroy_model_parallel()

    def test_basic_construction(self):
        """Verify the layer builds and has the expected sub-modules."""
        from megatron.core.transformer.experimental_attention_variant.deepseek_v4_hybrid_attention import (
            DSv4HybridSelfAttention,
        )

        torch.manual_seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)

        config = _make_config()
        pg = ProcessGroupCollection.use_mpu_process_groups()
        attn = _build_attention(config, layer_number=1, pg_collection=pg)

        assert isinstance(attn, DSv4HybridSelfAttention)
        assert hasattr(attn, 'linear_q_down_proj')
        assert hasattr(attn, 'linear_q_up_proj')
        assert hasattr(attn, 'linear_kv_proj')
        assert hasattr(attn, 'linear_proj')
        assert hasattr(attn, 'linear_o_group_proj')
        assert hasattr(attn, 'core_attention')
        assert hasattr(attn, 'q_layernorm')
        assert hasattr(attn, 'kv_layernorm')

        # Q is head-sharded, while the single MQA KV projection is duplicated.
        assert attn.num_local_q_heads == config.num_attention_heads
        assert attn.query_projection_size == config.num_attention_heads * config.v_head_dim
        assert attn.query_projection_size_per_partition == (
            config.num_attention_heads * config.v_head_dim
        )
        assert attn.linear_q_up_proj.weight.shape[0] == attn.query_projection_size_per_partition
        assert attn.linear_kv_proj.weight.shape[0] == config.v_head_dim
        assert not getattr(attn.linear_kv_proj.weight, 'tensor_model_parallel', False)

    def test_q_head_dim_equals_v_head_dim(self):
        """q_head_dim must equal v_head_dim for DSv4 hybrid."""
        torch.manual_seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)

        config = _make_config()
        pg = ProcessGroupCollection.use_mpu_process_groups()
        attn = _build_attention(config, layer_number=1, pg_collection=pg)

        assert attn.q_head_dim == config.v_head_dim

    def test_compressor_owns_tp_output_gradient_contract(self):
        """Main and indexer compressors expose different TP backward semantics."""
        torch.manual_seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)

        config = _make_config()
        pg = ProcessGroupCollection.use_mpu_process_groups()
        attn = _build_attention(config, layer_number=2, pg_collection=pg)

        core = attn.core_attention
        assert core.compress_ratio == 4
        assert core.compressor.reduce_output_grad_across_tp
        assert core.indexer is not None
        assert not core.indexer.compressor.reduce_output_grad_across_tp

    @pytest.mark.parametrize("layer_number", [1, 2, 3, 4])
    def test_rope_base_varies_with_compress_ratio(self, layer_number):
        """Layers with compress_ratio > 1 should use csa_compress_rotary_base."""
        torch.manual_seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)

        ratios = [0, 4, 128, 4]
        config = _make_config(csa_compress_ratios=ratios)
        pg = ProcessGroupCollection.use_mpu_process_groups()
        attn = _build_attention(config, layer_number=layer_number, pg_collection=pg)

        ratio = ratios[layer_number - 1]
        if ratio > 1:
            expected_base = config.csa_compress_rotary_base
        else:
            expected_base = config.rotary_base

        # inv_freq is derived from rotary_base; verify the correct base was used
        dim = config.qk_pos_emb_head_dim
        recomputed_inv_freq = 1.0 / (
            expected_base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )
        assert torch.allclose(
            attn.rotary_pos_emb.inv_freq.cpu(), recomputed_inv_freq, rtol=1e-5, atol=1e-5
        )


# ===========================================================================
# Forward / backward tests
# ===========================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not HAVE_TE, reason="transformer_engine not available")
class TestDSv4HybridAttentionForwardBackward:
    """Test forward and backward passes of DSv4HybridSelfAttention."""

    @pytest.fixture(scope='class', autouse=True)
    def setup_method(self, request):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        torch.manual_seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)

        cls = request.cls
        cls.config = _make_config(dsa_indexer_loss_coeff=1.0)
        cls.pg = ProcessGroupCollection.use_mpu_process_groups()

        yield
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("layer_number", [1, 2, 3, 4])
    def test_forward_output_shape(self, layer_number):
        """Forward should produce [sq, b, hidden_size] output."""
        seq_len = 256
        batch_size = 2

        torch.manual_seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)

        attn = _build_attention(
            self.config, layer_number=layer_number, pg_collection=self.pg
        ).cuda()

        hidden = torch.randn(
            seq_len, batch_size, self.config.hidden_size, dtype=torch.bfloat16
        ).cuda()

        output, bias = attn(hidden_states=hidden, attention_mask=None)

        assert output.shape == (seq_len, batch_size, self.config.hidden_size)
        assert output.dtype == torch.bfloat16
        assert not torch.isnan(output).any()

    @pytest.mark.parametrize("layer_number", [1, 2])
    def test_backward_gradient_flow(self, layer_number):
        """Backward should produce gradients for all trainable parameters."""
        seq_len = 256
        batch_size = 2

        torch.manual_seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)

        attn = _build_attention(
            self.config, layer_number=layer_number, pg_collection=self.pg
        ).cuda()
        attn.train()

        hidden = (
            torch.randn(seq_len, batch_size, self.config.hidden_size, dtype=torch.bfloat16)
            .cuda()
            .requires_grad_(True)
        )

        output, bias = attn(hidden_states=hidden, attention_mask=None)
        loss = output.sum()
        loss.backward()

        assert hidden.grad is not None, "No gradient on hidden_states"
        for name, param in attn.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for parameter {name}"

    def test_eval_mode(self):
        """Forward should work in eval mode."""
        seq_len = 128
        batch_size = 2

        torch.manual_seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)

        attn = _build_attention(self.config, layer_number=1, pg_collection=self.pg).cuda()
        attn.eval()

        hidden = torch.randn(
            seq_len, batch_size, self.config.hidden_size, dtype=torch.bfloat16
        ).cuda()

        with torch.no_grad():
            output, bias = attn(hidden_states=hidden, attention_mask=None)

        assert output.shape == (seq_len, batch_size, self.config.hidden_size)
        assert not torch.isnan(output).any()

    def test_different_seq_lengths(self):
        """Forward should handle various sequence lengths."""
        batch_size = 2

        torch.manual_seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)

        attn = _build_attention(self.config, layer_number=2, pg_collection=self.pg).cuda()

        for seq_len in [64, 128, 256]:
            hidden = torch.randn(
                seq_len, batch_size, self.config.hidden_size, dtype=torch.bfloat16
            ).cuda()
            output, bias = attn(hidden_states=hidden, attention_mask=None)
            assert output.shape == (seq_len, batch_size, self.config.hidden_size)


# ===========================================================================
# get_query_key_value_tensors tests
# ===========================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not HAVE_TE, reason="transformer_engine not available")
class TestDSv4HybridQKV:
    """Test get_query_key_value_tensors internals."""

    @pytest.fixture(scope='class', autouse=True)
    def setup_method(self, request):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        torch.manual_seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)

        cls = request.cls
        cls.config = _make_config()
        cls.pg = ProcessGroupCollection.use_mpu_process_groups()

        yield
        Utils.destroy_model_parallel()

    def test_qkv_shapes(self):
        """Query, key, value should have correct shapes."""
        seq_len = 64
        batch_size = 2

        torch.manual_seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)

        attn = _build_attention(self.config, layer_number=1, pg_collection=self.pg).cuda()
        hidden = torch.randn(
            seq_len, batch_size, self.config.hidden_size, dtype=torch.bfloat16
        ).cuda()

        q, k, v, q_compressed, gathered_hidden_states = attn.get_query_key_value_tensors(hidden)

        n_heads = self.config.num_attention_heads
        v_dim = self.config.v_head_dim

        assert q.shape == (seq_len, batch_size, n_heads, v_dim)
        # key and value are single-head (MQA-style) with an extra head dim
        assert k.shape[-1] == v_dim
        assert v.shape[-1] == v_dim
        assert q_compressed.shape[:2] == (seq_len, batch_size)
        assert q_compressed.requires_grad
        assert gathered_hidden_states.shape == hidden.shape

    def test_key_equals_value(self):
        """In the wkv path, key and value should be the same tensor."""
        seq_len = 64
        batch_size = 2

        torch.manual_seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)

        attn = _build_attention(self.config, layer_number=1, pg_collection=self.pg).cuda()
        hidden = torch.randn(
            seq_len, batch_size, self.config.hidden_size, dtype=torch.bfloat16
        ).cuda()

        q, k, v, _, _ = attn.get_query_key_value_tensors(hidden)
        assert torch.equal(k, v), "key and value should be identical in wkv path"


# ===========================================================================
# Grouped output projection tests
# ===========================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not HAVE_TE, reason="transformer_engine not available")
class TestDSv4HybridGroupedOutput:
    """Test that grouped output projection (wo_a) parameters are created."""

    @pytest.fixture(scope='class', autouse=True)
    def setup_method(self):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        yield
        Utils.destroy_model_parallel()

    def test_o_group_proj_shape(self):
        """linear_o_group_proj should have the correct shape."""
        torch.manual_seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)

        o_groups = 8
        o_lora_rank = 64
        config = _make_config(o_groups=o_groups, o_lora_rank=o_lora_rank)
        pg = ProcessGroupCollection.use_mpu_process_groups()
        attn = _build_attention(config, layer_number=1, pg_collection=pg)

        expected_out = o_groups * o_lora_rank
        expected_in = (config.v_head_dim * config.num_attention_heads) // o_groups
        assert attn.linear_o_group_proj.shape == (expected_out, expected_in)
        assert attn.linear_o_group_proj.requires_grad


# ===========================================================================
# DSv4 Hybrid Attention + Hash MoE integration tests
# ===========================================================================


def _make_dsv4_hash_moe_config():
    """Create a compact DSv4 config that combines CSA/HCA with hash MoE."""
    return _make_config(
        num_layers=2,
        hidden_size=128,
        num_attention_heads=8,
        v_head_dim=32,
        qk_pos_emb_head_dim=16,
        q_lora_rank=32,
        o_groups=4,
        o_lora_rank=32,
        csa_compress_ratios=[4, 128],
        csa_window_size=16,
        dsa_indexer_n_heads=4,
        dsa_indexer_head_dim=32,
        dsa_indexer_topk=8,
        ffn_hidden_size=256,
        num_moe_experts=4,
        moe_ffn_hidden_size=256,
        moe_layer_freq=1,
        moe_router_topk=2,
        moe_router_load_balancing_type="aux_loss",
        moe_aux_loss_coeff=0.0,
        moe_router_dtype="fp32",
        moe_router_score_function="sqrtsoftplus",
        moe_n_hash_layers=1,
        actual_vocab_size=128,
        activation_func=F.silu,
        gated_linear_unit=True,
        activation_func_clamp_value=10.0,
        bias_activation_fusion=False,
        moe_grouped_gemm=False,
    )


def _build_dsv4_moe_layer(config, layer_number, pg_collection):
    """Instantiate a TransformerLayer from the DSv4 experimental attention spec."""
    from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
    from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
        get_transformer_layer_with_experimental_attention_variant_spec,
    )
    from megatron.core.transformer.spec_utils import build_module

    layer_specs = get_transformer_layer_with_experimental_attention_variant_spec(
        config=config, backend=TESpecProvider()
    )
    return build_module(
        layer_specs[layer_number - 1],
        config=config,
        layer_number=layer_number,
        pg_collection=pg_collection,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not HAVE_TE, reason="transformer_engine not available")
class TestDSv4HybridHashMoEIntegration:
    """Integration coverage for DSv4 hybrid attention with hash MoE and clamped SwiGLU."""

    @pytest.fixture(scope='class', autouse=True)
    def setup_method(self, request):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=1,
        )
        torch.manual_seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)

        cls = request.cls
        cls.config = _make_dsv4_hash_moe_config()
        cls.pg = ProcessGroupCollection.use_mpu_process_groups()

        yield
        Utils.destroy_model_parallel()

    def test_csa_hash_moe_layer_forward_backward(self):
        """Layer 1 should combine DSv4 CSA, hash routing, and clamped SwiGLU."""
        from megatron.core.transformer.experimental_attention_variant.deepseek_v4_hybrid_attention import (
            DSv4HybridSelfAttention,
        )
        from megatron.core.transformer.moe.moe_layer import MoELayer

        seq_len = 256
        batch_size = 1

        torch.manual_seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)

        layer = _build_dsv4_moe_layer(self.config, layer_number=1, pg_collection=self.pg).cuda()
        layer.train()

        assert isinstance(layer.self_attention, DSv4HybridSelfAttention)
        assert layer.self_attention.core_attention.compress_ratio == 4
        assert isinstance(layer.mlp, MoELayer)
        assert layer.mlp.router.is_hash_layer is True
        assert layer.mlp.router.tid2eid is not None
        assert layer.config.activation_func_clamp_value == 10.0
        assert layer.config.activation_func is F.silu
        assert layer.config.gated_linear_unit is True

        hidden = torch.randn(
            seq_len,
            batch_size,
            self.config.hidden_size,
            dtype=torch.bfloat16,
            device="cuda",
            requires_grad=True,
        )
        input_ids = torch.randint(
            0, self.config.actual_vocab_size, (batch_size, seq_len), device="cuda"
        )

        output, context = layer(hidden_states=hidden, attention_mask=None, input_ids=input_ids)
        loss = output.float().square().mean()
        loss.backward()

        assert context is None
        assert output.shape == hidden.shape
        assert output.dtype == torch.bfloat16
        assert torch.isfinite(output).all()
        assert hidden.grad is not None
        assert torch.isfinite(hidden.grad).all()
        assert any(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in layer.self_attention.parameters()
            if p.requires_grad
        )
        assert any(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in layer.mlp.parameters()
            if p.requires_grad
        )

    def test_hash_moe_layer_requires_input_ids_but_hca_layer_does_not(self):
        """Hash routing is limited to leading layers while later HCA MoE layers remain runnable."""
        seq_len = 256
        batch_size = 1

        torch.manual_seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)

        hash_layer = _build_dsv4_moe_layer(
            self.config, layer_number=1, pg_collection=self.pg
        ).cuda()
        hidden = torch.randn(
            seq_len, batch_size, self.config.hidden_size, dtype=torch.bfloat16, device="cuda"
        )
        with pytest.raises(AssertionError, match="input_ids is required for hash-based routing"):
            hash_layer(hidden_states=hidden, attention_mask=None)

        hca_layer = _build_dsv4_moe_layer(self.config, layer_number=2, pg_collection=self.pg).cuda()
        assert hca_layer.self_attention.core_attention.compress_ratio == 128
        assert hca_layer.mlp.router.is_hash_layer is False

        output, context = hca_layer(hidden_states=hidden, attention_mask=None)

        assert context is None
        assert output.shape == hidden.shape
        assert torch.isfinite(output).all()


# ===========================================================================
# apply_rope_fusion tests
# ===========================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not HAVE_TE, reason="transformer_engine not available")
class TestDSv4HybridRopeFusion:
    """Test that apply_rope_fusion=True works for both yarn and non-yarn layers.

    DSv4 Hybrid uses YarnRotaryEmbedding for layers with compress_ratio > 1
    and standard RotaryEmbedding for layers with compress_ratio <= 1. The
    fused RoPE path must obtain cos/sin from both embedding classes via
    get_cached_cos_sin.

    compress_ratios=[0, 4, 128, 4]: layer 1 has ratio 0 (standard
    RotaryEmbedding), layers 2-4 have ratio > 1 (YarnRotaryEmbedding).
    """

    @pytest.fixture(scope='class', autouse=True)
    def setup_method(self, request):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        torch.manual_seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)

        cls = request.cls
        cls.pg = ProcessGroupCollection.use_mpu_process_groups()

        yield
        Utils.destroy_model_parallel()

    def test_rope_fusion_forward_backward_parity(self):
        """Fused RoPE forward/backward succeeds and matches the unfused path."""
        seq_len = 128
        batch_size = 2

        torch.manual_seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)
        fused_config = _make_config(apply_rope_fusion=True)
        attn_fused = _build_attention(fused_config, layer_number=4, pg_collection=self.pg).cuda()
        attn_fused.train()

        torch.manual_seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)
        unfused_config = _make_config(apply_rope_fusion=False)
        attn_unfused = _build_attention(
            unfused_config, layer_number=4, pg_collection=self.pg
        ).cuda()
        attn_unfused.train()

        hidden = torch.randn(
            seq_len, batch_size, fused_config.hidden_size, dtype=torch.bfloat16
        ).cuda()

        out_fused, _ = attn_fused(hidden_states=hidden, attention_mask=None)
        out_unfused, _ = attn_unfused(hidden_states=hidden, attention_mask=None)

        assert out_fused.shape == (seq_len, batch_size, fused_config.hidden_size)
        assert torch.isfinite(out_fused).all()
        # Production code forces ``mscale=1.0`` (DSv4 contract) in both
        # fused and unfused paths, so the only residual is bf16 noise from
        # the fused Triton kernel's different accumulation order vs the
        # PyTorch eager ops. The residual concentrates at output positions
        # whose values are near zero (sign flips on tiny magnitudes drive
        # the worst-case max-abs-diff).
        torch.testing.assert_close(out_fused, out_unfused, atol=3e-2, rtol=3e-2)

        hidden_fused = hidden.detach().clone().requires_grad_(True)
        hidden_unfused = hidden.detach().clone().requires_grad_(True)

        attn_fused(hidden_states=hidden_fused, attention_mask=None)[0].sum().backward()
        attn_unfused(hidden_states=hidden_unfused, attention_mask=None)[0].sum().backward()

        assert hidden_fused.grad is not None
        for name, param in attn_fused.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for parameter {name}"


def _load_tp1_parameters_into_tpn(module, tp1_parameters, tp_rank, tp_size):
    """Load a TP1 parameter snapshot into a TP-sharded module."""
    with torch.no_grad():
        for name, param in module.named_parameters():
            source = tp1_parameters[name].to(device=param.device, dtype=param.dtype)
            if tuple(source.shape) == tuple(param.shape):
                param.copy_(source)
                continue

            assert getattr(param, 'tensor_model_parallel', False), (
                f"{name}: shape changed from {tuple(source.shape)} to {tuple(param.shape)} "
                "without tensor_model_parallel metadata"
            )
            partition_dim = getattr(param, 'partition_dim')
            local_width = param.shape[partition_dim]
            assert source.shape[partition_dim] == local_width * tp_size
            param.copy_(source.narrow(partition_dim, tp_rank * local_width, local_width))


def _gather_sequence(tensor, tp_group):
    gathered = torch.empty(
        tensor.shape[0] * tp_group.size(),
        *tensor.shape[1:],
        device=tensor.device,
        dtype=tensor.dtype,
    )
    torch.distributed.all_gather_into_tensor(gathered, tensor.contiguous(), group=tp_group)
    return gathered


def _relative_l2_error(actual, expected):
    numerator = torch.linalg.vector_norm(actual.float() - expected.float())
    denominator = torch.linalg.vector_norm(expected.float()).clamp_min(1e-12)
    return (numerator / denominator).item()


def _is_first_data_parallel_replica():
    """Keep distributed diagnostics on one DP replica."""
    return parallel_state.get_data_parallel_rank(with_context_parallel=True) == 0


def _format_grad_metrics(actual, reference):
    """Compact precision metrics used only in failure messages."""
    actual_flat = actual.float().reshape(-1)
    reference_flat = reference.float().reshape(-1)
    reference_norm = torch.linalg.vector_norm(reference_flat).clamp_min(1e-12)
    cosine_similarity = F.cosine_similarity(actual_flat, reference_flat, dim=0).item()
    least_squares_scale = (
        torch.dot(actual_flat, reference_flat)
        / torch.dot(reference_flat, reference_flat).clamp_min(1e-12)
    ).item()
    norm_ratio = (torch.linalg.vector_norm(actual_flat) / reference_norm).item()
    relative_l2 = (
        torch.linalg.vector_norm(actual_flat - reference_flat) / reference_norm
    ).item()
    max_abs = (actual_flat - reference_flat).abs().max().item()
    minimum_tolerance = (
        (actual_flat - reference_flat).abs() / (1.0 + reference_flat.abs())
    ).max().item()
    return (
        f"cosine_similarity={cosine_similarity:.9f}; "
        f"least_squares_scale={least_squares_scale:.9f}; "
        f"norm_ratio={norm_ratio:.9f}; relative_l2={relative_l2:.9e}; "
        f"max_abs={max_abs:.9e}; minimum_atol_eq_rtol={minimum_tolerance:.9e}"
    )


def _tracked_indexer_loss(layer_number):
    from megatron.core.transformer.experimental_attention_variant.dsa import (
        DSAIndexerLossLoggingHelper,
    )

    return DSAIndexerLossLoggingHelper.tracker['values'][layer_number - 1].detach().clone()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not HAVE_TE, reason="transformer_engine not available")
@pytest.mark.experimental
@pytest.mark.parametrize(
    "apply_dsa_kernel_fusion", [False, True], ids=["unfused-dsa", "fused-dsa"]
)
@pytest.mark.parametrize(
    ("tp", "sp"),
    [
        (2, False),  # TP w/o SP
        (2, True),  # TP w/ SP
        (4, False),  # TP w/o SP
        (4, True),  # TP w/ SP
        (8, False),  # TP w/o SP
        (8, True),  # TP w/ SP
    ],
)
def test_parallel_dsv4_hybrid_sparse_attention_correctness(
    tmp_path_dist_ckpt, tp, sp, apply_dsa_kernel_fusion
):
    """A small GPT's DSv4 attention must match after TP checkpoint resharding.

    This follows ``test_parallel_multi_latent_attention_correctness``: build a
    TP1 GPT model, save its distributed checkpoint, rebuild the same GPT under
    TP, load the checkpoint, and compare attention forward and backward results.
    Both the fused and unfused DSA implementations pass through the complete
    hybrid-attention path, including QKV projections, SP gather, CSA, inverse
    RoPE, grouped output projection, and the row-parallel output projection.
    """
    if apply_dsa_kernel_fusion and torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("Fused DSA TP correctness currently requires the SM90 Triton backend")

    from megatron.core.transformer.experimental_attention_variant.dsa import (
        DSAIndexerLossLoggingHelper,
    )

    seed = 123
    sequence_length = 64
    micro_batch_size = 2
    hidden_size = 128
    layer_number = 1

    def initialize_gpt_model(
        config, pre_process=True, post_process=True, vp_stage=None, pg_collection=None
    ):
        layer_spec = get_transformer_block_with_experimental_attention_variant_spec(
            config=config, vp_stage=None, pp_rank=None
        )
        return GPTModel(
            config=config,
            transformer_layer_spec=layer_spec,
            vocab_size=128,
            max_sequence_length=sequence_length,
            pre_process=pre_process,
            post_process=post_process,
            vp_stage=vp_stage,
            pg_collection=pg_collection,
        )

    transformer_config = _make_config(
        num_layers=1,
        hidden_size=hidden_size,
        num_attention_heads=8,
        v_head_dim=32,
        qk_pos_emb_head_dim=16,
        q_lora_rank=32,
        o_groups=8,
        o_lora_rank=32,
        csa_compress_ratios=[4],
        csa_window_size=8,
        dsa_indexer_n_heads=4,
        dsa_indexer_head_dim=32,
        dsa_indexer_topk=8,
        apply_rope_fusion=True,
        apply_dsa_kernel_fusion=apply_dsa_kernel_fusion,
        dsa_indexer_loss_coeff=1.0,
        dsa_indexer_use_sparse_loss=True,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        ffn_hidden_size=256,
        normalization="RMSNorm",
        transformer_impl="transformer_engine",
    )
    compress_ratio = transformer_config.csa_compress_ratios[layer_number - 1]
    assert compress_ratio * transformer_config.dsa_indexer_topk <= sequence_length, (
        "This correctness test must stay in the normal training regime where "
        "compress_ratio * topk <= sequence_length; the sparse-attention implementation "
        "does not handle the degenerate short-sequence boundary."
    )

    try:
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        torch.manual_seed(seed)
        model_parallel_cuda_manual_seed(seed)
        input_hidden_states = (
            torch.rand((sequence_length, micro_batch_size, hidden_size), device="cuda")
            .bfloat16()
            .requires_grad_(True)
        )

        with TempNamedDir(tmp_path_dist_ckpt / "test_parallel_dsv4", sync=True) as ckpt_dir:
            mock_args = parse_args(ignore_unknown_args=True)
            set_args(mock_args)
            init_basic_mock_args(mock_args, 1, 1, bf16=True)
            mock_args.context_parallel_size = 1
            mock_args.sequence_parallel = False
            gpt_model = unwrap_model(
                get_model(initialize_gpt_model, config=transformer_config)
            )

            init_checkpointing_mock_args(mock_args, ckpt_dir, False)
            mock_args.no_save_optim = True
            mock_args.no_save_rng = True
            mock_args.no_load_optim = True
            mock_args.no_load_rng = True
            save_checkpoint(10, gpt_model, None, None, 0)

            attention = gpt_model[0].decoder.layers[0].self_attention
            assert (
                attention.core_attention.apply_dsa_kernel_fusion
                is apply_dsa_kernel_fusion
            )
            DSAIndexerLossLoggingHelper.clean_loss_in_tracker()
            output_baseline, bias_baseline = attention(
                input_hidden_states, attention_mask=None
            )
            indexer_loss_baseline = _tracked_indexer_loss(layer_number)
            output_baseline.sum().backward()
            input_grad_baseline = input_hidden_states.grad.detach()
            output_baseline = output_baseline.detach()

            Utils.destroy_model_parallel()
            Utils.initialize_model_parallel(
                tensor_model_parallel_size=tp, pipeline_model_parallel_size=1
            )
            torch.manual_seed(seed)
            model_parallel_cuda_manual_seed(seed)
            transformer_config.tensor_model_parallel_size = tp
            transformer_config.sequence_parallel = sp
            init_basic_mock_args(mock_args, tp, 1, bf16=True)
            mock_args.context_parallel_size = 1
            mock_args.sequence_parallel = sp
            pg_collection = ProcessGroupCollection.use_mpu_process_groups()
            pg_collection.embd = parallel_state.get_embedding_group()
            gpt_model = unwrap_model(
                get_model(
                    initialize_gpt_model,
                    config=transformer_config,
                    pg_collection=pg_collection,
                )
            )
            with mock.patch("megatron.training.checkpointing.check_checkpoint_args"):
                with mock.patch("megatron.training.checkpointing.update_num_microbatches"):
                    load_checkpoint(gpt_model, None, None)

            tp_rank = parallel_state.get_tensor_model_parallel_rank()

            def get_tensor_on_this_rank(tensor):
                if tp > 1 and sp:
                    sequence_per_tp_rank = sequence_length // tp
                    tensor = tensor[
                        tp_rank * sequence_per_tp_rank : (tp_rank + 1) * sequence_per_tp_rank
                    ]
                return tensor

            input_parallel = (
                get_tensor_on_this_rank(input_hidden_states).detach().requires_grad_(True)
            )
            parallel_attention = gpt_model[0].decoder.layers[0].self_attention
            assert (
                parallel_attention.core_attention.apply_dsa_kernel_fusion
                is apply_dsa_kernel_fusion
            )
            DSAIndexerLossLoggingHelper.clean_loss_in_tracker()
            output_parallel, bias_parallel = parallel_attention(
                input_parallel, attention_mask=None
            )
            indexer_loss_parallel = _tracked_indexer_loss(layer_number)
            output_parallel.sum().backward()
            input_grad_parallel = input_parallel.grad.detach()

            output_baseline = get_tensor_on_this_rank(output_baseline)
            input_grad_baseline = get_tensor_on_this_rank(input_grad_baseline)
            torch.testing.assert_close(
                indexer_loss_parallel, indexer_loss_baseline, rtol=2e-4, atol=2e-4
            )
            assert bias_baseline is None
            assert bias_parallel is None

            for name, tensor in (
                ("output_baseline", output_baseline),
                ("output_parallel", output_parallel),
                ("input_grad_baseline", input_grad_baseline),
                ("input_grad_parallel", input_grad_parallel),
            ):
                assert torch.isfinite(tensor).all(), f"{name} contains NaN or Inf"

            # Fixed tolerance contract (see docs/diagnostics notes); the unified
            # precision metrics are only reported when an assertion fails.
            # TP1 and TPN use different Triton launch shapes in the fused path.
            # Its direct TP tests permit one BF16 output ULP (up to 0.015625),
            # while the unfused hybrid path retains its tighter contract.
            atol = rtol = 2e-2 if apply_dsa_kernel_fusion else 5e-3
            rank = torch.distributed.get_rank()
            should_report = _is_first_data_parallel_replica() and (
                sp or parallel_state.get_tensor_model_parallel_rank() == 0
            )
            try:
                torch.testing.assert_close(
                    output_parallel,
                    output_baseline,
                    atol=atol,
                    rtol=rtol,
                    msg=lambda msg: f"Mismatch in output_hidden_states: {msg}",
                )
            except AssertionError as error:
                if should_report:
                    print(
                        f"[rank{rank}] output_hidden_states mismatch at atol=rtol={atol:.0e}: "
                        f"{_format_grad_metrics(output_parallel, output_baseline)}"
                    )
                raise

            try:
                if apply_dsa_kernel_fusion:
                    input_grad_relative_l2 = _relative_l2_error(
                        input_grad_parallel, input_grad_baseline
                    )
                    input_grad_cosine = F.cosine_similarity(
                        input_grad_parallel.float().reshape(1, -1),
                        input_grad_baseline.float().reshape(1, -1),
                    ).item()
                    assert input_grad_relative_l2 < 7e-2 and input_grad_cosine > 0.995, (
                        f"fused input_grad mismatch: relative_l2={input_grad_relative_l2:.6g}, "
                        f"cosine={input_grad_cosine:.9g}"
                    )
                else:
                    torch.testing.assert_close(
                        input_grad_parallel,
                        input_grad_baseline,
                        atol=atol,
                        rtol=rtol,
                        msg=lambda msg: f"Mismatch in input_grad: {msg}",
                    )
            except AssertionError as initial_error:
                if should_report:
                    print(
                        f"[rank{rank}] input_grad mismatch at atol=rtol={atol:.0e}: "
                        f"{_format_grad_metrics(input_grad_parallel, input_grad_baseline)}"
                    )
                raise AssertionError(
                    f"input_grad mismatch at atol=rtol={atol:.0e}; "
                    f"{_format_grad_metrics(input_grad_parallel, input_grad_baseline)}"
                ) from initial_error
    finally:
        DSAIndexerLossLoggingHelper.clean_loss_in_tracker()
        Utils.destroy_model_parallel()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not HAVE_TE, reason="transformer_engine not available")
@pytest.mark.parametrize("tp", (2, 4, 8), ids=lambda tp: f"tp{tp}")
def test_dsv4_tp_full_sequence_duplicated_param_grads_match_tp1(tp):
    """TP+SP duplicated CSA/indexer parameter gradients must match TP1.

    In the fused tp_sp path the sequence-parallel gather feeds *complete*
    sequences to the duplicated CSA/indexer operators, so they must not use
    TE's sequence-parallel parameter-gradient reduction. The main compressor
    is nevertheless consumed by TP-local query heads: its output mapping must
    SUM those local-head activation gradients during backward before they
    enter the replicated compressor. Indexer-private operators already compute
    their replicated loss from the complete sequence and need no such mapping.
    """
    seq_len, batch_size, layer_number = 64, 2, 2
    common = dict(
        apply_rope_fusion=True,
        apply_dsa_kernel_fusion=False,
        csa_compress_ratios=[0, 4, 0, 0],
        qk_layernorm=True,
        dsa_indexer_loss_coeff=1.0,
        dsa_indexer_use_sparse_loss=True,
        attention_dropout=0.0,
        hidden_dropout=0.0,
    )
    target_names = [
        "core_attention.compressor.linear_wkv.weight",
        "core_attention.compressor.linear_wgate.weight",
        "core_attention.compressor.norm.weight",
        "core_attention.compressor.ape",
        "core_attention.indexer.linear_wq_b.weight",
        "core_attention.indexer.linear_weights_proj.weight",
        "core_attention.indexer.compressor.linear_wkv.weight",
        "core_attention.indexer.compressor.linear_wgate.weight",
        "core_attention.indexer.compressor.norm.weight",
        "core_attention.indexer.compressor.ape",
        "kv_layernorm.weight",
    ]

    try:
        # Each process independently computes the same TP1 reference.
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        pg_tp1 = ProcessGroupCollection.use_mpu_process_groups()
        torch.manual_seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)
        config_tp1 = _make_config(
            tensor_model_parallel_size=1, sequence_parallel=False, **common
        )
        attn_tp1 = _build_attention(config_tp1, layer_number, pg_tp1).cuda().train()
        tp1_parameters = {
            name: param.detach().cpu().clone() for name, param in attn_tp1.named_parameters()
        }

        torch.manual_seed(_SEED + 1)
        hidden_full = torch.randn(
            seq_len,
            batch_size,
            config_tp1.hidden_size,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        output_tp1, _ = attn_tp1(hidden_states=hidden_full, attention_mask=None)
        torch.manual_seed(_SEED + 2)
        output_grad = torch.randn_like(output_tp1)
        output_tp1.backward(output_grad)
        param_grads_tp1 = {
            name: param.grad.detach().cpu().clone()
            for name, param in attn_tp1.named_parameters()
            if param.grad is not None
        }
        del attn_tp1
        Utils.destroy_model_parallel()

        # Rebuild with TP+SP and load exact slices of the TP1 global weights.
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp, pipeline_model_parallel_size=1
        )
        pg_tp = ProcessGroupCollection.use_mpu_process_groups()
        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        torch.manual_seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)
        config_tp = _make_config(
            tensor_model_parallel_size=tp, sequence_parallel=True, **common
        )
        attn_tp = _build_attention(config_tp, layer_number, pg_tp).cuda().train()
        _load_tp1_parameters_into_tpn(attn_tp, tp1_parameters, tp_rank, tp)
        main_compressor = attn_tp.core_attention.compressor
        assert main_compressor.reduce_output_grad_across_tp, (
            "The main CSA compressor must reduce its output gradient across TP"
        )
        assert main_compressor.pg_collection.tp.size() == tp

        seq_per_rank = seq_len // tp
        seq_slice = slice(tp_rank * seq_per_rank, (tp_rank + 1) * seq_per_rank)
        hidden_local = hidden_full.detach()[seq_slice].clone().requires_grad_(True)
        csa_input_sequence_lengths = []

        def capture_csa_input_sequence_lengths(module, args, kwargs):
            csa_input_sequence_lengths.append(
                {
                    "query": args[0].shape[0],
                    "x": kwargs["x"].shape[0],
                    "qr": kwargs["qr"].shape[0],
                }
            )

        hook = attn_tp.core_attention.register_forward_pre_hook(
            capture_csa_input_sequence_lengths, with_kwargs=True
        )
        try:
            output_local, _ = attn_tp(hidden_states=hidden_local, attention_mask=None)
            output_local.backward(output_grad[seq_slice])
        finally:
            hook.remove()
        assert csa_input_sequence_lengths == [
            {"query": seq_len, "x": seq_len, "qr": seq_len}
        ], (
            "DSv4HybridSelfAttention must gather the SP sequence before entering CSA; "
            f"got {csa_input_sequence_lengths} from local input length {seq_per_rank}"
        )

        # Emulate the TP part of finalize_model_grads. Full-sequence CSA/indexer
        # parameters intentionally carry sequence_parallel=False; the main
        # compressor's local-head contributions must already have been summed
        # by its output activation-gradient mapping above.
        for param in attn_tp.parameters():
            if param.grad is not None and getattr(param, "sequence_parallel", False):
                torch.distributed.all_reduce(param.grad, group=pg_tp.tp)

        for name in target_names:
            assert name in param_grads_tp1, f"TP1 reference has no gradient for {name}"
            param = dict(attn_tp.named_parameters())[name]
            assert param.grad is not None, f"TP{tp} has no gradient for {name}"
            assert tuple(param.grad.shape) == tuple(param_grads_tp1[name].shape), name
            reference = param_grads_tp1[name].to(device=param.device, dtype=param.grad.dtype)
            # A double-gradient bug shows up as ~2x here and fails loudly.
            try:
                torch.testing.assert_close(
                    param.grad,
                    reference,
                    rtol=3e-2,
                    atol=3e-2,
                    msg=lambda msg, name=name: f"Mismatch in parameter gradient {name}: {msg}",
                )
                relative_l2 = _relative_l2_error(param.grad, reference)
                assert relative_l2 < 2e-2, f"{name}: relative_l2={relative_l2:.9e}"
            except AssertionError as error:
                rank = torch.distributed.get_rank()
                if _is_first_data_parallel_replica():
                    print(
                        f"[rank{rank}] TP{tp} parameter-gradient mismatch for {name}: "
                        f"{_format_grad_metrics(param.grad, reference)}"
                    )
                raise AssertionError(
                    f"TP{tp} parameter-gradient mismatch for {name}; "
                    f"{_format_grad_metrics(param.grad, reference)}"
                ) from error
    finally:
        Utils.destroy_model_parallel()
