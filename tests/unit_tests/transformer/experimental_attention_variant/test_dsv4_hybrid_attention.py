# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import os
from copy import copy
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

import megatron.core.parallel_state as parallel_state
from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_config import MLATransformerConfig
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
# TP communication adapter tests
# ===========================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestDSv4TPRopeExchange:
    """Verify SP gather and non-SP replicated-KV backward contracts."""

    @pytest.fixture(scope='class', autouse=True)
    def setup_method(self):
        if not torch.distributed.is_available() or int(os.environ.get('WORLD_SIZE', '1')) < 2:
            pytest.skip("requires torchrun with at least two ranks")
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=2, pipeline_model_parallel_size=1
        )
        yield
        Utils.destroy_model_parallel()

    @staticmethod
    def _fake_fused_rope():
        class FakeFusedRope:
            @staticmethod
            def forward(ctx, q, *args):
                return q * 2

            @staticmethod
            def backward(ctx, grad_query):
                return (grad_query * 2,)

        return FakeFusedRope

    def test_sp_field_backward_policies(self):
        from megatron.core.transformer.experimental_attention_variant import (
            deepseek_v4_hybrid_attention as dsv4_attention,
        )

        pg = ProcessGroupCollection.use_mpu_process_groups()
        tp_rank = pg.tp.rank()
        device = torch.device('cuda', torch.cuda.current_device())
        q = torch.ones(4, 1, 3, device=device, requires_grad=True)
        hidden = torch.full((2, 1, 2), float(tp_rank + 1), device=device, requires_grad=True)
        kv = torch.ones(2, 1, 3, device=device, requires_grad=True)
        q_compressed = torch.ones(2, 1, 1, device=device, requires_grad=True)
        fields = (
            dsv4_attention._DSv4TPField(
                "hidden_states", hidden, dsv4_attention._DSv4TPBackwardPolicy.SCATTER
            ),
            dsv4_attention._DSv4TPField(
                "kv", kv, dsv4_attention._DSv4TPBackwardPolicy.REDUCE_SCATTER
            ),
            dsv4_attention._DSv4TPField(
                "q_compressed",
                q_compressed,
                dsv4_attention._DSv4TPBackwardPolicy.NO_GRAD,
            ),
        )

        with patch.object(
            dsv4_attention, '_FusedMLARoPEInplace', self._fake_fused_rope()
        ):
            query, exchanged = dsv4_attention._dsv4_tp_rope_exchange(
                q,
                fields,
                torch.empty(0, device=device),
                torch.empty(0, device=device),
                0,
                0,
                torch.tensor([0, 4], dtype=torch.int32, device=device),
                0,
                1,
                pg.tp,
                gather_sequence=True,
                is_thd=True,
                apply_fused_rope=True,
            )
            assert exchanged["hidden_states"].shape[0] == 4
            assert not exchanged["q_compressed"].requires_grad
            (query.sum() + exchanged["hidden_states"].sum() + exchanged["kv"].sum()).backward()

        torch.testing.assert_close(q.grad, torch.full_like(q, 2))
        torch.testing.assert_close(hidden.grad, torch.ones_like(hidden))
        torch.testing.assert_close(kv.grad, torch.full_like(kv, 2))
        torch.testing.assert_close(q_compressed.grad, torch.zeros_like(q_compressed))

    def test_non_sp_kv_backward_all_reduce(self):
        from megatron.core.transformer.experimental_attention_variant import (
            deepseek_v4_hybrid_attention as dsv4_attention,
        )

        pg = ProcessGroupCollection.use_mpu_process_groups()
        device = torch.device('cuda', torch.cuda.current_device())
        q = torch.ones(2, 1, 1, 3, device=device, requires_grad=True)
        kv = torch.ones(2, 1, 3, device=device, requires_grad=True)
        field = dsv4_attention._DSv4TPField(
            "kv", kv, dsv4_attention._DSv4TPBackwardPolicy.ALL_REDUCE
        )

        with patch.object(
            dsv4_attention, '_FusedMLARoPEInplace', self._fake_fused_rope()
        ):
            query, exchanged = dsv4_attention._dsv4_tp_rope_exchange(
                q,
                (field,),
                torch.empty(0, device=device),
                torch.empty(0, device=device),
                0,
                0,
                None,
                0,
                1,
                pg.tp,
                gather_sequence=False,
                is_thd=False,
                apply_fused_rope=True,
            )
            rank_scale = float(pg.tp.rank() + 1)
            (query.sum() + exchanged["kv"].sum() * rank_scale).backward()

        torch.testing.assert_close(q.grad, torch.full_like(q, 2))
        torch.testing.assert_close(kv.grad, torch.full_like(kv, 3))

    def test_thd_cp_fused_rope_uses_contiguous_position_ids(self):
        from megatron.core.transformer.experimental_attention_variant import (
            deepseek_v4_hybrid_attention as dsv4_attention,
        )

        pg = ProcessGroupCollection.use_mpu_process_groups()
        device = torch.device('cuda', torch.cuda.current_device())
        captured = {}

        class CaptureFusedRope:
            @staticmethod
            def forward(ctx, q, *args):
                captured["position_ids"] = args[-1]
                return q

            @staticmethod
            def backward(ctx, grad_query):
                return (grad_query,)

        q = torch.ones(4, 1, 3, device=device, requires_grad=True)
        kv = torch.ones(4, 1, 2, device=device, requires_grad=True)
        field = dsv4_attention._DSv4TPField(
            "kv", kv, dsv4_attention._DSv4TPBackwardPolicy.ALL_REDUCE
        )
        cu_seqlens_q = torch.tensor([0, 3, 8], dtype=torch.int32, device=device)

        with patch.object(dsv4_attention, '_FusedMLARoPEInplace', CaptureFusedRope):
            dsv4_attention._dsv4_tp_rope_exchange(
                q,
                (field,),
                torch.empty(0, device=device),
                torch.empty(0, device=device),
                0,
                0,
                cu_seqlens_q,
                1,
                2,
                pg.tp,
                gather_sequence=False,
                is_thd=True,
                apply_fused_rope=True,
            )

        torch.testing.assert_close(
            captured["position_ids"],
            torch.tensor([1, 2, 3, 4], dtype=torch.int32, device=device),
        )


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

    def test_q_head_dim_equals_v_head_dim(self):
        """q_head_dim must equal v_head_dim for DSv4 hybrid."""
        torch.manual_seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)

        config = _make_config()
        pg = ProcessGroupCollection.use_mpu_process_groups()
        attn = _build_attention(config, layer_number=1, pg_collection=pg)

        assert attn.q_head_dim == config.v_head_dim

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

        q, k, v, q_compressed, kv_compressed = attn.get_query_key_value_tensors(hidden)

        n_heads = self.config.num_attention_heads
        v_dim = self.config.v_head_dim

        assert q.shape == (seq_len, batch_size, n_heads, v_dim)
        # key and value are single-head (MQA-style) with an extra head dim
        assert k.shape[-1] == v_dim
        assert v.shape[-1] == v_dim

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


# ===========================================================================
# THD packed-sequence end-to-end
# ===========================================================================
#
# Closes the highest-level integration gap: even though the CSA THD path
# is independently tested in test_attention_variant_csa.py
# (TestCompressedSparseAttentionThd), the DSv4HybridSelfAttention module
# adds its own THD-aware glue around CSA — packed_seq_params propagation
# through get_query_key_value_tensors, the output reshape at line ~290
# (``core_attn_out.reshape(total, 1, -1)`` to recover the 3-D contract),
# and the inverse-RoPE call with cu_seqlens. These tests verify the full
# DSv4Hybrid forward/backward works end-to-end for each ``compress_ratio``.


from megatron.core.packed_seq_params import PackedSeqParams  # noqa: E402


def _make_thd_packed_seq_params(seg_lens, device='cuda'):
    """Build ``PackedSeqParams(qkv_format='thd', ...)`` for self-attention
    (``cu_seqlens_q == cu_seqlens_kv``) from a list of per-segment lengths.
    """
    cu_seqlens = torch.tensor(
        [0] + list(torch.tensor(seg_lens, dtype=torch.int64).cumsum(0).tolist()),
        dtype=torch.int32,
        device=device,
    )
    max_len = int(max(seg_lens)) if seg_lens else 0
    return PackedSeqParams(
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_q_padded=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        cu_seqlens_kv_padded=cu_seqlens,
        max_seqlen_q=max_len,
        max_seqlen_kv=max_len,
        qkv_format='thd',
    )


_DSV4_TP_ATOL = 5e-3
_DSV4_TP_RTOL = 5e-3


def _copy_tp_parameters_to_tp1_reference(tp_module, ref_module, tp_group):
    """Gather TP shards and load the equivalent parameters into a TP1 module."""
    tp_size = tp_group.size()
    tp_params = dict(tp_module.named_parameters())
    for name, ref_param in ref_module.named_parameters():
        assert name in tp_params, f"Missing TP parameter for TP1 reference parameter {name}"
        tp_param = tp_params[name]
        if tp_param.shape == ref_param.shape:
            ref_param.data.copy_(tp_param.data)
            continue

        partition_dims = [
            dim
            for dim in range(tp_param.ndim)
            if ref_param.size(dim) == tp_param.size(dim) * tp_size
            and all(
                ref_param.size(other_dim) == tp_param.size(other_dim)
                for other_dim in range(tp_param.ndim)
                if other_dim != dim
            )
        ]
        assert len(partition_dims) == 1, (
            f"Cannot infer TP partition dimension for {name}: "
            f"local={tuple(tp_param.shape)}, full={tuple(ref_param.shape)}"
        )
        shards = [torch.empty_like(tp_param) for _ in range(tp_size)]
        torch.distributed.all_gather(shards, tp_param.data, group=tp_group)
        ref_param.data.copy_(torch.cat(shards, dim=partition_dims[0]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not HAVE_TE, reason="transformer_engine not available")
class TestDSv4HybridAttentionThd:
    """End-to-end THD forward/backward of :class:`DSv4HybridSelfAttention`
    across all configured ``compress_ratio`` values (0/4/128).

    Each test runs a multi-segment THD batch through the full
    ``attn(hidden_states, packed_seq_params=...)`` pipeline and verifies:

      * Output shape ``(total_tokens, 1, hidden_size)`` — the layer
        re-adds the dummy ``b=1`` axis at line ~293 of
        ``deepseek_v4_hybrid_attention.py``.
      * No NaN.
      * (Backward test) grads flow on ``hidden_states`` and every
        learnable parameter.

    This is the highest-level integration test for THD; lower-level
    parity vs SBHD is established by ``TestCompressedSparseAttentionThd``
    in ``test_attention_variant_csa.py``.
    """

    @pytest.fixture(scope='class', autouse=True)
    def setup_method(self, request):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        torch.manual_seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)

        cls = request.cls
        # ``csa_compress_ratios=[0, 4, 128, 4]`` → layer_number ∈ {1,2,3,4}
        # cover all three CSA ratios (0 = window-only; 4 = full
        # indexer+compressed; 128 = compressor-only, no indexer).
        cls.config = _make_config(dsa_indexer_loss_coeff=1.0)
        cls.pg = ProcessGroupCollection.use_mpu_process_groups()

        yield
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize(
        "layer_number",
        [1, 2, 3, 4],
        ids=[
            "ratio_0_window_only",  # layer 1 → ratio=0
            "ratio_4_with_indexer",  # layer 2 → ratio=4
            "ratio_128_compressor_only",  # layer 3 → ratio=128
            "ratio_4_with_indexer_alt",  # layer 4 → ratio=4
        ],
    )
    def test_thd_forward_output_shape(self, layer_number):
        """THD forward through DSv4HybridSelfAttention produces
        ``(total_tokens, 1, hidden_size)`` output, no NaN, for every
        configured ``compress_ratio``.
        """
        seg_lens = [128, 96, 64]  # multi-segment, varied lengths
        total = sum(seg_lens)

        torch.manual_seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)

        attn = _build_attention(
            self.config, layer_number=layer_number, pg_collection=self.pg
        ).cuda()
        attn.eval()

        # THD hidden_states shape is ``(total_tokens, 1, hidden_size)``
        # per the DSv4 hybrid contract.
        hidden = torch.randn(total, 1, self.config.hidden_size, dtype=torch.bfloat16, device='cuda')
        packed = _make_thd_packed_seq_params(seg_lens)

        with torch.no_grad():
            output, _bias = attn(
                hidden_states=hidden, attention_mask=None, packed_seq_params=packed
            )

        assert output.shape == (total, 1, self.config.hidden_size), (
            f"layer {layer_number}: shape {tuple(output.shape)} != "
            f"expected {(total, 1, self.config.hidden_size)}"
        )
        assert output.dtype == torch.bfloat16
        assert not torch.isnan(output).any(), f"layer {layer_number}: NaN in THD forward output"

    @pytest.mark.parametrize(
        "layer_number",
        [1, 2],  # ratio=0 (window-only) and ratio=4 (full indexer pipeline)
        ids=["ratio_0_window_only", "ratio_4_with_indexer"],
    )
    def test_thd_backward_gradient_flow(self, layer_number):
        """THD backward produces grads on ``hidden_states`` and every
        learnable parameter (covers the full indexer-loss path in Path
        B THD when ``layer_number=2`` triggers ratio=4).
        """
        # Both packed segments contain a complete ratio-128 compression group;
        # the SP midpoint still falls inside the first segment.
        seg_lens = [160, 128]
        total = sum(seg_lens)

        torch.manual_seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)

        attn = _build_attention(
            self.config, layer_number=layer_number, pg_collection=self.pg
        ).cuda()
        attn.train()

        hidden = torch.randn(
            total, 1, self.config.hidden_size, dtype=torch.bfloat16, device='cuda'
        ).requires_grad_(True)
        packed = _make_thd_packed_seq_params(seg_lens)

        output, _bias = attn(hidden_states=hidden, attention_mask=None, packed_seq_params=packed)
        output.sum().backward()

        assert hidden.grad is not None, "no grad on hidden_states"
        assert not torch.isnan(hidden.grad).any()
        for name, param in attn.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"no grad on {name}"
                assert not torch.isnan(param.grad).any(), f"NaN grad on {name}"

    def test_thd_single_segment_matches_sbhd_b1(self):
        """B=1 single-segment THD output matches the SBHD-b=1 output on
        identical hidden states (sanity check that the DSv4Hybrid
        THD-vs-SBHD glue doesn't silently change the math).

        Uses ``layer_number=1`` (ratio=0, window-only) for determinism —
        no indexer top-K tie-breaking nondeterminism.
        """
        layer_number = 1  # ratio=0 → window-only path, no cuDNN topk
        sq = 128

        torch.manual_seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)

        attn = _build_attention(
            self.config, layer_number=layer_number, pg_collection=self.pg
        ).cuda()
        attn.eval()

        hidden = torch.randn(sq, 1, self.config.hidden_size, dtype=torch.bfloat16, device='cuda')

        with torch.no_grad():
            out_sbhd, _ = attn(hidden_states=hidden, attention_mask=None, packed_seq_params=None)
            packed = _make_thd_packed_seq_params([sq])
            out_thd, _ = attn(hidden_states=hidden, attention_mask=None, packed_seq_params=packed)

        assert out_sbhd.shape == out_thd.shape
        # Generous tol: the full DSv4Hybrid forward chains many bf16 ops
        # (QKV down/up proj, RoPE, attn, output proj). We're testing for
        # no plumbing bug, not bit-exactness.
        assert torch.allclose(out_sbhd.float(), out_thd.float(), atol=5e-2, rtol=5e-2), (
            f"B=1 SBHD/THD parity failed: max abs diff = "
            f"{(out_sbhd.float() - out_thd.float()).abs().max().item():.4e}"
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not HAVE_TE, reason="transformer_engine not available")
class TestDSv4HybridAttentionThdTP:
    """Compare CP-disabled THD TP paths with a genuine TP1/SP-off reference."""

    tp_size = 2

    @pytest.fixture(scope='class', autouse=True)
    def setup_method(self, request):
        if not torch.distributed.is_available() or int(os.environ.get('WORLD_SIZE', '1')) < 2:
            pytest.skip("THD TP parity requires torchrun with at least two ranks")

        Utils.initialize_model_parallel(
            tensor_model_parallel_size=self.tp_size,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
        )
        torch.manual_seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)

        cls = request.cls
        cls.tp_rank = parallel_state.get_tensor_model_parallel_rank()
        cls.pg = ProcessGroupCollection.use_mpu_process_groups()

        # Transformer Engine needs a real ProcessGroup to execute its TP1
        # column/row-linear paths. Every distributed rank creates the singleton
        # groups in the same order, then keeps the group containing itself.
        cls.ref_tp_group = None
        for rank in range(torch.distributed.get_world_size()):
            singleton_group = torch.distributed.new_group(ranks=[rank])
            if rank == torch.distributed.get_rank():
                cls.ref_tp_group = singleton_group
        assert cls.ref_tp_group is not None
        cls.ref_pg = copy(cls.pg)
        cls.ref_pg.tp = cls.ref_tp_group

        yield
        torch.distributed.destroy_process_group(cls.ref_tp_group)
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("sequence_parallel", [False, True], ids=["sp_off", "sp_on"])
    @pytest.mark.parametrize("layer_number", [1, 3], ids=["window_only", "compressor"])
    def test_forward_backward_matches_tp1_reference(self, sequence_parallel, layer_number):
        """TP2 output/input-grad parity follows MLA's BF16 5e-3 tolerance."""
        config_tp = _make_config(
            tensor_model_parallel_size=self.tp_size,
            sequence_parallel=sequence_parallel,
            dsa_indexer_loss_coeff=0.0,
            apply_dsa_kernel_fusion=False,
            apply_rope_fusion=True,
        )
        config_ref = _make_config(
            tensor_model_parallel_size=1,
            sequence_parallel=False,
            dsa_indexer_loss_coeff=0.0,
            apply_dsa_kernel_fusion=False,
            apply_rope_fusion=True,
        )
        tp_attn = _build_attention(
            config_tp, layer_number=layer_number, pg_collection=self.pg
        ).cuda()
        ref_attn = _build_attention(
            config_ref, layer_number=layer_number, pg_collection=self.ref_pg
        ).cuda()
        _copy_tp_parameters_to_tp1_reference(tp_attn, ref_attn, self.pg.tp)

        seg_lens = [128, 96]
        total_tokens = sum(seg_lens)
        packed = _make_thd_packed_seq_params(seg_lens)
        rows = torch.arange(total_tokens, dtype=torch.float32, device='cuda')
        full_hidden = (rows.remainder(17) / 17).to(torch.bfloat16).view(-1, 1, 1)
        full_hidden = full_hidden.expand(-1, 1, config_tp.hidden_size).contiguous()

        if sequence_parallel:
            local_tokens = total_tokens // self.tp_size
            local_start = self.tp_rank * local_tokens
            local_hidden_values = full_hidden.narrow(0, local_start, local_tokens)
        else:
            local_start = 0
            local_tokens = total_tokens
            local_hidden_values = full_hidden

        local_hidden = local_hidden_values.detach().clone().requires_grad_(True)
        ref_hidden = full_hidden.detach().clone().requires_grad_(True)
        output, _ = tp_attn(
            hidden_states=local_hidden,
            attention_mask=None,
            packed_seq_params=packed,
        )
        ref_output, _ = ref_attn(
            hidden_states=ref_hidden,
            attention_mask=None,
            packed_seq_params=packed,
        )
        ref_local_output = ref_output.narrow(0, local_start, local_tokens)

        label = f"layer={layer_number}:tp{self.tp_size}:sp={sequence_parallel}:cp1"
        torch.testing.assert_close(
            output,
            ref_local_output,
            atol=_DSV4_TP_ATOL,
            rtol=_DSV4_TP_RTOL,
            msg=lambda msg: f"{label}: output mismatch: {msg}",
        )

        # Slice one deterministic full-sequence gradient with the same SP
        # ownership rule so both executions differentiate the same objective.
        full_grad = ((rows * 3).remainder(23) / 23).to(torch.bfloat16).view(-1, 1, 1)
        full_grad = full_grad.expand_as(ref_output).contiguous()
        output.backward(full_grad.narrow(0, local_start, local_tokens))
        ref_output.backward(full_grad)
        ref_local_hidden_grad = ref_hidden.grad.narrow(0, local_start, local_tokens)
        torch.testing.assert_close(
            local_hidden.grad,
            ref_local_hidden_grad,
            atol=_DSV4_TP_ATOL,
            rtol=_DSV4_TP_RTOL,
            msg=lambda msg: f"{label}: input gradient mismatch: {msg}",
        )
