# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Fork configuration, precision and caller contracts for the upstream mHC implementation."""

from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest
import torch

from megatron.core import tensor_parallel
from megatron.core.models.gpt.fine_grained_callables import build_transformer_layer_callables
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.hyper_connection import (
    HyperConnectionModule,
    learned_output_contract,
    native_h_aggregate,
)
from megatron.core.transformer.module import Float16Module
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import HyperConnectionTransformerLayer
from tests.unit_tests.test_utilities import Utils


def _config(**kwargs):
    return TransformerConfig(
        num_layers=3,
        hidden_size=64,
        num_attention_heads=4,
        use_cpu_initialization=True,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        **kwargs,
    )


def test_fork_configuration_and_exports():
    config = _config(num_residual_streams=2, mhc_fused_backend="native")
    module = HyperConnectionModule(config, 1)
    assert module.mapping_proj.weight.shape == (8, 128)
    assert config.mhc_sinkhorn_iterations == 20
    assert replace(config).num_residual_streams == 2
    assert "mhc_num_residual_streams" not in asdict(config)
    assert tensor_parallel.CheckpointManager.__name__ == "CheckpointManager"
    with pytest.raises(ValueError, match="mhc_fused_backend"):
        _config(mhc_fused_backend="invalid")


@pytest.mark.parametrize("backend", [None, "native", "triton"])
def test_aggregation_matches_fp32_gradient_oracle(backend):
    from megatron.core.fusions.fused_mhc_kernels import fused_h_aggregate

    torch.manual_seed(190)
    x = torch.randn(8, 2, 4, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weights = torch.randn(8, 2, 4, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    actual = (
        native_h_aggregate(x, weights)
        if backend is None
        else fused_h_aggregate(x, weights, backend=backend)
    )
    expected = (x.float() * weights.float().unsqueeze(-1)).sum(-2).to(x.dtype)
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=2e-2)
    grad = torch.randn_like(actual)
    actual.backward(grad)
    expected_wgrad = (grad.float().unsqueeze(-2) * x.detach().float()).sum(-1).to(weights.dtype)
    expected_xgrad = (grad.float().unsqueeze(-2) * weights.detach().float().unsqueeze(-1)).to(
        x.dtype
    )
    torch.testing.assert_close(weights.grad, expected_wgrad, rtol=0, atol=0)
    torch.testing.assert_close(x.grad, expected_xgrad, rtol=0, atol=0)


@pytest.fixture
def model_parallel():
    Utils.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(123)
    yield
    Utils.destroy_model_parallel()


@pytest.mark.parametrize("backend", ["native", "triton"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_mhc_marked_parameters_and_backend_gradients(model_parallel, backend, dtype):
    config = _config(
        enable_hyper_connections=True,
        fp16=dtype == torch.float16,
        bf16=dtype == torch.bfloat16,
        params_dtype=dtype,
    )
    native = Float16Module(config, HyperConnectionModule(config, 1).cuda()).module
    assert set(native.state_dict()) == {
        "mapping_proj.weight",
        "alpha_pre",
        "alpha_post",
        "alpha_res",
        "bias",
    }
    fused_config = replace(config, use_fused_mhc=True, mhc_fused_backend=backend)
    fused = Float16Module(fused_config, HyperConnectionModule(fused_config, 1).cuda()).module
    fused.load_state_dict(native.state_dict())
    assert fused._sinkhorn_op.keywords["backend"] == backend
    hidden = torch.randn(16, 2, 256, device="cuda", dtype=dtype)
    results = []
    for module in (native, fused):
        assert all(p.dtype == torch.float32 for p in module.parameters())
        inputs = hidden.detach().clone().requires_grad_()
        outputs = module(inputs, return_residual=True)
        assert len(module(inputs)) == 3
        assert outputs[0].shape == (16, 2, 64)
        assert outputs[3].shape == hidden.shape
        assert all(output.dtype == dtype for output in outputs)
        sum(t.float().square().mean() for t in outputs).backward()
        results.append((outputs[:3], inputs.grad, dict(module.named_parameters())))
    for actual, expected in zip(results[1][0], results[0][0]):
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-3)
    torch.testing.assert_close(results[1][1], results[0][1], rtol=2e-2, atol=2e-3)
    for name, expected in results[0][2].items():
        assert expected.grad is not None, name
        torch.testing.assert_close(results[1][2][name].grad, expected.grad, rtol=2e-2, atol=2e-4)


def test_retained_input_alias_and_forward_order_replay(model_parallel):
    x = torch.randn(4, 4, device="cuda", requires_grad=True)
    reference = x.detach().clone().requires_grad_()
    (reference.view(2, 8).square() + reference.view(2, 8).sin()).sum().backward()
    calls = []
    manager = tensor_parallel.CheckpointManager()

    def first(value):
        calls.append("first")
        return value.view(2, 8), value.sin()

    def second(view, sine):
        calls.append("second")
        return view.square() + sine.view(2, 8)

    ckpt = tensor_parallel.CheckpointWithoutOutput(ckpt_manager=manager, retain_input_tensors=True)
    view, sine = ckpt.checkpoint(first, x)
    output = tensor_parallel.CheckpointWithoutOutput(ckpt_manager=manager).checkpoint(
        second, view, sine
    )
    loss = output.sum()
    manager.discard_all_outputs_and_register_unified_recompute(loss)
    assert x.untyped_storage().nbytes() > 0
    assert view.untyped_storage().data_ptr() == x.untyped_storage().data_ptr()
    assert sine.untyped_storage().nbytes() == 0
    loss.backward()
    assert calls == ["first", "second", "first", "second"]
    torch.testing.assert_close(x.grad, reference.grad)


def _block(config, moe=False):
    spec = get_gpt_layer_with_transformer_engine_spec(num_experts=2 if moe else None)
    if config.enable_hyper_connections:
        spec.module = HyperConnectionTransformerLayer
        spec.submodules.self_attention_hyper_connection = HyperConnectionModule
        spec.submodules.mlp_hyper_connection = HyperConnectionModule
    return TransformerBlock(config, spec, pre_process=True, post_process=True).cuda()


@pytest.mark.parametrize("group_size", [None, 1, 2])
@pytest.mark.parametrize("fused", [False, True])
def test_block_recompute_outputs_and_gradients(model_parallel, group_size, fused):
    config = _config(enable_hyper_connections=True, use_fused_mhc=fused, mhc_fused_backend="triton")
    reference = _block(config)
    candidate = _block(
        replace(
            config,
            recompute_granularity="selective",
            recompute_modules=["mhc"],
            mhc_recompute_layer_num=group_size,
        )
    )
    candidate.load_state_dict(reference.state_dict())
    hidden = torch.randn(8, 2, 64, device="cuda")
    grad_output = torch.randn_like(hidden)
    mask = torch.zeros(1, 1, 8, 8, device="cuda", dtype=torch.bool)
    results = []
    for block in (reference, candidate):
        x = hidden.detach().clone().requires_grad_()
        output = block(x, mask)
        output.backward(grad_output)
        results.append((output, x.grad, dict(block.named_parameters())))
    torch.testing.assert_close(results[1][0], results[0][0])
    torch.testing.assert_close(results[1][1], results[0][1])
    for name, expected in results[0][2].items():
        assert expected.grad is not None, name
        torch.testing.assert_close(results[1][2][name].grad, expected.grad)


def test_plain_transformer_forward_backward(model_parallel):
    block = _block(_config())
    x = torch.randn(8, 2, 64, device="cuda", requires_grad=True)
    block(x, None).square().sum().backward()
    assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_output_head_preserves_fp32_parameters(model_parallel, dtype):
    config = _config(
        enable_hyper_connections=True,
        fp16=dtype == torch.float16,
        bf16=dtype == torch.bfloat16,
        params_dtype=dtype,
        attention_backend=AttnBackend.local,
    )
    block = TransformerBlock(config, get_gpt_layer_local_spec(enable_hyper_connection=True)).cuda()
    parameters = {
        name: getattr(block, name) for name in ("hc_head_fn", "hc_head_base", "hc_head_scale")
    }
    wrapped = Float16Module(config, block)
    for name, parameter in parameters.items():
        assert getattr(wrapped.module, name) is parameter
        assert parameter.dtype == torch.float32
    # Exercise the actual contraction consumer with the real block's converted parameters.
    x = torch.randn(8, 2, 256, device="cuda", dtype=dtype, requires_grad=True)
    output = learned_output_contract(
        x,
        block.hc_head_fn,
        block.hc_head_base,
        block.hc_head_scale,
        config.num_residual_streams,
        config.layernorm_epsilon,
    )
    assert output.dtype == dtype and output.shape == (8, 2, 64)
    output.backward(torch.randn_like(output))
    assert torch.isfinite(x.grad).all()
    for parameter in parameters.values():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    block.zero_grad(set_to_none=True)
    hidden = torch.randn(8, 2, 64, device="cuda", dtype=dtype, requires_grad=True)
    result = wrapped.module(hidden, None)
    result.backward(torch.randn_like(result))
    assert torch.isfinite(hidden.grad).all()
    for parameter in parameters.values():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()


@pytest.mark.parametrize("moe", [False, True])
def test_existing_fine_grained_callables(model_parallel, moe):
    config = _config(
        enable_hyper_connections=True,
        num_moe_experts=2 if moe else None,
        recompute_granularity="selective",
        recompute_modules=["mhc"],
    )
    layer = _block(config, moe).layers[0]
    functions, _ = build_transformer_layer_callables(layer)
    manager = tensor_parallel.CheckpointManager()
    manager.is_last_layer_in_recompute_block = True
    node = SimpleNamespace(
        layer_state=SimpleNamespace(mhc_recompute_manager=manager),
        chunk_state=SimpleNamespace(
            attention_mask=None,
            rotary_pos_emb=None,
            rotary_pos_cos=None,
            rotary_pos_sin=None,
            rotary_pos_cos_sin=None,
            packed_seq_params=None,
            sequence_len_offset=None,
            input_ids=None,
            model=SimpleNamespace(decoder=SimpleNamespace(final_layernorm=None)),
        ),
        detach=lambda tensor: tensor,
        is_mtp=False,
        is_last_layer=False,
    )
    x = torch.randn(8, 2, 256, device="cuda", requires_grad=True)
    output = functions[0](node, x)
    if moe:
        output = functions[1](node, *output)
        output = functions[2](node, output)
        output = functions[3](node, output)
    else:
        output = functions[2](node, output)
    manager.discard_all_outputs_and_register_unified_recompute(output)
    output.square().sum().backward()
    assert output.shape == x.shape
    assert torch.isfinite(x.grad).all()
    assert layer.mlp_hyper_connection.mapping_proj.weight.grad is not None
