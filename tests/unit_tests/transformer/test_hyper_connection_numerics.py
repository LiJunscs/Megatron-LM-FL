# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Precision and native/fused contracts for mHC used by DSv4."""
from dataclasses import replace

import pytest
import torch

from megatron.core.transformer.hyper_connection import HyperConnectionModule
from megatron.core.transformer.module import Float16Module
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def test_mhc_fp32_mapping_and_native_fused_gradients():
    Utils.initialize_model_parallel()
    try:
        torch.manual_seed(182)
        config = TransformerConfig(
            num_layers=1,
            hidden_size=64,
            num_attention_heads=4,
            use_cpu_initialization=True,
            bf16=True,
            params_dtype=torch.bfloat16,
            enable_hyper_connections=True,
            num_residual_streams=4,
            use_fused_mhc=False,
        )
        native = Float16Module(config, HyperConnectionModule(config, 1).cuda()).module
        fused_config = replace(config, use_fused_mhc=True)
        fused = Float16Module(fused_config, HyperConnectionModule(fused_config, 1).cuda()).module
        fused.load_state_dict(native.state_dict())
        for module in (native, fused):
            assert all(p.dtype == torch.float32 for p in module.parameters())
        hidden = torch.randn(16, 2, 256, device="cuda", dtype=torch.bfloat16)
        results = []
        for module in (native, fused):
            inputs = hidden.detach().clone().requires_grad_()
            aggregated, h_res, h_post, residual = module(inputs)
            assert aggregated.shape == (16, 2, 64)
            assert residual.shape == hidden.shape
            assert h_res.dtype == h_post.dtype == aggregated.dtype == torch.bfloat16
            # Exercise aggregation, mixing and residual gradient branches together.
            loss = aggregated.float().square().mean() + h_res.float().square().mean()
            loss = loss + h_post.float().square().mean() + residual.float().square().mean()
            loss.backward()
            results.append(
                (
                    aggregated.detach(),
                    h_res.detach(),
                    h_post.detach(),
                    inputs.grad,
                    {name: param.grad for name, param in module.named_parameters()},
                )
            )
        for actual, expected in zip(results[1][:-1], results[0][:-1]):
            torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-3)
        for name, expected in results[0][-1].items():
            assert expected is not None, name
            actual = results[1][-1][name]
            assert actual is not None, name
            torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-4, msg=name)
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.parametrize("fused", [False, True])
def test_mhc_aggregate_into_matches_fp32_gradient_oracle(fused):
    from megatron.core.fusions.fused_mhc_kernels import fused_h_aggregate_into
    from megatron.core.transformer.hyper_connection import native_h_aggregate_into

    torch.manual_seed(190)
    x = torch.randn(8, 2, 4, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weights = torch.randn(8, 2, 4, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    output = torch.empty(8, 2, 64, device="cuda", dtype=torch.bfloat16)
    fn = fused_h_aggregate_into if fused else native_h_aggregate_into
    actual = fn(x, weights, output)
    assert actual.data_ptr() == output.data_ptr()
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
