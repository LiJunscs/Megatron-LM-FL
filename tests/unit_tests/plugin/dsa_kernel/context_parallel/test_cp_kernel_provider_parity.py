# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Exact contract parity across the PyTorch, Triton, and CuTe CP providers.

CP layout kernels do not perform model arithmetic. ``build_attention_indices``
only constructs integer metadata; ``compact_compressor_input`` only copies or
zero-fills floating-point payload rows and scatters their gradients. Therefore
all caller-visible tensors must be bitwise equal to the PyTorch oracle.
"""

import pytest
import torch

from megatron.core.transformer.experimental_attention_variant.csa_utils import (
    cp_layout as torch_cp,
)
from megatron.plugin.dsa_kernel.context_parallel.backends import cute as cute_cp
from megatron.plugin.dsa_kernel.context_parallel.backends import triton as triton_cp


_PROVIDERS = (
    pytest.param("torch", id="torch"),
    pytest.param("triton", id="triton"),
    pytest.param("cute", id="cute"),
)


def _provider(name):
    if not torch.cuda.is_available():
        pytest.skip("CP provider parity requires CUDA")
    if name == "torch":
        return torch_cp
    if name == "triton":
        if not triton_cp._TRITON_AVAILABLE:
            pytest.skip("real Triton CP provider is unavailable")
        return triton_cp
    if name == "cute":
        if not cute_cp._CUTE_AVAILABLE:
            pytest.skip("real CuTe CP provider is unavailable")
        return cute_cp
    raise AssertionError(f"unknown CP provider {name!r}")


def _assert_exact_tuple(actual, expected, label):
    assert len(actual) == len(expected)
    for index, (actual_tensor, expected_tensor) in enumerate(zip(actual, expected)):
        if expected_tensor is None:
            assert actual_tensor is None, f"{label}[{index}] must be None"
            continue
        assert actual_tensor is not None, f"{label}[{index}] unexpectedly returned None"
        assert actual_tensor.dtype == expected_tensor.dtype
        assert actual_tensor.shape == expected_tensor.shape
        assert torch.equal(actual_tensor, expected_tensor), (
            f"{label}[{index}] is not bitwise equal to the PyTorch CP oracle"
        )


@pytest.mark.parametrize("provider_name", _PROVIDERS)
@pytest.mark.parametrize(
    "cu_values,global_start,local_rows,ratio,d_comp,c_cap",
    (
        pytest.param([0, 5, 21, 40], 11, 18, 4, 8, 8, id="ratio4-ragged"),
        pytest.param([0, 256], 128, 128, 128, 128, 2, id="ratio128"),
    ),
)
def test_cp_provider_compaction_forward_backward_is_bitwise_equal(
    provider_name, cu_values, global_start, local_rows, ratio, d_comp, c_cap
):
    """Copied payloads, compaction ids, and one-to-one scatter grads are exact."""
    provider = _provider(provider_name)
    cu = torch.tensor(cu_values, dtype=torch.int32, device="cuda")

    # Integer-valued BF16 payloads make the copy-only contract explicit while
    # still exercising the dtype used by training.
    hidden_values = torch.arange(
        local_rows * 6, dtype=torch.float32, device="cuda"
    ).reshape(local_rows, 2, 3).to(torch.bfloat16)
    boundary_values = torch.arange(
        -d_comp * 6, 0, dtype=torch.float32, device="cuda"
    ).reshape(d_comp, 2, 3).to(torch.bfloat16)

    hidden_ref = hidden_values.clone().requires_grad_(True)
    boundary_ref = boundary_values.clone().requires_grad_(True)
    compact_ref, ids_ref = torch_cp.compact_compressor_input(
        hidden_ref, boundary_ref, cu, global_start, ratio, d_comp, c_cap
    )

    hidden_actual = hidden_values.clone().requires_grad_(True)
    boundary_actual = boundary_values.clone().requires_grad_(True)
    compact_actual, ids_actual = provider.compact_compressor_input(
        hidden_actual,
        boundary_actual,
        cu,
        global_start,
        ratio,
        d_comp,
        c_cap,
    )

    assert torch.equal(compact_actual, compact_ref)
    assert torch.equal(ids_actual, ids_ref)

    grad = torch.arange(
        compact_ref.numel(), dtype=torch.float32, device="cuda"
    ).reshape_as(compact_ref).to(torch.bfloat16)
    grad_hidden_ref, grad_boundary_ref = torch.autograd.grad(
        compact_ref, (hidden_ref, boundary_ref), grad
    )
    grad_hidden_actual, grad_boundary_actual = torch.autograd.grad(
        compact_actual, (hidden_actual, boundary_actual), grad
    )
    assert torch.equal(grad_hidden_actual, grad_hidden_ref)
    assert torch.equal(grad_boundary_actual, grad_boundary_ref)


@pytest.mark.parametrize("provider_name", _PROVIDERS)
@pytest.mark.parametrize("mode", (0, 1, 2), ids=("selected-topk", "all-visible", "indexer-loss"))
def test_cp_provider_attention_indices_are_bitwise_equal(provider_name, mode):
    """All three integer index-lowering modes exactly match the oracle."""
    provider = _provider(provider_name)
    cu = torch.tensor([0, 5, 21, 40], dtype=torch.int32, device="cuda")
    cu_compressed = torch.tensor([0, 1, 5, 9], dtype=torch.int32, device="cuda")
    global_start = 11
    local_rows = 18
    d_window = 8
    window_size = 6
    ratio = 4
    compressed_width = 3
    seq_to_rank_row = torch.arange(8, -1, -1, dtype=torch.int32, device="cuda")
    compressed_topk = (
        torch.arange(compressed_width, dtype=torch.int32, device="cuda")
        .unsqueeze(0)
        .expand(local_rows, -1)
        .clone()
    )
    compressed_topk[1::4, -1] = -1

    kwargs = dict(
        compressed_topk=None if mode == 1 else compressed_topk,
        cu_seqlens_compressed=cu_compressed,
        seq_to_rank_row=seq_to_rank_row,
        for_indexer_loss=mode == 2,
    )
    expected = torch_cp.build_attention_indices(
        cu,
        global_start,
        local_rows,
        d_window,
        window_size,
        ratio,
        compressed_width,
        **kwargs,
    )
    actual = provider.build_attention_indices(
        cu,
        global_start,
        local_rows,
        d_window,
        window_size,
        ratio,
        compressed_width,
        **kwargs,
    )
    _assert_exact_tuple(actual, expected, f"{provider_name} mode={mode}")
