# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Physical compressor group ABI, fallback, and SM100-only fused numerics."""

import pytest
import torch

from megatron.core.transformer.experimental_attention_variant.csa import (
    Compressor,
    _pool_compressor_values,
)
from megatron.core.transformer.experimental_attention_variant.csa_utils import cp_utils
from megatron.core.transformer.experimental_attention_variant.csa_utils import (
    fused_compressor as fused,
)

pytestmark = pytest.mark.experimental


class _PoolShape:
    def __init__(self, head_dim):
        self.head_dim = head_dim


def _public_pool(kv, score, ape, ids, ratio, head_dim):
    groups = ids.numel()
    kv = kv.reshape(groups, ratio, 1, -1)
    score = score.reshape_as(kv) + ape.view(1, ratio, 1, -1)
    if ratio == 4:
        owner = _PoolShape(head_dim)
        kv = Compressor._overlap_transform_thd(owner, kv, ids == 0, 0)
        score = Compressor._overlap_transform_thd(owner, score, ids == 0, float('-inf'))
    return _pool_compressor_values(kv, score, kv.dtype)


def _physical_oracle(kv, score, ape, ids, ratio, head_dim):
    """Independent per-output oracle; includes the public physical roll semantics."""
    coff = 2 if ratio == 4 else 1
    kv = kv.reshape(len(ids), ratio, coff * head_dim)
    gates = score.reshape_as(kv).double() + ape.double()
    outputs = []
    for row, group in enumerate(ids):
        values = kv[row, :, -head_dim:].double()
        logits = gates[row, :, -head_dim:]
        if coff == 2 and group != 0:
            values = torch.cat((kv[row - 1, :, :head_dim].double(), values))
            logits = torch.cat((gates[row - 1, :, :head_dim], logits))
        outputs.append((values * logits.softmax(0)).sum(0))
    return torch.stack(outputs).unsqueeze(1).to(kv.dtype)


def _segment_oracle(kv, score, ape, cu, cu_comp, ratio, head_dim):
    """Interpret frontend offsets directly, independently of physical group IDs."""
    coff = kv.shape[-1] // head_dim
    outputs = []
    for start, end, cs, ce in zip(
        cu.tolist(), cu.tolist()[1:], cu_comp.tolist(), cu_comp.tolist()[1:]
    ):
        assert end - start == (ce - cs) * ratio
        for group in range(ce - cs):
            begin = start + group * ratio
            values = kv[begin : begin + ratio, 0, -head_dim:].double()
            logits = (
                score[begin : begin + ratio, 0, -head_dim:].double() + ape[:, -head_dim:].double()
            )
            if coff == 2 and group:
                values = torch.cat((kv[begin - ratio : begin, 0, :head_dim].double(), values))
                prior = (
                    score[begin - ratio : begin, 0, :head_dim].double() + ape[:, :head_dim].double()
                )
                logits = torch.cat((prior, logits))
            outputs.append((values * logits.softmax(0)).sum(0))
    return torch.stack(outputs).unsqueeze(1).to(kv.dtype)


@pytest.mark.parametrize('ratio', [4, 128])
@pytest.mark.parametrize('ids', [[0, 1, 0, 1, -1, -1], [7, 8, 0, -1], [-1, -1]])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_group_adapter_matches_independent_oracle_and_gradients(ratio, ids, device):
    """Offsets preserve boundaries, halos, empty capacity, and every input gradient."""
    head_dim = 8
    coff = 2 if ratio == 4 else 1
    torch.manual_seed(17)
    # Strided source tensors protect contiguous-copy autograd in the frontend shim.
    kv = torch.randn(len(ids) * ratio, 1, coff * head_dim * 2, device=device)[
        ..., ::2
    ].requires_grad_()
    score = torch.randn_like(kv).requires_grad_()
    ape = torch.randn(ratio, coff * head_dim, device=device, requires_grad=True)
    group_ids = torch.tensor(ids, dtype=torch.int32, device=device)
    k, s, cu, cuc = fused._prepare_grouped_compressor_inputs(kv, score, group_ids, ratio, coff)
    actual = _segment_oracle(k, s, ape, cu, cuc, ratio, head_dim)[int(coff == 2) :]
    expected = _physical_oracle(kv, score, ape, ids, ratio, head_dim)
    grad = torch.randn_like(actual) * 0.7
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        _public_pool(kv, score, ape, group_ids, ratio, head_dim), expected, rtol=1e-5, atol=1e-6
    )
    for actual_grad, expected_grad in zip(
        torch.autograd.grad(actual, (kv, score, ape), grad, retain_graph=True),
        torch.autograd.grad(expected, (kv, score, ape), grad),
    ):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize('ratio', [4, 128])
def test_cp_preparation_adapter_preserves_producer_gradients(ratio):
    """Complete/partial/empty documents use the existing CP producer gather once."""
    for start in (0, 256):
        width = 16 if ratio == 4 else 8
        hidden = torch.randn(256, 1, width, device='cuda', requires_grad=True)
        boundary = torch.randn(256, 1, width, device='cuda', requires_grad=True)
        # Empty documents, incomplete groups, and a group crossing a rank boundary.
        cu = torch.tensor([0, 0, 131, 385, 385, 512], device='cuda', dtype=torch.int32)
        x, ids, positions, _, _ = cp_utils.prepare_cp_compressor_input(
            hidden, boundary, cu, start, 2, ratio
        )
        score = (x * 0.13).to(x.dtype)
        ape = torch.randn(ratio, width, device='cuda', requires_grad=True)
        k, s, offsets, comp_offsets = fused._prepare_grouped_compressor_inputs(
            x, score, ids, ratio, 2 if ratio == 4 else 1
        )
        actual = _segment_oracle(k, s, ape, offsets, comp_offsets, ratio, 8)[int(ratio == 4) :]
        # Use a separate compiled preparation graph: AOTAutograd donated buffers
        # require retain_graph=False in the existing environment.
        x_ref, ids_ref, _, _, _ = cp_utils.prepare_cp_compressor_input(
            hidden, boundary, cu, start, 2, ratio
        )
        expected = _physical_oracle(x_ref, x_ref * 0.13, ape, ids_ref.tolist(), ratio, 8)
        grad = torch.randn_like(actual)
        # Only canonical rows are consumed by CSA; testing every physical row is stronger.
        for got, want in zip(
            torch.autograd.grad(actual, (hidden, boundary, ape), grad),
            torch.autograd.grad(expected, (hidden, boundary, ape), grad),
        ):
            torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)
        assert torch.equal(positions[ids < 0], torch.zeros_like(positions[ids < 0]))


def test_sm90_fallback_does_not_import_frontend(monkeypatch):
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip('SM90 fallback boundary')

    def forbidden():
        raise AssertionError('unsupported device imported the compressor frontend')

    monkeypatch.setattr(fused, '_get_frontend', forbidden)
    kv = torch.randn(8, 1, 256, device='cuda', dtype=torch.bfloat16)
    assert (
        fused.maybe_compress_grouped_fused(
            kv,
            kv,
            kv.new_zeros((4, 256), dtype=torch.float32),
            torch.tensor([0, 1], device='cuda'),
            ratio=4,
            head_dim=128,
            coff=2,
        )
        is None
    )


@pytest.mark.parametrize('boundary', ['disabled', 'deterministic', 'compile'])
def test_dispatch_boundaries_do_not_launch(monkeypatch, boundary):
    monkeypatch.setattr(fused, 'fused_compressor_available', lambda device: True)
    monkeypatch.setattr(torch.compiler, 'is_compiling', lambda: boundary == 'compile')

    def forbidden(*args, **kwargs):
        raise AssertionError('boundary must retain eager')

    monkeypatch.setattr(fused, '_prepare_grouped_compressor_inputs', forbidden)
    prior = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(boundary == 'deterministic')
        kv = torch.zeros(8, 1, 256)
        assert (
            fused.maybe_compress_grouped_fused(
                kv,
                kv,
                torch.zeros(4, 256),
                torch.tensor([0, 1]),
                ratio=4,
                head_dim=128,
                coff=2,
                enabled=boundary != 'disabled',
            )
            is None
        )
    finally:
        torch.use_deterministic_algorithms(prior)


@pytest.mark.parametrize('ratio,head_dim', [(4, 128), (4, 512), (128, 128), (128, 512)])
@pytest.mark.parametrize('ids', [[0, 1, 0, 1, -1], [7, 8, 0, -1], [-1, -1]])
def test_real_sm100_fused_output_and_all_gradients(ratio, head_dim, ids):
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip('actual fused compressor requires SM100; SM90 validates fallback only')
    if not fused.fused_compressor_available(torch.device('cuda')):
        pytest.skip('cuDNN compressor API unavailable')
    coff = 2 if ratio == 4 else 1
    torch.manual_seed(19)
    kv = torch.randn(
        len(ids) * ratio,
        1,
        coff * head_dim,
        device='cuda',
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    score = torch.randn_like(kv, requires_grad=True)
    ape = torch.randn(ratio, coff * head_dim, device='cuda', requires_grad=True)
    group_ids = torch.tensor(ids, device='cuda', dtype=torch.int32)
    actual = fused.maybe_compress_grouped_fused(
        kv, score, ape, group_ids, ratio=ratio, head_dim=head_dim, coff=coff
    )
    assert actual is not None
    expected = _physical_oracle(kv, score, ape, ids, ratio, head_dim)
    grad = torch.randn_like(actual) * 0.7
    grads = torch.autograd.grad(actual, (kv, score, ape), grad)
    references = torch.autograd.grad(expected, (kv, score, ape), grad)
    public = _public_pool(kv, score, ape, group_ids, ratio, head_dim)
    public_grads = torch.autograd.grad(public, (kv, score, ape), grad)
    for got, want in [(actual, public), *zip(grads, public_grads)]:
        torch.testing.assert_close(got, want, rtol=1.6e-2, atol=1.6e-2)
    for got, want in [(actual, expected), *zip(grads, references)]:
        torch.testing.assert_close(got, want, rtol=1.6e-2, atol=1.6e-2)
        assert (
            torch.nn.functional.cosine_similarity(
                got.float().flatten(), want.float().flatten(), dim=0
            )
            > 0.999
        )
    # Saved output storage and run-to-run output/dKV/dScore must stay invariant.
    saved = actual.detach().clone()
    again = fused.maybe_compress_grouped_fused(
        kv, score, ape, group_ids, ratio=ratio, head_dim=head_dim, coff=coff
    )
    again_grads = torch.autograd.grad(again, (kv, score, ape), grad)
    assert torch.equal(actual, saved) and torch.equal(actual, again)
    assert torch.equal(grads[0], again_grads[0]) and torch.equal(grads[1], again_grads[1])
    # Warmed frontend configurations support CUDA graph capture; first-use JIT is not forced under capture.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = fused.maybe_compress_grouped_fused(
            kv, score, ape, group_ids, ratio=ratio, head_dim=head_dim, coff=coff
        )
        captured_grads = torch.autograd.grad(captured, (kv, score, ape), grad)
    graph.replay()
    torch.testing.assert_close(captured, actual, rtol=0, atol=0)
    for got, want in zip(captured_grads, grads):
        torch.testing.assert_close(got, want, rtol=1.6e-2, atol=1.6e-2)


@pytest.mark.parametrize(
    'ratio,head_dim,coff,dtype,ape_dtype',
    [
        (2, 128, 2, torch.bfloat16, torch.float32),
        (128, 64, 1, torch.bfloat16, torch.float32),
        (4, 128, 3, torch.bfloat16, torch.float32),
        (4, 128, 2, torch.float16, torch.float32),
        (4, 128, 2, torch.bfloat16, torch.bfloat16),
    ],
)
def test_unsupported_envelope_keeps_eager(monkeypatch, ratio, head_dim, coff, dtype, ape_dtype):
    # Bypass only device availability so unsupported-shape gates execute on H100.
    # This does not call or stand in for frontend numerical validation.
    monkeypatch.setattr(fused, 'fused_compressor_available', lambda device: True)

    def forbidden(*args, **kwargs):
        raise AssertionError('unsupported envelope launched a fused kernel')

    monkeypatch.setattr(fused._CompressThdFused, 'apply', forbidden)
    kv = torch.zeros(ratio * 2, 1, coff * head_dim, device='cuda', dtype=dtype)
    ape = torch.zeros(ratio, coff * head_dim, device='cuda', dtype=ape_dtype)
    assert (
        fused.maybe_compress_grouped_fused(
            kv,
            kv,
            ape,
            torch.tensor([0, 1], device='cuda'),
            ratio=ratio,
            head_dim=head_dim,
            coff=coff,
        )
        is None
    )


@pytest.mark.parametrize('ratio', [4, 128])
def test_real_compile_keeps_public_pooling(monkeypatch, ratio):
    """A real fullgraph trace must avoid frontend probing and raw-pointer launch."""

    def forbidden(*args, **kwargs):
        raise AssertionError('compile tracing reached frontend capability detection')

    monkeypatch.setattr(fused, 'fused_compressor_available', forbidden)
    head_dim = 8
    coff = 2 if ratio == 4 else 1
    ids = torch.tensor([0, 1, 0, -1], device='cuda', dtype=torch.int32)
    kv = torch.randn(4 * ratio, 1, coff * head_dim, device='cuda', requires_grad=True)
    score = torch.randn_like(kv, requires_grad=True)
    ape = torch.randn(ratio, coff * head_dim, device='cuda', requires_grad=True)

    def pool(kv, score, ape, ids):
        output = fused.maybe_compress_grouped_fused(
            kv, score, ape, ids, ratio=ratio, head_dim=head_dim, coff=coff
        )
        return _public_pool(kv, score, ape, ids, ratio, head_dim) if output is None else output

    compiled = torch.compile(pool, fullgraph=True)
    actual = compiled(kv, score, ape, ids)
    expected = _public_pool(kv, score, ape, ids, ratio, head_dim)
    grad = torch.randn_like(actual) * 0.7
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    for got, want in zip(
        torch.autograd.grad(actual, (kv, score, ape), grad),
        torch.autograd.grad(expected, (kv, score, ape), grad),
    ):
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)
    torch.compiler.reset()


@pytest.mark.parametrize('ratio', [4, 128])
@pytest.mark.parametrize('ids', [[0, 1, 0, -1], [7, 8, 0, -1], [-1, -1]])
def test_group_adapter_cuda_graph_capture_and_replay(ratio, ids):
    """Capture only the public group ABI adapter; this does not validate fused SM100."""
    coff = 2 if ratio == 4 else 1
    group_ids = torch.tensor(ids, device='cuda', dtype=torch.int32)
    kv = torch.randn(len(ids) * ratio, 1, coff * 8, device='cuda')
    score = torch.randn_like(kv)
    # Prime allocator/dispatch outside capture; no frontend is involved.
    fused._prepare_grouped_compressor_inputs(kv, score, group_ids, ratio, coff)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = fused._prepare_grouped_compressor_inputs(kv, score, group_ids, ratio, coff)
    for step in range(2):
        kv.add_(0.25)
        score.add_(0.125)
        graph.replay()
        expected = fused._prepare_grouped_compressor_inputs(kv, score, group_ids, ratio, coff)
        for got, want in zip(captured, expected):
            torch.testing.assert_close(got, want, rtol=0, atol=0)
