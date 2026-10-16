# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Real SBHD providers against an independent FP32 attention/teacher oracle."""

import pytest
import torch

from megatron.core.transformer.experimental_attention_variant.csa import get_window_topk_idxs
from megatron.core.transformer.experimental_attention_variant.csa_utils import (
    fused_sparse_attention,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.mark.parametrize("legacy_api", [False, True])
def test_indexer_score_recomputation_keeps_fp32(monkeypatch, legacy_api):
    from megatron.plugin.dsa_kernel.triton_dsa_kernels import _score_bmm_fp32

    query = torch.randn(2, 16, 128, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(2, 128, 32, device="cuda", dtype=torch.bfloat16)
    original_bmm = torch.bmm
    if legacy_api:

        def legacy_bmm(left, right, **kwargs):
            if "out_dtype" in kwargs:
                raise TypeError("bmm() got an unexpected keyword argument 'out_dtype'")
            return original_bmm(left, right, **kwargs)

        monkeypatch.setattr(torch, "bmm", legacy_bmm)

    actual = _score_bmm_fp32(query, key)
    expected = original_bmm(query.float(), key.float())
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    "backend,sparse",
    [("triton", True), ("triton", False), ("cudnn", True), ("none", True), ("none", False)],
)
@pytest.mark.parametrize("per_token", [False, True])
def test_real_sbhd_indexer_loss_and_all_gradients(backend, sparse, per_token):
    torch.manual_seed(927)
    seq, heads, dim, nh, ihd, ratio, topk = 128, 2, 512, 64, 128, 4, 16
    nc = seq // ratio
    q = (
        torch.randn(seq, 1, heads, dim, device="cuda", dtype=torch.bfloat16) * 0.2
    ).requires_grad_()
    kv = (torch.randn(seq + nc, 1, dim, device="cuda", dtype=torch.bfloat16) * 0.2).requires_grad_()
    sink = torch.tensor([-2.0, 2.0], device="cuda", requires_grad=True)
    qi = (torch.randn(seq, 1, nh, ihd, device="cuda", dtype=torch.bfloat16) * 0.1).requires_grad_()
    ki = (torch.randn(nc, 1, ihd, device="cuda", dtype=torch.bfloat16) * 0.1).requires_grad_()
    w = (torch.rand(seq, 1, nh, device="cuda", dtype=torch.bfloat16) * 0.1).requires_grad_()
    inputs = (q, kv, sink, qi, ki, w)
    reference = tuple(x.detach().float().requires_grad_() for x in inputs)
    window = get_window_topk_idxs(16, 1, seq, q.device).int()
    scale, index_scale, coeff = dim**-0.5, ihd**-0.5, 0.2
    if backend == "none":
        from types import SimpleNamespace

        from megatron.core.transformer.experimental_attention_variant.csa import (
            unfused_compressed_sparse_attn,
        )
        from megatron.core.transformer.experimental_attention_variant.dsa import (
            FusedDSAIndexerLoss,
            fused_qk_topk_naive,
        )

        causal = (
            torch.arange(nc, device="cuda")[None, :]
            < (torch.arange(seq, device="cuda")[:, None] + 1) // ratio
        )
        index_mask = torch.where(causal, 0.0, -torch.inf)

        def select(q, k, weights, width, ratio, index_scale):
            scores, indices = fused_qk_topk_naive(
                q, k, weights.float() * index_scale, width, mask=index_mask
            )
            return indices, None

        def run(
            q,
            kv,
            sink,
            window,
            qi,
            ki,
            w,
            topk,
            ratio,
            scale,
            index_scale,
            coeff,
            sparse_loss,
            kv_offset,
            calculate_per_token_loss,
        ):
            window_scores = (
                torch.einsum("qbhd,kbd->bhqk", q.detach().float(), kv[:seq].detach().float())
                * scale
            )
            visible_window = torch.zeros(seq, seq, device="cuda", dtype=torch.bool)
            for row in range(seq):
                visible_window[row, max(0, row - 15) : row + 1] = True
            window_scores = window_scores.masked_fill(~visible_window[None, None], -torch.inf)
            noncomp = torch.logsumexp(
                torch.cat(
                    (window_scores, sink.detach()[None, :, None, None].expand(1, heads, seq, 1)),
                    dim=-1,
                ),
                dim=-1,
            )
            selected, loss = FusedDSAIndexerLoss.apply(
                qi,
                w.float() * index_scale,
                ki,
                q.detach(),
                kv[seq:].detach().unsqueeze(2).expand(-1, -1, heads, -1),
                scale,
                topk,
                coeff,
                index_mask,
                sparse_loss,
                SimpleNamespace(tp=SimpleNamespace(size=lambda: 1)),
                None,
                None,
                None,
                None,
                calculate_per_token_loss,
                True,
                noncomp,
            )
            indices = torch.cat((torch.where(selected >= 0, selected + seq, -1), window), dim=-1)
            return unfused_compressed_sparse_attn(q, kv, sink, indices, scale), loss

        provider = SimpleNamespace(indexer_topk=select)
    elif backend == "triton":
        from megatron.plugin.dsa_kernel import triton_dsa_kernels as provider

        run = provider.fused_indexer_sparse_attn
    else:
        provider = fused_sparse_attention
        run = provider.fused_csa_indexer_sparse_attn
    selected, _ = provider.indexer_topk(qi, ki, w, topk, ratio, index_scale)
    out, loss = run(
        q,
        kv,
        sink,
        window,
        qi,
        ki,
        w,
        topk,
        ratio,
        scale,
        index_scale,
        coeff,
        sparse_loss=sparse,
        kv_offset=seq,
        calculate_per_token_loss=per_token,
    )
    grad_out = torch.randn_like(out)
    torch.autograd.backward((out, loss), (grad_out, torch.tensor(0.7, device="cuda")))

    rq, rkv, rs, rqi, rki, rw = reference
    scores = torch.einsum("qbhd,kbd->bqhk", rq, rkv) * scale
    # All visible compressed keys, window and sink define the teacher denominator.
    rows = torch.arange(seq, device="cuda")
    visible = torch.arange(nc, device="cuda")[None, :] < (rows[:, None] + 1) // ratio
    wm = torch.zeros(seq, seq, device="cuda", dtype=torch.bool)
    for i in range(seq):
        wm[i, max(0, i - 15) : i + 1] = True
    full_valid = torch.cat((wm, visible), dim=-1)
    teacher_logits = scores.detach().masked_fill(~full_valid[None, :, None, :], -torch.inf)
    denominator = torch.logsumexp(
        torch.cat(
            (teacher_logits, rs.detach()[None, None, :, None].expand(1, seq, heads, 1)), dim=-1
        ),
        dim=-1,
    )
    teacher = torch.exp(teacher_logits[..., seq:] - denominator[..., None]).sum(dim=2)
    logits = (
        torch.relu(torch.einsum("qbhd,kbd->bqhk", rqi, rki))
        * rw.permute(1, 0, 2)[..., None]
        * index_scale
    ).sum(dim=2)
    if sparse:
        idx = selected.long()
        valid = idx >= 0
        target = teacher.gather(-1, idx.clamp_min(0)).masked_fill(~valid, 0)
        logits = logits.gather(-1, idx.clamp_min(0)).masked_fill(~valid, -torch.inf)
    else:
        valid = visible[None, :, :]
        target = teacher.masked_fill(~valid, 0)
        logits = logits.masked_fill(~valid, -torch.inf)
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-10)
    predict = torch.softmax(
        logits.masked_fill(~valid.any(-1, keepdim=True), 0), dim=-1
    ).masked_fill(~valid, 0)
    kl = (target * (torch.log(target + 1e-6) - torch.log(predict + 1e-6))).sum(-1)
    ref_loss = coeff * (kl.sum() if per_token else kl.mean())
    attention_valid = torch.zeros_like(full_valid)
    attention_valid[:, :seq] = wm
    cmp = selected[0].long()
    selected_mask = torch.zeros(seq, nc, dtype=torch.int32, device="cuda")
    selected_mask.scatter_add_(1, cmp.clamp_min(0), (cmp >= 0).int())
    attention_valid[:, seq:] = selected_mask > 0
    alogits = scores.masked_fill(~attention_valid[None, :, None, :], -torch.inf)
    probs = torch.softmax(
        torch.cat((alogits, rs[None, None, :, None].expand(1, seq, heads, 1)), dim=-1), dim=-1
    )[..., :-1]
    ref_out = torch.einsum("bqhk,kbd->qbhd", probs, rkv).reshape_as(out)
    torch.autograd.backward(
        (ref_out, ref_loss), (grad_out.float(), torch.tensor(0.7, device="cuda"))
    )
    torch.testing.assert_close(out.float(), ref_out, rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(loss, ref_loss, rtol=2e-2, atol=2e-4)
    for name, actual, expected in zip(
        ("q", "kv", "sink", "index_q", "index_k", "weights"), inputs, reference
    ):
        assert actual.grad is not None, name
        a, e = actual.grad.float().flatten(), expected.grad.flatten()
        cosine = torch.nn.functional.cosine_similarity(a, e, dim=0)
        assert cosine > 0.999, f"{name}: cosine={cosine.item()}"
        torch.testing.assert_close(a, e, rtol=5e-2, atol=5e-2)


@pytest.mark.parametrize("ratio", [0, 4, 128])
@pytest.mark.parametrize("training", [False, True])
def test_sbhd_model_preserves_backend_routes_and_gradients(ratio, training):
    """Reuse one process across Triton and cuDNN without provider cross-talk."""
    from dataclasses import replace

    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.transformer.experimental_attention_variant.dsa import (
        DSAIndexerLossAutoScaler,
    )
    from tests.unit_tests.test_utilities import Utils
    from tests.unit_tests.transformer.experimental_attention_variant.test_dsv4_hybrid_attention import (
        _build_attention,
        _make_config,
    )

    Utils.initialize_model_parallel()
    try:
        pg = ProcessGroupCollection.use_mpu_process_groups()
        torch.manual_seed(127)
        cfg = _make_config(
            num_layers=1,
            csa_compress_ratios=[ratio],
            num_attention_heads=2,
            v_head_dim=512,
            qk_pos_emb_head_dim=64,
            q_lora_rank=128,
            csa_window_size=16,
            dsa_indexer_n_heads=64,
            dsa_indexer_head_dim=128,
            dsa_indexer_topk=16,
            dsa_indexer_loss_coeff=0.2,
            dsa_indexer_use_sparse_loss=True,
            gradient_accumulation_fusion=False,
            dsa_kernel_backend="none",
        )
        baseline = _build_attention(cfg, 1, pg).cuda().to(torch.bfloat16)
        baseline.train(training)
        hidden = torch.randn(256, 1, 256, device="cuda", dtype=torch.bfloat16)
        grad = torch.randn_like(hidden)
        DSAIndexerLossAutoScaler.set_loss_scale(torch.ones((), device="cuda"))

        def evaluate(model):
            x = hidden.detach().clone().requires_grad_(training)
            with torch.set_grad_enabled(training):
                out, _ = model(x, attention_mask=None)
                if training:
                    out.backward(grad)
            result = {"output": out.detach()}
            if training:
                result['input'] = x.grad
                for name, param in model.named_parameters():
                    assert param.grad is not None, name
                    result[name] = param.grad
            return result

        reference = evaluate(baseline)
        for backend in ("triton", "cudnn"):
            model = (
                _build_attention(replace(cfg, dsa_kernel_backend=backend), 1, pg)
                .cuda()
                .to(torch.bfloat16)
            )
            model.load_state_dict(baseline.state_dict())
            model.train(training)
            actual = evaluate(model)
            assert actual.keys() == reference.keys()
            for name, value in actual.items():
                expected = reference[name]
                a, e = value.float().flatten(), expected.float().flatten()
                if torch.count_nonzero(e):
                    cosine = torch.nn.functional.cosine_similarity(a.double(), e.double(), dim=0)
                    assert cosine > 0.999, f"{backend}/{name}: {cosine.item()}"
                torch.testing.assert_close(
                    value.float(),
                    expected.float(),
                    rtol=5e-2,
                    atol=5e-2,
                    msg=lambda message: f"{backend}/{name}: {message}",
                )
    finally:
        Utils.destroy_model_parallel()
