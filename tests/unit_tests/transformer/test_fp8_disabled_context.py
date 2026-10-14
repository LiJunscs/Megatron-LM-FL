# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Verify the higher precision boundary with real nested TE contexts."""

from types import SimpleNamespace

import pytest

from megatron.core.fp8_utils import get_fp8_disabled_context


@pytest.mark.parametrize("is_init", [False, True])
@pytest.mark.parametrize("flag", ["fp8", "fp4", None])
def test_disabled_context_restores_outer_quantization(is_init, flag):
    import transformer_engine.pytorch as te
    from transformer_engine.pytorch.fp8 import FP8GlobalStateManager

    config = SimpleNamespace(fp8=None, fp4=None, fp8_param=False, fp4_param=False)
    if flag is not None:
        setattr(config, flag + ("_param" if is_init else ""), True)
    outer = te.fp8_model_init if is_init else te.fp8_autocast
    enabled = (
        FP8GlobalStateManager.with_fp8_parameters
        if is_init
        else FP8GlobalStateManager.is_fp8_enabled
    )
    with outer(enabled=True):
        assert enabled()
        with get_fp8_disabled_context(config, is_init=is_init):
            assert enabled() == (flag is None)
        assert enabled()
    assert not enabled()


def test_context_without_te(monkeypatch):
    from megatron.core import fp8_utils

    monkeypatch.setattr(fp8_utils, "HAVE_TE", False)
    config = SimpleNamespace(fp8="hybrid", fp4=None, fp8_param=True, fp4_param=False)
    for is_init in (False, True):
        with get_fp8_disabled_context(config, is_init=is_init):
            pass
