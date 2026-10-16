# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Packed provider boundaries and real/physical metadata regression tests."""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.context_parallel.utils import get_packed_seq_padding_mask
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.experimental_attention_variant.csa_utils import cp_utils
from megatron.core.utils import flatten_batch_for_packed_sequences


class _Group:
    def __init__(self, size=1):
        self._size = size

    def size(self):
        return self._size

    def rank(self):
        return 0


def _packed():
    real = torch.tensor([0, 3, 3, 7], dtype=torch.int32)
    physical = torch.tensor([0, 4, 4, 10], dtype=torch.int32)
    return PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=real,
        cu_seqlens_kv=real,
        cu_seqlens_q_padded=physical,
        cu_seqlens_kv_padded=physical,
        max_seqlen_q=6,
        max_seqlen_kv=6,
    )


@pytest.mark.parametrize("cp_size", [1, 2])
@pytest.mark.parametrize("backend", ["triton", "none", "tilelang"])
def test_thd_rejects_unsupported_provider_even_at_cp1(backend, cp_size):
    config = SimpleNamespace(
        dsa_kernel_backend=backend, attention_cp_layout="contiguous", deterministic_mode=False
    )
    with pytest.raises(ValueError, match="requires dsa_kernel_backend='cudnn'"):
        cp_utils.validate_packed_inputs(_packed(), config, _Group(cp_size))


@pytest.mark.parametrize("start,count", [(0, 10), (0, 5), (5, 5)])
def test_router_padding_uses_real_lengths_and_physical_offsets(start, count):
    actual = get_packed_seq_padding_mask(_packed(), start, count)
    oracle = torch.tensor([False, False, False, True, False, False, False, False, True, True])
    torch.testing.assert_close(actual, oracle[start : start + count])


def test_packed_batch_merges_real_and_physical_prefixes_together():
    tokens = torch.arange(16).view(2, 8)
    batch = dict(
        tokens=tokens,
        labels=tokens.clone(),
        position_ids=tokens.clone(),
        loss_mask=torch.ones_like(tokens),
        cu_seqlens=torch.tensor([[0, 3, 6], [0, 2, 5]], dtype=torch.int32),
        cu_seqlens_padded=torch.tensor([[0, 4, 8], [0, 3, 8]], dtype=torch.int32),
        max_seqlen=torch.tensor([4, 5]),
    )
    actual = flatten_batch_for_packed_sequences(batch)
    assert actual['cu_seqlens'].tolist() == [[0, 3, 6, 8, 11]]
    assert actual['cu_seqlens_padded'].tolist() == [[0, 4, 8, 11, 16]]
    assert actual['tokens'].tolist() == [list(range(16))]
    assert actual['max_seqlen'].tolist() == [5]
    repeated = flatten_batch_for_packed_sequences(actual)
    assert repeated['cu_seqlens'].tolist() == [[0, 3, 6, 8, 11]]
    assert repeated['cu_seqlens_padded'].tolist() == [[0, 4, 8, 11, 16]]


def test_packed_batch_leaves_sbhd_tensor_layout_unchanged():
    tokens = torch.arange(16).view(2, 8)
    batch = dict(tokens=tokens, cu_seqlens=None)
    assert flatten_batch_for_packed_sequences(batch)['tokens'] is tokens


def test_gpt_dataset_emits_actual_document_lengths_before_padding():
    import numpy as np

    from megatron.core.datasets.gpt_dataset import GPTDataset

    config = SimpleNamespace(
        add_extra_token_to_sequence=True,
        inter_document_masking=True,
        preserve_packed_sequence_lengths=True,
        sequence_length=8,
        tokenizer=SimpleNamespace(eod=99),
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
        create_attention_mask=False,
    )
    sample = SimpleNamespace(
        config=config,
        _pad_token_id=-1,
        masks_and_position_ids_are_cacheable=False,
        masks_and_position_ids_are_cached=False,
        shuffle_index=np.array([0]),
        sample_index=np.array([[0, 0], [2, 1]]),
        document_index=np.array([0, 1, 2]),
        dataset=SimpleNamespace(sequence_lengths=np.array([3, 2, 2])),
        _query_document_sample_shuffle_indices=lambda idx: (
            np.array([1, 2, 3, 4, 5, 6, 7, -1, -1]),
            np.array([0, 1, 2]),
        ),
    )
    result = GPTDataset.__getitem__(sample, 0)
    # The last document has one query token, followed by physical sample padding.
    assert result['cu_seqlens'][:4].tolist() == [0, 3, 5, 6]
    assert result['cu_seqlens_padded'][:4].tolist() == [0, 3, 5, 8]
    assert result['position_ids'].tolist() == [0, 1, 2, 0, 1, 0, 1, 2]
