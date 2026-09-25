# SPDX-License-Identifier: Apache-2.0
"""Receipt-chain tests against KERNEL ReceiptChain (SHA3-256)."""
from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.kernels_ci


def test_receipt_verifies_and_tamper_breaks(km):
    chain = km.ReceiptChain()
    q = k = v = torch.randn(1, 1, 4, 8)
    km.maskmod_attn(q, k, v, causal=True, chain=chain)
    ok, depth, brk = chain.verify()
    assert ok is True
    assert depth == 1
    assert brk == -1
    chain._rows[0]["score_mod"] = "tampered"
    ok2, _, first = chain.verify()
    assert ok2 is False
    assert first == 0


def test_mask_identity_changes_receipt(km):
    q = k = v = torch.randn(1, 1, 4, 8)
    a = km.ReceiptChain()
    b = km.ReceiptChain()
    km.maskmod_attn(q, k, v, causal=False, chain=a)
    km.maskmod_attn(q, k, v, causal=True, chain=b)
    assert a._rows[0]["digest"] != b._rows[0]["digest"]
    keep = km.create_dense_mask(4, 4, causal=True)
    c = km.ReceiptChain()
    km.maskmod_attn(q, k, v, block_mask=keep, chain=c)
    assert c._rows[0]["block_mask"] != a._rows[0]["block_mask"]


def test_equal_count_different_topology_masks_have_distinct_receipts(km):
    q = k = v = torch.randn(1, 1, 4, 8)
    mask_a = torch.tensor(
        [
            [True, True, False, False],
            [True, False, True, False],
            [False, True, False, True],
            [False, False, True, True],
        ]
    )
    mask_b = torch.tensor(
        [
            [True, False, False, True],
            [False, True, True, False],
            [True, True, False, False],
            [False, False, True, True],
        ]
    )
    assert int(mask_a.sum()) == int(mask_b.sum())
    assert not torch.equal(mask_a, mask_b)

    chain_a = km.ReceiptChain()
    chain_b = km.ReceiptChain()
    km.maskmod_attn(q, k, v, block_mask=mask_a, chain=chain_a)
    km.maskmod_attn(q, k, v, block_mask=mask_b, chain=chain_b)

    assert chain_a._rows[0]["block_mask"] != chain_b._rows[0]["block_mask"]
    assert chain_a._rows[0]["digest"] != chain_b._rows[0]["digest"]
    assert chain_a.verify() == (True, 1, -1)
    assert chain_b.verify() == (True, 1, -1)


def test_equal_mask_values_are_layout_independent(km):
    q = k = v = torch.randn(1, 1, 4, 8)
    contiguous = torch.tensor(
        [
            [True, True, False, False],
            [True, False, True, False],
            [False, True, False, True],
            [False, False, True, True],
        ]
    )
    noncontiguous = contiguous.t().contiguous().t()
    assert torch.equal(contiguous, noncontiguous)
    assert contiguous.is_contiguous()
    assert not noncontiguous.is_contiguous()

    chain_a = km.ReceiptChain()
    chain_b = km.ReceiptChain()
    km.maskmod_attn(q, k, v, block_mask=contiguous, chain=chain_a)
    km.maskmod_attn(q, k, v, block_mask=noncontiguous, chain=chain_b)

    assert chain_a._rows[0]["block_mask"] == chain_b._rows[0]["block_mask"]
    assert chain_a._rows[0]["digest"] == chain_b._rows[0]["digest"]
    assert chain_a.verify() == (True, 1, -1)
    assert chain_b.verify() == (True, 1, -1)


def test_score_mod_identity_on_chain(km):
    def alibi_unit(scores, q_idx, k_idx):
        return scores - (q_idx - k_idx).abs().to(scores.dtype)

    q = k = v = torch.randn(1, 1, 4, 8)
    chain = km.ReceiptChain()
    km.maskmod_attn(q, k, v, score_mod=alibi_unit, chain=chain)
    assert chain._rows[0]["score_mod"] == "alibi_unit"
    ok, depth, brk = chain.verify()
    assert ok and depth == 1 and brk == -1
