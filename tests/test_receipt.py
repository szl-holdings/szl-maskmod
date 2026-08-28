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


def test_score_mod_identity_on_chain(km):
    def alibi_unit(scores, q_idx, k_idx):
        return scores - (q_idx - k_idx).abs().to(scores.dtype)

    q = k = v = torch.randn(1, 1, 4, 8)
    chain = km.ReceiptChain()
    km.maskmod_attn(q, k, v, score_mod=alibi_unit, chain=chain)
    assert chain._rows[0]["score_mod"] == "alibi_unit"
    ok, depth, brk = chain.verify()
    assert ok and depth == 1 and brk == -1
