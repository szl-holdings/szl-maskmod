# SPDX-License-Identifier: Apache-2.0
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "torch-ext"))
import torch
from szl_maskmod import maskmod_attn, ReceiptChain, selfcheck

def test_causal_matches_sdpa():
    torch.manual_seed(1)
    q = k = v = torch.randn(1, 2, 8, 16)
    y = maskmod_attn(q, k, v, causal=True)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
    assert torch.allclose(y, ref, atol=1e-5, rtol=1e-5)

def test_score_mod_bias():
    q = k = v = torch.randn(1, 1, 4, 8)
    def plus_one(scores, q_idx, k_idx):
        return scores + 1.0
    y0 = maskmod_attn(q, k, v, causal=False)
    y1 = maskmod_attn(q, k, v, causal=False, score_mod=plus_one)
    # softmax(s+1) == softmax(s); output must match
    assert torch.allclose(y0, y1, atol=1e-5, rtol=1e-5)

def test_selfcheck():
    assert selfcheck()["ok"] is True
