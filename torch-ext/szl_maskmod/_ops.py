# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SZL Holdings
"""Original score_mod + block-mask attention. Not copied from flex_attention.py."""
from __future__ import annotations
import hashlib
from typing import Callable, Optional
import torch
from ._chain import ReceiptChain

ScoreMod = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]

def _block_mask_identity(block_mask: Optional[torch.Tensor]) -> str:
    if block_mask is None:
        return "none"
    canonical = block_mask.detach().to(device="cpu", dtype=torch.bool).contiguous()
    shape = "x".join(str(int(dim)) for dim in canonical.shape)
    payload = bytes(canonical.view(-1).to(torch.uint8).tolist())
    return f"shape={shape};sha256={hashlib.sha256(payload).hexdigest()}"

def _dense_attn(q, k, v, *, causal, score_mod, block_mask, scale):
    b, h, tq, d = q.shape
    tkv = k.shape[2]
    sm = (d ** -0.5) if scale is None else scale
    scores = torch.matmul(q, k.transpose(-2, -1)) * sm
    q_idx = torch.arange(tq, device=q.device)[:, None]
    k_idx = torch.arange(tkv, device=q.device)[None, :]
    if causal:
        scores = scores.masked_fill(q_idx < k_idx, float("-inf"))
    if block_mask is not None:
        scores = scores.masked_fill(~block_mask, float("-inf"))
    if score_mod is not None:
        scores = score_mod(scores, q_idx, k_idx)
    p = torch.softmax(scores, dim=-1)
    p = torch.nan_to_num(p, nan=0.0)
    return torch.matmul(p, v)

def maskmod_attn(
    q, k, v, *,
    score_mod: Optional[ScoreMod] = None,
    block_mask: Optional[torch.Tensor] = None,
    causal: bool = False,
    chain: Optional[ReceiptChain] = None,
    scale: Optional[float] = None,
):
    y = _dense_attn(q, k, v, causal=causal, score_mod=score_mod, block_mask=block_mask, scale=scale)
    if chain is not None:
        mid = "none" if score_mod is None else getattr(score_mod, "__name__", type(score_mod).__name__)
        bdigest = _block_mask_identity(block_mask)
        chain.emit({"op": "maskmod_attn", "score_mod": mid, "block_mask": bdigest,
                    "causal": causal, "q_shape": list(q.shape), "lambda": "Conjecture 1"})
    return y

def selfcheck() -> dict:
    torch.manual_seed(20260828)
    q = k = v = torch.randn(1, 2, 8, 16)
    chain = ReceiptChain()
    y = maskmod_attn(q, k, v, causal=True, chain=chain)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
    err = float((y - ref).abs().max().item())
    ok_c, depth, brk = chain.verify()
    ok = bool(err < 1e-5 and ok_c)
    return {"ok": ok, "max_abs_vs_sdpa_causal": err, "chain_ok": ok_c, "chain_depth": depth,
            "lambda": "Conjecture 1", "note": "correctness vs SDPA causal; no speedup claimed"}
