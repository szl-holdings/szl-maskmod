# SPDX-License-Identifier: Apache-2.0
"""Block-mask and documented dense-reference tests (no speedup claims)."""
from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.kernels_ci


def _dense_ref(q, k, v, *, causal=False, score_mod=None, block_mask=None, scale=None):
    """Documented dense oracle matching KERNEL order: scores, causal, mask, score_mod, softmax."""
    d = q.shape[-1]
    sm = (d ** -0.5) if scale is None else scale
    scores = torch.matmul(q, k.transpose(-2, -1)) * sm
    q_idx = torch.arange(q.shape[2], device=q.device)[:, None]
    k_idx = torch.arange(k.shape[2], device=q.device)[None, :]
    if causal:
        scores = scores.masked_fill(q_idx < k_idx, float("-inf"))
    if block_mask is not None:
        scores = scores.masked_fill(~block_mask, float("-inf"))
    if score_mod is not None:
        scores = score_mod(scores, q_idx, k_idx)
    p = torch.softmax(scores, dim=-1)
    p = torch.nan_to_num(p, nan=0.0)
    return torch.matmul(p, v)


def test_block_mask_matches_dense_reference(km):
    torch.manual_seed(2)
    q = k = v = torch.randn(1, 2, 8, 16)
    keep = km.create_dense_mask(8, 8, causal=True, sliding_window=4)
    y = km.maskmod_attn(q, k, v, block_mask=keep)
    ref = _dense_ref(q, k, v, block_mask=keep)
    torch.testing.assert_close(y, ref, atol=1e-5, rtol=1e-5)


def test_compile_block_occupancy(km):
    keep = km.create_dense_mask(16, 16, causal=True)
    occ = km.compile_block_mask(keep, block_m=8, block_n=8)
    assert occ.dtype == torch.uint8
    assert occ.shape == (2, 2)
    assert int(occ[0, 1].item()) == 0
    assert int(occ[1, 0].item()) == 1


def test_score_mod_then_mask_order_is_kernel_order(km):
    torch.manual_seed(3)
    q = k = v = torch.randn(1, 1, 6, 8)
    keep = torch.ones(6, 6, dtype=torch.bool)
    keep[:, 3:] = False

    def times_two(scores, q_idx, k_idx):
        return scores * 2.0

    y = km.maskmod_attn(q, k, v, score_mod=times_two, block_mask=keep)
    ref = _dense_ref(q, k, v, score_mod=times_two, block_mask=keep)
    torch.testing.assert_close(y, ref, atol=1e-5, rtol=1e-5)


def test_cuda_matches_cpu_or_skip(km):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA GPU")
    torch.manual_seed(4)
    q = k = v = torch.randn(1, 2, 8, 16)
    y_cpu = km.maskmod_attn(q, k, v, causal=True)
    y_cuda = km.maskmod_attn(q.cuda(), k.cuda(), v.cuda(), causal=True)
    torch.testing.assert_close(y_cuda.cpu(), y_cpu, atol=1e-5, rtol=1e-5)
