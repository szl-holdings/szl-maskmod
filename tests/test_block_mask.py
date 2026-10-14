# SPDX-License-Identifier: Apache-2.0
"""Block-mask and documented dense-reference tests (no speedup claims)."""
from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.kernels_ci


def _dense_ref(q, k, v, *, causal=False, score_mod=None, block_mask=None, scale=None):
    """Documented dense oracle, hard-constraint semantics (0.2.0).

    Order: scores, causal+mask -> -inf, score_mod, causal+mask re-applied -> -inf,
    softmax. Excluded positions stay excluded whatever score_mod returns.
    """
    d = q.shape[-1]
    sm = (d ** -0.5) if scale is None else scale
    scores = torch.matmul(q, k.transpose(-2, -1)) * sm
    q_idx = torch.arange(q.shape[2], device=q.device)[:, None]
    k_idx = torch.arange(k.shape[2], device=q.device)[None, :]
    excluded = torch.zeros(q.shape[2], k.shape[2], dtype=torch.bool)
    if causal:
        excluded = excluded | (q_idx < k_idx)
    if block_mask is not None:
        excluded = excluded | ~block_mask
    scores = scores.masked_fill(excluded, float("-inf"))
    if score_mod is not None:
        scores = score_mod(scores, q_idx, k_idx)
        scores = scores.masked_fill(excluded, float("-inf"))
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


def test_score_mod_then_mask_reapplied_is_kernel_order(km):
    torch.manual_seed(3)
    q = k = v = torch.randn(1, 1, 6, 8)
    keep = torch.ones(6, 6, dtype=torch.bool)
    keep[:, 3:] = False

    def times_two(scores, q_idx, k_idx):
        return scores * 2.0

    y = km.maskmod_attn(q, k, v, score_mod=times_two, block_mask=keep)
    ref = _dense_ref(q, k, v, score_mod=times_two, block_mask=keep)
    torch.testing.assert_close(y, ref, atol=1e-5, rtol=1e-5)



def _independent_weights(scores_raw, score_mod, allowed):
    """Row-by-row softmax over the allowed set only (no -inf arithmetic)."""
    tq, tkv = allowed.shape
    q_idx = torch.arange(tq)[:, None]
    k_idx = torch.arange(tkv)[None, :]
    modded = score_mod(scores_raw.clone(), q_idx, k_idx).to(torch.float64)
    w = torch.zeros(scores_raw.shape, dtype=torch.float64)
    for i in range(tq):
        cols = [j for j in range(tkv) if bool(allowed[i, j])]
        if not cols:
            continue
        row = modded[..., i, cols]
        row = row - row.max(dim=-1, keepdim=True).values
        e = torch.exp(row)
        w[..., i, cols] = e / e.sum(dim=-1, keepdim=True)
    return w


def _replace_with_zero(scores, q_idx, k_idx):
    return torch.zeros_like(scores)


def _clamp_neg_inf(scores, q_idx, k_idx):
    return scores.clamp(min=-1.0e4)


def _inf_to_zero(scores, q_idx, k_idx):
    return torch.where(torch.isinf(scores), torch.zeros_like(scores), scores)


def _nan_at_masked(scores, q_idx, k_idx):
    return scores * 0.0  # -inf * 0 -> NaN at masked entries


ADVERSARIAL = [_replace_with_zero, _clamp_neg_inf, _inf_to_zero, _nan_at_masked]


@pytest.mark.parametrize("score_mod", ADVERSARIAL, ids=lambda f: f.__name__)
@pytest.mark.parametrize("causal", [False, True])
def test_adversarial_score_mod_cannot_restore_masked_positions(km, score_mod, causal):
    torch.manual_seed(5)
    t = 6
    q = k = torch.randn(1, 2, t, 8)
    v = torch.eye(t).expand(1, 2, t, t).contiguous()  # output == attention weights
    keep = torch.ones(t, t, dtype=torch.bool)
    keep[:, 4:] = False
    keep[2, :] = False  # one fully masked query row
    allowed = keep & ~(torch.arange(t)[:, None] < torch.arange(t)[None, :]) if causal else keep

    w = km.maskmod_attn(q, k, v, score_mod=score_mod, block_mask=keep, causal=causal)

    assert torch.isfinite(w).all()
    forbidden = ~allowed.expand_as(w)
    assert (w[forbidden] == 0.0).all(), "masked position received attention weight"
    assert (w[..., 2, :] == 0.0).all(), "fully masked row must be a well-defined zero row"
    live = allowed.any(dim=-1)
    torch.testing.assert_close(w.sum(dim=-1)[..., live], torch.ones_like(w.sum(dim=-1)[..., live]),
                               atol=1e-5, rtol=1e-5)

    sm = 8 ** -0.5
    scores_raw = torch.matmul(q, k.transpose(-2, -1)) * sm
    oracle = _independent_weights(scores_raw, score_mod if score_mod is not _nan_at_masked
                                  else _replace_with_zero, allowed)
    torch.testing.assert_close(w.to(torch.float64), oracle, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(w, _dense_ref(q, k, v, causal=causal, score_mod=score_mod,
                                             block_mask=keep), atol=1e-5, rtol=1e-5)


def test_malicious_zero_score_mod_gives_exactly_zero_weight(km):
    """Regression for RQ-16 / C07: a score_mod returning 0.0 at masked positions."""
    torch.manual_seed(6)
    t = 5
    q = k = torch.randn(1, 1, t, 4)
    v = torch.eye(t).expand(1, 1, t, t).contiguous()
    keep = torch.ones(t, t, dtype=torch.bool)
    keep[:, 1] = False

    def restore_masked(scores, q_idx, k_idx):
        return torch.where(torch.isneginf(scores), torch.zeros_like(scores), scores)

    w = km.maskmod_attn(q, k, v, score_mod=restore_masked, block_mask=keep, causal=True)
    assert (w[..., :, 1] == 0.0).all()
    upper = (torch.arange(t)[:, None] < torch.arange(t)[None, :]).expand_as(w)
    assert (w[upper] == 0.0).all()


def test_cuda_matches_cpu_or_skip(km):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA GPU")
    torch.manual_seed(4)
    q = k = v = torch.randn(1, 2, 8, 16)
    y_cpu = km.maskmod_attn(q, k, v, causal=True)
    y_cuda = km.maskmod_attn(q.cuda(), k.cuda(), v.cuda(), causal=True)
    torch.testing.assert_close(y_cuda.cpu(), y_cpu, atol=1e-5, rtol=1e-5)
