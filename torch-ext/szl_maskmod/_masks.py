# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SZL Holdings
"""Helpers to build a dense keep-mask and a block occupancy map.

Additive polish on KERNEL's maskmod_attn, which applies ``block_mask`` as a
bool keep-tensor (True = participate). Occupancy is for inspection / receipts;
it is not a FlexAttention BlockMask rehost.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch


def create_dense_mask(
    q_len: int,
    kv_len: int,
    *,
    causal: bool = False,
    sliding_window: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Bool keep-mask ``[q_len, kv_len]``. True means the score is kept."""
    if device is None:
        device = torch.device("cpu")
    q_idx = torch.arange(q_len, device=device)[:, None]
    k_idx = torch.arange(kv_len, device=device)[None, :]
    keep = torch.ones(q_len, kv_len, dtype=torch.bool, device=device)
    if causal:
        keep = keep & ~(q_idx < k_idx)
    if sliding_window is not None:
        if sliding_window < 1:
            raise ValueError("sliding_window must be >= 1")
        keep = keep & ((q_idx - k_idx) < sliding_window)
    return keep


def compile_block_mask(
    keep: torch.Tensor,
    *,
    block_m: int = 8,
    block_n: int = 8,
) -> torch.Tensor:
    """Tile occupancy ``[..., n_q_blocks, n_kv_blocks]`` as uint8 (0/1)."""
    if keep.dim() < 2:
        raise ValueError("keep mask must have at least 2 dims")
    if block_m < 1 or block_n < 1:
        raise ValueError("block sizes must be >= 1")
    q_len, kv_len = int(keep.shape[-2]), int(keep.shape[-1])
    n_qb = (q_len + block_m - 1) // block_m
    n_kb = (kv_len + block_n - 1) // block_n
    pad_q = n_qb * block_m - q_len
    pad_k = n_kb * block_n - kv_len
    tiled = torch.nn.functional.pad(keep.to(torch.bool), (0, pad_k, 0, pad_q), value=False)
    lead = tiled.shape[:-2]
    tiled = tiled.reshape(*lead, n_qb, block_m, n_kb, block_n)
    return tiled.any(dim=-1).any(dim=-2).to(torch.uint8)
