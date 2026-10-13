# SPDX-License-Identifier: Apache-2.0
"""Smoke hook for hf-kernel-publish (szl-maskmod).

`smoke(kernel)` receives the module that the `kernels` loader imported, from a local
build (check-local) or from the Hub (verify). Raise, or return False, to fail the run.
CPU-only, deterministic and fast. The result's repr is written to the receipt.

Besides selfcheck (causal output vs SDPA plus a verifying receipt chain), it checks the
#5 fix: two block masks with the same number of kept scores but a different topology
must produce different receipts.
"""
from __future__ import annotations


def smoke(kernel):
    import torch

    report = kernel.selfcheck()
    if report.get("ok") is not True:
        raise AssertionError(f"selfcheck failed: {report}")

    torch.manual_seed(5)
    q = k = v = torch.randn(1, 1, 4, 8)
    mask_a = torch.tensor([[1, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 1]], dtype=torch.bool)
    mask_b = torch.tensor([[1, 0, 0, 1], [0, 1, 1, 0], [1, 1, 0, 0], [0, 0, 1, 1]], dtype=torch.bool)
    chain_a, chain_b = kernel.ReceiptChain(), kernel.ReceiptChain()
    kernel.maskmod_attn(q, k, v, block_mask=mask_a, chain=chain_a)
    kernel.maskmod_attn(q, k, v, block_mask=mask_b, chain=chain_b)
    if chain_a.head() == chain_b.head():
        raise AssertionError("block-mask topology is not bound into the receipt (pre-#5 code)")
    if chain_a.verify() != (True, 1, -1) or chain_b.verify() != (True, 1, -1):
        raise AssertionError("receipt chain does not verify")

    keep = kernel.create_dense_mask(16, 16, causal=True)
    occupancy = kernel.compile_block_mask(keep, block_m=8, block_n=8)
    if occupancy.tolist() != [[1, 0], [1, 1]]:
        raise AssertionError(f"unexpected causal block occupancy: {occupancy.tolist()}")

    return {"selfcheck_ok": True, "max_abs_vs_sdpa_causal": report["max_abs_vs_sdpa_causal"],
            "topology_bound": True, "occupancy": occupancy.tolist()}
