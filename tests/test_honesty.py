# SPDX-License-Identifier: Apache-2.0
"""Honesty tests: no vendored FlexAttention/CuTeDSL imports; selfcheck labels."""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.kernels_ci

BANNED = (
    "flash_attn",
    "sageattn",
    "sage_attn",
    "vllm",
    "cutlass",
    "cute",
    "torch.nn.attention.flex_attention",
)


def test_selfcheck_no_speedup_claim(km):
    report = km.selfcheck()
    assert report["ok"] is True
    assert "Conjecture 1" in str(report.get("lambda", ""))
    note = str(report.get("note", "")).lower()
    assert "speedup" not in note or "no speedup" in note


def test_no_vendored_attention_imports():
    root = Path(__file__).resolve().parents[1] / "torch-ext" / "szl_maskmod"
    for path in sorted(root.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    for banned in BANNED:
                        assert not alias.name.startswith(banned), path.name
            elif isinstance(node, ast.ImportFrom) and node.module:
                for banned in BANNED:
                    assert not node.module.startswith(banned), path.name
