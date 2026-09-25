# SPDX-License-Identifier: Apache-2.0
"""Source-owned workflow provenance contract."""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_kernel_smoke_binds_exact_event_source() -> None:
    workflow = (ROOT / ".github/workflows/kernel-smoke.yml").read_text(encoding="utf-8")
    revision = "${{ github.event.pull_request.head.sha || github.sha }}"

    assert f"SOURCE_REVISION: {revision}" in workflow
    assert f"ref: {revision}" in workflow
    assert "persist-credentials: false" in workflow
    assert 'actual="$(git rev-parse HEAD)"' in workflow
    assert 'test "${actual}" = "${SOURCE_REVISION}"' in workflow
