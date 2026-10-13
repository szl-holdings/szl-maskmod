# SPDX-License-Identifier: Apache-2.0
"""Load the KERNEL package from this checkout's torch-ext/ (the GitHub source).

Tests always exercise the source under review. What the Hub serves is checked
separately by hf-kernel-publish's verify job, through get_kernel(version=1).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "torch-ext"
if str(EXT) not in sys.path:
    sys.path.insert(0, str(EXT))


def load_kernel():
    import szl_maskmod

    loaded = Path(szl_maskmod.__file__).resolve().parent
    if loaded != (EXT / "szl_maskmod").resolve():
        raise RuntimeError(f"szl_maskmod imported from {loaded}, not this checkout's torch-ext")
    return szl_maskmod


@pytest.fixture(scope="module")
def km():
    return load_kernel()
