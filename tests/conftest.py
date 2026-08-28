# SPDX-License-Identifier: Apache-2.0
"""Load KERNEL package: get_kernel when available, else torch-ext."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "torch-ext"
if str(EXT) not in sys.path:
    sys.path.insert(0, str(EXT))


def load_kernel():
    try:
        from kernels import get_kernel

        return get_kernel("SZLHOLDINGS/szl-maskmod", revision="main", trust_remote_code=True)
    except Exception:
        import szl_maskmod

        return szl_maskmod


@pytest.fixture(scope="module")
def km():
    return load_kernel()
