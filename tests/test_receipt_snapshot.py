# SPDX-License-Identifier: Apache-2.0
"""Receipt history owns its input values; these tests require no torch."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "torch-ext" / "szl_maskmod" / "_chain.py"
_SPEC = importlib.util.spec_from_file_location("maskmod_receipt_snapshot", _PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
ReceiptChain = _MODULE.ReceiptChain


def test_emit_preserves_nested_values_after_caller_mutation():
    payload = {"q_shape": [1, 2, 8, 16], "metadata": {"values": [1, {"x": 2}]},
               "tags": ("fixture", ["original"])}
    expected = copy.deepcopy(payload)
    expected.update(seq=0, prev="0" * 64)
    encoded = json.dumps(expected, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha3_256(encoded).hexdigest()
    chain = ReceiptChain()
    assert chain.emit(payload) == digest
    payload["q_shape"][2] = 999
    payload["metadata"]["values"][1]["x"] = -1
    payload["tags"][1].append("changed")
    assert chain._rows == [{**expected, "digest": digest}]
    assert chain.verify() == (True, 1, -1)
    assert chain.head() == digest


def test_reused_payload_cannot_rewrite_earlier_rows():
    chain = ReceiptChain()
    payload = {"values": [1]}
    first = chain.emit(payload)
    payload["values"].append(2)
    second = chain.emit(payload)
    payload["values"].clear()
    assert chain._rows[0]["values"] == [1]
    assert chain._rows[1]["values"] == [1, 2]
    assert chain._rows[1]["prev"] == first
    assert chain.head() == second
    assert chain.verify() == (True, 2, -1)


def test_failed_emit_leaves_predecessor_unchanged():
    chain = ReceiptChain()
    chain.emit({"values": [1]})
    before = copy.deepcopy(chain._rows)
    with pytest.raises(TypeError):
        chain.emit({"unsupported": object()})
    assert chain._rows == before
    assert chain.verify() == (True, 1, -1)


def test_internal_tampering_is_still_detected():
    chain = ReceiptChain()
    chain.emit({"values": [1]})
    chain._rows[0]["values"].append(2)
    assert chain.verify() == (False, 0, 0)

