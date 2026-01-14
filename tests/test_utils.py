from __future__ import annotations

from spectraqc.utils.canonical_json import canonical_dumps
from spectraqc.utils.hashing import sha256_hex_canonical_json
from spectraqc.utils.quantize import q, q_list


def test_canonical_dumps_is_deterministic():
    obj = {"b": 1, "a": 2, "nested": {"z": 1, "y": 2}}
    assert canonical_dumps(obj) == '{"a":2,"b":1,"nested":{"y":2,"z":1}}'


def test_sha256_hex_canonical_json_matches_order():
    obj1 = {"b": 1, "a": 2}
    obj2 = {"a": 2, "b": 1}
    assert sha256_hex_canonical_json(obj1) == sha256_hex_canonical_json(obj2)


def test_quantize_rounding():
    assert q(1.234, 0.01) == 1.23
    assert q(1.235, 0.01) == 1.24
    assert q(-1.235, 0.01) == -1.24


def test_quantize_list():
    assert q_list([0.004, 0.005, 0.006], 0.01) == [0.0, 0.01, 0.01]
