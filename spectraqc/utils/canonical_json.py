from __future__ import annotations
import json


def canonical_dumps(obj) -> str:
    """Serialize object to canonical JSON (sorted keys, minimal whitespace)."""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
