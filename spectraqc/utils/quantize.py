from __future__ import annotations
import math


def q(x: float, step: float) -> float:
    """Quantize a float to the nearest step for stable hashing."""
    if x is None or math.isnan(x) or math.isinf(x):
        return x
    inv = 1.0 / step
    y = x * inv
    if y >= 0:
        yq = math.floor(y + 0.5)
    else:
        yq = -math.floor(-y + 0.5)
    return yq / inv


def q_list(xs: list[float], step: float) -> list[float]:
    """Quantize a list of floats."""
    return [q(float(v), step) for v in xs]
