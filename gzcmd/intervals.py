"""Descriptive binomial Wilson intervals (95%, without continuity correction)."""
from math import sqrt


def wilson95(successes: int, total: int) -> tuple[float, float]:
    if total <= 0 or not 0 <= successes <= total:
        raise ValueError("Wilson interval requires 0 <= successes <= total and total > 0")
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return (0. if successes == 0 else max(0., center - half),
            1. if successes == total else min(1., center + half))
