"""Dependency-free helpers for latency distribution reporting."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def percentile_stats(samples: Iterable[float]) -> dict[str, float | int]:
    """Return average/p50/p95/p99/max in seconds for one stage."""

    values = np.asarray(tuple(float(sample) for sample in samples), dtype=np.float64)
    if values.size == 0:
        return {
            "count": 0,
            "average": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0.0,
        }
    return {
        "count": int(values.size),
        "average": float(np.mean(values)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }
