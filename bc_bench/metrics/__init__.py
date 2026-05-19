"""Metric registry for bc-bench."""

from __future__ import annotations

from .base import Metric
from .latency import ColdLatency
from .mrr import MRR
from .ndcg import NDCGAtK
from .precision import PrecisionAtK
from .recall import RecallAtK


def default_metrics() -> list[Metric]:
    return [
        PrecisionAtK(10),
        RecallAtK(10),
        NDCGAtK(10),
        MRR(),
        ColdLatency(),
    ]
