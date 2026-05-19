"""Latency metric."""

from __future__ import annotations

from typing import Sequence

from ..types import MetricResult
from .base import QueryPair, _avg, _percentiles


class ColdLatency:
    def __init__(self) -> None:
        self.name = "cold-latency"
        self.label = "Cold Latency"

    def compute(self, pairs: Sequence[QueryPair]) -> tuple[MetricResult, ...]:
        values = [execution.duration_ms / 1000 for execution, _ in pairs]
        return (
            MetricResult(
                name=self.name,
                label=self.label,
                value=_avg(values),
                unit="s",
                percentiles=_percentiles(values),
            ),
        )
