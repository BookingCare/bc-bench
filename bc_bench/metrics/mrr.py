"""MRR metric."""

from __future__ import annotations

from typing import Sequence

from ..types import MetricResult
from .base import QueryPair, _avg, _relevant_rank


class MRR:
    def __init__(self) -> None:
        self.name = "mrr"
        self.label = "MRR"

    def compute(self, pairs: Sequence[QueryPair]) -> tuple[MetricResult, ...]:
        values = []
        for execution, entry in pairs:
            rank = _relevant_rank(execution, entry)
            values.append(1 / rank if rank else 0.0)
        return (
            MetricResult(
                name=self.name,
                label=self.label,
                value=_avg(values),
                unit="ratio",
            ),
        )
