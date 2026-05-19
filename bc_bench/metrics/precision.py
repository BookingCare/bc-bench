"""Precision@K metric."""

from __future__ import annotations

from typing import Sequence

from ..types import MetricResult
from .base import Metric, QueryPair, _avg, _retrieved_ids


class PrecisionAtK:
    def __init__(self, k: int = 10) -> None:
        self.k = k
        self.name = f"precision@{k}"
        self.label = f"Precision@{k}"

    def compute(self, pairs: Sequence[QueryPair]) -> tuple[MetricResult, ...]:
        values = []
        for execution, entry in pairs:
            retrieved = _retrieved_ids(execution)[: self.k]
            relevant = set(entry.expected_doc_ids)
            hits = sum(1 for doc_id in retrieved if doc_id in relevant)
            values.append(hits / self.k if self.k else 0.0)
        return (
            MetricResult(
                name=self.name,
                label=self.label,
                value=_avg(values),
                unit="ratio",
            ),
        )
