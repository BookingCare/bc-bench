"""Recall@K metric."""

from __future__ import annotations

from typing import Sequence

from ..types import MetricResult
from .base import QueryPair, _avg, _retrieved_ids


class RecallAtK:
    def __init__(self, k: int = 10) -> None:
        self.k = k
        self.name = f"recall@{k}"
        self.label = f"Recall@{k}"

    def compute(self, pairs: Sequence[QueryPair]) -> tuple[MetricResult, ...]:
        values = []
        for execution, entry in pairs:
            retrieved = set(_retrieved_ids(execution)[: self.k])
            relevant = set(entry.expected_doc_ids)
            hits = len(retrieved & relevant)
            values.append(hits / len(relevant) if relevant else 0.0)
        return (
            MetricResult(
                name=self.name,
                label=self.label,
                value=_avg(values),
                unit="ratio",
            ),
        )
