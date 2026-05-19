"""NDCG@K metric."""

from __future__ import annotations

from math import log2
from typing import Sequence

from ..types import MetricResult
from .base import QueryPair, _avg, _retrieved_ids


class NDCGAtK:
    def __init__(self, k: int = 10) -> None:
        self.k = k
        self.name = f"ndcg@{k}"
        self.label = f"NDCG@{k}"

    def compute(self, pairs: Sequence[QueryPair]) -> tuple[MetricResult, ...]:
        values = []
        for execution, entry in pairs:
            retrieved = _retrieved_ids(execution)[: self.k]
            relevant = set(entry.expected_doc_ids)
            dcg = 0.0
            for index, doc_id in enumerate(retrieved, start=1):
                if doc_id in relevant:
                    dcg += 1 / log2(index + 1)
            ideal_hits = min(len(relevant), self.k)
            idcg = sum(1 / log2(index + 1) for index in range(1, ideal_hits + 1))
            values.append(dcg / idcg if idcg else 0.0)
        return (
            MetricResult(
                name=self.name,
                label=self.label,
                value=_avg(values),
                unit="ratio",
            ),
        )
