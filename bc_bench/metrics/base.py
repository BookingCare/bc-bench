"""Metric base types for bc-bench."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Protocol, Sequence

from ..types import GroundTruthEntry, MetricResult, Percentiles, QueryExecution

QueryPair = tuple[QueryExecution, GroundTruthEntry]


class Metric(Protocol):
    name: str
    label: str

    def compute(self, pairs: Sequence[QueryPair]) -> tuple[MetricResult, ...]: ...


def _avg(values: Sequence[float]) -> float:
    return mean(values) if values else 0.0


def _percentiles(values: Sequence[float]) -> Percentiles:
    if not values:
        return Percentiles(p50=0.0, p95=0.0, p99=0.0)

    ordered = sorted(values)
    last = len(ordered) - 1
    return Percentiles(
        p50=ordered[int(round(last * 0.50))],
        p95=ordered[int(round(last * 0.95))],
        p99=ordered[int(round(last * 0.99))],
    )


def _relevant_rank(execution: QueryExecution, entry: GroundTruthEntry) -> int | None:
    relevant = set(entry.expected_doc_ids)
    for index, result in enumerate(execution.results, start=1):
        if result.path in relevant:
            return index
    return None


def _retrieved_ids(execution: QueryExecution) -> tuple[str, ...]:
    return tuple(result.path for result in execution.results)
