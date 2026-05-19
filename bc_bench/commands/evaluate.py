"""Evaluation helpers for bc-bench."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from ..metrics import default_metrics
from ..metrics.base import Metric
from ..reporting.terminal import format_report
from ..types import (
    BenchmarkDataset,
    BenchmarkReport,
    CategoryResult,
    CorpusDocument,
    GroundTruthEntry,
    MetricResult,
    QueryExecution,
    SearchResult,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO_ROOT / "assets" / "sample_data" / "bc-bench.json"
DEFAULT_MEMORY_SYSTEM = "demo"


class CliOptions(argparse.Namespace):
    input_path: Path
    memory_system: str
    dry_run: bool

QueryPair = tuple[QueryExecution, GroundTruthEntry]


def compute_metrics(
    metrics: Sequence[Metric],
    pairs: Sequence[QueryPair],
) -> list[MetricResult]:
    results: list[MetricResult] = []
    for metric in metrics:
        results.extend(metric.compute(pairs))
    return results


def compute_category_breakdown(
    metrics: Sequence[Metric],
    pairs: Sequence[QueryPair],
) -> tuple[CategoryResult, ...]:
    groups: dict[str, list[QueryPair]] = defaultdict(list)
    for qe, gt in pairs:
        groups[gt.category].append((qe, gt))

    breakdown: list[CategoryResult] = []
    for category in sorted(groups):
        group_pairs = groups[category]
        breakdown.append(
            CategoryResult(
                category=category,
                query_count=len(group_pairs),
                metrics=tuple(compute_metrics(metrics, group_pairs)),
            )
        )
    return tuple(breakdown)


def build_report(
    dataset: BenchmarkDataset,
    memory_system: str,
    pairs: Sequence[QueryPair],
    metrics: Sequence[Metric] | None = None,
    duration_ms: float = 0.0,
) -> BenchmarkReport:
    metric_list = list(metrics) if metrics is not None else default_metrics()
    return BenchmarkReport(
        name=dataset.name,
        memory_system=memory_system,
        context_tree_docs=len(dataset.corpus),
        query_count=len(dataset.entries),
        duration_ms=duration_ms,
        metrics=tuple(compute_metrics(metric_list, pairs)),
        category_breakdown=compute_category_breakdown(metric_list, pairs),
    )


def save_partial(
    output_path: Path,
    pairs: Sequence[QueryPair],
) -> None:
    output_path.write_text(f"completed={len(pairs)}\n")


def benchmark_pairs(
    dataset: BenchmarkDataset,
    limit: int = 10,
) -> list[QueryPair]:
    """Create a placeholder query-result pair list for tests and examples."""
    pairs: list[QueryPair] = []
    for entry in dataset.entries:
        results = tuple(
            SearchResult(path=doc_id, title="", score=1.0, excerpt="")
            for doc_id in entry.expected_doc_ids[:limit]
        )
        pairs.append(
            (
                QueryExecution(
                    query=entry.query,
                    results=results,
                    total_found=len(results),
                    duration_ms=1.0,
                    answer=entry.expected_answer,
                ),
                entry,
            )
        )
    return pairs


def parse_args(argv: list[str] | None = None) -> CliOptions:
    parser = argparse.ArgumentParser(description="Evaluate a benchmark dataset.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--memory-system", default=DEFAULT_MEMORY_SYSTEM)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _require_mapping(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object")
    return value


def _require_string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"{label} must be a non-empty string")
    return value.strip()


def _optional_string(value: Any, *, label: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise RuntimeError(f"{label} must be a string")
    return value.strip()


def _require_string_tuple(value: Any, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise RuntimeError(f"{label} must be a non-empty array of strings")
    items: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise RuntimeError(f"{label}[{index}] must be a non-empty string")
        items.append(item.strip())
    return tuple(items)


def load_dataset(path: Path) -> BenchmarkDataset:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise RuntimeError(f"Failed to read dataset: {path}") from error
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Invalid dataset JSON in {path}: {error}") from error

    data = _require_mapping(payload, label="Dataset")
    name = _require_string(data.get("name"), label="Dataset name")

    corpus_raw = data.get("corpus")
    if not isinstance(corpus_raw, list):
        raise RuntimeError("Dataset corpus must be an array")
    corpus: list[CorpusDocument] = []
    for index, item in enumerate(corpus_raw):
        doc = _require_mapping(item, label=f"Corpus item {index}")
        corpus.append(
            CorpusDocument(
                doc_id=_require_string(doc.get("doc_id"), label=f"Corpus item {index} doc_id"),
                content=_require_string(doc.get("content"), label=f"Corpus item {index} content"),
                source=_optional_string(doc.get("source"), label=f"Corpus item {index} source"),
            )
        )

    entries_raw = data.get("entries")
    if not isinstance(entries_raw, list):
        raise RuntimeError("Dataset entries must be an array")
    entries: list[GroundTruthEntry] = []
    for index, item in enumerate(entries_raw):
        entry = _require_mapping(item, label=f"Entry {index}")
        expected_answer = entry.get("expected_answer")
        if expected_answer is None:
            answer: str | None = None
        elif isinstance(expected_answer, str) and expected_answer.strip():
            answer = expected_answer.strip()
        else:
            raise RuntimeError(f"Entry {index} expected_answer must be a non-empty string or null")

        category = _require_string(entry.get("category", "unspecified"), label=f"Entry {index} category")
        entries.append(
            GroundTruthEntry(
                query=_require_string(entry.get("query"), label=f"Entry {index} query"),
                expected_doc_ids=_require_string_tuple(
                    entry.get("expected_doc_ids"),
                    label=f"Entry {index} expected_doc_ids",
                ),
                category=category,
                expected_answer=answer,
            )
        )

    metadata = data.get("metadata")
    if metadata is None:
        metadata = {}
    elif not isinstance(metadata, dict):
        raise RuntimeError("Dataset metadata must be a JSON object")

    return BenchmarkDataset(
        name=name,
        corpus=tuple(corpus),
        entries=tuple(entries),
        metadata=metadata,
    )


def main(argv: list[str] | None = None) -> int:
    options = parse_args(argv)

    if not options.input.exists():
        print(f"Input file not found: {options.input}", file=sys.stderr)
        return 1

    try:
        dataset = load_dataset(options.input)
    except RuntimeError as error:
        print(str(error), file=sys.stderr)
        return 1

    if options.dry_run:
        print(f"Dataset: {dataset.name}")
        print(f"Corpus docs: {len(dataset.corpus)}")
        print(f"Queries: {len(dataset.entries)}")
        print(f"Input: {options.input}")
        return 0

    pairs = benchmark_pairs(dataset)
    report = build_report(dataset, options.memory_system, pairs)
    print(format_report(report))
    return 0
