"""Terminal report formatter."""

from __future__ import annotations

from ..types import BenchmarkReport, MetricResult


def _format_metric(metric: MetricResult, query_count: int) -> str:
    if metric.percentiles is None:
        return f"  {metric.label:<18} {metric.value:>8.2%}"
    p = metric.percentiles
    return (
        f"  {metric.label:<18} {metric.value:>8.2f}s"
        f"  (p50={p.p50:.2f}s, p95={p.p95:.2f}s, p99={p.p99:.2f}s)"
    )


def format_report(report: BenchmarkReport) -> str:
    lines = ["=" * 60]
    lines.append(f"Dataset:        {report.name}")
    lines.append(f"Memory System:   {report.memory_system}")
    lines.append(f"Corpus Docs:     {report.context_tree_docs}")
    lines.append(f"Queries:         {report.query_count}")
    lines.append(f"Duration:        {report.duration_ms / 1000:.1f}s")
    lines.append("-" * 60)
    for metric in report.metrics:
        lines.append(_format_metric(metric, report.query_count))
    if report.category_breakdown:
        lines.append("-" * 60)
        for category in report.category_breakdown:
            lines.append(f"{category.category} ({category.query_count})")
            for metric in category.metrics:
                lines.append(_format_metric(metric, category.query_count))
    lines.append("=" * 60)
    return "\n".join(lines)
