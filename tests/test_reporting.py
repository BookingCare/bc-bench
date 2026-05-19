import unittest

from bc_bench.reporting.terminal import format_report
from bc_bench.types import BenchmarkReport, MetricResult


class ReportingTests(unittest.TestCase):
    def test_format_report_contains_metrics(self):
        report = BenchmarkReport(
            name="bc-bench",
            memory_system="demo",
            context_tree_docs=2,
            query_count=1,
            duration_ms=1234.0,
            metrics=(MetricResult(name="precision@10", label="Precision@10", value=0.5, unit="ratio"),),
        )

        output = format_report(report)
        self.assertIn("bc-bench", output)
        self.assertIn("Precision@10", output)
