import unittest

from bc_bench.metrics import default_metrics
from bc_bench.metrics.precision import PrecisionAtK
from bc_bench.types import GroundTruthEntry, QueryExecution, SearchResult


class MetricsTests(unittest.TestCase):
    def test_precision_and_default_metrics(self):
        entry = GroundTruthEntry(query="q", expected_doc_ids=("d1", "d2"))
        execution = QueryExecution(
            query="q",
            results=(
                SearchResult(path="d1", title="", score=1.0, excerpt=""),
                SearchResult(path="x", title="", score=0.5, excerpt=""),
            ),
            total_found=2,
            duration_ms=10.0,
        )

        metric = PrecisionAtK(2)
        result = metric.compute([(execution, entry)])[0]
        self.assertEqual(result.value, 0.5)
        self.assertTrue(default_metrics())
