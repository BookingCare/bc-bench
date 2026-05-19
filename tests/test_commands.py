import unittest

from bc_bench.commands.curate import curate_documents
from bc_bench.commands.evaluate import benchmark_pairs, build_report
from bc_bench.types import BenchmarkDataset, CorpusDocument, GroundTruthEntry


class CommandTests(unittest.TestCase):
    def test_curate_documents_and_report(self):
        corpus = (CorpusDocument(doc_id="d1", content="hello", source="hanoi"),)
        entries = (GroundTruthEntry(query="q", expected_doc_ids=("d1",)),)
        dataset = BenchmarkDataset(name="bc-bench", corpus=corpus, entries=entries)

        curated = curate_documents(corpus)
        pairs = benchmark_pairs(dataset)
        report = build_report(dataset, "demo", pairs)

        self.assertEqual(curated[0]["doc_id"], "d1")
        self.assertEqual(report.query_count, 1)
