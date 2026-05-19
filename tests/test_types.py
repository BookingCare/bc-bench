import unittest

from bc_bench.types import BenchmarkDataset, CorpusDocument, GroundTruthEntry


class TypesTests(unittest.TestCase):
    def test_types_round_trip(self):
        dataset = BenchmarkDataset(
            name="bc-bench",
            corpus=(CorpusDocument(doc_id="d1", content="c1", source="s1"),),
            entries=(GroundTruthEntry(query="q1", expected_doc_ids=("d1",)),),
        )

        self.assertEqual(dataset.name, "bc-bench")
        self.assertEqual(dataset.corpus[0].doc_id, "d1")
        self.assertEqual(dataset.entries[0].expected_doc_ids, ("d1",))
        self.assertEqual(dataset.metadata, {})
