import json
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import asdict
from io import StringIO
from pathlib import Path

from bc_bench.cli import main
from bc_bench.types import BenchmarkDataset, CorpusDocument, GroundTruthEntry


class CliTests(unittest.TestCase):
    def _write_dataset(self) -> Path:
        dataset = BenchmarkDataset(
            name="demo-bench",
            corpus=(CorpusDocument(doc_id="d1", content="hello", source="hanoi"),),
            entries=(GroundTruthEntry(query="q", expected_doc_ids=("d1",), category="overview"),),
        )
        tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(tempdir.cleanup)
        path = Path(tempdir.name) / "dataset.json"
        path.write_text(json.dumps(asdict(dataset)), encoding="utf-8")
        return path

    def test_evaluate_dry_run_is_wired_up(self):
        dataset_path = self._write_dataset()
        buffer = StringIO()
        with redirect_stdout(buffer):
            code = main(["evaluate", "--input", str(dataset_path), "--dry-run"])

        output = buffer.getvalue()
        self.assertEqual(code, 0)
        self.assertIn("Dataset: demo-bench", output)
        self.assertIn("Corpus docs: 1", output)
        self.assertIn("Queries: 1", output)

    def test_generate_subcommand_is_still_supported(self):
        buffer = StringIO()
        with redirect_stdout(buffer):
            code = main(["generate", "--dry-run", "--max-rows", "1"])

        output = buffer.getvalue()
        self.assertEqual(code, 0)
        self.assertIn("Rows: 1", output)

    def test_evaluate_prints_report(self):
        dataset_path = self._write_dataset()
        buffer = StringIO()
        with redirect_stdout(buffer):
            code = main(["evaluate", "--input", str(dataset_path), "--memory-system", "demo"])

        output = buffer.getvalue()
        self.assertEqual(code, 0)
        self.assertIn("Memory System:   demo", output)
        self.assertIn("Precision@10", output)
