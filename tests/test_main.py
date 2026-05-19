import unittest
from contextlib import redirect_stdout
from io import StringIO

from bc_bench.commands.generate import main


class MainTests(unittest.TestCase):
    def test_main_dry_run(self):
        buffer = StringIO()
        with redirect_stdout(buffer):
            code = main(["--dry-run", "--max-rows", "1"])

        out = buffer.getvalue()
        self.assertEqual(code, 0)
        self.assertIn("Rows: 1", out)
        self.assertIn("assets/sample_data/bc-bench.json", out)
