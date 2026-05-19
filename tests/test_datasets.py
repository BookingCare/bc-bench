import unittest
from io import StringIO
from contextlib import redirect_stdout

from bc_bench.datasets import available_datasets, get_prompt_config
from bc_bench.datasets.bc_bench import (
    ALLOWED_CATEGORIES,
    ClinicRow,
    build_corpus,
    build_document,
    build_generation_prompt,
    validate_cases,
)


class DatasetTests(unittest.TestCase):
    def test_document_builder_normalizes_html(self):
        doc = build_document(
            ClinicRow(
                type="Clinic",
                name="Alpha Dental",
                content="<p>Hello<br/>world</p>",
                province_city="Hanoi",
                image="",
                additional="Open daily",
                market="Vietnam",
            ),
            0,
        )

        self.assertEqual(doc.doc_id, "clinic-001-alpha-dental")
        self.assertIn("Hello world", doc.content)
        self.assertEqual(doc.source, "Hanoi")

    def test_build_corpus_and_prompt_registration(self):
        rows = [
            ClinicRow("Clinic", "A", "X", "Hanoi", "", "", "Vietnam"),
            ClinicRow("Place", "B", "Y", "", "", "", "Vietnam"),
        ]
        corpus = build_corpus(rows)
        prompt = get_prompt_config("bc-bench")

        self.assertEqual(len(corpus), 2)
        self.assertTrue(corpus[0].doc_id.startswith("clinic-001"))
        self.assertEqual(prompt.query_template, "{question}")
        self.assertIn("bc-bench", available_datasets())

        messages = build_generation_prompt(list(corpus))
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["content"].count("doc_id="), 2)

    def test_validate_cases_accepts_allowed_categories(self):
        payload = {
            "cases": [
                {
                    "query": "Where is it?",
                    "expected_answer": "Hanoi",
                    "category": ALLOWED_CATEGORIES[0],
                }
            ]
        }

        cases = validate_cases(payload, 1)
        self.assertEqual(cases[0]["category"], ALLOWED_CATEGORIES[0])
