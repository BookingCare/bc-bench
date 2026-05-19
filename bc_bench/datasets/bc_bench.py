"""Dataset helpers for bc-bench."""

from __future__ import annotations

import csv
import html
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..types import BenchmarkDataset, CorpusDocument, GroundTruthEntry, PromptConfig
from . import register

ALLOWED_CATEGORIES = (
    "location",
    "pricing",
    "services",
    "team",
    "overview",
    "testimonial",
    "mixed",
)

HTML_TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class ClinicRow:
    type: str
    name: str
    content: str
    province_city: str
    image: str
    additional: str
    market: str


CURATE_TEMPLATE = """\
You are indexing a dental clinic profile into a context tree.
Follow the existing file structure exactly and keep the extracted facts concise.

Now index this content:

doc_id: {doc_id}
source: {source}

{content}
"""

QUERY_TEMPLATE = "{question}"

JUSTIFIER_TEMPLATE = """\
Answer the question using only the provided clinic context.
Question: {question}
Context:
{context}
Answer:
"""

PROMPT_CONFIG = PromptConfig(
    curate_template=CURATE_TEMPLATE,
    query_template=QUERY_TEMPLATE,
    justifier_template=JUSTIFIER_TEMPLATE,
)

register("bc-bench", PROMPT_CONFIG)


def strip_html(value: str) -> str:
    text = html.unescape(value)
    text = HTML_TAG_RE.sub(" ", text)
    text = text.replace("\r", " ").replace("\n", " ")
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


def truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 1].rstrip() + "…"


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower())
    return re.sub(r"-+", "-", slug).strip("-")


def read_rows(path: Path) -> list[ClinicRow]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows: list[ClinicRow] = []
        for raw in reader:
            rows.append(
                ClinicRow(
                    type=(raw.get("Type") or "").strip(),
                    name=(raw.get("Name") or "").strip(),
                    content=(raw.get("Content") or "").strip(),
                    province_city=(raw.get("Province/City") or "").strip(),
                    image=(raw.get("Image") or "").strip(),
                    additional=(raw.get("Additional") or "").strip(),
                    market=(raw.get("Market") or "").strip(),
                )
            )
    return rows


def build_document(row: ClinicRow, index: int) -> CorpusDocument:
    name = row.name or f"row-{index + 1}"
    slug = slugify(name) or f"row-{index + 1}"
    parts = [
        f"Name: {name}",
        f"Type: {row.type}",
    ]
    if row.province_city:
        parts.append(f"Province/City: {row.province_city}")
    if row.additional:
        parts.append(f"Additional: {row.additional}")
    if row.market:
        parts.append(f"Market: {row.market}")
    parts.append(f"Content: {truncate(strip_html(row.content), 2400)}")

    return CorpusDocument(
        doc_id=f"clinic-{index + 1:03d}-{slug}",
        content="\n\n".join(parts),
        source=row.province_city or row.market or row.type,
    )


def build_corpus(rows: list[ClinicRow]) -> tuple[CorpusDocument, ...]:
    return tuple(build_document(row, index) for index, row in enumerate(rows))


def chunk(values: list[Any], size: int) -> list[list[Any]]:
    return [values[i : i + size] for i in range(0, len(values), size)]


def build_generation_prompt(batch: list[CorpusDocument]) -> list[dict[str, str]]:
    system = (
        "You generate synthetic benchmark items for a retrieval benchmark. "
        "Return strict JSON only. Produce exactly one case for each input document, "
        "in the same order. Each case must be answerable from the corresponding "
        "document alone. Use concise, specific questions. Category must be one of: "
        + ", ".join(ALLOWED_CATEGORIES)
        + ". Expected answers should be short phrases or sentences, not full explanations."
    )

    user_lines = [
        "Create one benchmark case per document below.",
        "Return JSON with this exact shape:",
        '{"cases":[{"query":"...","expected_answer":"...","category":"..."}]}',
        "Rules:",
        f"- Return exactly {len(batch)} cases.",
        "- Preserve the input order.",
        "- Use only the facts present in each document.",
        "- Vary the query style across the batch when possible.",
        "- Do not mention document IDs in the query.",
        "- Do not add markdown fences or commentary.",
        "Documents:",
    ]

    for index, doc in enumerate(batch, start=1):
        user_lines.append(
            f"{index}. doc_id={doc.doc_id}\nsource={doc.source}\ncontent={doc.content}"
        )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n\n".join(user_lines)},
    ]


def parse_json_block(text: str) -> dict[str, Any]:
    stripped = text.strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise RuntimeError(f"Model response was not JSON: {stripped[:200]}")
    parsed = json.loads(stripped[start : end + 1])
    if not isinstance(parsed, dict):
        raise RuntimeError("Model output was not a JSON object")
    return parsed


def validate_cases(payload: dict[str, Any], expected_count: int) -> list[dict[str, str]]:
    cases = payload.get("cases")
    if not isinstance(cases, list):
        raise RuntimeError("Model output is missing cases array")
    if len(cases) != expected_count:
        raise RuntimeError(f"Model returned {len(cases)} cases, expected {expected_count}")

    normalized: list[dict[str, str]] = []
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise RuntimeError(f"Invalid case at index {index}")
        query = case.get("query")
        expected_answer = case.get("expected_answer")
        category = case.get("category")
        if not all(
            isinstance(value, str) and value.strip()
            for value in (query, expected_answer, category)
        ):
            raise RuntimeError(f"Invalid case at index {index}")
        normalized_category = category.strip()
        if normalized_category not in ALLOWED_CATEGORIES:
            raise RuntimeError(f"Invalid category at index {index}: {normalized_category}")
        normalized.append(
            {
                "query": query.strip(),
                "expected_answer": expected_answer.strip(),
                "category": normalized_category,
            }
        )
    return normalized


def build_dataset(
    *,
    name: str,
    corpus: tuple[CorpusDocument, ...],
    entries: tuple[GroundTruthEntry, ...],
    metadata: dict[str, Any] | None = None,
) -> BenchmarkDataset:
    return BenchmarkDataset(
        name=name,
        corpus=corpus,
        entries=entries,
        metadata=metadata or {},
    )
