#!/usr/bin/env python3
"""Generate a synthetic retrieval benchmark from data.csv using Azure OpenAI.

The output matches the brv-bench ground-truth shape:

{
  "name": "...",
  "corpus": [{"doc_id": "...", "content": "...", "source": "..."}],
  "entries": [{
    "query": "...",
    "expected_doc_ids": ["..."],
    "expected_answer": "...",
    "category": "..."
  }],
  "metadata": {...}
}
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import re
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "data.csv"
DEFAULT_OUTPUT = REPO_ROOT / "generated" / "bc-bench.json"
DEFAULT_DEPLOYMENT = "gpt-5.4-nano"
DEFAULT_API_VERSION = "2024-12-01-preview"
DEFAULT_BATCH_SIZE = 4
DEFAULT_MAX_TOKENS = 4000
DEFAULT_TEMPERATURE = 0.2

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


@dataclass(frozen=True)
class ClinicDocument:
    doc_id: str
    content: str
    source: str


@dataclass(frozen=True)
class BenchmarkEntry:
    query: str
    expected_doc_ids: list[str]
    expected_answer: str
    category: str


@dataclass(frozen=True)
class BenchmarkDataset:
    name: str
    corpus: list[ClinicDocument]
    entries: list[BenchmarkEntry]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class CliOptions:
    input_path: Path
    output_path: Path
    name: str
    deployment: str
    endpoint: str | None
    api_key: str | None
    api_version: str
    batch_size: int
    max_rows: int | None
    temperature: float
    max_tokens: int
    dry_run: bool


def parse_args() -> CliOptions:
    parser = argparse.ArgumentParser(
        description="Generate synthetic benchmark cases from data.csv.",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--name", default="bc-bench")
    parser.add_argument("--deployment", default=DEFAULT_DEPLOYMENT)
    parser.add_argument("--endpoint", default=os.getenv("AZURE_OPENAI_ENDPOINT"))
    parser.add_argument("--api-key", default=os.getenv("AZURE_OPENAI_API_KEY"))
    parser.add_argument(
        "--api-version",
        default=os.getenv("AZURE_OPENAI_API_VERSION", DEFAULT_API_VERSION),
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    if args.max_rows is not None and args.max_rows < 1:
        parser.error("--max-rows must be >= 1")
    if not 0 <= args.temperature <= 2:
        parser.error("--temperature must be between 0 and 2")

    return CliOptions(
        input_path=args.input,
        output_path=args.output,
        name=args.name,
        deployment=args.deployment,
        endpoint=args.endpoint,
        api_key=args.api_key,
        api_version=args.api_version,
        batch_size=args.batch_size,
        max_rows=args.max_rows,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        dry_run=args.dry_run,
    )


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


def build_document(row: ClinicRow, index: int) -> ClinicDocument:
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

    return ClinicDocument(
        doc_id=f"clinic-{index + 1:03d}-{slug}",
        content="\n\n".join(parts),
        source=row.province_city or row.market or row.type,
    )


def chunk(values: list[Any], size: int) -> list[list[Any]]:
    return [values[i : i + size] for i in range(0, len(values), size)]


def build_prompt(batch: list[ClinicDocument]) -> list[dict[str, str]]:
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


def call_azure_chat_completion(
    *,
    endpoint: str,
    api_key: str,
    api_version: str,
    deployment: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
) -> str:
    base = endpoint.rstrip("/")
    url = (
        f"{base}/openai/deployments/{deployment}/chat/completions"
        f"?api-version={api_version}"
    )
    body = json.dumps(
        {
            "messages": messages,
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
        }
    ).encode("utf-8")

    request = Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "api-key": api_key,
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=120) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        raise RuntimeError(f"Azure OpenAI request failed: {error.read().decode('utf-8')}") from error
    except URLError as error:
        raise RuntimeError(f"Azure OpenAI request failed: {error}") from error

    choices = payload.get("choices") or []
    if not choices:
        raise RuntimeError("Azure OpenAI response did not include any choices")

    message = choices[0].get("message") or {}
    content = message.get("content")
    if not isinstance(content, str):
        raise RuntimeError("Azure OpenAI response did not include text content")

    return content


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
        if not all(isinstance(value, str) and value.strip() for value in (query, expected_answer, category)):
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


def generate_dataset(
    *,
    name: str,
    input_path: Path,
    docs: list[ClinicDocument],
    endpoint: str,
    api_key: str,
    api_version: str,
    deployment: str,
    batch_size: int,
    temperature: float,
    max_tokens: int,
) -> BenchmarkDataset:
    corpus = docs
    entries: list[BenchmarkEntry] = []
    batches = chunk(docs, batch_size)

    for batch_index, batch in enumerate(batches, start=1):
        print(f"Generating batch {batch_index}/{len(batches)} ({len(batch)} docs)")
        messages = build_prompt(batch)
        response_text = call_azure_chat_completion(
            endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
            deployment=deployment,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        payload = parse_json_block(response_text)
        cases = validate_cases(payload, len(batch))

        for doc, case in zip(batch, cases, strict=True):
            entries.append(
                BenchmarkEntry(
                    query=case["query"],
                    expected_doc_ids=[doc.doc_id],
                    expected_answer=case["expected_answer"],
                    category=case["category"],
                )
            )

    metadata = {
        "source_path": str(input_path),
        "deployment": deployment,
        "api_version": api_version,
        "batch_size": batch_size,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    return BenchmarkDataset(name=name, corpus=corpus, entries=entries, metadata=metadata)


def main() -> int:
    options = parse_args()

    if not options.input_path.exists():
        print(f"Input file not found: {options.input_path}", file=sys.stderr)
        return 1

    rows = read_rows(options.input_path)
    if options.max_rows is not None:
        rows = rows[: options.max_rows]

    docs = [build_document(row, index) for index, row in enumerate(rows)]
    if not docs:
        print(f"No rows found in {options.input_path}", file=sys.stderr)
        return 1

    if options.dry_run:
        print(f"Rows: {len(rows)}")
        print(f"Documents: {len(docs)}")
        print(f"Batches: {(len(docs) + options.batch_size - 1) // options.batch_size}")
        print(f"Output: {options.output_path}")
        return 0

    if not options.endpoint or not options.api_key:
        print(
            "Missing Azure OpenAI configuration. Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY, "
            "or pass --endpoint and --api-key.",
            file=sys.stderr,
        )
        return 1

    dataset = generate_dataset(
        name=options.name,
        input_path=options.input_path,
        docs=docs,
        endpoint=options.endpoint,
        api_key=options.api_key,
        api_version=options.api_version,
        deployment=options.deployment,
        batch_size=options.batch_size,
        temperature=options.temperature,
        max_tokens=options.max_tokens,
    )

    options.output_path.parent.mkdir(parents=True, exist_ok=True)
    options.output_path.write_text(
        json.dumps(asdict(dataset), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {len(dataset.entries)} cases from {len(dataset.corpus)} documents.")
    print(f"Output: {options.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
