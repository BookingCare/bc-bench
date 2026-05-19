"""Generate a synthetic benchmark from data.csv using Azure OpenAI."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ..datasets.bc_bench import (
    build_corpus,
    build_generation_prompt,
    build_dataset,
    chunk,
    parse_json_block,
    read_rows,
    validate_cases,
)
from ..types import GroundTruthEntry, CorpusDocument

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO_ROOT / "data.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "assets" / "sample_data"
DEFAULT_OUTPUT = DEFAULT_OUTPUT_DIR / "bc-bench.json"
DEFAULT_DEPLOYMENT = "gpt-5.4-nano"
DEFAULT_API_VERSION = "2024-12-01-preview"
DEFAULT_BATCH_SIZE = 4
DEFAULT_MAX_TOKENS = 4000
DEFAULT_TEMPERATURE = 0.2


class CliOptions(argparse.Namespace):
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


def parse_args(argv: list[str] | None = None) -> CliOptions:
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

    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    if args.max_rows is not None and args.max_rows < 1:
        parser.error("--max-rows must be >= 1")
    if not 0 <= args.temperature <= 2:
        parser.error("--temperature must be between 0 and 2")

    return args  # type: ignore[return-value]


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
        headers={"Content-Type": "application/json", "api-key": api_key},
        method="POST",
    )

    try:
        with urlopen(request, timeout=120) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        raise RuntimeError(
            f"Azure OpenAI request failed: {error.read().decode('utf-8')}"
        ) from error
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


def generate_dataset(
    *,
    name: str,
    input_path: Path,
    docs: tuple[CorpusDocument, ...],
    endpoint: str,
    api_key: str,
    api_version: str,
    deployment: str,
    batch_size: int,
    temperature: float,
    max_tokens: int,
) -> BenchmarkDataset:
    entries: list[GroundTruthEntry] = []
    batches = chunk(list(docs), batch_size)

    for batch_index, batch in enumerate(batches, start=1):
        print(f"Generating batch {batch_index}/{len(batches)} ({len(batch)} docs)")
        messages = build_generation_prompt(batch)
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
                GroundTruthEntry(
                    query=case["query"],
                    expected_doc_ids=(doc.doc_id,),
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

    return build_dataset(name=name, corpus=docs, entries=tuple(entries), metadata=metadata)


def main(argv: list[str] | None = None) -> int:
    options = parse_args(argv)

    if not options.input.exists():
        print(f"Input file not found: {options.input}", file=sys.stderr)
        return 1

    rows = read_rows(options.input)
    if options.max_rows is not None:
        rows = rows[: options.max_rows]

    docs = build_corpus(rows)
    if not docs:
        print(f"No rows found in {options.input}", file=sys.stderr)
        return 1

    if options.dry_run:
        print(f"Rows: {len(rows)}")
        print(f"Documents: {len(docs)}")
        print(f"Batches: {(len(docs) + options.batch_size - 1) // options.batch_size}")
        print(f"Output: {options.output}")
        return 0

    if not options.endpoint or not options.api_key:
        print(
            "Missing Azure OpenAI configuration. Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY, or pass --endpoint and --api-key.",
            file=sys.stderr,
        )
        return 1

    dataset = generate_dataset(
        name=options.name,
        input_path=options.input,
        docs=docs,
        endpoint=options.endpoint,
        api_key=options.api_key,
        api_version=options.api_version,
        deployment=options.deployment,
        batch_size=options.batch_size,
        temperature=options.temperature,
        max_tokens=options.max_tokens,
    )

    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(
        json.dumps(asdict(dataset), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {len(dataset.entries)} cases from {len(dataset.corpus)} documents.")
    print(f"Output: {options.output}")
    return 0
