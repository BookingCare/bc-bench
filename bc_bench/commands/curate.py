"""Curate helpers for bc-bench."""

from __future__ import annotations

from ..types import CorpusDocument


def curate_documents(corpus: tuple[CorpusDocument, ...]) -> tuple[dict[str, str], ...]:
    return tuple(
        {
            "doc_id": doc.doc_id,
            "source": doc.source,
            "content": doc.content,
        }
        for doc in corpus
    )
