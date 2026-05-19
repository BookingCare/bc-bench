"""CLI wrapper for bc-bench."""

from __future__ import annotations

import sys

from .commands import evaluate, generate


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args[:1] == ["evaluate"]:
        return evaluate.main(args[1:])
    if args[:1] == ["generate"]:
        return generate.main(args[1:])
    return generate.main(args)


if __name__ == "__main__":
    raise SystemExit(main())
