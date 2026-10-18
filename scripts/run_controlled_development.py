#!/usr/bin/env python
"""Separate validation-only development entry point; historical CLI is untouched.

Use --plan-only for a deterministic, standard-library-only JSON plan. Without
that flag this is a real training operation (including possible ImageNet weight
downloads), which must be separately authorized by the operator.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.development.config import make_plan, parse_args


def main(argv: list[str] | None = None) -> int:
    try:
        config, plan_only = parse_args(argv)
        if plan_only:
            print(json.dumps(make_plan(config), sort_keys=True, indent=2, allow_nan=False))
        else:
            from scripts.development.runner import execute

            execute(config)
        return 0
    except KeyboardInterrupt:
        print("Controlled development interrupted; reserved output is not reusable.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Controlled development refused/failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
