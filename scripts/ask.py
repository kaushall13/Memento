"""Single query against Postgres -> print prompt + ranked ids/scores."""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _lib import build_real_service  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--question-time", default=None)
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--include-expired", action="store_true")
    args = ap.parse_args()

    from longmem.config import load_settings

    settings = load_settings(args.config)
    service = build_real_service(settings)
    qt = (
        datetime.strptime(args.question_time, "%Y/%m/%d (%a) %H:%M")
        if args.question_time
        else None
    )
    ctx, scored = service.answer_context(
        args.query, question_time=qt, include_expired=args.include_expired
    )
    print(ctx.to_prompt())
    rows = scored if args.k is None else scored[: args.k]
    for r in rows:
        print(
            json.dumps(
                {
                    "memory_id": r.memory.memory_id,
                    "final_score": round(r.final_score, 4),
                    "scores": {k: round(v, 4) for k, v in r.scores.items()},
                }
            )
        )


if __name__ == "__main__":
    main()
