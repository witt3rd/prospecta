"""`prospecta eval QUESTIONS.md` — score recall, ablate channels, report cost and latency."""
from __future__ import annotations

import sys


def cmd_eval(args) -> int:
    from prospecta import evaluation
    from prospecta.cli import _common

    try:
        questions = evaluation.load_questions(args.questions)
    except (OSError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    try:
        memory = _common.make_memory(args)
    except SystemExit as e:
        return int(e.code) if isinstance(e.code, int) else 1
    if memory._embed is None:
        print("error: eval needs an embedder", file=sys.stderr)
        return 2
    if (args.synth or args.judge) and memory._llm is None:
        print("error: --synth / --judge need an llm", file=sys.stderr)
        return 2
    try:
        report = evaluation.run_eval(
            memory, questions, ablate=args.ablate, synth=args.synth or args.judge,
            judge=args.judge,
            rerank_blend=getattr(args, "rerank_blend", None),
            scope_promote=getattr(args, "scope_promote", None))
    except Exception as e:
        print(f"error: eval failed: {e}", file=sys.stderr)
        return 2
    print(evaluation.report_json(report) if args.json else evaluation.format_report(report))
    return 0
