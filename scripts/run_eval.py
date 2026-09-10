#!/usr/bin/env python3
"""Run the full RAG evaluation (golden dataset + judge) from the CLI.

This wires the *real* production pipeline from the environment and evaluates it
against ``data/evals/golden_set.json``. If Ollama is available in the docker
network it uses :class:`OllamaJudge` for LLM-as-judge scoring; otherwise it
falls back to the deterministic :class:`HeuristicJudge` so the command always
produces a report (and a CI-safe status code).

Usage:
    PYTHONPATH=src python scripts/run_eval.py [--golden data/evals/golden_set.json]
                                            [--chroma-dir data/chroma]
                                            [--json]
                                            [--topic reentrancy]

Exit codes:
    0  PASS   (regression guard green)
    1  FAIL   (regression guard red — a metric breached a threshold)
    2  WARN   (a metric is approaching a threshold)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from smart_contract_rag.builders import build_pipeline
from smart_contract_rag.config import Settings
from smart_contract_rag.evals.dataset import GoldenDataset
from smart_contract_rag.evals.judge import HeuristicJudge, OllamaJudge
from smart_contract_rag.evals.runner import RegressionStatus, run_eval


def _select_judge(settings: Settings) -> HeuristicJudge | OllamaJudge:
    """Prefer the LLM judge when configured, else the deterministic one."""
    try:
        return OllamaJudge(base_url=settings.ollama_url, model=settings.ollama_model)
    except ValueError:
        return HeuristicJudge()


def _looks_available() -> bool:
    """Cheap heuristic: skip the LLM judge in CI-like environments."""
    import os

    return os.environ.get("SMART_CONTRACT_RAG_USE_LLM_JUDGE", "1").lower() not in {
        "0",
        "false",
        "no",
    }


def main(argv: list[str] | None = None) -> int:
    default_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--golden",
        default=str(default_root / "data" / "evals" / "golden_set.json"),
        help="Path to the golden dataset JSON.",
    )
    parser.add_argument(
        "--chroma-dir",
        default="data/chroma",
        help="Persistent ChromaDB directory (default: data/chroma).",
    )
    parser.add_argument(
        "--json", action="store_true", help="Print the report as JSON instead of markdown."
    )
    parser.add_argument(
        "--topic",
        default=None,
        help="Restrict the eval to a single golden topic (e.g. reentrancy).",
    )
    parser.add_argument(
        "--no-llm-judge",
        action="store_true",
        help="Force the deterministic heuristic judge (no Ollama).",
    )
    args = parser.parse_args(argv)

    settings = Settings.from_env()
    dataset = GoldenDataset.from_json(args.golden)
    if args.topic:
        dataset = dataset.subset_by_topic(args.topic)

    pipeline = build_pipeline(settings, persist_dir=args.chroma_dir)
    judge = HeuristicJudge()
    if not args.no_llm_judge and _looks_available():
        judge = _select_judge(settings)

    report = run_eval(pipeline, dataset, judge=judge)

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(report.to_markdown())

    # Map verdict to exit code so CI can gate on it.
    code_map = {
        RegressionStatus.PASS: 0,
        RegressionStatus.FAIL: 1,
        RegressionStatus.WARN: 2,
    }
    return code_map[report.status]


if __name__ == "__main__":
    raise SystemExit(main())
