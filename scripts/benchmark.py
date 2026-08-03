"""Per-stage timing benchmark for the OCR pipeline.

Runs the production code paths stage by stage over a representative subset of the
synthetic corpus and reports average/p95 wall-clock per stage. This is the offline
replacement for a production timing log.

    docker compose exec worker python scripts/benchmark.py
    docker compose exec worker python scripts/benchmark.py --limit 3 --json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402

from app.core.extraction import Extraction, extract, fast_path_sufficient  # noqa: E402
from app.core.ocr_engine import OcrBundle, read_pooled, read_variant  # noqa: E402
from app.core.preprocessing import build_variants  # noqa: E402

DEFAULT_SUBSET = [
    "syn-0000-ph_nonvat_si-clean.jpg",
    "syn-0001-us_pos-clean.jpg",
    "syn-0002-ph_nonvat_si-clean.jpg",
    "syn-0003-ph_pos-clean.jpg",
    "syn-0004-us_pos-clean.jpg",
    "syn-0011-ph_pos-clean.jpg",
    "syn-0012-my_pos-clean.jpg",
    "syn-0013-my_pos-clean.jpg",
    "syn-0014-ph_nonvat_si-clean.jpg",
    "syn-0155-ph_vat_or-moderate.jpg",
    "syn-0156-ph_vat_or-moderate.jpg",
    "syn-0157-ph_vat_or-moderate.jpg",
    "syn-0158-ph_vat_or-severe.jpg",
    "syn-0159-ph_vat_or-severe.jpg",
]


def run_receipt(path: Path) -> tuple[dict[str, float], OcrBundle | None, Extraction | None]:
    """Time each production stage over one receipt image.

    Mirrors ``_process_page`` in app/core/pipeline.py: fast path, extract,
    sufficiency gate, escalation to the full pool, then the consolidated
    model-assist call. ``early_exit`` and ``escalated`` report which path ran.
    """
    bundle: OcrBundle | None = None
    result: Extraction | None = None
    stage = {"early_exit": False, "escalated": False, "llm_assist": 0.0}

    with Image.open(path) as img:
        t_pre = _elapsed(lambda: build_variants(img))

        async def pooled_fast():
            return await read_pooled(img, lang="eng")

        t_engine, bundle = _elapsed_async(pooled_fast())
        stage["early_exit"] = bool(bundle.early_exit)

        def run_extract():
            nonlocal result
            result = extract(bundle)
            return result

        t_extract = _elapsed(run_extract)

        if bundle.early_exit and not fast_path_sufficient(result):
            async def pooled_full():
                return await read_pooled(img, lang="eng", fast_path=False)

            t_escalate, bundle = _elapsed_async(pooled_full())
            t_engine += t_escalate
            stage["escalated"] = True

            def run_extract_full():
                nonlocal result
                result = extract(bundle)
                return result

            t_extract += _elapsed(run_extract_full)

        t_single = _elapsed(lambda: read_variant(img, psm=6))

    return {"preprocess": t_pre, "engine_total": t_engine, "extract": t_extract,
            "single_pass": t_single}, bundle, result, stage


def _elapsed(fn) -> float:
    started = time.perf_counter()
    fn()
    return time.perf_counter() - started


def _elapsed_async(coro) -> tuple[float, OcrBundle]:
    started = time.perf_counter()
    value = asyncio.run(coro)
    return time.perf_counter() - started, value


def summarize(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    keys = rows[0].keys()
    out: dict[str, dict[str, float]] = {}
    for key in keys:
        values = sorted(r[key] for r in rows)
        out[key] = {
            "avg": statistics.mean(values),
            "p50": statistics.median(values),
            "p95": values[min(len(values) - 1, int(len(values) * 0.95))],
            "min": values[0],
            "max": values[-1],
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="OCR pipeline stage timing benchmark")
    parser.add_argument("--fixtures", type=Path,
                        default=ROOT / "tests" / "fixtures" / "synthetic")
    parser.add_argument("--limit", type=int, default=None,
                        help="benchmark only the first N fixtures")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    names = [n for n in DEFAULT_SUBSET if (args.fixtures / n).exists()]
    if args.limit:
        names = names[:args.limit]
    if not names:
        print(f"no fixtures found under {args.fixtures}", file=sys.stderr)
        return 1

    rows: list[dict[str, float]] = []
    for name in names:
        path = args.fixtures / name
        print(f"benchmarking {name} ...", file=sys.stderr, flush=True)
        timings, _, _, stage = run_receipt(path)
        rows.append(timings)
        print(f"  engine_total={timings['engine_total']:.1f}s "
              f"extract={timings['extract']:.2f}s "
              f"single_pass={timings['single_pass']:.2f}s "
              f"[early_exit={stage['early_exit']} escalated={stage['escalated']}]",
              file=sys.stderr)

    summary = summarize(rows)
    if args.json:
        print(json.dumps({"receipts": rows, "summary": summary}, indent=2))
        return 0

    print("\nPER-STAGE TIMING (seconds)")
    print(f"{'stage':<16}{'avg':>8}{'p50':>8}{'p95':>8}{'min':>8}{'max':>8}")
    print("-" * 56)
    for key, stats in summary.items():
        print(f"{key:<16}{stats['avg']:>8.2f}{stats['p50']:>8.2f}{stats['p95']:>8.2f}"
              f"{stats['min']:>8.2f}{stats['max']:>8.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
