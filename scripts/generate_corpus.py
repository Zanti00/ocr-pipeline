"""Generate the synthetic receipt corpus.

    python scripts/generate_corpus.py
    python scripts/generate_corpus.py --count 300 --seed 7

Output is reproducible for a given seed, so the corpus is regenerable rather than
something to commit.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.eval.groundtruth import SYNTHETIC_DIR  # noqa: E402
from app.eval.synthetic import generate_corpus  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic receipts")
    parser.add_argument("--count", type=int, default=150)
    parser.add_argument("--corrupted", type=int, default=10,
                        help="receipts with deliberately broken arithmetic")
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--output", type=Path, default=SYNTHETIC_DIR)
    args = parser.parse_args()

    stats = generate_corpus(
        output_dir=args.output,
        count=args.count,
        corrupted_count=args.corrupted,
        seed=args.seed,
    )

    print(f"\nWrote {stats.written} receipts to {args.output}")
    print(f"  arithmetic-corrupt set: {stats.corrupted}")
    print("\n  by template")
    for template, number in sorted(stats.by_template.items()):
        print(f"    {template:<16}{number:>4}")
    print("\n  by degradation")
    for tier, number in sorted(stats.by_degradation.items()):
        print(f"    {tier:<16}{number:>4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
