"""Generate the synthetic receipt corpus.

    python scripts/generate_synthetic.py --count 150 --corrupted 12

Writes image + ground-truth pairs into ``tests/fixtures/synthetic/``. Generation is
seeded, so the corpus is reproducible and can be regenerated larger at any time.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.eval.groundtruth import SYNTHETIC_DIR  # noqa: E402
from app.eval.synthetic.corpus import generate_corpus  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic receipts")
    parser.add_argument("--count", type=int, default=150,
                        help="arithmetically sound receipts")
    parser.add_argument("--corrupted", type=int, default=12,
                        help="receipts whose VAT identity deliberately fails")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", type=Path, default=SYNTHETIC_DIR)
    parser.add_argument("--keep", action="store_true",
                        help="keep existing files instead of clearing the directory")
    args = parser.parse_args()

    if not args.keep and args.output.exists():
        shutil.rmtree(args.output)

    stats = generate_corpus(
        args.output, count=args.count, corrupted_count=args.corrupted, seed=args.seed
    )

    print(f"\nWrote {stats.written} receipts to {args.output}\n")
    print("  by template")
    for name in sorted(stats.by_template):
        print(f"    {name:<24}{stats.by_template[name]:>4}")
    print("  by degradation")
    for name in sorted(stats.by_tier):
        print(f"    {name:<24}{stats.by_tier[name]:>4}")
    print(f"  deliberately inconsistent {stats.corrupted:>4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
