"""Pack the unified MRL + DDD dataset into a single SQLite .db file.

Run once on the machine that has both source datasets downloaded. The output
file is self-contained — hand it to a teammate and they can train/evaluate
without needing the original MRL/DDD folders.

Usage:
    py export_dataset.py
    py export_dataset.py --out data/drowsiness.db --seed 42

Load the bundle from downstream code:

    from src.datasets import SQLiteDrowsinessDataset, make_weighted_sampler
    train_ds = SQLiteDrowsinessDataset("data/drowsiness.db", split="train",
                                       augment=True)
    sampler  = make_weighted_sampler(train_ds.samples)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.datasets import export_to_sqlite, read_sqlite_meta


DEFAULT_MRL = r"C:\Users\adith\OneDrive\Documents\engg2112\datasets\MRL"
DEFAULT_DDD = r"C:\Users\adith\OneDrive\Documents\engg2112\datasets\DDD"
DEFAULT_OUT = Path("data") / "drowsiness.db"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mrl", default=DEFAULT_MRL, help="MRL dataset root")
    parser.add_argument("--ddd", default=DEFAULT_DDD, help="DDD dataset root")
    parser.add_argument("--out", default=str(DEFAULT_OUT),
                        help="Output SQLite file path")
    parser.add_argument("--val-frac",  type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--seed",      type=int,   default=42)
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace the output file if it exists")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    info = export_to_sqlite(
        mrl_root=args.mrl,
        ddd_root=args.ddd,
        db_path=args.out,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
        overwrite=args.overwrite,
    )

    mb = info["size_bytes"] / (1024 ** 2)
    print()
    print(f"Wrote {info['db_path']}  ({mb:.1f} MB)")
    print(f"  source_counts : {info['source_counts']}")
    print(f"  split counts  : {info['counts']}")
    print(f"  pos_weight    : {info['pos_weight']:.4f}")
    print()
    print("Metadata:")
    for k, v in read_sqlite_meta(args.out).items():
        print(f"  {k:<14} {v}")


if __name__ == "__main__":
    main()
