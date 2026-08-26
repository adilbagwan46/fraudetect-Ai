from __future__ import annotations

import argparse
import json
from pathlib import Path

from ml.fraudetect_ml.data.relationship_index import build_relationship_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the ignored, label-free causal relationship lookup index."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/manifest.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/relationship/history.sqlite"),
    )
    parser.add_argument("--chunksize", type=int, default=250_000)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing generated index.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_relationship_index(
        args.manifest,
        args.output,
        chunksize=args.chunksize,
        overwrite=args.force,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
