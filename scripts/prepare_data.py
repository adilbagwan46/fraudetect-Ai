from __future__ import annotations

import argparse
from pathlib import Path

from backend.app.core.config import get_settings
from ml.fraudetect_ml.data.pipeline import prepare_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate, enrich, feature, and split transaction data"
    )
    parser.add_argument("--input", type=Path, default=Path("data/raw/paysim.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--source-kind",
        choices=("paysim_public_synthetic", "generated_demo_only"),
        default="paysim_public_synthetic",
    )
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(
            f"dataset not found: {args.input}. Place the genuine PaySim CSV at "
            "data/raw/paysim.csv or pass --input explicitly."
        )
    settings = get_settings()
    manifest = prepare_dataset(
        args.input,
        args.output_dir,
        enrichment_seed=settings.enrichment_seed,
        device_buckets=settings.enrichment_device_buckets,
        ip_buckets=settings.enrichment_ip_buckets,
        source_kind=args.source_kind,
    )
    print(
        f"Prepared {manifest['dataset']['rows']} rows; "
        f"manifest: {args.output_dir / 'manifest.json'}"
    )


if __name__ == "__main__":
    main()
