from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.app.services.risk_service import load_active_bundle
from ml.fraudetect_ml.modeling.artifacts import write_json
from ml.fraudetect_ml.modeling.reference_profile import build_reference_profile


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build deterministic Phase 2B evidence reference statistics"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/manifest.json"),
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("artifacts/models"),
    )
    parser.add_argument("--sample-size", type=int, default=250_000)
    args = parser.parse_args()

    latest = json.loads((args.artifact_root / "latest.json").read_text(encoding="utf-8"))
    bundle = load_active_bundle(args.artifact_root, require_reference_profile=False)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    training_path = Path(manifest["splits"]["train"]["path"])
    profile = build_reference_profile(
        training_path=training_path,
        manifest_path=args.manifest,
        model=bundle.model,
        model_version=latest["model_version"],
        sample_size=args.sample_size,
    )
    output = bundle.model_dir / "reference-profile.json"
    write_json(output, profile)
    print(
        f"Saved {profile['reference_profile_version']} from training steps "
        f"{profile['source_boundary']['min_step']}-"
        f"{profile['source_boundary']['max_step']}: {output}"
    )


if __name__ == "__main__":
    main()
