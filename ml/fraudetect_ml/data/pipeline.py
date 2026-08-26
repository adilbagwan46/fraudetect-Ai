from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from ml.fraudetect_ml.data.contracts import (
    ML_EXCLUDED_COLUMNS,
    ML_FEATURE_COLUMNS,
    ML_TARGET_COLUMN,
)
from ml.fraudetect_ml.data.enrichment import add_demo_relationship_fields
from ml.fraudetect_ml.data.features import (
    add_investigation_only_features,
    add_safe_model_features,
)
from ml.fraudetect_ml.data.ingestion import load_paysim
from ml.fraudetect_ml.data.splitting import chronological_split


def prepare_dataset(
    source_path: Path,
    output_dir: Path,
    *,
    enrichment_seed: str,
    device_buckets: int = 10_000,
    ip_buckets: int = 5_000,
    source_kind: str = "paysim_public_synthetic",
) -> dict:
    raw = load_paysim(source_path)
    safe_featured = add_safe_model_features(raw)
    splits = chronological_split(safe_featured)

    output_dir.mkdir(parents=True, exist_ok=True)
    split_files: dict[str, dict] = {}
    for name, split in (
        ("train", splits.train),
        ("validation", splits.validation),
        ("test", splits.test),
    ):
        split = add_demo_relationship_fields(
            split,
            seed=enrichment_seed,
            device_buckets=device_buckets,
            ip_buckets=ip_buckets,
            source_kind=source_kind,
        )
        split = add_investigation_only_features(split)
        path = output_dir / f"{name}.csv"
        split.to_csv(path, index=False)
        split_files[name] = {
            "path": str(path),
            "rows": len(split),
            "fraud_rows": int(split["is_fraud"].sum()),
            "min_step": int(split["step"].min()),
            "max_step": int(split["step"].max()),
            "step_count": int(split["step"].nunique()),
            "fraction": len(split) / len(safe_featured),
        }

    digest = hashlib.sha256()
    with source_path.open("rb") as source_handle:
        for chunk in iter(lambda: source_handle.read(1024 * 1024), b""):
            digest.update(chunk)

    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "source": {
            "kind": source_kind,
            "path": str(source_path),
            "size_bytes": source_path.stat().st_size,
            "sha256": digest.hexdigest(),
            "relationship_fields": "deterministic_synthetic_enrichment",
            "enrichment_uses_label": False,
            "enrichment_configuration": {
                "device_buckets": device_buckets,
                "ip_buckets": ip_buckets,
                "depends_on_dataset_cardinality": False,
            },
        },
        "dataset": {
            "rows": len(safe_featured),
            "fraud_rows": int(safe_featured["is_fraud"].sum()),
            "columns": list(split.columns),
        },
        "ml_feature_contract": {
            "features": list(ML_FEATURE_COLUMNS),
            "target": ML_TARGET_COLUMN,
            "explicitly_excluded": list(ML_EXCLUDED_COLUMNS),
            "scoring_time": "transaction_submission_before_balance_mutation",
        },
        "split_strategy": {
            "kind": "chronological_complete_step_boundaries",
            "target_train_fraction": 0.70,
            "target_validation_fraction": 0.15,
            "target_test_fraction": 0.15,
            "complete_steps_are_atomic": True,
            "held_out_test_policy": "Do not use for model or threshold selection.",
        },
        "splits": split_files,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
