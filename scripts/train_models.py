from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

from ml.fraudetect_ml.data.contracts import (
    ML_EXCLUDED_COLUMNS,
    ML_FEATURE_COLUMNS,
    ML_TARGET_COLUMN,
)
from ml.fraudetect_ml.modeling.artifacts import (
    environment_metadata,
    save_model_bundle,
)
from ml.fraudetect_ml.modeling.candidates import (
    balanced_class_weights,
    build_candidate_pipeline,
    candidate_specs,
)
from ml.fraudetect_ml.modeling.data import (
    features_and_target,
    load_prepared_split,
    split_fit_calibration,
)
from ml.fraudetect_ml.modeling.evaluation import (
    metrics_at_threshold,
    probability_metrics,
)
from ml.fraudetect_ml.modeling.thresholds import (
    downsample_threshold_curve,
    select_threshold_policies,
    threshold_curve,
)

RANDOM_STATE = 20260826
RECOMMENDED_MODE = "BALANCED"
CALIBRATION_METHOD = "sigmoid"
MODEL_FIT_FRACTION = 0.90


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("source", {}).get("kind") != "paysim_public_synthetic":
        raise ValueError("Final model training requires the genuine prepared PaySim dataset")
    manifest_features = tuple(manifest.get("ml_feature_contract", {}).get("features", ()))
    if manifest_features != ML_FEATURE_COLUMNS:
        raise ValueError("Manifest feature contract does not match application code")
    return manifest


def calibrate_frozen_model(model: Any, features: Any, target: Any) -> Any:
    calibrator = CalibratedClassifierCV(
        FrozenEstimator(model),
        method=CALIBRATION_METHOD,
    )
    calibrator.fit(features, target)
    return calibrator


def train_and_select(
    *,
    manifest_path: Path,
    artifact_root: Path,
) -> Path:
    started_at = datetime.now(UTC)
    manifest = load_manifest(manifest_path)
    processed_dir = manifest_path.parent

    print("Loading training split with explicit feature contract...", flush=True)
    training_frame = load_prepared_split(processed_dir / "train.csv")
    partitions = split_fit_calibration(training_frame, fit_fraction=MODEL_FIT_FRACTION)
    del training_frame
    fit_target_array = partitions.fit_target.to_numpy(dtype="int8", copy=False)
    class_weights = balanced_class_weights(fit_target_array)
    calibration_target_array = partitions.calibration_target.to_numpy(dtype="int8", copy=False)
    print(
        "Chronological model-fit/calibration boundary: "
        f"steps {partitions.fit_min_step}-{partitions.fit_max_step} / "
        f"{partitions.calibration_min_step}-{partitions.calibration_max_step}",
        flush=True,
    )

    print("Loading validation split (never used for fitting)...", flush=True)
    validation_frame = load_prepared_split(processed_dir / "validation.csv")
    validation_features, validation_target = features_and_target(validation_frame)
    validation_target_array = validation_target.to_numpy(dtype="int8", copy=False)
    del validation_frame, validation_target

    candidate_results: dict[str, Any] = {}
    calibrated_models: dict[str, Any] = {}
    validation_curves: dict[str, Any] = {}
    for spec in candidate_specs():
        candidate_started = time.monotonic()
        print(f"Training {spec.name}...", flush=True)
        pipeline = build_candidate_pipeline(
            spec,
            class_weights=class_weights,
            random_state=RANDOM_STATE,
        )
        pipeline.fit(partitions.fit_features, fit_target_array)
        calibrated = calibrate_frozen_model(
            pipeline,
            partitions.calibration_features,
            calibration_target_array,
        )
        probabilities = calibrated.predict_proba(validation_features)[:, 1]
        probability_summary = probability_metrics(validation_target_array, probabilities)
        policies = select_threshold_policies(validation_target_array, probabilities)
        default_metrics = metrics_at_threshold(validation_target_array, probabilities, 0.5)
        curve = threshold_curve(validation_target_array, probabilities)
        duration = time.monotonic() - candidate_started
        candidate_results[spec.name] = {
            "name": spec.name,
            "family": spec.family,
            "weighted": spec.weighted,
            "calibration": CALIBRATION_METHOD,
            "validation_probability_metrics": probability_summary,
            "validation_at_0_5": default_metrics,
            "validation_threshold_policies": policies,
            "training_duration_seconds": duration,
        }
        calibrated_models[spec.name] = calibrated
        validation_curves[spec.name] = downsample_threshold_curve(curve)
        print(
            f"{spec.name}: PR-AUC={probability_summary['pr_auc']:.6f}, "
            f"balanced F1={policies['BALANCED']['f1']:.6f}",
            flush=True,
        )

    winner_name = max(
        candidate_results,
        key=lambda name: (
            candidate_results[name]["validation_threshold_policies"][RECOMMENDED_MODE]["f1"],
            candidate_results[name]["validation_probability_metrics"]["pr_auc"],
            candidate_results[name]["validation_probability_metrics"]["roc_auc"],
        ),
    )
    winner_result = candidate_results[winner_name]
    winner_model = calibrated_models[winner_name]
    frozen_policies = winner_result["validation_threshold_policies"]
    frozen_threshold = frozen_policies[RECOMMENDED_MODE]["threshold"]
    frozen_selection = {
        "candidate": winner_name,
        "recommended_operating_mode": RECOMMENDED_MODE,
        "classification_threshold": frozen_threshold,
        "selection_dataset": "validation",
        "selection_logic": (
            "Highest validation BALANCED-policy F1; ties use PR-AUC then ROC-AUC. "
            "The BALANCED policy maximizes validation F1."
        ),
    }
    print(
        f"Frozen validation winner: {winner_name}, threshold={frozen_threshold:.12f}",
        flush=True,
    )

    # Test labels and outcomes are loaded only after every selection decision above is frozen.
    print("Loading held-out test once for final frozen evaluation...", flush=True)
    test_frame = load_prepared_split(processed_dir / "test.csv")
    test_features, test_target = features_and_target(test_frame)
    test_target_array = test_target.to_numpy(dtype="int8", copy=False)
    del test_frame, test_target
    test_probabilities = winner_model.predict_proba(test_features)[:, 1]
    test_probability_summary = probability_metrics(test_target_array, test_probabilities)
    test_policy_metrics = {
        mode: metrics_at_threshold(test_target_array, test_probabilities, policy["threshold"])
        for mode, policy in frozen_policies.items()
    }
    test_metrics = {
        "evaluation_kind": "single_held_out_chronological_test",
        "selection_was_frozen_before_test_load": True,
        "probability_metrics": test_probability_summary,
        "recommended_policy": RECOMMENDED_MODE,
        "recommended_policy_metrics": test_policy_metrics[RECOMMENDED_MODE],
        "all_frozen_policy_metrics": test_policy_metrics,
        "temporal_drift_note": (
            "Fraud prevalence changes materially across chronological periods: "
            "training 0.081616%, validation 0.059367%, test 0.419568%. "
            "This shift is intentionally preserved and not normalized away."
        ),
    }

    created_at = datetime.now(UTC)
    model_version = (
        f"paysim-{created_at.strftime('%Y%m%dT%H%M%SZ')}-"
        f"{manifest['source']['sha256'][:8]}"
    )
    feature_contract = {
        "features": list(ML_FEATURE_COLUMNS),
        "target": ML_TARGET_COLUMN,
        "explicitly_excluded": list(ML_EXCLUDED_COLUMNS),
        "api_inputs": [
            "transaction_type",
            "amount",
            "origin_balance_before",
            "hour_of_day",
        ],
        "derived_by_backend": ["log_amount", "amount_to_origin_balance"],
    }
    metadata = {
        "model_version": model_version,
        "created_at": created_at.isoformat(),
        "training_started_at": started_at.isoformat(),
        "model_family": winner_result["family"],
        "model_candidate": winner_name,
        "weighted": winner_result["weighted"],
        "random_state": RANDOM_STATE,
        "dataset": {
            "kind": manifest["source"]["kind"],
            "sha256": manifest["source"]["sha256"],
            "rows": manifest["dataset"]["rows"],
            "splits": manifest["splits"],
        },
        "feature_contract": feature_contract,
        "preprocessing": {
            "logistic_regression": (
                "Training-only median imputation, standard scaling, and one-hot encoding."
            ),
            "hist_gradient_boosting": (
                "Training-only median imputation and dense one-hot encoding; no scaling."
            ),
        },
        "calibration": {
            "method": CALIBRATION_METHOD,
            "fit_fraction": MODEL_FIT_FRACTION,
            "model_fit_steps": [partitions.fit_min_step, partitions.fit_max_step],
            "calibration_steps": [
                partitions.calibration_min_step,
                partitions.calibration_max_step,
            ],
            "model_fit_rows": len(partitions.fit_target),
            "calibration_rows": len(partitions.calibration_target),
            "model_fit_fraud_rows": int(partitions.fit_target.sum()),
            "calibration_fraud_rows": int(partitions.calibration_target.sum()),
        },
        "class_weights_from_model_fit_only": class_weights,
        "selection": frozen_selection,
        "recommended_action_policy": {
            "LOW": "NORMAL_PROCESSING",
            "MEDIUM": "MANUAL_REVIEW",
            "HIGH": "HOLD_FOR_INVESTIGATION",
            "simulated": True,
        },
    }
    validation_metrics = {
        "selected_candidate": winner_name,
        "recommended_operating_mode": RECOMMENDED_MODE,
        "probability_metrics": winner_result["validation_probability_metrics"],
        "at_0_5": winner_result["validation_at_0_5"],
        "threshold_policies": frozen_policies,
    }
    candidate_comparison = {
        "selection": frozen_selection,
        "candidates": candidate_results,
        "test_metrics_used_for_selection": False,
    }
    operating_points = {
        "methodology": (
            "Capacity metrics rank risk scores descending and inspect exactly "
            "floor(N * rate) rows. "
            "Threshold policies operate on validation score thresholds and are frozen before test."
        ),
        "review_capacities": [0.001, 0.005, 0.01],
        "candidate_validation_curves_downsampled": validation_curves,
        "selected_candidate_validation_operating_points": winner_result[
            "validation_probability_metrics"
        ]["operating_points"],
        "held_out_test_operating_points": test_probability_summary["operating_points"],
    }
    false_positive_cost = {
        "enabled_for_model_selection": False,
        "currency": "USD",
        "manual_review_cost_assumption": 2.5,
        "customer_friction_cost_assumption": 5.0,
        "assumption_status": "illustrative_demo_only_not_merchant_data",
        "recommended_test_policy_estimate": test_policy_metrics[RECOMMENDED_MODE][
            "false_positives"
        ]
        * 7.5,
    }
    files = {
        "metadata.json": metadata,
        "feature-contract.json": feature_contract,
        "threshold-policy.json": {
            "recommended_mode": RECOMMENDED_MODE,
            "policies_selected_on_validation": frozen_policies,
        },
        "validation-metrics.json": validation_metrics,
        "test-metrics.json": test_metrics,
        "candidate-comparison.json": candidate_comparison,
        "operating-points.json": operating_points,
        "confusion-matrix.json": {
            "validation": frozen_policies[RECOMMENDED_MODE]["confusion_matrix"],
            "test": test_policy_metrics[RECOMMENDED_MODE]["confusion_matrix"],
            "labels": ["non_fraud", "fraud"],
        },
        "environment.json": environment_metadata(),
        "false-positive-cost.json": false_positive_cost,
    }
    model_dir = save_model_bundle(
        artifact_root,
        model_version,
        model=winner_model,
        files=files,
    )
    print(f"Saved frozen model bundle: {model_dir}", flush=True)
    return model_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate Phase 2A PaySim candidates")
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
    args = parser.parse_args()
    train_and_select(manifest_path=args.manifest, artifact_root=args.artifact_root)


if __name__ == "__main__":
    main()
