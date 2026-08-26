from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from backend.app.schemas.risk import (
    DerivedFeatures,
    EvidenceItem,
    InvestigationContext,
    ModelOutputContext,
    RiskPredictionRequest,
)

SEVERITY_RANK = {"CRITICAL": 5, "HIGH": 4, "MEDIUM": 3, "LOW": 2, "INFO": 1}


@dataclass(frozen=True)
class EvidenceCandidate:
    item: EvidenceItem
    relevance: float


def _estimated_percentile(value: float, percentiles: dict[str, float]) -> float:
    points = sorted(
        (float(percentile), float(boundary))
        for percentile, boundary in percentiles.items()
    )
    if value <= points[0][1]:
        return points[0][0]
    if value >= points[-1][1]:
        return points[-1][0]
    for (lower_p, lower_v), (upper_p, upper_v) in zip(points, points[1:], strict=True):
        if lower_v <= value <= upper_v:
            if upper_v == lower_v:
                return upper_p
            fraction = (value - lower_v) / (upper_v - lower_v)
            return lower_p + fraction * (upper_p - lower_p)
    return points[-1][0]


def _importance(profile: dict[str, Any], *features: str) -> float:
    importance = profile["global_model_importance"]["features"]
    return max(
        float(importance.get(feature, {}).get("normalized_positive_importance", 0.0))
        for feature in features
    )


def _model_risk_evidence(model: ModelOutputContext) -> EvidenceCandidate:
    margin = model.fraud_probability - model.classification_threshold
    if model.fraud_prediction:
        severity = "CRITICAL" if model.fraud_probability >= 0.99 else "HIGH"
        title = "Predicted risk exceeds the active threshold"
        description = (
            "The frozen ML model score is above the BALANCED classification threshold. "
            "This is model output, not a causal explanation."
        )
    else:
        severity = "INFO"
        title = "Predicted risk remains below the active threshold"
        description = (
            "The frozen ML model score is below the BALANCED classification threshold; "
            "no fraud classification was produced."
        )
    return EvidenceCandidate(
        item=EvidenceItem(
            id=(
                "model_risk_above_threshold"
                if model.fraud_prediction
                else "model_risk_below_threshold"
            ),
            category="MODEL_RISK",
            severity=severity,
            title=title,
            description=description,
            facts={
                "fraud_probability": model.fraud_probability,
                "classification_threshold": model.classification_threshold,
                "margin_from_threshold": margin,
                "operating_mode": model.operating_mode,
            },
        ),
        relevance=1.0 + abs(margin),
    )


def _amount_evidence(
    transaction: RiskPredictionRequest,
    profile: dict[str, Any],
) -> EvidenceCandidate:
    statistics = profile["statistics"]
    global_amount = statistics["amount"]
    type_amount = statistics["transaction_types"][transaction.transaction_type]["amount"]
    global_percentile = _estimated_percentile(
        transaction.amount, global_amount["percentiles"]
    )
    type_percentile = _estimated_percentile(transaction.amount, type_amount["percentiles"])
    percentile = max(global_percentile, type_percentile)
    if percentile >= 0.995:
        severity, title = "HIGH", "Transaction amount is exceptionally high"
        description = (
            "The amount is at or above the estimated 99.5th percentile of an approved "
            "PaySim training reference distribution."
        )
    elif percentile >= 0.95:
        severity, title = "MEDIUM", "Transaction amount is higher than typical"
        description = (
            "The amount is above the estimated 95th percentile of the approved training "
            "reference distribution."
        )
    else:
        severity, title = "INFO", "Transaction amount is within the reference range"
        description = (
            "The amount is not above the estimated 95th percentile of the approved "
            "training reference distribution."
        )
    return EvidenceCandidate(
        item=EvidenceItem(
            id="amount_reference_context",
            category="AMOUNT_CONTEXT",
            severity=severity,
            title=title,
            description=description,
            facts={
                "amount": transaction.amount,
                "estimated_global_percentile": global_percentile,
                "estimated_transaction_type_percentile": type_percentile,
                "global_training_median": global_amount["percentiles"]["0.500"],
                "transaction_type_training_median": type_amount["percentiles"]["0.500"],
                "reference_split": "train",
            },
        ),
        relevance=percentile + _importance(profile, "amount", "log_amount"),
    )


def _balance_evidence(
    transaction: RiskPredictionRequest,
    derived: DerivedFeatures,
    profile: dict[str, Any],
) -> EvidenceCandidate:
    ratio = derived.amount_to_origin_balance
    if transaction.origin_balance_before == 0 and transaction.amount > 0:
        severity, title = "HIGH", "Transaction starts from a zero recorded origin balance"
        description = (
            "The submitted amount is positive while the recorded pre-transaction origin "
            "balance is zero. The derived ratio is safely represented as zero, not infinity."
        )
    elif ratio >= 1:
        severity, title = "HIGH", "Amount meets or exceeds the recorded origin balance"
        description = (
            "The submitted amount is at least as large as the recorded pre-transaction "
            "origin balance."
        )
    elif ratio >= 0.75:
        severity, title = "MEDIUM", "Transaction consumes most of the origin balance"
        description = "The amount represents at least 75% of the recorded origin balance."
    else:
        severity, title = "INFO", "Amount is below 75% of the origin balance"
        description = (
            "The amount does not consume most of the recorded pre-transaction origin balance."
        )
    return EvidenceCandidate(
        item=EvidenceItem(
            id="origin_balance_ratio_context",
            category="BALANCE_CONTEXT",
            severity=severity,
            title=title,
            description=description,
            facts={
                "amount": transaction.amount,
                "origin_balance_before": transaction.origin_balance_before,
                "amount_to_origin_balance": ratio,
                "zero_balance_handling": (
                    "ratio_set_to_zero"
                    if transaction.origin_balance_before == 0
                    else "not_applicable"
                ),
                "numeric_ratio_cap": 1_000_000_000_000.0,
            },
        ),
        relevance=min(1.5, ratio) + _importance(profile, "amount_to_origin_balance"),
    )


def _transaction_type_evidence(
    transaction: RiskPredictionRequest,
    profile: dict[str, Any],
) -> EvidenceCandidate:
    statistics = profile["statistics"]
    type_stats = statistics["transaction_types"][transaction.transaction_type]
    prevalence = float(type_stats["historical_fraud_prevalence"])
    overall = float(statistics["historical_fraud_prevalence"])
    ratio = prevalence / overall if overall else 0.0
    elevated = ratio >= 2 and int(type_stats["fraud_rows"]) >= 10
    severity = "MEDIUM" if elevated else "INFO"
    title = (
        "Transaction type has elevated training fraud prevalence"
        if elevated
        else "Transaction type reference context"
    )
    description = (
        "This transaction type has elevated historical fraud prevalence in the approved "
        "training reference data. This is population context, not this transaction's probability."
        if elevated
        else (
            "Historical training prevalence for this transaction type is provided as "
            "factual context, not an individual prediction."
        )
    )
    return EvidenceCandidate(
        item=EvidenceItem(
            id="transaction_type_training_context",
            category="TRANSACTION_TYPE_CONTEXT",
            severity=severity,
            title=title,
            description=description,
            facts={
                "transaction_type": transaction.transaction_type,
                "training_rows": type_stats["rows"],
                "training_share": type_stats["share"],
                "historical_fraud_prevalence_in_training_reference_data": prevalence,
                "overall_training_fraud_prevalence": overall,
                "prevalence_ratio": ratio,
            },
        ),
        relevance=min(2.0, ratio / 2) + _importance(profile, "transaction_type"),
    )


def _time_evidence(
    transaction: RiskPredictionRequest,
    profile: dict[str, Any],
) -> EvidenceCandidate:
    statistics = profile["statistics"]
    hour_stats = statistics["hours"][str(transaction.hour_of_day)]
    type_hour_stats = statistics["transaction_type_hours"][transaction.transaction_type][
        str(transaction.hour_of_day)
    ]
    prevalence = float(hour_stats["historical_fraud_prevalence"])
    overall = float(statistics["historical_fraud_prevalence"])
    ratio = prevalence / overall if overall else 0.0
    elevated = ratio >= 1.5 and int(hour_stats["fraud_rows"]) >= 10
    unusual_for_type = float(type_hour_stats["share_within_type"]) < 0.02
    if elevated:
        severity, title = "MEDIUM", "Hour has elevated training fraud prevalence"
        description = (
            "This hour has elevated historical fraud prevalence in the approved training "
            "reference. It does not determine this transaction's fraud probability."
        )
    elif unusual_for_type:
        severity, title = "LOW", "Hour is uncommon for this transaction type"
        description = (
            "Fewer than 2% of training transactions of this type occurred during this hour."
        )
    else:
        severity, title = "INFO", "Transaction hour is within the training activity pattern"
        description = (
            "The hour is not an elevated-prevalence or unusually sparse time segment in the "
            "approved training reference."
        )
    return EvidenceCandidate(
        item=EvidenceItem(
            id="hour_training_context",
            category="TIME_CONTEXT",
            severity=severity,
            title=title,
            description=description,
            facts={
                "hour_of_day": transaction.hour_of_day,
                "historical_fraud_prevalence_in_training_reference_data": prevalence,
                "overall_training_fraud_prevalence": overall,
                "prevalence_ratio": ratio,
                "share_within_transaction_type": type_hour_stats["share_within_type"],
            },
        ),
        relevance=min(1.5, ratio / 1.5) + _importance(profile, "hour_of_day"),
    )


def generate_evidence(
    *,
    transaction: RiskPredictionRequest,
    derived_features: DerivedFeatures,
    model_output: ModelOutputContext,
    reference_profile: dict[str, Any],
    max_items: int = 5,
) -> list[EvidenceItem]:
    candidates = [
        _model_risk_evidence(model_output),
        _amount_evidence(transaction, reference_profile),
        _balance_evidence(transaction, derived_features, reference_profile),
        _transaction_type_evidence(transaction, reference_profile),
        _time_evidence(transaction, reference_profile),
    ]
    ordered = sorted(
        candidates,
        key=lambda candidate: (
            -SEVERITY_RANK[candidate.item.severity],
            -candidate.relevance,
            candidate.item.id,
        ),
    )
    return [candidate.item for candidate in ordered[:max_items]]


def build_investigation_context(
    *,
    transaction: RiskPredictionRequest,
    derived_features: DerivedFeatures,
    model_output: ModelOutputContext,
    reference_profile: dict[str, Any],
) -> InvestigationContext:
    evidence = generate_evidence(
        transaction=transaction,
        derived_features=derived_features,
        model_output=model_output,
        reference_profile=reference_profile,
    )
    statistics = reference_profile["statistics"]
    return InvestigationContext(
        transaction=transaction,
        derived_features=derived_features,
        model_output=model_output,
        evidence=evidence,
        reference_profile_version=reference_profile["reference_profile_version"],
        approved_reference_statistics={
            "source_split": reference_profile["source_boundary"]["split"],
            "source_step_range": [
                reference_profile["source_boundary"]["min_step"],
                reference_profile["source_boundary"]["max_step"],
            ],
            "overall_training_fraud_prevalence": statistics[
                "historical_fraud_prevalence"
            ],
            "global_model_importance": reference_profile["global_model_importance"],
        },
    )
