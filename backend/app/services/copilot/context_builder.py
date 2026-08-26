from __future__ import annotations

import math
from typing import Any

from backend.app.schemas.copilot import (
    SanitizedEvidence,
    SanitizedInvestigationContext,
    SanitizedModelOutput,
    SanitizedReferenceContext,
    SanitizedTransaction,
)
from backend.app.schemas.risk import (
    BehavioralContext,
    CurrentAmountBehavior,
    CurrentRelationshipAmountContext,
    DestinationNetworkContext,
    InvestigationContext,
    OriginNetworkContext,
    PriorAmountContext,
    RecentActivityContext,
    RecentWindowActivity,
    RecommendedAction,
    RelationshipAmountContext,
    RelationshipContext,
    TransactionTypeBehavior,
)

SAFE_EVIDENCE_FACTS: dict[str, frozenset[str]] = {
    "model_risk_above_threshold": frozenset(
        {
            "fraud_probability",
            "classification_threshold",
            "margin_from_threshold",
            "operating_mode",
        }
    ),
    "model_risk_below_threshold": frozenset(
        {
            "fraud_probability",
            "classification_threshold",
            "margin_from_threshold",
            "operating_mode",
        }
    ),
    "amount_reference_context": frozenset(
        {
            "amount",
            "estimated_global_percentile",
            "estimated_transaction_type_percentile",
            "global_training_median",
            "transaction_type_training_median",
            "reference_split",
        }
    ),
    "origin_balance_ratio_context": frozenset(
        {
            "amount",
            "origin_balance_before",
            "amount_to_origin_balance",
            "zero_balance_handling",
            "numeric_ratio_cap",
        }
    ),
    "transaction_type_training_context": frozenset(
        {
            "transaction_type",
            "training_rows",
            "training_share",
            "historical_fraud_prevalence_in_training_reference_data",
            "overall_training_fraud_prevalence",
            "prevalence_ratio",
        }
    ),
    "hour_training_context": frozenset(
        {
            "hour_of_day",
            "historical_fraud_prevalence_in_training_reference_data",
            "overall_training_fraud_prevalence",
            "prevalence_ratio",
            "share_within_transaction_type",
        }
    ),
    "behavior_history_unavailable": frozenset({"history_available"}),
    "behavior_amount_above_typical": frozenset(
        {"amount_vs_prior_average", "prior_transaction_count", "baseline_is_limited"}
    ),
    "behavior_amount_above_prior_maximum": frozenset(
        {"exceeds_prior_maximum", "amount_vs_prior_maximum"}
    ),
    "behavior_recent_activity": frozenset(
        {
            "window_steps",
            "prior_transaction_count",
            "prior_amount_total",
            "steps_since_previous_transaction",
        }
    ),
    "behavior_new_transaction_type": frozenset(
        {"is_new_transaction_type_for_origin", "prior_transaction_type_count"}
    ),
    "relationship_context_unavailable": frozenset({"context_available"}),
    "relationship_new_counterparty": frozenset(
        {
            "relationship_seen_before",
            "relationship_first_seen",
            "prior_interaction_count",
            "prior_unique_counterparty_count",
            "prior_unique_origin_count",
            "current_destination_is_new",
            "current_origin_is_new_for_destination",
        }
    ),
    "relationship_previously_observed": frozenset(
        {
            "relationship_seen_before",
            "prior_interaction_count",
            "steps_since_previous_interaction",
        }
    ),
    "relationship_limited_history": frozenset(
        {"prior_interaction_count", "baseline_is_limited"}
    ),
    "relationship_amount_deviation": frozenset(
        {"amount_vs_prior_average", "prior_interaction_count", "baseline_is_limited"}
    ),
    "relationship_exceeds_prior_maximum": frozenset(
        {"exceeds_prior_relationship_maximum", "amount_vs_prior_maximum"}
    ),
}
SAFE_STRING_FACT_VALUES = frozenset(
    {
        "BALANCED",
        "train",
        "CASH_IN",
        "CASH_OUT",
        "DEBIT",
        "PAYMENT",
        "TRANSFER",
        "ratio_set_to_zero",
        "not_applicable",
    }
)


def _approved_fact_value(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    return isinstance(value, str) and value in SAFE_STRING_FACT_VALUES


def _safe_evidence_facts(evidence_id: str, facts: dict[str, Any]) -> dict[str, Any]:
    allowed = SAFE_EVIDENCE_FACTS.get(evidence_id, frozenset())
    return {
        key: facts[key]
        for key in sorted(allowed)
        if key in facts and _approved_fact_value(facts[key])
    }


def _sanitize_behavior(context: BehavioralContext) -> BehavioralContext:
    prior_amount = context.prior_amount
    current_amount = context.current_amount_context
    return BehavioralContext(
        history_available=context.history_available,
        availability_explanation=(
            "Aggregates use only transactions for this origin with step < current step."
            if context.history_available
            else "No eligible earlier-step origin history is available."
        ),
        prior_transaction_count=context.prior_transaction_count,
        prior_total_amount=context.prior_total_amount,
        prior_amount=(
            PriorAmountContext(
                average=prior_amount.average,
                median=prior_amount.median,
                maximum=prior_amount.maximum,
            )
            if prior_amount is not None
            else None
        ),
        current_amount_context=(
            CurrentAmountBehavior(
                amount_vs_prior_average=current_amount.amount_vs_prior_average,
                amount_vs_prior_median=current_amount.amount_vs_prior_median,
                amount_vs_prior_maximum=current_amount.amount_vs_prior_maximum,
                prior_empirical_percentile=current_amount.prior_empirical_percentile,
                exceeds_prior_maximum=current_amount.exceeds_prior_maximum,
            )
            if current_amount is not None
            else None
        ),
        recent_activity=RecentActivityContext(
            windows=[
                RecentWindowActivity(
                    window_steps=window.window_steps,
                    prior_transaction_count=window.prior_transaction_count,
                    prior_amount_total=window.prior_amount_total,
                )
                for window in context.recent_activity.windows
            ],
            steps_since_previous_transaction=(
                context.recent_activity.steps_since_previous_transaction
            ),
        ),
        transaction_type_context=TransactionTypeBehavior(
            prior_transaction_type_count=(
                context.transaction_type_context.prior_transaction_type_count
            ),
            is_new_transaction_type_for_origin=(
                context.transaction_type_context.is_new_transaction_type_for_origin
            ),
        ),
    )


def _sanitize_relationship(context: RelationshipContext) -> RelationshipContext:
    prior = context.prior_amount
    current = context.current_amount_context
    return RelationshipContext(
        context_available=context.context_available,
        history_available=context.history_available,
        availability_explanation=(
            "Aggregates use only origin-destination interactions with step < current step."
            if context.history_available
            else (
                "No earlier-step origin-destination relationship was observed."
                if context.context_available
                else "Relationship context is unavailable for this input."
            )
        ),
        relationship_seen_before=context.relationship_seen_before,
        relationship_first_seen=context.relationship_first_seen,
        prior_interaction_count=context.prior_interaction_count,
        prior_total_amount=context.prior_total_amount,
        prior_amount=(
            RelationshipAmountContext(
                average=prior.average,
                median=prior.median,
                maximum=prior.maximum,
            )
            if prior is not None
            else None
        ),
        current_amount_context=(
            CurrentRelationshipAmountContext(
                amount_vs_prior_average=current.amount_vs_prior_average,
                amount_vs_prior_median=current.amount_vs_prior_median,
                amount_vs_prior_maximum=current.amount_vs_prior_maximum,
                prior_empirical_percentile=current.prior_empirical_percentile,
                exceeds_prior_relationship_maximum=(
                    current.exceeds_prior_relationship_maximum
                ),
            )
            if current is not None
            else None
        ),
        steps_since_previous_interaction=context.steps_since_previous_interaction,
        baseline_is_limited=context.baseline_is_limited,
        origin_network=OriginNetworkContext(
            prior_unique_counterparty_count=(
                context.origin_network.prior_unique_counterparty_count
            ),
            prior_transaction_count=context.origin_network.prior_transaction_count,
            current_destination_is_new=context.origin_network.current_destination_is_new,
        ),
        destination_network=DestinationNetworkContext(
            prior_unique_origin_count=context.destination_network.prior_unique_origin_count,
            prior_transaction_count=context.destination_network.prior_transaction_count,
            current_origin_is_new_for_destination=(
                context.destination_network.current_origin_is_new_for_destination
            ),
        ),
    )


def build_sanitized_context(
    investigation: InvestigationContext,
    recommended_action: RecommendedAction,
) -> SanitizedInvestigationContext:
    """Construct the provider payload through explicit positive field selection."""

    transaction = investigation.transaction
    model = investigation.model_output
    reference = investigation.approved_reference_statistics
    return SanitizedInvestigationContext(
        transaction=SanitizedTransaction(
            transaction_type=transaction.transaction_type,
            amount=transaction.amount,
            origin_balance_before=transaction.origin_balance_before,
            hour_of_day=transaction.hour_of_day,
        ),
        model_output=SanitizedModelOutput(
            fraud_probability=model.fraud_probability,
            risk_score=model.risk_score,
            risk_level=model.risk_level,
            fraud_prediction=model.fraud_prediction,
            classification_threshold=model.classification_threshold,
            operating_mode=model.operating_mode,
            recommended_simulated_action=recommended_action,
        ),
        evidence=[
            SanitizedEvidence(
                evidence_id=item.id,
                title=item.title,
                severity=item.severity,
                category=item.category,
                facts=_safe_evidence_facts(item.id, item.facts),
            )
            for item in [*investigation.evidence, *investigation.relationship_evidence]
        ],
        reference_context=SanitizedReferenceContext(
            source_split=reference["source_split"],
            source_step_range=tuple(reference["source_step_range"]),
            overall_training_fraud_prevalence=reference[
                "overall_training_fraud_prevalence"
            ],
        ),
        behavioral_context=_sanitize_behavior(investigation.behavioral_context),
        relationship_context=_sanitize_relationship(investigation.relationship_context),
    )
