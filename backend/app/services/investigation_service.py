from __future__ import annotations

from backend.app.core.config import Settings
from backend.app.schemas.risk import (
    InvestigationContext,
    RecommendedAction,
    RiskInvestigationRequest,
)
from backend.app.services.behavioral_service import SQLitePaySimHistoryProvider
from backend.app.services.relationship_service import SQLiteRelationshipHistoryProvider
from backend.app.services.risk_service import investigate_risk, load_active_bundle


def build_investigation(
    request: RiskInvestigationRequest,
    settings: Settings,
) -> tuple[InvestigationContext, RecommendedAction]:
    """Compose existing deterministic engines without changing their calculations."""

    bundle = load_active_bundle(settings.model_artifact_root)
    if request.transaction_reference is None:
        return investigate_risk(bundle, request.manual_transaction())

    history = SQLitePaySimHistoryProvider(settings.behavioral_history_db)
    historical_transaction, behavioral_context = history.context_for(
        request.transaction_reference
    )
    relationship_context = SQLiteRelationshipHistoryProvider(
        settings.relationship_history_db
    ).context_for(request.transaction_reference)
    return investigate_risk(
        bundle,
        historical_transaction.scoring_request(),
        behavioral_context=behavioral_context,
        relationship_context=relationship_context,
    )
