from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status

from backend.app.core.config import Settings, get_settings
from backend.app.schemas.case import (
    AnalystDisposition,
    CaseCreateRequest,
    CaseDetailResponse,
    CaseListResponse,
    CasePriority,
    CaseStatus,
    CaseUpdateRequest,
)
from backend.app.schemas.copilot import CopilotInvestigationResponse
from backend.app.schemas.risk import RiskLevel
from backend.app.services.behavioral_service import (
    BehaviorHistoryUnavailableError,
    TransactionReferenceNotFoundError,
)
from backend.app.services.case_service import (
    CaseNotFoundError,
    CaseRepository,
    CaseStoreUnavailableError,
    InvalidCaseTransitionError,
    SQLiteCaseRepository,
    assign_case_priority,
    evidence_summary,
    investigation_limitations,
)
from backend.app.services.copilot.context_builder import build_sanitized_context
from backend.app.services.copilot.service import CopilotService, create_copilot_service
from backend.app.services.investigation_service import build_investigation
from backend.app.services.relationship_service import (
    RelationshipHistoryUnavailableError,
    RelationshipTransactionNotFoundError,
)
from backend.app.services.risk_service import ModelUnavailableError

router = APIRouter(prefix="/cases", tags=["cases"])
SettingsDependency = Annotated[Settings, Depends(get_settings)]
CaseId = Annotated[str, Path(pattern=r"^CASE-[A-F0-9]{16}$")]


def get_case_repository(settings: SettingsDependency) -> CaseRepository:
    return SQLiteCaseRepository(settings.case_database)


def get_case_copilot(settings: SettingsDependency) -> CopilotService:
    return create_copilot_service(settings)


CaseRepositoryDependency = Annotated[CaseRepository, Depends(get_case_repository)]
CaseCopilotDependency = Annotated[CopilotService, Depends(get_case_copilot)]


def _raise_investigation_error(error: Exception) -> None:
    if isinstance(
        error,
        (TransactionReferenceNotFoundError, RelationshipTransactionNotFoundError),
    ):
        raise HTTPException(
            status_code=404, detail="Transaction reference was not found"
        ) from error
    raise HTTPException(status_code=503, detail=str(error)) from error


@router.post("", response_model=CaseDetailResponse, status_code=status.HTTP_201_CREATED)
def create_case(
    request: CaseCreateRequest,
    settings: SettingsDependency,
    repository: CaseRepositoryDependency,
) -> CaseDetailResponse:
    try:
        context, action = build_investigation(request, settings)
        snapshot = build_sanitized_context(context, action)
        return repository.create(
            source_type=(
                "REFERENCE" if request.transaction_reference is not None else "MANUAL"
            ),
            transaction_reference_available=request.transaction_reference is not None,
            model_version=context.model_output.model_version,
            snapshot=snapshot,
            priority=assign_case_priority(snapshot),
            evidence_summary=evidence_summary(snapshot),
            limitations=investigation_limitations(snapshot),
        )
    except (
        TransactionReferenceNotFoundError,
        BehaviorHistoryUnavailableError,
        RelationshipTransactionNotFoundError,
        RelationshipHistoryUnavailableError,
        ModelUnavailableError,
    ) as error:
        _raise_investigation_error(error)
    except CaseStoreUnavailableError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@router.get("", response_model=CaseListResponse)
def list_cases(
    repository: CaseRepositoryDependency,
    status_filter: Annotated[CaseStatus | None, Query(alias="status")] = None,
    risk_level: RiskLevel | None = None,
    priority: CasePriority | None = None,
    disposition: AnalystDisposition | None = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> CaseListResponse:
    try:
        return repository.list(
            status=status_filter,
            risk_level=risk_level,
            priority=priority,
            disposition=disposition,
            limit=limit,
            offset=offset,
        )
    except CaseStoreUnavailableError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@router.get("/{case_id}", response_model=CaseDetailResponse)
def get_case(
    case_id: CaseId,
    repository: CaseRepositoryDependency,
) -> CaseDetailResponse:
    try:
        return repository.get(case_id)
    except CaseNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except CaseStoreUnavailableError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@router.patch("/{case_id}", response_model=CaseDetailResponse)
def update_case(
    case_id: CaseId,
    request: CaseUpdateRequest,
    repository: CaseRepositoryDependency,
) -> CaseDetailResponse:
    try:
        return repository.update(case_id, request)
    except CaseNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except InvalidCaseTransitionError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except CaseStoreUnavailableError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@router.post("/{case_id}/copilot", response_model=CopilotInvestigationResponse)
def generate_case_copilot(
    case_id: CaseId,
    repository: CaseRepositoryDependency,
    copilot: CaseCopilotDependency,
) -> CopilotInvestigationResponse:
    try:
        detail = repository.get(case_id)
        if detail.case.status == "CLOSED":
            raise InvalidCaseTransitionError("Closed cases are immutable")
        response = copilot.investigate(detail.intelligence_snapshot)
        repository.save_copilot(case_id, response)
        return response
    except CaseNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except InvalidCaseTransitionError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except CaseStoreUnavailableError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
