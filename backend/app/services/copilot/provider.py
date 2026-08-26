from __future__ import annotations

from typing import Any, Protocol

from backend.app.schemas.copilot import InvestigationReport, SanitizedInvestigationContext
from backend.app.services.copilot.prompts import SYSTEM_PROMPT, build_context_prompt


class CopilotProviderError(RuntimeError):
    """Base class for controlled provider failures."""


class CopilotProviderUnavailableError(CopilotProviderError):
    """Raised when a configured provider cannot complete a request."""


class CopilotProviderInvalidOutputError(CopilotProviderError):
    """Raised when provider output does not satisfy the report contract."""


class InvestigationLLMProvider(Protocol):
    name: str
    model: str | None

    def generate(self, context: SanitizedInvestigationContext) -> InvestigationReport: ...


class OpenAIInvestigationProvider:
    """Server-side OpenAI Responses API adapter using Pydantic Structured Outputs."""

    name = "openai"

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        timeout_seconds: float,
        client: Any | None = None,
    ) -> None:
        if not api_key:
            raise CopilotProviderUnavailableError("OpenAI API key is not configured")
        self.model = model
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as error:
                raise CopilotProviderUnavailableError(
                    "OpenAI SDK is unavailable; install the llm dependency"
                ) from error
            client = OpenAI(api_key=api_key, timeout=timeout_seconds)
        self._client = client

    def generate(self, context: SanitizedInvestigationContext) -> InvestigationReport:
        try:
            response = self._client.responses.parse(
                model=self.model,
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_context_prompt(context)},
                ],
                text_format=InvestigationReport,
                store=False,
            )
        except Exception as error:
            raise CopilotProviderUnavailableError("OpenAI request failed") from error

        parsed = getattr(response, "output_parsed", None)
        if parsed is None:
            raise CopilotProviderInvalidOutputError(
                "OpenAI response contained no validated structured report"
            )
        try:
            return InvestigationReport.model_validate(parsed)
        except Exception as error:
            raise CopilotProviderInvalidOutputError(
                "OpenAI response failed InvestigationReport validation"
            ) from error
