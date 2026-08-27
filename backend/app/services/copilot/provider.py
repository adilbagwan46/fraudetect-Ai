from __future__ import annotations

from typing import Any, Protocol

from backend.app.schemas.copilot import InvestigationReport, SanitizedInvestigationContext
from backend.app.services.copilot.prompts import SYSTEM_PROMPT, build_context_prompt


class CopilotProviderError(RuntimeError):
    """Base class for controlled provider failures."""


class CopilotProviderUnavailableError(CopilotProviderError):
    """Raised when a configured provider cannot complete a request."""


class CopilotProviderTimeoutError(CopilotProviderUnavailableError):
    """Raised when a configured provider exceeds its request timeout."""


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
        self._timeout_errors: tuple[type[BaseException], ...] = (TimeoutError,)
        if client is None:
            try:
                from openai import APITimeoutError, OpenAI
            except ImportError as error:
                raise CopilotProviderUnavailableError(
                    "OpenAI SDK is unavailable; install the llm dependency"
                ) from error
            self._timeout_errors = (TimeoutError, APITimeoutError)
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
        except self._timeout_errors as error:
            raise CopilotProviderTimeoutError("OpenAI request timed out") from error
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


class GeminiInvestigationProvider:
    """Server-side Google Gen AI adapter using Pydantic structured output."""

    name = "gemini"

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        timeout_seconds: float,
        client: Any | None = None,
    ) -> None:
        if not api_key:
            raise CopilotProviderUnavailableError("Gemini API key is not configured")
        self.model = model
        self._timeout_errors: tuple[type[BaseException], ...] = (TimeoutError,)
        if client is None:
            try:
                import httpx
                from google import genai
                from google.genai import types
            except ImportError as error:
                raise CopilotProviderUnavailableError(
                    "Google Gen AI SDK is unavailable; install the llm dependency"
                ) from error
            self._timeout_errors = (TimeoutError, httpx.TimeoutException)
            client = genai.Client(
                api_key=api_key,
                http_options=types.HttpOptions(timeout=int(timeout_seconds * 1000)),
            )
        self._client = client

    def generate(self, context: SanitizedInvestigationContext) -> InvestigationReport:
        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=build_context_prompt(context),
                config={
                    "system_instruction": SYSTEM_PROMPT,
                    "response_mime_type": "application/json",
                    "response_schema": InvestigationReport,
                },
            )
        except self._timeout_errors as error:
            raise CopilotProviderTimeoutError("Gemini request timed out") from error
        except Exception as error:
            raise CopilotProviderUnavailableError("Gemini request failed") from error

        parsed = getattr(response, "parsed", None)
        if parsed is None:
            raise CopilotProviderInvalidOutputError(
                "Gemini response contained no validated structured report"
            )
        try:
            return InvestigationReport.model_validate(parsed)
        except Exception as error:
            raise CopilotProviderInvalidOutputError(
                "Gemini response failed InvestigationReport validation"
            ) from error
