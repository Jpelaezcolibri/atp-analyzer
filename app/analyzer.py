import base64
import json
import re
from typing import Any

import httpx
from anthropic import AsyncAnthropic, APIError, APIStatusError, APITimeoutError
from fastapi import HTTPException, status
from openai import APIError as OpenAIAPIError
from openai import APIStatusError as OpenAIAPIStatusError
from openai import APITimeoutError as OpenAIAPITimeoutError
from openai import AsyncOpenAI
from pydantic import ValidationError

from app.config import Settings
from app.models import ClaudeAnalysis, PortEvidenceAudit


SYSTEM_PROMPT = (
    "You are a specialized computer vision system for analyzing ATP fiber "
    "optic distribution boxes used in telecommunications field operations "
    "in Latin America. Your only job is to examine the image and determine "
    "which ports are occupied (have a fiber optic connector inserted) and "
    "which are available (empty). Always respond ONLY with valid JSON, "
    "no markdown, no explanation."
)

USER_PROMPT = """Analyze this ATP fiber optic distribution box image.
The box has 8 ports numbered 1 through 8 from left to right
(numbers are printed above or below each port).
An OCCUPIED port has a fiber optic connector physically inserted into that
numbered port adapter, with a cable/plug protruding from the port.
An AVAILABLE port is an empty adapter/slot with no connector inserted.

Respond ONLY with this exact JSON structure, no other text:
{
  "total_puertos": 8,
  "total_ocupados": <integer>,
  "puertos_ocupados": [<list of port numbers>],
  "total_disponibles": <integer>,
  "puertos_disponibles": [<list of port numbers>],
  "codigo_dispositivo": "<text from device label or null>",
  "ubicacion": "<location text from label or null>",
  "observaciones": "<any visible damage, special notes, or null>",
  "confidence": "high" | "medium" | "low"
}

Rules:
- puertos_ocupados + puertos_disponibles must always equal [1,2,3,4,5,6,7,8]
- total_ocupados = length of puertos_ocupados
- total_disponibles = length of puertos_disponibles
- total_puertos is always 8
- Evaluate each port individually from left to right using the printed numbers 1 through 8
- Do NOT count the green rectangular SC/APC adapters, dust caps, or green port faces as occupied by themselves
- A port is occupied only when a connector plug is seated inside that exact numbered port
- Green color alone is not evidence of occupancy; many available ATP ports have green adapters
- A row of 8 green rectangular tabs/adapters usually means the ports are present, not occupied
- Dark holes, shadows, black internal plastic, labels, screws, separators, and gaps under the adapters are not connectors
- Do not infer occupancy from a cable passing near, behind, below, or beside a port
- Count a port as occupied only if you can see an external connector body protruding from that numbered adapter and visually associated cable
- If you cannot clearly see the external connector body seated in the exact port, classify that port as available
- Prefer false negatives over false positives: uncertain ports must be marked available with confidence lowered
- Cables, connectors, or test leads held by a technician or connected to an optical power meter do not count unless visibly inserted into a numbered port on the ATP box
- If a connector appears near the box but is outside the port, in the technician's hand, or connected only to test equipment, do not count it as occupied
- If a port is partially hidden and you cannot verify a seated connector, mark it available and lower confidence
- In observaciones, mention technician test activity separately from permanent occupied ports
- For images with an optical power meter, the connector inserted into the meter is not an ATP box port connection
- If image quality is too low to be certain, set confidence to 'low'
- Return ONLY the JSON object, no markdown fences"""

STRICT_RETRY_PROMPT = (
    f"{USER_PROMPT}\n\nYour previous response was not valid JSON. "
    "Return exactly one parseable JSON object. Do not include markdown, "
    "comments, prose, or trailing commas."
)

EVIDENCE_AUDIT_PROMPT = """Analyze this ATP fiber optic distribution box image with a conservative anti-hallucination checklist.

The box has exactly 8 numbered ports from left to right. Your task is NOT to count green parts. Your task is to inspect each numbered adapter and decide whether there is visible evidence of an external connector body seated in that exact port.

Definitions:
- external_connector_body_visible: a real connector plug body is visibly inserted in the numbered adapter, protruding outward/downward, with a cable visually associated to that plug.
- empty_green_adapter_only: the port shows only the green rectangular SC/APC adapter, face, flap, or cap. This is AVAILABLE.
- shadow_or_internal_plastic_only: the port shows dark gaps, black plastic, shadow, separator, screw, or internal slot material. This is AVAILABLE.
- nearby_cable_not_inserted: a cable passes near/behind/below/beside the port but no connector body is visibly seated in that exact adapter. This is AVAILABLE.
- technician_or_meter_connector_only: the visible connector is held by the technician or connected to the orange optical power meter, not seated in a numbered ATP port. This is AVAILABLE for the ATP port.
- occluded_or_uncertain: the exact port is hidden or ambiguous. This is AVAILABLE and confidence should be medium or low.

Critical rules:
- The horizontal row of 8 green rectangles is usually the adapter row, not 8 occupied ports.
- Green color is never enough to mark occupied.
- Do not infer occupancy from cable proximity.
- Do not infer occupancy from identical-looking green adapters.
- If you cannot point to the external connector body protruding from that exact numbered adapter, mark it available.
- Prefer false negatives over false positives.
- A connector plugged into the orange meter is not plugged into the ATP port.

Respond ONLY with valid JSON in this exact structure:
{
  "total_puertos": 8,
  "puertos": [
    {
      "puerto": 1,
      "estado": "occupied" | "available",
      "evidencia": "external_connector_body_visible" | "empty_green_adapter_only" | "shadow_or_internal_plastic_only" | "nearby_cable_not_inserted" | "technician_or_meter_connector_only" | "occluded_or_uncertain",
      "razon": "<short visual reason>"
    }
  ],
  "codigo_dispositivo": "<text from device label or null>",
  "ubicacion": "<location text from label or null>",
  "observaciones": "<technician activity, visible damage, special notes, or null>",
  "confidence": "high" | "medium" | "low"
}

Return exactly 8 items in puertos, one for each port 1 through 8. A port can be "occupied" only when evidencia is "external_connector_body_visible"; otherwise estado must be "available". Return ONLY JSON, no markdown."""

MAX_IMAGE_BYTES = 10 * 1024 * 1024
ACCEPTED_IMAGE_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
}


class ATPAnalyzer:
    def __init__(
        self,
        settings: Settings,
        anthropic_client: AsyncAnthropic | None = None,
        openai_client: AsyncOpenAI | None = None,
    ) -> None:
        self.settings = settings
        self.anthropic_client = anthropic_client
        self.openai_client = openai_client

        if settings.llm_provider == "anthropic" and self.anthropic_client is None:
            self.anthropic_client = AsyncAnthropic(api_key=settings.anthropic_api_key)
        if settings.llm_provider == "openai" and self.openai_client is None:
            self.openai_client = AsyncOpenAI(api_key=settings.openai_api_key)

    async def analyze(self, image_url: str) -> ClaudeAnalysis:
        image_bytes, media_type = await self.fetch_image(image_url)
        encoded_image = base64.b64encode(image_bytes).decode("utf-8")
        if self.settings.llm_provider == "openai":
            return await self.call_openai_audit(encoded_image, media_type)
        return await self.call_claude_audit_with_retry(encoded_image, media_type)

    async def fetch_image(self, image_url: str) -> tuple[bytes, str]:
        try:
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                response = await client.get(image_url)
                response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    "Image URL returned an unsuccessful status code: "
                    f"{exc.response.status_code}"
                ),
            ) from exc
        except httpx.RequestError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Image URL unreachable: {exc}",
            ) from exc

        media_type = response.headers.get("content-type", "").split(";")[0].lower()
        if media_type not in ACCEPTED_IMAGE_TYPES:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    "URL must return an accepted image content-type "
                    "(image/jpeg, image/png, or image/webp)"
                ),
            )

        image_bytes = response.content
        if len(image_bytes) > MAX_IMAGE_BYTES:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Image too large. Maximum size is 10MB",
            )

        return image_bytes, media_type

    async def call_claude_with_retry(
        self,
        encoded_image: str,
        media_type: str,
    ) -> ClaudeAnalysis:
        first_response = await self.call_claude(encoded_image, media_type, USER_PROMPT)
        try:
            return self.parse_claude_response(first_response)
        except (json.JSONDecodeError, ValidationError, ValueError) as first_error:
            retry_response = await self.call_claude(
                encoded_image,
                media_type,
                STRICT_RETRY_PROMPT,
            )
            try:
                return self.parse_claude_response(retry_response)
            except (json.JSONDecodeError, ValidationError, ValueError) as retry_error:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=(
                        "Claude response could not be parsed after retry. "
                        f"First error: {first_error}. Retry error: {retry_error}"
                    ),
                ) from retry_error

    async def call_claude_audit_with_retry(
        self,
        encoded_image: str,
        media_type: str,
    ) -> ClaudeAnalysis:
        first_response = await self.call_claude(
            encoded_image,
            media_type,
            EVIDENCE_AUDIT_PROMPT,
        )
        try:
            return self.parse_evidence_audit_response(first_response)
        except (json.JSONDecodeError, ValidationError, ValueError) as first_error:
            retry_response = await self.call_claude(
                encoded_image,
                media_type,
                (
                    f"{EVIDENCE_AUDIT_PROMPT}\n\nYour previous response failed "
                    "validation. Return exactly one valid JSON object. Remember: "
                    "occupied is allowed only with external_connector_body_visible."
                ),
            )
            try:
                return self.parse_evidence_audit_response(retry_response)
            except (json.JSONDecodeError, ValidationError, ValueError) as retry_error:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=(
                        "Claude evidence audit could not be parsed after retry. "
                        f"First error: {first_error}. Retry error: {retry_error}"
                    ),
                ) from retry_error

    async def call_claude(
        self,
        encoded_image: str,
        media_type: str,
        prompt: str,
    ) -> str:
        if self.anthropic_client is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Anthropic client is not configured",
            )

        try:
            message = await self.anthropic_client.messages.create(
                model=self.settings.anthropic_model,
                max_tokens=1000,
                temperature=0,
                system=SYSTEM_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": encoded_image,
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
            )
        except (APIError, APIStatusError, APITimeoutError) as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Claude API error: {exc}",
            ) from exc

        text_parts = [
            block.text
            for block in message.content
            if getattr(block, "type", None) == "text" and hasattr(block, "text")
        ]
        if not text_parts:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Claude response did not contain text content",
            )

        return "\n".join(text_parts)

    async def call_openai(
        self,
        encoded_image: str,
        media_type: str,
    ) -> ClaudeAnalysis:
        if self.openai_client is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="OpenAI client is not configured",
            )

        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "total_puertos": {"type": "integer", "enum": [8]},
                "total_ocupados": {"type": "integer", "minimum": 0, "maximum": 8},
                "puertos_ocupados": {
                    "type": "array",
                    "items": {"type": "integer", "minimum": 1, "maximum": 8},
                },
                "total_disponibles": {"type": "integer", "minimum": 0, "maximum": 8},
                "puertos_disponibles": {
                    "type": "array",
                    "items": {"type": "integer", "minimum": 1, "maximum": 8},
                },
                "codigo_dispositivo": {"type": ["string", "null"]},
                "ubicacion": {"type": ["string", "null"]},
                "observaciones": {"type": ["string", "null"]},
                "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
            },
            "required": [
                "total_puertos",
                "total_ocupados",
                "puertos_ocupados",
                "total_disponibles",
                "puertos_disponibles",
                "codigo_dispositivo",
                "ubicacion",
                "observaciones",
                "confidence",
            ],
        }

        try:
            response = await self.openai_client.responses.create(
                model=self.settings.openai_model,
                instructions=SYSTEM_PROMPT,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": USER_PROMPT},
                            {
                                "type": "input_image",
                                "image_url": (
                                    f"data:{media_type};base64,{encoded_image}"
                                ),
                                "detail": "high",
                            },
                        ],
                    }
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "atp_port_analysis",
                        "strict": True,
                        "schema": schema,
                    }
                },
            )
        except (
            OpenAIAPIError,
            OpenAIAPIStatusError,
            OpenAIAPITimeoutError,
        ) as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"OpenAI API error: {exc}",
            ) from exc

        output_text = getattr(response, "output_text", None)
        if not output_text:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="OpenAI response did not contain text content",
            )

        try:
            return self.parse_claude_response(output_text)
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"OpenAI structured response failed validation: {exc}",
            ) from exc

    async def call_openai_audit(
        self,
        encoded_image: str,
        media_type: str,
    ) -> ClaudeAnalysis:
        if self.openai_client is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="OpenAI client is not configured",
            )

        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "total_puertos": {"type": "integer", "enum": [8]},
                "puertos": {
                    "type": "array",
                    "minItems": 8,
                    "maxItems": 8,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "puerto": {"type": "integer", "minimum": 1, "maximum": 8},
                            "estado": {
                                "type": "string",
                                "enum": ["occupied", "available"],
                            },
                            "evidencia": {
                                "type": "string",
                                "enum": [
                                    "external_connector_body_visible",
                                    "empty_green_adapter_only",
                                    "shadow_or_internal_plastic_only",
                                    "nearby_cable_not_inserted",
                                    "technician_or_meter_connector_only",
                                    "occluded_or_uncertain",
                                ],
                            },
                            "razon": {"type": "string"},
                        },
                        "required": ["puerto", "estado", "evidencia", "razon"],
                    },
                },
                "codigo_dispositivo": {"type": ["string", "null"]},
                "ubicacion": {"type": ["string", "null"]},
                "observaciones": {"type": ["string", "null"]},
                "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
            },
            "required": [
                "total_puertos",
                "puertos",
                "codigo_dispositivo",
                "ubicacion",
                "observaciones",
                "confidence",
            ],
        }

        try:
            response = await self.openai_client.responses.create(
                model=self.settings.openai_model,
                instructions=SYSTEM_PROMPT,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": EVIDENCE_AUDIT_PROMPT},
                            {
                                "type": "input_image",
                                "image_url": (
                                    f"data:{media_type};base64,{encoded_image}"
                                ),
                                "detail": "high",
                            },
                        ],
                    }
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "atp_port_evidence_audit",
                        "strict": True,
                        "schema": schema,
                    }
                },
            )
        except (
            OpenAIAPIError,
            OpenAIAPIStatusError,
            OpenAIAPITimeoutError,
        ) as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"OpenAI API error: {exc}",
            ) from exc

        output_text = getattr(response, "output_text", None)
        if not output_text:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="OpenAI response did not contain text content",
            )

        try:
            return self.parse_evidence_audit_response(output_text)
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"OpenAI evidence audit failed validation: {exc}",
            ) from exc

    def parse_claude_response(self, response_text: str) -> ClaudeAnalysis:
        cleaned = self.strip_markdown_fences(response_text)
        data = json.loads(cleaned)
        if not isinstance(data, dict):
            raise ValueError("Claude response JSON must be an object")

        analysis = ClaudeAnalysis.model_validate(data)
        self.validate_port_logic(data)
        return analysis

    def parse_evidence_audit_response(self, response_text: str) -> ClaudeAnalysis:
        cleaned = self.strip_markdown_fences(response_text)
        data = json.loads(cleaned)
        if not isinstance(data, dict):
            raise ValueError("Evidence audit JSON must be an object")

        audit = PortEvidenceAudit.model_validate(data)
        for port in audit.puertos:
            if (
                port.estado == "occupied"
                and port.evidencia != "external_connector_body_visible"
            ):
                raise ValueError(
                    "Evidence audit validation failed: occupied ports must have "
                    "external_connector_body_visible evidence"
                )

        analysis = audit.to_claude_analysis()
        self.validate_port_logic(analysis.model_dump())
        return analysis

    @staticmethod
    def strip_markdown_fences(text: str) -> str:
        stripped = text.strip()
        fenced_match = re.fullmatch(
            r"```(?:json)?\s*(.*?)\s*```",
            stripped,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if fenced_match:
            return fenced_match.group(1).strip()
        return stripped

    @staticmethod
    def validate_port_logic(data: dict[str, Any]) -> None:
        occupied = data.get("puertos_ocupados")
        available = data.get("puertos_disponibles")
        if not isinstance(occupied, list) or not isinstance(available, list):
            raise ValueError("Port fields must be lists")

        all_ports = occupied + available
        if len(occupied) + len(available) != 8:
            raise ValueError("Port validation failed: occupied + available != 8")
        if any(not isinstance(port, int) or port < 1 or port > 8 for port in all_ports):
            raise ValueError("Port validation failed: ports must be integers 1..8")
        if len(set(occupied).intersection(available)) > 0:
            raise ValueError(
                "Port validation failed: duplicate port between occupied and available"
            )
        if len(set(all_ports)) != 8:
            raise ValueError("Port validation failed: duplicate port detected")
        if sorted(all_ports) != list(range(1, 9)):
            raise ValueError(
                "Port validation failed: ports must cover exactly [1,2,3,4,5,6,7,8]"
            )
