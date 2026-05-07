import base64
import io
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
from PIL import Image
from pydantic import ValidationError

from app.config import Settings
from app.models import ClaudeAnalysis, OccupancyVerification, PortEvidenceAudit


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
You will receive two images:
- Image 1: a zoomed crop focused on the ATP port row. Use this as the primary source for occupancy.
- Image 2: the full original photo. Use this for label text, technician context, and anything outside the crop.

Numbering procedure:
- First identify the printed port numbers on the box, even if they are faint, embossed, or printed above the green adapter row.
- Only after locating the printed numbers, map each visible connector to the nearest printed port number.
- Never assume that the leftmost visible connector is port 1 unless the printed numbering confirms it.
- If the image framing cuts off part of the row or the numbering starts later than expected, still trust the printed labels over visual position alone.
- If a technician jumper is inserted, assign it to the printed port number directly above or below that exact adapter, not to the nearest occupied cluster.

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
- A connector plugged into the orange meter is not automatically an ATP port connection.
- However, if a test jumper visibly originates from a numbered ATP port and continues to the orange optical power meter, that numbered ATP port DOES count as occupied.
- For technician testing, count the ATP-side inserted connector, not the meter-side connector.
- Do not infer port numbers from connector order alone. Port numbering must come from the printed labels on the ATP housing.
- Faint embossed numbers above the adapter row take priority over where the connector cluster begins.
- Do not mark a contiguous block of ports occupied just because the adapter row is green. Each occupied port must have its own visible plug/cable leaving that exact numbered adapter.
- In this ATP box style, the strongest occupancy evidence is a cable or connector body exiting downward from the lower edge of the numbered adapter. If only the top green face is visible, mark available.
- Many ATP boxes have green hinged shutters or rectangular hanging adapter flaps below each port. These flaps can look like connector bodies, but they are part of the empty adapter. Do NOT count a hanging green flap as occupied unless a separate fiber cable is clearly attached to it and can be traced away from that exact port.
- Empty ports may show vertical green plastic pieces below the adapter row. These are not SC/APC connector plugs by themselves.
- For each occupied decision, require two visible facts together: (1) a plug body distinct from the built-in adapter/shutter, and (2) an individual cable physically attached to that plug.
- If several adjacent ports appear occupied only because they have identical green hanging shapes, they are probably adapter shutters, not inserted connectors. Mark them available unless their individual cables are clearly visible.

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


def build_verification_prompt(candidate_ports: list[int]) -> str:
    return f"""You are a skeptical verifier for ATP fiber optic port occupancy.

The previous analysis may contain false positives. Review the image again and verify ONLY these candidate occupied ports: {candidate_ports}.
You will receive two images:
- Image 1: zoomed crop of the ATP port row. Use this as the primary source for connector verification.
- Image 2: full original photo. Use it only as supporting context.

For each candidate port, answer whether it is truly occupied under this strict rule:
- confirmed_occupied is true ONLY if a physical external connector body is clearly seated in that exact numbered adapter and protrudes outward/downward from the port.
- visible_connector_body must be true only when the connector body itself is visible, not just a green adapter face.
- If a technician test jumper is visibly inserted into the ATP-side numbered port and routes to the orange optical power meter, confirmed_occupied must be true for that ATP port.
- Verify the printed port number for each candidate before confirming it. Do not confirm a port if the visible connector actually aligns with a different printed number.

Reject these as NOT occupied:
- green rectangular SC/APC adapter row
- green port faces, flaps, caps, dust covers
- dark holes, black internal plastic, shadows, separators, screws
- cables passing nearby, behind, below, or beside the port
- connector plugged into the orange optical power meter when the ATP-side port insertion is not visible
- connector held by the technician
- anything ambiguous or partially occluded

Important ATP test-photo rule:
- The meter-side plug itself is not a port, but the ATP-side port where the test jumper is inserted is occupied.
- If the visible cable leaves the numbered adapter and goes toward the meter, count that adapter port as occupied.
- Do not count ports 3, 4, 5, or 6 as occupied merely because they have the same vertical green shutter shape as their neighbors. Count them only if their own cable can be visually traced from that exact port.
- If the printed labels indicate the technician jumper is in port 5, do not relabel it as port 4 just because it is the fourth occupied-looking connector from the left.

If unsure, set confirmed_occupied=false and visible_connector_body=false.

Respond ONLY with valid JSON:
{{
  "verifications": [
    {{
      "puerto": <candidate port number>,
      "confirmed_occupied": true | false,
      "visible_connector_body": true | false,
      "reason": "<short reason>"
    }}
  ]
}}

Return one verification item for every candidate port in {candidate_ports}. No markdown."""

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
        cropped_bytes, cropped_media_type = self.create_port_zoom(image_bytes, media_type)
        encoded_image = base64.b64encode(image_bytes).decode("utf-8")
        encoded_zoom = base64.b64encode(cropped_bytes).decode("utf-8")
        analyses = [
            await self.analyze_encoded_once(
                encoded_image,
                media_type,
                encoded_zoom,
                cropped_media_type,
            )
            for _ in range(self.settings.analysis_passes)
        ]
        return self.build_consensus_analysis(analyses)

    async def analyze_debug(self, image_url: str) -> dict[str, Any]:
        image_bytes, media_type = await self.fetch_image(image_url)
        cropped_bytes, cropped_media_type = self.create_port_zoom(image_bytes, media_type)
        encoded_image = base64.b64encode(image_bytes).decode("utf-8")
        encoded_zoom = base64.b64encode(cropped_bytes).decode("utf-8")
        passes = [
            await self.analyze_encoded_debug_once(
                encoded_image,
                media_type,
                encoded_zoom,
                cropped_media_type,
            )
            for _ in range(self.settings.analysis_passes)
        ]
        final = self.build_consensus_analysis(
            [ClaudeAnalysis.model_validate(item["final_analysis"]) for item in passes]
        )
        return {
            "analysis_passes": self.settings.analysis_passes,
            "consensus_rule": (
                "A port is returned occupied only if every pass confirms it after "
                "skeptical verification."
            ),
            "passes": passes,
            "final_consensus": final.model_dump(mode="json"),
        }

    async def analyze_encoded_once(
        self,
        encoded_image: str,
        media_type: str,
        encoded_zoom: str,
        zoom_media_type: str,
    ) -> ClaudeAnalysis:
        if self.settings.llm_provider == "openai":
            initial_analysis = await self.call_openai_audit(
                encoded_image,
                media_type,
                encoded_zoom,
                zoom_media_type,
            )
            return await self.verify_openai_occupied_ports(
                encoded_image,
                media_type,
                encoded_zoom,
                zoom_media_type,
                initial_analysis,
            )

        initial_analysis = await self.call_claude_audit_with_retry(
            encoded_image,
            media_type,
            encoded_zoom,
            zoom_media_type,
        )
        return await self.verify_claude_occupied_ports(
            encoded_image,
            media_type,
            encoded_zoom,
            zoom_media_type,
            initial_analysis,
        )

    async def analyze_encoded_debug_once(
        self,
        encoded_image: str,
        media_type: str,
        encoded_zoom: str,
        zoom_media_type: str,
    ) -> dict[str, Any]:
        if self.settings.llm_provider == "openai":
            audit = await self.call_openai_audit_raw(
                encoded_image,
                media_type,
                encoded_zoom,
                zoom_media_type,
            )
            initial_analysis = audit.to_claude_analysis()
            verification = await self.verify_openai_occupied_ports_raw(
                encoded_image,
                media_type,
                encoded_zoom,
                zoom_media_type,
                initial_analysis,
            )
            final_analysis = self.apply_occupancy_verification(
                initial_analysis,
                verification,
            )
        else:
            audit = await self.call_claude_audit_raw_with_retry(
                encoded_image,
                media_type,
                encoded_zoom,
                zoom_media_type,
            )
            initial_analysis = audit.to_claude_analysis()
            verification = await self.verify_claude_occupied_ports_raw(
                encoded_image,
                media_type,
                encoded_zoom,
                zoom_media_type,
                initial_analysis,
            )
            final_analysis = self.apply_occupancy_verification(
                initial_analysis,
                verification,
            )

        return {
            "zoom_used": True,
            "audit": audit.model_dump(mode="json"),
            "initial_analysis": initial_analysis.model_dump(mode="json"),
            "verification": verification.model_dump(mode="json"),
            "final_analysis": final_analysis.model_dump(mode="json"),
        }

    @staticmethod
    def create_port_zoom(image_bytes: bytes, media_type: str) -> tuple[bytes, str]:
        try:
            image = Image.open(io.BytesIO(image_bytes))
            image.load()
        except Exception as exc:  # pragma: no cover - defensive fallback
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Unable to process image for zoom analysis: {exc}",
            ) from exc

        width, height = image.size
        left = int(width * 0.30)
        top = int(height * 0.12)
        right = int(width * 0.98)
        bottom = int(height * 0.62)

        cropped = image.crop((left, top, right, bottom))
        resized = cropped.resize((cropped.width * 2, cropped.height * 2))

        output = io.BytesIO()
        output_format = "PNG" if media_type == "image/png" else "JPEG"
        save_image = resized.convert("RGB") if output_format == "JPEG" else resized
        save_image.save(output, format=output_format, quality=95)
        return output.getvalue(), "image/png" if output_format == "PNG" else "image/jpeg"

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
        encoded_zoom: str,
        zoom_media_type: str,
    ) -> ClaudeAnalysis:
        audit = await self.call_claude_audit_raw_with_retry(
            encoded_image,
            media_type,
            encoded_zoom,
            zoom_media_type,
        )
        analysis = audit.to_claude_analysis()
        self.validate_port_logic(analysis.model_dump())
        return analysis

    async def call_claude_audit_raw_with_retry(
        self,
        encoded_image: str,
        media_type: str,
        encoded_zoom: str,
        zoom_media_type: str,
    ) -> PortEvidenceAudit:
        first_response = await self.call_claude(
            encoded_image,
            media_type,
            encoded_zoom,
            zoom_media_type,
            EVIDENCE_AUDIT_PROMPT,
        )
        try:
            return self.parse_evidence_audit_raw_response(first_response)
        except (json.JSONDecodeError, ValidationError, ValueError) as first_error:
            retry_response = await self.call_claude(
                encoded_image,
                media_type,
                encoded_zoom,
                zoom_media_type,
                (
                    f"{EVIDENCE_AUDIT_PROMPT}\n\nYour previous response failed "
                    "validation. Return exactly one valid JSON object. Remember: "
                    "occupied is allowed only with external_connector_body_visible."
                ),
            )
            try:
                return self.parse_evidence_audit_raw_response(retry_response)
            except (json.JSONDecodeError, ValidationError, ValueError) as retry_error:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=(
                        "Claude evidence audit could not be parsed after retry. "
                        f"First error: {first_error}. Retry error: {retry_error}"
                    ),
                ) from retry_error

    async def verify_claude_occupied_ports(
        self,
        encoded_image: str,
        media_type: str,
        encoded_zoom: str,
        zoom_media_type: str,
        analysis: ClaudeAnalysis,
    ) -> ClaudeAnalysis:
        if not analysis.puertos_ocupados:
            return analysis

        verification = await self.verify_claude_occupied_ports_raw(
            encoded_image,
            media_type,
            encoded_zoom,
            zoom_media_type,
            analysis,
        )
        return self.apply_occupancy_verification(analysis, verification)

    async def verify_claude_occupied_ports_raw(
        self,
        encoded_image: str,
        media_type: str,
        encoded_zoom: str,
        zoom_media_type: str,
        analysis: ClaudeAnalysis,
    ) -> OccupancyVerification:
        prompt = build_verification_prompt(analysis.puertos_ocupados)
        response = await self.call_claude(
            encoded_image,
            media_type,
            encoded_zoom,
            zoom_media_type,
            prompt,
        )
        try:
            return self.parse_verification_response(response)
        except (json.JSONDecodeError, ValidationError, ValueError):
            return OccupancyVerification(verifications=[])

    async def call_claude(
        self,
        encoded_image: str,
        media_type: str,
        encoded_zoom: str,
        zoom_media_type: str,
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
                                    "media_type": zoom_media_type,
                                    "data": encoded_zoom,
                                },
                            },
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
        encoded_zoom: str,
        zoom_media_type: str,
    ) -> ClaudeAnalysis:
        audit = await self.call_openai_audit_raw(
            encoded_image,
            media_type,
            encoded_zoom,
            zoom_media_type,
        )
        analysis = audit.to_claude_analysis()
        self.validate_port_logic(analysis.model_dump())
        return analysis

    async def call_openai_audit_raw(
        self,
        encoded_image: str,
        media_type: str,
        encoded_zoom: str,
        zoom_media_type: str,
    ) -> PortEvidenceAudit:
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
                                    f"data:{zoom_media_type};base64,{encoded_zoom}"
                                ),
                                "detail": "high",
                            },
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
            return self.parse_evidence_audit_raw_response(output_text)
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"OpenAI evidence audit failed validation: {exc}",
            ) from exc

    async def verify_openai_occupied_ports(
        self,
        encoded_image: str,
        media_type: str,
        encoded_zoom: str,
        zoom_media_type: str,
        analysis: ClaudeAnalysis,
    ) -> ClaudeAnalysis:
        if not analysis.puertos_ocupados:
            return analysis
        verification = await self.verify_openai_occupied_ports_raw(
            encoded_image,
            media_type,
            encoded_zoom,
            zoom_media_type,
            analysis,
        )
        return self.apply_occupancy_verification(analysis, verification)

    async def verify_openai_occupied_ports_raw(
        self,
        encoded_image: str,
        media_type: str,
        encoded_zoom: str,
        zoom_media_type: str,
        analysis: ClaudeAnalysis,
    ) -> OccupancyVerification:
        if not analysis.puertos_ocupados:
            return OccupancyVerification(verifications=[])
        if self.openai_client is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="OpenAI client is not configured",
            )

        prompt = build_verification_prompt(analysis.puertos_ocupados)
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "verifications": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "puerto": {"type": "integer", "minimum": 1, "maximum": 8},
                            "confirmed_occupied": {"type": "boolean"},
                            "visible_connector_body": {"type": "boolean"},
                            "reason": {"type": "string"},
                        },
                        "required": [
                            "puerto",
                            "confirmed_occupied",
                            "visible_connector_body",
                            "reason",
                        ],
                    },
                }
            },
            "required": ["verifications"],
        }

        try:
            response = await self.openai_client.responses.create(
                model=self.settings.openai_model,
                instructions=SYSTEM_PROMPT,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {
                                "type": "input_image",
                                "image_url": (
                                    f"data:{zoom_media_type};base64,{encoded_zoom}"
                                ),
                                "detail": "high",
                            },
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
                        "name": "atp_occupancy_verification",
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
            return OccupancyVerification(verifications=[])

        try:
            return self.parse_verification_response(output_text)
        except (json.JSONDecodeError, ValidationError, ValueError):
            return OccupancyVerification(verifications=[])

    def parse_claude_response(self, response_text: str) -> ClaudeAnalysis:
        cleaned = self.strip_markdown_fences(response_text)
        data = json.loads(cleaned)
        if not isinstance(data, dict):
            raise ValueError("Claude response JSON must be an object")

        analysis = ClaudeAnalysis.model_validate(data)
        self.validate_port_logic(data)
        return analysis

    def parse_evidence_audit_response(self, response_text: str) -> ClaudeAnalysis:
        audit = self.parse_evidence_audit_raw_response(response_text)
        analysis = audit.to_claude_analysis()
        self.validate_port_logic(analysis.model_dump())
        return analysis

    def parse_evidence_audit_raw_response(
        self,
        response_text: str,
    ) -> PortEvidenceAudit:
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

        return audit

    def parse_verification_response(self, response_text: str) -> OccupancyVerification:
        cleaned = self.strip_markdown_fences(response_text)
        data = json.loads(cleaned)
        if not isinstance(data, dict):
            raise ValueError("Occupancy verification JSON must be an object")

        return OccupancyVerification.model_validate(data)

    @staticmethod
    def apply_occupancy_verification(
        analysis: ClaudeAnalysis,
        verification: OccupancyVerification,
    ) -> ClaudeAnalysis:
        confirmed = {
            item.puerto
            for item in verification.verifications
            if item.confirmed_occupied and item.visible_connector_body
        }
        occupied = [port for port in analysis.puertos_ocupados if port in confirmed]
        available = [port for port in range(1, 9) if port not in occupied]
        rejected_count = len(analysis.puertos_ocupados) - len(occupied)
        confidence = analysis.confidence
        if rejected_count > 0 and confidence == "high":
            confidence = "medium"

        return ClaudeAnalysis(
            total_puertos=8,
            total_ocupados=len(occupied),
            puertos_ocupados=occupied,
            total_disponibles=len(available),
            puertos_disponibles=available,
            codigo_dispositivo=analysis.codigo_dispositivo,
            ubicacion=analysis.ubicacion,
            observaciones=analysis.observaciones,
            confidence=confidence,
        )

    @staticmethod
    def lower_confidence(analysis: ClaudeAnalysis) -> ClaudeAnalysis:
        confidence = "medium" if analysis.confidence == "high" else analysis.confidence
        return ClaudeAnalysis(
            total_puertos=analysis.total_puertos,
            total_ocupados=analysis.total_ocupados,
            puertos_ocupados=analysis.puertos_ocupados,
            total_disponibles=analysis.total_disponibles,
            puertos_disponibles=analysis.puertos_disponibles,
            codigo_dispositivo=analysis.codigo_dispositivo,
            ubicacion=analysis.ubicacion,
            observaciones=analysis.observaciones,
            confidence=confidence,
        )

    @staticmethod
    def build_consensus_analysis(analyses: list[ClaudeAnalysis]) -> ClaudeAnalysis:
        if not analyses:
            raise ValueError("At least one analysis pass is required")

        occupied_sets = [set(analysis.puertos_ocupados) for analysis in analyses]
        consensus_occupied = sorted(set.intersection(*occupied_sets))
        consensus_available = [
            port for port in range(1, 9) if port not in consensus_occupied
        ]
        first = analyses[0]
        any_disagreement = len({tuple(sorted(item)) for item in occupied_sets}) > 1
        confidence = first.confidence
        if any_disagreement:
            confidence = "low"
        elif any(analysis.confidence != "high" for analysis in analyses):
            confidence = "medium"

        observations = first.observaciones
        if any_disagreement:
            observations = (
                f"{observations or ''} Consensus warning: analysis passes disagreed; "
                "only ports confirmed in every pass were returned as occupied."
            ).strip()

        return ClaudeAnalysis(
            total_puertos=8,
            total_ocupados=len(consensus_occupied),
            puertos_ocupados=consensus_occupied,
            total_disponibles=len(consensus_available),
            puertos_disponibles=consensus_available,
            codigo_dispositivo=first.codigo_dispositivo,
            ubicacion=first.ubicacion,
            observaciones=observations,
            confidence=confidence,
        )

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
