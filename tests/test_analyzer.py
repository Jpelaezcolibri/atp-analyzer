import json

import httpx
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.analyzer import ATPAnalyzer
from app.config import Settings, get_settings
from app.main import app, get_analyzer
from app.models import ClaudeAnalysis


class DummyAnalyzer:
    async def analyze(self, image_url: str) -> ClaudeAnalysis:
        return ClaudeAnalysis(
            total_puertos=8,
            total_ocupados=4,
            puertos_ocupados=[1, 2, 3, 4],
            total_disponibles=4,
            puertos_disponibles=[5, 6, 7, 8],
            codigo_dispositivo="NE-312099-CO2-TARAP-CL6",
            ubicacion="RESERVA SERRAT (SELVA) TO 2",
            observaciones=None,
            confidence="high",
        )


@pytest.fixture(autouse=True)
def override_settings() -> None:
    app.dependency_overrides[get_settings] = lambda: Settings(
        LLM_PROVIDER="anthropic",
        ANTHROPIC_API_KEY="test-anthropic-key",
        ANTHROPIC_MODEL="claude-opus-4-5",
        OPENAI_API_KEY="test-openai-key",
        OPENAI_MODEL="gpt-5.1",
        API_KEY="test-api-key",
        PORT=8000,
        ENVIRONMENT="test",
    )
    app.dependency_overrides[get_analyzer] = lambda: DummyAnalyzer()
    yield
    app.dependency_overrides.clear()


def test_health_public() -> None:
    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": "1.0.0"}


def test_analyze_requires_api_key() -> None:
    client = TestClient(app)
    response = client.post(
        "/analyze",
        json={"image_url": "https://example.com/atp.jpg"},
    )

    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid or missing API key"}


def test_analyze_success() -> None:
    client = TestClient(app)
    response = client.post(
        "/analyze",
        headers={"X-API-Key": "test-api-key"},
        json={"image_url": "https://example.com/atp.jpg"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["image_url"] == "https://example.com/atp.jpg"
    assert body["result"]["total_puertos"] == 8
    assert body["result"]["puertos_ocupados"] == [1, 2, 3, 4]
    assert body["device_info"]["codigo_dispositivo"] == "NE-312099-CO2-TARAP-CL6"
    assert body["confidence"] == "high"


def test_strip_markdown_fences() -> None:
    raw = '```json\n{"total_puertos": 8}\n```'

    assert ATPAnalyzer.strip_markdown_fences(raw) == '{"total_puertos": 8}'


def test_parse_valid_claude_response() -> None:
    settings = Settings(
        LLM_PROVIDER="anthropic",
        ANTHROPIC_API_KEY="test-anthropic-key",
        API_KEY="test-api-key",
    )
    analyzer = ATPAnalyzer(settings=settings, anthropic_client=object())
    payload = {
        "total_puertos": 8,
        "total_ocupados": 2,
        "puertos_ocupados": [1, 8],
        "total_disponibles": 6,
        "puertos_disponibles": [2, 3, 4, 5, 6, 7],
        "codigo_dispositivo": None,
        "ubicacion": None,
        "observaciones": None,
        "confidence": "medium",
    }

    analysis = analyzer.parse_claude_response(json.dumps(payload))

    assert analysis.total_ocupados == 2
    assert analysis.puertos_disponibles == [2, 3, 4, 5, 6, 7]


def test_parse_invalid_ports_fails() -> None:
    settings = Settings(
        LLM_PROVIDER="anthropic",
        ANTHROPIC_API_KEY="test-anthropic-key",
        API_KEY="test-api-key",
    )
    analyzer = ATPAnalyzer(settings=settings, anthropic_client=object())
    payload = {
        "total_puertos": 8,
        "total_ocupados": 2,
        "puertos_ocupados": [1, 1],
        "total_disponibles": 6,
        "puertos_disponibles": [2, 3, 4, 5, 6, 7],
        "codigo_dispositivo": None,
        "ubicacion": None,
        "observaciones": None,
        "confidence": "low",
    }

    with pytest.raises(ValueError):
        analyzer.parse_claude_response(json.dumps(payload))


@pytest.mark.asyncio
async def test_fetch_image_rejects_non_image(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(
        LLM_PROVIDER="anthropic",
        ANTHROPIC_API_KEY="test-anthropic-key",
        API_KEY="test-api-key",
    )
    analyzer = ATPAnalyzer(settings=settings, anthropic_client=object())

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/html"},
            content=b"<html></html>",
        )

    transport = httpx.MockTransport(handler)
    original_client = httpx.AsyncClient

    def mock_client(*args: object, **kwargs: object) -> httpx.AsyncClient:
        kwargs["transport"] = transport
        return original_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", mock_client)

    with pytest.raises(HTTPException) as exc_info:
        await analyzer.fetch_image("https://example.com/not-image")

    assert exc_info.value.status_code == 422
