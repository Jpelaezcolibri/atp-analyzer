from datetime import UTC, datetime
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError

from app.analyzer import ATPAnalyzer
from app.auth import verify_api_key
from app.config import Settings, get_settings
from app.models import AnalyzeRequest, AnalyzeResponse, ErrorResponse


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    get_settings()
    yield


app = FastAPI(
    title="ATP Port Analyzer",
    version="1.0.0",
    description="Analyze ATP fiber optic distribution box port occupancy from image URLs.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def utc_now() -> datetime:
    return datetime.now(UTC)


def get_analyzer(settings: Settings = Depends(get_settings)) -> ATPAnalyzer:
    return ATPAnalyzer(settings=settings)


def extract_image_url_from_request(request: Request) -> str | None:
    return getattr(request.state, "image_url", None)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    if request.url.path == "/analyze" and exc.status_code != 401:
        error = ErrorResponse(
            error=str(exc.detail),
            image_url=extract_image_url_from_request(request),
            analyzed_at=utc_now(),
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=error.model_dump(mode="json"),
        )

    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )


@app.exception_handler(ValidationError)
async def validation_exception_handler(
    request: Request,
    exc: ValidationError,
) -> JSONResponse:
    error = ErrorResponse(
        error=f"Validation error: {exc}",
        image_url=extract_image_url_from_request(request),
        analyzed_at=utc_now(),
    )
    return JSONResponse(
        status_code=500,
        content=error.model_dump(mode="json"),
    )


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    if request.url.path == "/analyze":
        error = ErrorResponse(
            error=f"Request validation error: {exc}",
            image_url=extract_image_url_from_request(request),
            analyzed_at=utc_now(),
        )
        return JSONResponse(
            status_code=422,
            content=error.model_dump(mode="json"),
        )

    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    if request.url.path == "/analyze":
        error = ErrorResponse(
            error=f"Internal server error: {exc}",
            image_url=extract_image_url_from_request(request),
            analyzed_at=utc_now(),
        )
        return JSONResponse(
            status_code=500,
            content=error.model_dump(mode="json"),
        )

    raise exc


@app.get("/health")
async def health(settings: Settings = Depends(get_settings)) -> dict[str, str]:
    return {"status": "ok", "version": settings.app_version}


@app.post(
    "/analyze",
    response_model=AnalyzeResponse,
    dependencies=[Depends(verify_api_key)],
)
async def analyze(
    request_body: AnalyzeRequest,
    request: Request,
    analyzer: ATPAnalyzer = Depends(get_analyzer),
) -> AnalyzeResponse:
    image_url = str(request_body.image_url)
    request.state.image_url = image_url
    analyzed_at = utc_now()

    analysis = await analyzer.analyze(image_url)
    result, device_info, confidence = analysis.to_response_parts()

    return AnalyzeResponse(
        image_url=image_url,
        analyzed_at=analyzed_at,
        result=result,
        device_info=device_info,
        confidence=confidence,
    )
