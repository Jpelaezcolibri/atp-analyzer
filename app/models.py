from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, HttpUrl, field_validator, model_validator


PortNumber = Literal[1, 2, 3, 4, 5, 6, 7, 8]
Confidence = Literal["high", "medium", "low"]


class AnalyzeRequest(BaseModel):
    image_url: HttpUrl


class PortAnalysisResult(BaseModel):
    total_puertos: int = Field(8)
    total_ocupados: int
    puertos_ocupados: list[PortNumber]
    total_disponibles: int
    puertos_disponibles: list[PortNumber]

    @field_validator("puertos_ocupados", "puertos_disponibles")
    @classmethod
    def sort_ports(cls, value: list[PortNumber]) -> list[PortNumber]:
        return sorted(value)

    @model_validator(mode="after")
    def validate_ports(self) -> "PortAnalysisResult":
        occupied = list(self.puertos_ocupados)
        available = list(self.puertos_disponibles)
        all_ports = occupied + available

        if self.total_puertos != 8:
            raise ValueError("total_puertos must be 8")
        if len(occupied) != self.total_ocupados:
            raise ValueError("total_ocupados must match puertos_ocupados length")
        if len(available) != self.total_disponibles:
            raise ValueError(
                "total_disponibles must match puertos_disponibles length"
            )
        if len(all_ports) != 8:
            raise ValueError("occupied and available ports must total 8")
        if len(set(all_ports)) != 8:
            raise ValueError("ports cannot contain duplicates")
        if sorted(all_ports) != list(range(1, 9)):
            raise ValueError("ports must cover exactly [1,2,3,4,5,6,7,8]")

        return self


class DeviceInfo(BaseModel):
    codigo_dispositivo: str | None
    ubicacion: str | None
    observaciones: str | None


class ClaudeAnalysis(BaseModel):
    total_puertos: int = Field(8)
    total_ocupados: int
    puertos_ocupados: list[PortNumber]
    total_disponibles: int
    puertos_disponibles: list[PortNumber]
    codigo_dispositivo: str | None
    ubicacion: str | None
    observaciones: str | None
    confidence: Confidence

    @model_validator(mode="after")
    def validate_complete_analysis(self) -> "ClaudeAnalysis":
        PortAnalysisResult(
            total_puertos=self.total_puertos,
            total_ocupados=self.total_ocupados,
            puertos_ocupados=self.puertos_ocupados,
            total_disponibles=self.total_disponibles,
            puertos_disponibles=self.puertos_disponibles,
        )
        return self

    def to_response_parts(self) -> tuple[PortAnalysisResult, DeviceInfo, Confidence]:
        result = PortAnalysisResult(
            total_puertos=self.total_puertos,
            total_ocupados=self.total_ocupados,
            puertos_ocupados=self.puertos_ocupados,
            total_disponibles=self.total_disponibles,
            puertos_disponibles=self.puertos_disponibles,
        )
        device_info = DeviceInfo(
            codigo_dispositivo=self.codigo_dispositivo,
            ubicacion=self.ubicacion,
            observaciones=self.observaciones,
        )
        return result, device_info, self.confidence


class PortEvidence(BaseModel):
    puerto: PortNumber
    estado: Literal["occupied", "available"]
    evidencia: Literal[
        "external_connector_body_visible",
        "empty_green_adapter_only",
        "shadow_or_internal_plastic_only",
        "nearby_cable_not_inserted",
        "technician_or_meter_connector_only",
        "occluded_or_uncertain",
    ]
    razon: str


class PortEvidenceAudit(BaseModel):
    total_puertos: int = Field(8)
    puertos: list[PortEvidence]
    codigo_dispositivo: str | None
    ubicacion: str | None
    observaciones: str | None
    confidence: Confidence

    @model_validator(mode="after")
    def validate_audit(self) -> "PortEvidenceAudit":
        port_numbers = [port.puerto for port in self.puertos]
        if self.total_puertos != 8:
            raise ValueError("total_puertos must be 8")
        if sorted(port_numbers) != list(range(1, 9)):
            raise ValueError("audit must include exactly ports 1..8")
        if len(set(port_numbers)) != 8:
            raise ValueError("audit ports cannot contain duplicates")
        return self

    def to_claude_analysis(self) -> ClaudeAnalysis:
        occupied = sorted(
            port.puerto
            for port in self.puertos
            if (
                port.estado == "occupied"
                and port.evidencia == "external_connector_body_visible"
            )
        )
        available = [port for port in range(1, 9) if port not in occupied]
        confidence: Confidence = self.confidence
        uncertain_count = sum(
            1 for port in self.puertos if port.evidencia == "occluded_or_uncertain"
        )
        if uncertain_count > 0 and confidence == "high":
            confidence = "medium"

        return ClaudeAnalysis(
            total_puertos=8,
            total_ocupados=len(occupied),
            puertos_ocupados=occupied,
            total_disponibles=len(available),
            puertos_disponibles=available,
            codigo_dispositivo=self.codigo_dispositivo,
            ubicacion=self.ubicacion,
            observaciones=self.observaciones,
            confidence=confidence,
        )


class AnalyzeResponse(BaseModel):
    success: Literal[True] = True
    image_url: str
    analyzed_at: datetime
    result: PortAnalysisResult
    device_info: DeviceInfo
    confidence: Confidence


class ErrorResponse(BaseModel):
    success: Literal[False] = False
    error: str
    image_url: str | None = None
    analyzed_at: datetime
