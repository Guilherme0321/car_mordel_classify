"""
Schemas Module
==============
Modelos de dados Pydantic
"""

from app.schemas.responses import (
    BrandPrediction,
    ModelPrediction,
    FileInfo,
    BrandPredictionResponse,
    CompletePredictionResponse,
    HealthResponse,
    ModelsInfoResponse,
    CleanupResponse
)

__all__ = [
    'BrandPrediction',
    'ModelPrediction',
    'FileInfo',
    'BrandPredictionResponse',
    'CompletePredictionResponse',
    'HealthResponse',
    'ModelsInfoResponse',
    'CleanupResponse'
]
