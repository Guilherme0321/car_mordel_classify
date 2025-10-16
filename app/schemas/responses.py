"""
Pydantic Schemas
================
Modelos de dados para request/response
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class BrandPrediction(BaseModel):
    """Predição de marca"""
    rank: int = Field(..., description="Ranking da predição")
    brand: str = Field(..., description="Nome da marca")
    confidence: float = Field(..., description="Confiança (0-1)")
    confidence_percent: float = Field(..., description="Confiança em porcentagem")


class ModelPrediction(BaseModel):
    """Predição de modelo"""
    rank: int = Field(..., description="Ranking da predição")
    brand: str = Field(..., description="Nome da marca")
    model: str = Field(..., description="Nome do modelo")
    brand_model: str = Field(..., description="Marca_Modelo")
    confidence: float = Field(..., description="Confiança (0-1)")
    confidence_percent: float = Field(..., description="Confiança em porcentagem")


class FileInfo(BaseModel):
    """Informações do arquivo"""
    filename: str = Field(..., description="Nome do arquivo")
    content_type: str = Field(..., description="Tipo MIME")
    size_bytes: int = Field(..., description="Tamanho em bytes")


class BrandPredictionResponse(BaseModel):
    """Resposta de predição de marca"""
    success: bool = Field(..., description="Status da operação")
    stage: str = Field(..., description="Estágio da predição")
    predictions: List[BrandPrediction] = Field(..., description="Lista de predições")
    file_info: FileInfo = Field(..., description="Informações do arquivo")


class ModelInfo(BaseModel):
    """Informações do modelo"""
    brand: str = Field(..., description="Marca")
    accuracy: float = Field(..., description="Acurácia do modelo")
    total_classes: int = Field(..., description="Total de classes")


class ModelPredictionResult(BaseModel):
    """Resultado da predição de modelo"""
    success: bool = Field(..., description="Status da operação")
    predictions: Optional[List[ModelPrediction]] = Field(None, description="Lista de predições")
    model_info: Optional[ModelInfo] = Field(None, description="Informações do modelo")
    error: Optional[str] = Field(None, description="Mensagem de erro")


class BrandPredictionInfo(BaseModel):
    """Informação de predição de marca"""
    top_brands: List[BrandPrediction] = Field(..., description="Top marcas")
    selected_brand: str = Field(..., description="Marca selecionada")


class CompletePredictionResponse(BaseModel):
    """Resposta de predição completa"""
    success: bool = Field(..., description="Status da operação")
    stage: str = Field(..., description="Estágio da predição")
    brand_prediction: BrandPredictionInfo = Field(..., description="Predição de marca")
    model_prediction: ModelPredictionResult = Field(..., description="Predição de modelo")
    file_info: FileInfo = Field(..., description="Informações do arquivo")
    cleanup_performed: bool = Field(..., description="Se limpeza foi realizada")


class HealthResponse(BaseModel):
    """Resposta de health check"""
    status: str = Field(..., description="Status da API")
    brand_classifier: str = Field(..., description="Status do classificador de marcas")
    model_classifier: str = Field(..., description="Status do classificador de modelos")
    device: str = Field(..., description="Dispositivo (CPU/CUDA)")
    auto_cleanup: bool = Field(..., description="Se limpeza automática está ativada")


class ModelInMemory(BaseModel):
    """Modelo carregado em memória"""
    brand: str = Field(..., description="Marca")
    status: str = Field(..., description="Status (loaded/on_disk)")
    in_memory: bool = Field(..., description="Se está em memória")
    size_mb: Optional[float] = Field(None, description="Tamanho em MB (se em disco)")


class ModelsInfoResponse(BaseModel):
    """Resposta com informações de modelos"""
    total_in_memory: int = Field(..., description="Total em memória")
    total_on_disk: int = Field(..., description="Total no disco")
    models: List[ModelInMemory] = Field(..., description="Lista de modelos")


class CleanupResponse(BaseModel):
    """Resposta de limpeza"""
    success: bool = Field(..., description="Status da operação")
    message: str = Field(..., description="Mensagem")
    brand: Optional[str] = Field(None, description="Marca removida")
    removed_from_cache: Optional[int] = Field(None, description="Removidos do cache")
    removed_from_disk: Optional[int] = Field(None, description="Removidos do disco")
