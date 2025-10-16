"""
API Routes Module
=================
Agregador de todas as rotas da API
"""

from fastapi import APIRouter
from app.api import health, predict, models

# Router principal
api_router = APIRouter()

# Incluir rotas
api_router.include_router(health.router)
api_router.include_router(predict.router)
api_router.include_router(models.router)

__all__ = ['api_router', 'health', 'predict', 'models']
