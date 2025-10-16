"""
Health Check Endpoints
======================
Endpoints para verificar a saúde da API
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from app.core.logging import get_logger
from app.services import BrandClassifier, CarModelClassifier

logger = get_logger(__name__)
router = APIRouter(prefix="", tags=["Health"])

# Classificadores globais (serão injetados)
brand_classifier: BrandClassifier = None
car_model_classifier: CarModelClassifier = None


def set_classifiers(brand_clf, car_clf):
    """Define os classificadores globais"""
    global brand_classifier, car_model_classifier
    brand_classifier = brand_clf
    car_model_classifier = car_clf


@router.get("/")
async def root():
    """Endpoint raiz com informações da API"""
    return {
        "message": "Car Classification API - Two-Stage Model",
        "version": "2.0.0",
        "description": "Classificação inteligente: primeiro a marca, depois o modelo específico",
        "architecture": {
            "stage_1": "Brand Classification (sempre carregado)",
            "stage_2": "Model Classification (download sob demanda)"
        },
        "endpoints": {
            "predict": "POST /predict - Classificação completa (marca + modelo)",
            "predict_brand": "POST /predict/brand - Apenas marca",
            "health": "GET /health - Status da API",
            "models": "GET /models - Modelos carregados",
            "cleanup_brand": "DELETE /models/{brand} - Limpar modelo específico",
            "cleanup_all": "DELETE /models - Limpar todos os modelos"
        },
        "documentation": "GET /docs - Documentação interativa Swagger"
    }


@router.get("/health")
async def health_check():
    """Verifica a saúde da API e status dos classificadores"""
    if brand_classifier is None or car_model_classifier is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "message": "Classificadores não carregados",
                "brand_classifier": brand_classifier is not None,
                "model_classifier": car_model_classifier is not None
            }
        )
    
    from app.core.config import settings
    
    return {
        "status": "healthy",
        "brand_classifier": "loaded",
        "model_classifier": "loaded",
        "device": str(brand_classifier.device),
        "auto_cleanup": settings.AUTO_CLEANUP_MODELS
    }


@router.get("/info")
async def api_info():
    """Retorna informações detalhadas sobre a API e modelos"""
    if brand_classifier is None or car_model_classifier is None:
        raise HTTPException(status_code=503, detail="Classificadores não carregados")
    
    brand_info = brand_classifier.get_info()
    models_info = car_model_classifier.get_loaded_models_info()
    
    return {
        "api_version": "2.0.0",
        "brand_classifier": brand_info,
        "car_models": models_info
    }
