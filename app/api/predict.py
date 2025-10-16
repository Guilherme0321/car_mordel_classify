"""
Prediction Endpoints
====================
Endpoints para classificação de carros
"""

from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from typing import Optional

from app.core.logging import get_logger
from app.core.config import settings
from app.utils import ImageProcessor
from app.services import BrandClassifier, CarModelClassifier

logger = get_logger(__name__)
router = APIRouter(prefix="/predict", tags=["Prediction"])

# Classificadores globais (serão injetados)
brand_classifier: BrandClassifier = None
car_model_classifier: CarModelClassifier = None


def set_classifiers(brand_clf, car_clf):
    """Define os classificadores globais"""
    global brand_classifier, car_model_classifier
    brand_classifier = brand_clf
    car_model_classifier = car_clf


@router.post("/brand")
async def predict_brand_only(
    file: UploadFile = File(..., description="Imagem do carro"),
    top_k: int = Query(3, ge=1, le=10, description="Número de marcas top a retornar")
):
    """
    Prediz apenas a marca do carro (Estágio 1)
    
    Este endpoint é rápido pois usa apenas o modelo de marcas que já está em memória.
    Útil quando você só precisa saber a marca sem identificar o modelo específico.
    """
    if brand_classifier is None:
        raise HTTPException(
            status_code=503,
            detail="Classificador de marcas não carregado"
        )
    
    # Validações
    if not ImageProcessor.validate_image_type(file.content_type):
        raise HTTPException(
            status_code=400,
            detail=f"Tipo de arquivo inválido. Tipos aceitos: {settings.ALLOWED_IMAGE_TYPES}"
        )
    
    try:
        # Ler imagem
        image_data = await file.read()
        
        if not ImageProcessor.validate_image_size(image_data):
            raise HTTPException(
                status_code=400,
                detail=f"Imagem muito grande. Tamanho máximo: {settings.MAX_IMAGE_SIZE / (1024*1024):.2f}MB"
            )
        
        image = ImageProcessor.load_image_from_bytes(image_data)
        
        # Predição
        predictions = brand_classifier.predict(image, top_k)
        
        return {
            'success': True,
            'stage': 'brand_only',
            'predictions': predictions,
            'file_info': {
                'filename': file.filename,
                'content_type': file.content_type,
                'size_bytes': len(image_data)
            }
        }
        
    except ValueError as e:
        logger.error(f"Erro de validação: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Erro ao processar imagem: {e}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")


@router.post("")
async def predict_car(
    file: UploadFile = File(..., description="Imagem do carro"),
    top_k: int = Query(5, ge=1, le=10, description="Número de modelos top a retornar"),
    cleanup_after: bool = Query(False, description="Remover modelo do disco após uso"),
    brand_hint: Optional[str] = Query(None, description="Marca específica (pula predição de marca)")
):
    """
    Classificação completa: marca + modelo (Dois Estágios)
    
    Processo:
    1. **Estágio 1**: Prediz a marca (ou usa brand_hint se fornecido)
    2. **Estágio 2**: Baixa o modelo específico da marca (se necessário) e prediz o modelo
    3. **Opcional**: Remove o modelo do disco após uso (se cleanup_after=true)
    
    Parâmetros:
    - file: Imagem do carro (JPG, PNG)
    - top_k: Quantos modelos retornar (1-10)
    - cleanup_after: Se true, remove o modelo após uso (economiza espaço)
    - brand_hint: Se você já sabe a marca, pula o estágio 1
    
    Exemplo com brand_hint:
    ```
    POST /predict?brand_hint=BMW&top_k=3
    ```
    
    Economia de espaço:
    - Use cleanup_after=true para remover o modelo após cada uso
    - Ou configure AUTO_CLEANUP_MODELS=true no .env para sempre limpar
    """
    if brand_classifier is None or car_model_classifier is None:
        raise HTTPException(
            status_code=503,
            detail="Classificadores não carregados"
        )
    
    # Validações
    if not ImageProcessor.validate_image_type(file.content_type):
        raise HTTPException(
            status_code=400,
            detail=f"Tipo de arquivo inválido. Tipos aceitos: {settings.ALLOWED_IMAGE_TYPES}"
        )
    
    try:
        # Ler imagem
        image_data = await file.read()
        
        if not ImageProcessor.validate_image_size(image_data):
            raise HTTPException(
                status_code=400,
                detail=f"Imagem muito grande. Tamanho máximo: {settings.MAX_IMAGE_SIZE / (1024*1024):.2f}MB"
            )
        
        image = ImageProcessor.load_image_from_bytes(image_data)
        
        # Estágio 1: Predizer marca (ou usar hint)
        if brand_hint:
            logger.info(f"Usando brand_hint fornecido: {brand_hint}")
            predicted_brand = brand_hint
            brand_predictions = [{'brand': brand_hint, 'confidence_percent': 100.0}]
        else:
            logger.info("Iniciando predição de marca...")
            brand_predictions = brand_classifier.predict(image, top_k=3)
            predicted_brand = brand_predictions[0]['brand']
        
        logger.info(f"Marca prevista: {predicted_brand}")
        
        # Estágio 2: Predizer modelo específico
        logger.info(f"Iniciando predição de modelo para marca: {predicted_brand}")
        model_result = car_model_classifier.predict(image, predicted_brand, top_k)
        
        # Limpeza (automática ou sob demanda)
        should_cleanup = cleanup_after or settings.AUTO_CLEANUP_MODELS
        cleanup_performed = False
        
        if should_cleanup and model_result.get('success'):
            cleanup_performed = car_model_classifier.cleanup_brand_model(predicted_brand)
            if cleanup_performed:
                logger.info(f"Modelo '{predicted_brand}' removido do disco (limpeza {'automática' if settings.AUTO_CLEANUP_MODELS else 'sob demanda'})")
        
        # Resposta completa
        return {
            'success': True,
            'stage': 'complete',
            'brand_prediction': {
                'top_brands': brand_predictions,
                'selected_brand': predicted_brand
            },
            'model_prediction': model_result,
            'file_info': {
                'filename': file.filename,
                'content_type': file.content_type,
                'size_bytes': len(image_data)
            },
            'cleanup_performed': cleanup_performed
        }
        
    except ValueError as e:
        logger.error(f"Erro de validação: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        logger.error(f"Erro ao carregar modelo: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Erro inesperado: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")
