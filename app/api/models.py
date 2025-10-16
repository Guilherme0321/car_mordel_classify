"""
Models Management Endpoints
============================
Endpoints para gerenciar modelos carregados
"""

from fastapi import APIRouter, HTTPException

from app.core.logging import get_logger
from app.services import CarModelClassifier

logger = get_logger(__name__)
router = APIRouter(prefix="/models", tags=["Models Management"])

# Classificador global (será injetado)
car_model_classifier: CarModelClassifier = None


def set_classifier(car_clf):
    """Define o classificador global"""
    global car_model_classifier
    car_model_classifier = car_clf


@router.get("")
async def get_models_info():
    """
    Lista todos os modelos carregados
    
    Retorna informações sobre:
    - Modelos em memória (cache)
    - Modelos no disco
    - Tamanho dos arquivos
    """
    if car_model_classifier is None:
        raise HTTPException(
            status_code=503,
            detail="Classificador não carregado"
        )
    
    return car_model_classifier.get_loaded_models_info()


@router.delete("/{brand}")
async def cleanup_brand_model(brand: str):
    """
    Remove o modelo de uma marca específica
    
    Remove o modelo do cache (memória) e do disco, liberando espaço.
    Útil para economizar espaço em disco quando você não precisa mais de um modelo específico.
    
    Exemplo:
    ```
    DELETE /models/BMW
    ```
    """
    if car_model_classifier is None:
        raise HTTPException(
            status_code=503,
            detail="Classificador não carregado"
        )
    
    logger.info(f"Requisição de limpeza para marca: {brand}")
    cleaned = car_model_classifier.cleanup_brand_model(brand)
    
    if cleaned:
        return {
            'success': True,
            'message': f"Modelo da marca '{brand}' removido com sucesso",
            'brand': brand
        }
    else:
        return {
            'success': False,
            'message': f"Modelo da marca '{brand}' não foi encontrado (não estava em cache ou disco)",
            'brand': brand
        }


@router.delete("")
async def cleanup_all_models():
    """
    Remove TODOS os modelos de marcas
    
    ⚠️ **ATENÇÃO**: Esta operação remove todos os modelos do cache e do disco.
    Use com cuidado! Os modelos precisarão ser baixados novamente do Google Drive quando necessários.
    
    Útil para:
    - Liberar espaço em disco
    - Resetar o cache de modelos
    - Forçar re-download de modelos atualizados
    """
    if car_model_classifier is None:
        raise HTTPException(
            status_code=503,
            detail="Classificador não carregado"
        )
    
    logger.warning("Requisição de limpeza TOTAL de modelos")
    result = car_model_classifier.cleanup_all_models()
    
    return {
        'success': True,
        'message': 'Todos os modelos foram removidos',
        'removed_from_cache': result['removed_from_cache'],
        'removed_from_disk': result['removed_from_disk']
    }
