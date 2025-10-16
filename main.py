import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.logging import setup_logging, get_logger
from app.services import BrandClassifier, CarModelClassifier
from app.api import api_router
from app.api import health, predict, models

setup_logging()
logger = get_logger(__name__)

brand_classifier = None
car_model_classifier = None

def create_app():
    app = FastAPI(
        title=settings.API_TITLE,
        description=settings.API_DESCRIPTION,
        version=settings.API_VERSION
    )
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"]
    )
    
    app.include_router(api_router)
    
    @app.on_event("startup")
    async def startup():
        logger.info("Iniciando API...")
        load_classifiers()
    
    return app

def load_classifiers():
    global brand_classifier, car_model_classifier
    try:
        brand_classifier = BrandClassifier()
        car_model_classifier = CarModelClassifier()
        health.set_classifiers(brand_classifier, car_model_classifier)
        predict.set_classifiers(brand_classifier, car_model_classifier)
        models.set_classifier(car_model_classifier)
    except Exception as e:
        logger.error(f"Erro ao carregar classificadores: {e}")

app = create_app()

if __name__ == "__main__":
    uvicorn.run(app, host=settings.API_HOST, port=settings.API_PORT)
