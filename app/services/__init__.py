"""
Services Module
===============
Serviços de classificação
"""

from app.services.brand_classifier import BrandClassifier
from app.services.car_model_classifier import CarModelClassifier

__all__ = ['BrandClassifier', 'CarModelClassifier']
