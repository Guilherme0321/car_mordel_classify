"""
Core Configuration Module
=========================
Gerencia todas as configurações da aplicação carregadas do .env
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()


class Settings:
    """Configurações centralizadas da aplicação"""
    
    # Diretórios
    BASE_DIR: Path = Path(__file__).parent.parent.parent
    MODELS_DIR: Path = BASE_DIR / os.getenv('MODELS_DIR', 'models/car_models')
    BRAND_MODEL_PATH: Path = BASE_DIR / os.getenv('BRAND_MODEL_PATH', 'models/mark_efficientnet_b3_acc_97.46.pth')
    
    # Google Drive URLs
    DRIVE_BRAND_MODEL_URL: str = os.getenv('DRIVE_BRAND_MODEL_URL', '')
    DRIVE_BRAND_MODELS_IDS: str = os.getenv('DRIVE_BRAND_MODELS_IDS', '')
    
    # API Configuration
    API_HOST: str = os.getenv('API_HOST', '0.0.0.0')
    API_PORT: int = int(os.getenv('API_PORT', '8000'))
    API_TITLE: str = "Car Classification API - Two-Stage"
    API_VERSION: str = "2.0.0"
    API_DESCRIPTION: str = "API para classificação de marca e modelo de carros usando abordagem em duas etapas"
    
    # CORS
    CORS_ORIGINS: list = os.getenv('CORS_ORIGINS', '*').split(',')
    CORS_CREDENTIALS: bool = True
    CORS_METHODS: list = ["*"]
    CORS_HEADERS: list = ["*"]
    
    # Model Settings
    AUTO_CLEANUP_MODELS: bool = os.getenv('AUTO_CLEANUP_MODELS', 'false').lower() == 'true'
    MAX_IMAGE_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_IMAGE_TYPES: set = {'image/jpeg', 'image/png', 'image/jpg'}
    
    # Logging
    LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')
    
    # Model Architecture
    MODEL_INPUT_SIZE: tuple = (384, 384)
    MODEL_CROP_SIZE: int = 336
    MODEL_NORMALIZE_MEAN: list = [0.485, 0.456, 0.406]
    MODEL_NORMALIZE_STD: list = [0.229, 0.224, 0.225]
    
    @classmethod
    def get_brand_model_ids_map(cls) -> dict:
        """
        Retorna mapeamento de marcas para IDs do Google Drive
        
        Returns:
            dict: {brand: file_id}
        """
        models_map = {}
        if cls.DRIVE_BRAND_MODELS_IDS:
            for entry in cls.DRIVE_BRAND_MODELS_IDS.split(','):
                if ':' in entry:
                    brand, file_id = entry.strip().split(':', 1)
                    models_map[brand.strip()] = file_id.strip()
        return models_map
    
    @classmethod
    def validate(cls) -> list:
        """
        Valida as configurações
        
        Returns:
            list: Lista de erros encontrados
        """
        errors = []
        
        if not cls.DRIVE_BRAND_MODEL_URL or 'SEU_FILE_ID' in cls.DRIVE_BRAND_MODEL_URL:
            errors.append("DRIVE_BRAND_MODEL_URL não configurada corretamente no .env")
        
        if not cls.DRIVE_BRAND_MODELS_IDS:
            errors.append("DRIVE_BRAND_MODELS_IDS não configurada no .env")
        
        if not (1000 <= cls.API_PORT <= 65535):
            errors.append(f"Porta inválida: {cls.API_PORT}")
        
        return errors


# Instância global de configurações
settings = Settings()
