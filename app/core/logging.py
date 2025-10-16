"""
Logging Configuration
=====================
Sistema de logging centralizado
"""

import logging
import sys
from pathlib import Path
from app.core.config import settings


def setup_logging():
    """Configura o sistema de logging da aplicação"""
    
    # Criar diretório de logs se não existir
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configurar formato
    log_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Logger raiz
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, settings.LOG_LEVEL))
    
    # Handler para console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    
    # Handler para arquivo
    file_handler = logging.FileHandler(log_dir / "api.log")
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Retorna um logger específico
    
    Args:
        name: Nome do módulo
        
    Returns:
        Logger configurado
    """
    return logging.getLogger(name)
