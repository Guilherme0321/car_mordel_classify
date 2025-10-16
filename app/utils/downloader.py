"""
Download Utilities
==================
Gerencia downloads de arquivos do Google Drive
"""

from pathlib import Path
from typing import Optional
import gdown

from app.core.logging import get_logger
from app.core.config import settings

logger = get_logger(__name__)


class ModelDownloader:
    """Gerenciador de download de modelos do Google Drive"""
    
    @staticmethod
    def download_from_drive(drive_url: str, destination: Path, quiet: bool = False) -> bool:
        """
        Baixa um arquivo do Google Drive
        
        Args:
            drive_url: URL do arquivo no Google Drive
            destination: Caminho de destino
            quiet: Se True, minimiza output
            
        Returns:
            bool: True se o download foi bem-sucedido
        """
        try:
            if not drive_url or 'SEU_FILE_ID' in drive_url:
                raise ValueError("URL do Google Drive não configurada corretamente")
            
            destination.parent.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Baixando de: {drive_url}")
            logger.info(f"Destino: {destination}")
            
            gdown.download(drive_url, str(destination), quiet=quiet, fuzzy=True)
            
            if not destination.exists():
                raise FileNotFoundError("Falha ao baixar o arquivo")
            
            size_mb = destination.stat().st_size / (1024 * 1024)
            logger.info(f"Download concluído: {destination.name} ({size_mb:.2f} MB)")
            return True
            
        except Exception as e:
            logger.error(f"Erro no download: {e}")
            return False
    
    @staticmethod
    def get_brand_model_url(brand: str) -> Optional[str]:
        """
        Obtém a URL do modelo específico de uma marca do Google Drive.
        Retorna None se não configurado (modelos locais serão usados).
        
        Args:
            brand: Nome da marca
            
        Returns:
            URL do Google Drive ou None
        """
        models_map = settings.get_brand_model_ids_map()
        
        if not models_map:
            # Não é mais um warning, pois é opcional (modelos locais)
            logger.debug("DRIVE_BRAND_MODELS_IDS não configurado (usando modelos locais)")
            return None
        
        file_id = models_map.get(brand)
        if file_id:
            return f"https://drive.google.com/uc?id={file_id}"
        
        logger.debug(f"ID do modelo para a marca '{brand}' não encontrado no .env")
        return None
    
    @staticmethod
    def is_model_cached(model_path: Path) -> bool:
        """
        Verifica se um modelo já está em cache local
        
        Args:
            model_path: Caminho do modelo
            
        Returns:
            bool: True se o modelo existe
        """
        return model_path.exists()
