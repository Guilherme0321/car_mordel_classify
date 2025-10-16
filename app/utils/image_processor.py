"""
Image Processing Utilities
===========================
Funções para processamento de imagens
"""

from PIL import Image
import io
from typing import Tuple

from app.core.logging import get_logger
from app.core.config import settings

logger = get_logger(__name__)


class ImageProcessor:
    """Processador de imagens"""
    
    @staticmethod
    def validate_image_type(content_type: str) -> bool:
        """
        Valida se o tipo de conteúdo é uma imagem válida
        
        Args:
            content_type: MIME type do arquivo
            
        Returns:
            bool: True se válido
        """
        return content_type in settings.ALLOWED_IMAGE_TYPES
    
    @staticmethod
    def validate_image_size(image_data: bytes) -> bool:
        """
        Valida se o tamanho da imagem está dentro do limite
        
        Args:
            image_data: Dados da imagem em bytes
            
        Returns:
            bool: True se válido
        """
        return len(image_data) <= settings.MAX_IMAGE_SIZE
    
    @staticmethod
    def load_image_from_bytes(image_data: bytes) -> Image.Image:
        """
        Carrega uma imagem PIL a partir de bytes
        
        Args:
            image_data: Dados da imagem
            
        Returns:
            Imagem PIL
        """
        try:
            image = Image.open(io.BytesIO(image_data))
            
            # Converter para RGB se necessário
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            return image
        except Exception as e:
            logger.error(f"Erro ao carregar imagem: {e}")
            raise ValueError(f"Não foi possível carregar a imagem: {str(e)}")
    
    @staticmethod
    def get_image_info(image: Image.Image) -> dict:
        """
        Retorna informações sobre a imagem
        
        Args:
            image: Imagem PIL
            
        Returns:
            Dicionário com informações
        """
        return {
            'size': image.size,
            'mode': image.mode,
            'format': image.format
        }
