"""
File Utilities
==============
Funções auxiliares para manipulação de arquivos
"""

from pathlib import Path
from typing import List
import shutil

from app.core.logging import get_logger

logger = get_logger(__name__)


class FileUtils:
    """Utilitários para manipulação de arquivos"""
    
    @staticmethod
    def ensure_directory(directory: Path) -> None:
        """
        Garante que um diretório existe
        
        Args:
            directory: Caminho do diretório
        """
        directory.mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def get_file_size_mb(file_path: Path) -> float:
        """
        Retorna o tamanho de um arquivo em MB
        
        Args:
            file_path: Caminho do arquivo
            
        Returns:
            Tamanho em MB
        """
        if not file_path.exists():
            return 0.0
        return file_path.stat().st_size / (1024 * 1024)
    
    @staticmethod
    def delete_file(file_path: Path) -> bool:
        """
        Remove um arquivo do disco
        
        Args:
            file_path: Caminho do arquivo
            
        Returns:
            bool: True se removido com sucesso
        """
        try:
            if file_path.exists():
                file_path.unlink()
                logger.info(f"Arquivo removido: {file_path}")
                return True
            return False
        except Exception as e:
            logger.error(f"Erro ao remover arquivo {file_path}: {e}")
            return False
    
    @staticmethod
    def list_files_by_extension(directory: Path, extension: str) -> List[Path]:
        """
        Lista arquivos em um diretório por extensão
        
        Args:
            directory: Diretório a pesquisar
            extension: Extensão (ex: '.pth')
            
        Returns:
            Lista de arquivos encontrados
        """
        if not directory.exists():
            return []
        return list(directory.glob(f"*{extension}"))
    
    @staticmethod
    def normalize_brand_name(brand: str) -> str:
        """
        Normaliza nome de marca para uso em nomes de arquivo
        
        Args:
            brand: Nome da marca
            
        Returns:
            Nome normalizado
        """
        return brand.replace(' ', '_').replace('-', '_').replace('/', '_')
    
    @staticmethod
    def clear_directory(directory: Path, pattern: str = "*") -> int:
        """
        Remove todos os arquivos de um diretório que correspondem ao padrão
        
        Args:
            directory: Diretório
            pattern: Padrão de arquivo (ex: "*.pth")
            
        Returns:
            Número de arquivos removidos
        """
        count = 0
        try:
            for file_path in directory.glob(pattern):
                if file_path.is_file():
                    file_path.unlink()
                    count += 1
            logger.info(f"Removidos {count} arquivos de {directory}")
            return count
        except Exception as e:
            logger.error(f"Erro ao limpar diretório {directory}: {e}")
            return count
