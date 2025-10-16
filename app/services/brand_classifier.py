"""
Brand Classifier Service
=========================
Serviço para classificação de marcas de carros (Estágio 1)
"""

import torch
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from PIL import Image
from pathlib import Path
from typing import List, Dict, Any

from app.core.config import settings
from app.core.logging import get_logger
from app.utils import ModelDownloader

logger = get_logger(__name__)


class BrandClassifier:
    """Classificador de marcas (primeiro estágio)"""
    
    def __init__(self, model_path: Path = None):
        """
        Inicializa o classificador de marcas
        
        Args:
            model_path: Caminho do modelo (padrão: settings.BRAND_MODEL_PATH)
        """
        self.model_path = model_path or settings.BRAND_MODEL_PATH
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Baixar modelo se necessário
        if not self.model_path.exists():
            logger.info("Modelo de marcas não encontrado. Baixando do Google Drive...")
            self._download_model()
        
        # Carregar modelo
        self._load_model()
        
        # Configurar transformações
        self._setup_transforms()
        
        logger.info(f"✅ Classificador de marcas carregado! Dispositivo: {self.device}")
    
    def _download_model(self) -> None:
        """Baixa o modelo de marcas do Google Drive"""
        drive_url = settings.DRIVE_BRAND_MODEL_URL
        
        if not ModelDownloader.download_from_drive(drive_url, self.model_path):
            raise RuntimeError("Falha ao baixar modelo de marcas do Google Drive")
    
    def _load_model(self) -> None:
        """Carrega o modelo de marcas"""
        try:
            logger.info(f"Carregando modelo de marcas: {self.model_path}")
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            
            # Extrair informações
            self.model_info = {
                'model_type': checkpoint.get('model_type', 'efficientnet_b3'),
                'num_classes': checkpoint.get('num_classes', 0),
                'best_val_acc': checkpoint.get('best_val_acc', 0),
            }
            
            self.class_to_idx = checkpoint.get('class_to_idx', {})
            self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
            
            # Detectar arquitetura automaticamente
            arch_name = self._detect_model_architecture(checkpoint)
            logger.info(f"🔍 Arquitetura do modelo de marcas: {arch_name}")
            
            # Detectar tamanho da camada oculta
            hidden_size = self._detect_classifier_hidden_size(checkpoint)
            
            # Criar modelo com arquitetura e hidden_size corretos
            self.model = self._create_model_architecture(arch_name, self.model_info['num_classes'], hidden_size)
            
            # Carregar pesos
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            self.model = self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"✅ Modelo de marcas carregado: {self.model_info['num_classes']} classes - "
                       f"Acurácia: {self.model_info['best_val_acc']:.2f}% - Arch: {arch_name}")
            
        except Exception as e:
            logger.error(f"❌ Erro ao carregar modelo de marcas: {e}")
            raise
    
    def _detect_model_architecture(self, checkpoint: dict) -> str:
        """
        Detecta a arquitetura do modelo baseada no checkpoint.
        
        Args:
            checkpoint: Checkpoint do modelo
            
        Returns:
            Nome da arquitetura ('efficientnet_b0', 'efficientnet_b3', etc.)
        """
        # Verifica se tem informação explícita sobre a arquitetura
        if 'architecture' in checkpoint:
            return checkpoint['architecture']
        
        # Detecta pela estrutura do state_dict
        state_dict = checkpoint.get('model_state_dict', {})
        
        # EfficientNet-B0: features.8.0.weight tem shape [1280, 320, 1, 1]
        # EfficientNet-B3: features.8.0.weight tem shape [1536, 384, 1, 1]
        if 'features.8.0.weight' in state_dict:
            shape = state_dict['features.8.0.weight'].shape
            if shape[0] == 1280:
                return 'efficientnet_b0'
            elif shape[0] == 1536:
                return 'efficientnet_b3'
        
        # Padrão: usar b3 (esperado para modelo de marcas)
        logger.warning("Não foi possível detectar arquitetura, usando efficientnet_b3")
        return 'efficientnet_b3'
    
    def _detect_classifier_hidden_size(self, checkpoint: dict) -> int:
        """
        Detecta o tamanho da camada oculta do classificador.
        
        Args:
            checkpoint: Checkpoint do modelo
            
        Returns:
            Tamanho da camada oculta (256 ou 512)
        """
        state_dict = checkpoint.get('model_state_dict', {})
        
        # Verificar o shape de classifier.1.weight
        # Shape: [hidden_size, num_features]
        if 'classifier.1.weight' in state_dict:
            hidden_size = state_dict['classifier.1.weight'].shape[0]
            logger.info(f"🔍 Camada oculta detectada: {hidden_size} features")
            return hidden_size
        
        # Padrão: 512 (modelos novos)
        logger.warning("Não foi possível detectar tamanho da camada oculta, usando 512")
        return 512
    
    def _create_model_architecture(self, arch_name: str, num_classes: int, hidden_size: int = 512):
        """
        Cria a arquitetura do modelo.
        
        Args:
            arch_name: Nome da arquitetura
            num_classes: Número de classes
            hidden_size: Tamanho da camada oculta (256 ou 512)
            
        Returns:
            Modelo PyTorch
        """
        # Criar backbone
        if arch_name == 'efficientnet_b0':
            model = models.efficientnet_b0(weights=None)
        elif arch_name == 'efficientnet_b3':
            model = models.efficientnet_b3(weights=None)
        else:
            logger.warning(f"Arquitetura desconhecida '{arch_name}', usando efficientnet_b3")
            model = models.efficientnet_b3(weights=None)
        
        # Obter número de features do classificador
        num_features = model.classifier[1].in_features
        
        # Substituir classificador com hidden_size detectado
        model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.3),
            torch.nn.Linear(num_features, hidden_size),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm1d(hidden_size),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(hidden_size, num_classes)
        )
        
        return model
    
    def _setup_transforms(self) -> None:
        """Configura as transformações de imagem"""
        self.transforms = transforms.Compose([
            transforms.Resize(settings.MODEL_INPUT_SIZE),
            transforms.CenterCrop(settings.MODEL_CROP_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(settings.MODEL_NORMALIZE_MEAN, settings.MODEL_NORMALIZE_STD)
        ])
    
    def predict(self, image: Image.Image, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Prediz as marcas mais prováveis
        
        Args:
            image: Imagem PIL
            top_k: Número de top marcas a retornar
            
        Returns:
            Lista de predições com rank, marca e confiança
        """
        try:
            # Garantir que a imagem está em RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Preprocessar
            input_tensor = self.transforms(image).unsqueeze(0).to(self.device)
            
            # Predição
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = F.softmax(outputs, dim=1)
            
            # Top-k
            top_probs, top_indices = torch.topk(probabilities, min(top_k, len(self.idx_to_class)))
            
            # Formatar resultados
            predictions = []
            for i in range(len(top_indices[0])):
                class_idx = top_indices[0][i].item()
                prob = top_probs[0][i].item()
                brand = self.idx_to_class.get(class_idx, f"Unknown_{class_idx}")
                
                predictions.append({
                    'rank': i + 1,
                    'brand': brand,
                    'confidence': float(prob),
                    'confidence_percent': round(float(prob * 100), 2)
                })
            
            logger.info(f"Predição de marca: {predictions[0]['brand']} ({predictions[0]['confidence_percent']}%)")
            return predictions
            
        except Exception as e:
            logger.error(f"Erro na predição de marca: {e}")
            raise
    
    def get_info(self) -> Dict[str, Any]:
        """Retorna informações sobre o modelo"""
        return {
            'model_type': self.model_info['model_type'],
            'num_classes': self.model_info['num_classes'],
            'accuracy': self.model_info['best_val_acc'],
            'device': str(self.device),
            'model_path': str(self.model_path)
        }
