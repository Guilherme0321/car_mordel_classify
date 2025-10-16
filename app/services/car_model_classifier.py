"""
Car Model Classifier Service
=============================
Serviço para classificação de modelos específicos por marca (Estágio 2)
"""

import torch
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from PIL import Image
from pathlib import Path
from typing import Dict, Any, Optional

from app.core.config import settings
from app.core.logging import get_logger
from app.utils import ModelDownloader, FileUtils

logger = get_logger(__name__)


class CarModelClassifier:
    """Classificador de modelos específicos (segundo estágio)"""
    
    def __init__(self, models_dir: Path = None):
        """
        Inicializa o classificador de modelos
        
        Args:
            models_dir: Diretório onde os modelos são armazenados
        """
        self.models_dir = models_dir or settings.MODELS_DIR
        FileUtils.ensure_directory(self.models_dir)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.loaded_models = {}  # Cache de modelos em memória
        
        # Configurar transformações
        self._setup_transforms()
        
        logger.info(f"✅ Classificador de modelos inicializado! Dispositivo: {self.device}")
    
    def _setup_transforms(self) -> None:
        """Configura as transformações de imagem"""
        self.transforms = transforms.Compose([
            transforms.Resize(settings.MODEL_INPUT_SIZE),
            transforms.CenterCrop(settings.MODEL_CROP_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(settings.MODEL_NORMALIZE_MEAN, settings.MODEL_NORMALIZE_STD)
        ])
    
    def _get_model_path(self, brand: str) -> Path:
        """
        Retorna o caminho do arquivo do modelo para uma marca.
        Busca por padrões de nome flexíveis.
        
        Args:
            brand: Nome da marca
            
        Returns:
            Path do modelo
        """
        safe_brand = FileUtils.normalize_brand_name(brand)
        
        # Buscar primeiro na pasta models/car_model/ (estrutura antiga)
        legacy_dir = settings.BASE_DIR / "models" / "car_model"
        if legacy_dir.exists():
            # Procurar por arquivo com o nome da marca (flexível)
            for pattern in [
                f"{safe_brand}_model_acc_*.pth",  # Ex: Audi_model_acc_90.89.pth
                f"{safe_brand}_model.pth",         # Ex: Audi_model.pth
                f"{safe_brand}_*.pth",             # Ex: Audi_efficientnet_b3.pth
            ]:
                matches = list(legacy_dir.glob(pattern))
                if matches:
                    logger.info(f"Modelo encontrado localmente: {matches[0].name}")
                    return matches[0]
        
        # Se não encontrou, usar caminho padrão em models/car_models/
        return self.models_dir / f"{safe_brand}_efficientnet_b3.pth"
    
    def _download_brand_model(self, brand: str) -> bool:
        """
        Baixa o modelo específico de uma marca.
        Só baixa se não existir localmente.
        
        Args:
            brand: Nome da marca
            
        Returns:
            bool: True se download bem-sucedido ou já existe
        """
        model_path = self._get_model_path(brand)
        
        if model_path.exists():
            logger.info(f"✅ Modelo para '{brand}' encontrado: {model_path.name}")
            return True
        
        # Não encontrou localmente, tentar baixar do Drive
        logger.warning(f"⚠️ Modelo '{brand}' não encontrado localmente")
        logger.info(f"Tentando baixar do Google Drive...")
        
        drive_url = ModelDownloader.get_brand_model_url(brand)
        
        if not drive_url:
            logger.error(f"❌ URL não configurada para marca: {brand}")
            logger.info(f"💡 Dica: Adicione {brand} no .env em DRIVE_BRAND_MODELS_IDS ou coloque o arquivo em models/car_model/")
            return False
        
        return ModelDownloader.download_from_drive(drive_url, model_path)
    
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
        
        # Padrão: usar b0 (mais comum nos modelos de marca)
        logger.warning("Não foi possível detectar arquitetura, usando efficientnet_b0")
        return 'efficientnet_b0'
    
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
            logger.warning(f"Arquitetura desconhecida '{arch_name}', usando efficientnet_b0")
            model = models.efficientnet_b0(weights=None)
        
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
    
    def _load_brand_model(self, brand: str) -> Dict[str, Any]:
        """
        Carrega o modelo de uma marca específica.
        Detecta automaticamente a arquitetura correta.
        
        Args:
            brand: Nome da marca
            
        Returns:
            Dicionário com modelo, classes e informações
        """
        # Verificar cache
        if brand in self.loaded_models:
            logger.info(f"Usando modelo em cache para: {brand}")
            return self.loaded_models[brand]
        
        model_path = self._get_model_path(brand)
        
        # Download se necessário
        if not model_path.exists():
            if not self._download_brand_model(brand):
                raise RuntimeError(f"Não foi possível obter modelo para: {brand}")
        
        try:
            # Carregar checkpoint
            logger.info(f"Carregando modelo de '{brand}' do disco...")
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            model_info = {
                'brand_name': checkpoint.get('brand_name', brand),
                'num_classes': checkpoint.get('num_classes', 0),
                'best_val_acc': checkpoint.get('best_val_acc', 0),
            }
            
            # Usar os campos corretos: model_to_idx e idx_to_model (não class_to_idx)
            model_to_idx = checkpoint.get('model_to_idx', {})
            idx_to_model = checkpoint.get('idx_to_model', {})
            
            # Se idx_to_model estiver vazio, tentar class_to_idx (fallback)
            if not idx_to_model:
                class_to_idx = checkpoint.get('class_to_idx', {})
                idx_to_model = {idx: cls for cls, idx in class_to_idx.items()}
            
            # Detectar arquitetura automaticamente
            arch_name = self._detect_model_architecture(checkpoint)
            logger.info(f"🔍 Arquitetura detectada: {arch_name}")
            
            # Detectar tamanho da camada oculta
            hidden_size = self._detect_classifier_hidden_size(checkpoint)
            
            # Criar modelo com arquitetura e hidden_size corretos
            model = self._create_model_architecture(arch_name, model_info['num_classes'], hidden_size)
            
            # Carregar pesos
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            model = model.to(self.device)
            model.eval()
            
            # Cachear
            self.loaded_models[brand] = {
                'model': model,
                'idx_to_model': idx_to_model,  # Corrigido: usar idx_to_model
                'model_info': model_info,
                'architecture': arch_name
            }
            
            logger.info(f"✅ Modelo '{brand}' carregado: {model_info['num_classes']} classes - "
                       f"Acurácia: {model_info['best_val_acc']:.2f}% - Arch: {arch_name}")
            
            return self.loaded_models[brand]
            
        except Exception as e:
            logger.error(f"❌ Erro ao carregar modelo de '{brand}': {e}")
            raise
    
    def predict(self, image: Image.Image, brand: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Prediz o modelo específico dentro de uma marca
        
        Args:
            image: Imagem PIL
            brand: Marca do carro
            top_k: Número de predições a retornar
            
        Returns:
            Dicionário com predições e metadados
        """
        try:
            # Carregar modelo da marca
            brand_model = self._load_brand_model(brand)
            model = brand_model['model']
            idx_to_model = brand_model['idx_to_model']  # Corrigido: usar idx_to_model
            model_info = brand_model['model_info']
            
            # Preprocessar (mesma imagem usada no stage 1)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            input_tensor = self.transforms(image).unsqueeze(0).to(self.device)
            
            # Predição (apenas o modelo, não marca_modelo)
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = F.softmax(outputs, dim=1)
            
            # Top-k
            top_probs, top_indices = torch.topk(probabilities, min(top_k, len(idx_to_model)))
            
            # Formatar resultados (modelo retorna apenas nome do modelo)
            predictions = []
            for i in range(len(top_indices[0])):
                class_idx = top_indices[0][i].item()
                prob = top_probs[0][i].item()
                
                # idx_to_model pode ser dict com chaves string ou int
                if isinstance(list(idx_to_model.keys())[0], str):
                    model_name = idx_to_model.get(str(class_idx), f"Unknown_{class_idx}")
                else:
                    model_name = idx_to_model.get(class_idx, f"Unknown_{class_idx}")
                
                predictions.append({
                    'rank': i + 1,
                    'brand': brand,  # Usar a marca passada como parâmetro
                    'model': model_name,  # Apenas o nome do modelo (ex: "A3", "A4")
                    'brand_model': f"{brand} {model_name}",  # Combinado para exibição
                    'confidence': float(prob),
                    'confidence_percent': round(float(prob * 100), 2)
                })
            
            if predictions:
                logger.info(f"Predição de modelo para '{brand}': {predictions[0]['model']} ({predictions[0]['confidence_percent']}%)")
            else:
                logger.warning("Nenhuma predição gerada")
            
            return {
                'success': True,
                'predictions': predictions,
                'model_info': {
                    'brand': brand,
                    'accuracy': model_info['best_val_acc'],
                    'total_classes': model_info['num_classes']
                }
            }
            
        except Exception as e:
            logger.error(f"Erro na predição de modelo: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def cleanup_brand_model(self, brand: str) -> bool:
        """
        Remove o modelo de uma marca do disco e cache
        
        Args:
            brand: Nome da marca
            
        Returns:
            bool: True se removido com sucesso
        """
        try:
            removed = False
            
            # Remover do cache
            if brand in self.loaded_models:
                del self.loaded_models[brand]
                logger.info(f"Modelo '{brand}' removido do cache")
                removed = True
            
            # Remover arquivo
            model_path = self._get_model_path(brand)
            if FileUtils.delete_file(model_path):
                removed = True
            
            return removed
            
        except Exception as e:
            logger.error(f"Erro ao limpar modelo de '{brand}': {e}")
            return False
    
    def cleanup_all_models(self) -> Dict[str, int]:
        """
        Remove todos os modelos de marcas
        
        Returns:
            Dicionário com contagens de remoção
        """
        try:
            # Limpar cache
            brands_in_cache = list(self.loaded_models.keys())
            for brand in brands_in_cache:
                if brand in self.loaded_models:
                    del self.loaded_models[brand]
            
            # Limpar disco
            removed_from_disk = FileUtils.clear_directory(self.models_dir, "*.pth")
            
            logger.info(f"Limpeza completa: {len(brands_in_cache)} do cache, {removed_from_disk} do disco")
            
            return {
                'removed_from_cache': len(brands_in_cache),
                'removed_from_disk': removed_from_disk
            }
            
        except Exception as e:
            logger.error(f"Erro ao limpar todos os modelos: {e}")
            return {'removed_from_cache': 0, 'removed_from_disk': 0}
    
    def get_loaded_models_info(self) -> Dict[str, Any]:
        """
        Retorna informações sobre modelos carregados
        
        Returns:
            Dicionário com informações de modelos
        """
        models_info = []
        
        # Modelos em cache
        for brand in self.loaded_models:
            models_info.append({
                'brand': brand,
                'status': 'loaded',
                'in_memory': True
            })
        
        # Modelos no disco
        disk_models = FileUtils.list_files_by_extension(self.models_dir, '.pth')
        for model_file in disk_models:
            brand = model_file.stem.replace('_efficientnet_b3', '')
            if brand not in self.loaded_models:
                size_mb = FileUtils.get_file_size_mb(model_file)
                models_info.append({
                    'brand': brand,
                    'status': 'on_disk',
                    'in_memory': False,
                    'size_mb': round(size_mb, 2)
                })
        
        return {
            'total_in_memory': len(self.loaded_models),
            'total_on_disk': len(disk_models),
            'models': models_info
        }
