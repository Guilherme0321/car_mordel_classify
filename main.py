"""
Car Classification API
====================

API simples para classificação de marca e modelo de carros usando FastAPI.
Desenvolvido para classificar imagens em 742 combinações de marca_modelo.

Author: Guilherme0321
Repository: https://github.com/Guilherme0321/car-classification-api
"""

import torch
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import io
import logging
from pathlib import Path
import uvicorn
from typing import List, Dict, Any
import os
from dotenv import load_dotenv
import gdown
import requests

# Carregar variáveis de ambiente
load_dotenv()

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CarClassificationModel:
    """
    Modelo de classificação de carros usando EfficientNet-B3
    """
    
    def __init__(self, model_path: str):
        """
        Inicializa o modelo de classificação
        
        Args:
            model_path (str): Caminho para o arquivo .pth do modelo
        """
        self.model_path = Path(model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Fazer download do modelo se não existir localmente
        if not self.model_path.exists():
            logger.info(f"Modelo não encontrado localmente. Tentando baixar do Google Drive...")
            self._download_model_from_drive()
        
        # Carregar modelo
        self._load_model()
        
        # Configurar transformações
        self._setup_transforms()
        
        logger.info(f"Modelo carregado com sucesso! Dispositivo: {self.device}")
    
    def _download_model_from_drive(self):
        """Baixa o modelo do Google Drive usando a URL do .env"""
        try:
            drive_url = os.getenv('DRIVE_MODEL_URL')
            
            if not drive_url or 'SEU_FILE_ID_AQUI' in drive_url:
                raise ValueError(
                    "DRIVE_MODEL_URL não configurada corretamente no arquivo .env. "
                    "Configure a URL do Google Drive com o ID do arquivo."
                )
            
            # Criar diretório se não existir
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Baixando modelo do Google Drive...")
            logger.info(f"URL: {drive_url}")
            logger.info(f"Destino: {self.model_path}")
            
            # Baixar arquivo usando gdown
            gdown.download(drive_url, str(self.model_path), quiet=False, fuzzy=True)
            
            if not self.model_path.exists():
                raise FileNotFoundError(
                    f"Falha ao baixar o modelo. Verifique se:\n"
                    f"1. O arquivo está compartilhado publicamente no Google Drive\n"
                    f"2. O ID do arquivo está correto no .env\n"
                    f"3. Você tem conexão com a internet"
                )
            
            logger.info(f"Modelo baixado com sucesso!")
            
        except Exception as e:
            logger.error(f"Erro ao baixar modelo do Google Drive: {e}")
            raise
    
    def _load_model(self):
        """Carrega o modelo e suas informações"""
        try:
            logger.info(f"Carregando modelo: {self.model_path}")
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Extrair informações do modelo
            self.model_info = {
                'model_type': checkpoint.get('model_type', 'efficientnet_b3'),
                'num_classes': checkpoint.get('num_classes', 742),
                'best_val_acc': checkpoint.get('best_val_acc', 0),
                'epoch': checkpoint.get('epoch', 0)
            }
            
            self.class_to_idx = checkpoint.get('class_to_idx', {})
            self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
            
            # Criar arquitetura EfficientNet-B3
            self.model = models.efficientnet_b3(weights=None)
            num_features = self.model.classifier[1].in_features
            
            # Modificar classifier para número correto de classes
            self.model.classifier = torch.nn.Sequential(
                torch.nn.Dropout(p=0.3),
                torch.nn.Linear(num_features, 512),
                torch.nn.ReLU(inplace=True),
                torch.nn.BatchNorm1d(512),
                torch.nn.Dropout(p=0.5),
                torch.nn.Linear(512, self.model_info['num_classes'])
            )
            
            # Carregar pesos treinados
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model = self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Modelo carregado - {self.model_info['num_classes']} classes - Acurácia: {self.model_info['best_val_acc']:.2f}%")
            
        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {e}")
            raise
    
    def _setup_transforms(self):
        """Configura transformações para preprocessamento"""
        self.transforms = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.CenterCrop(336),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def predict(self, image: Image.Image, top_k: int = 5) -> Dict[str, Any]:
        """
        Realiza predição em uma imagem
        
        Args:
            image (PIL.Image): Imagem para classificar
            top_k (int): Número de top predições para retornar
        
        Returns:
            dict: Dicionário com predições e metadados
        """
        try:
            # Preprocessar imagem
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            input_tensor = self.transforms(image).unsqueeze(0).to(self.device)
            
            # Fazer predição
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = F.softmax(outputs, dim=1)
            
            # Obter top-k predições
            top_probs, top_indices = torch.topk(probabilities, min(top_k, len(self.idx_to_class)))
            
            predictions = []
            for i in range(len(top_indices[0])):
                class_idx = top_indices[0][i].item()
                prob = top_probs[0][i].item()
                class_name = self.idx_to_class.get(class_idx, f"Unknown_{class_idx}")
                
                # Separar marca e modelo
                if '_' in class_name:
                    brand, model = class_name.split('_', 1)
                else:
                    brand, model = class_name, "Unknown"
                
                predictions.append({
                    'rank': i + 1,
                    'brand': brand,
                    'model': model,
                    'brand_model': class_name,
                    'confidence': float(prob),
                    'confidence_percent': round(float(prob * 100), 2)
                })
            
            return {
                'success': True,
                'predictions': predictions,
                'model_info': {
                    'accuracy': self.model_info['best_val_acc'],
                    'total_classes': self.model_info['num_classes']
                }
            }
            
        except Exception as e:
            logger.error(f"Erro na predição: {e}")
            return {
                'success': False,
                'error': str(e)
            }

# Configuração do modelo a partir do .env
MODEL_PATH = os.getenv('MODEL_PATH', 'models/best_model_efficientnet_b3_acc_84.81.pth')

# Inicializar modelo globalmente
car_model = None

def load_model():
    """Carrega o modelo na inicialização da aplicação"""
    global car_model
    try:
        car_model = CarClassificationModel(MODEL_PATH)
        return True
    except Exception as e:
        logger.error(f"Falha ao carregar modelo: {e}")
        return False

# Criar aplicação FastAPI
app = FastAPI(
    title="Car Classification API",
    description="API para classificação de marca e modelo de carros usando Deep Learning",
    version="1.0.0",
    contact={
        "name": "Guilherme0321",
        "url": "https://github.com/Guilherme0321",
    }
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite todas as origens. Para produção, especifique as origens permitidas
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos os métodos (GET, POST, etc.)
    allow_headers=["*"],  # Permite todos os headers
)

@app.on_event("startup")
async def startup_event():
    """Evento executado na inicialização da API"""
    logger.info("Inicializando Car Classification API...")
    success = load_model()
    if not success:
        logger.error("Falha ao carregar modelo na inicialização!")

@app.get("/")
async def root():
    """Endpoint raiz com informações da API"""
    return {
        "message": "Car Classification API",
        "version": "1.0.0",
        "description": "API para classificação de marca e modelo de carros",
        "endpoints": {
            "predict": "POST /predict - Classificar imagem de carro",
            "health": "GET /health - Verificar saúde da API",
            "info": "GET /info - Informações do modelo"
        }
    }

# @app.get("/health")
# async def health_check():
#     """Endpoint para verificar a saúde da API"""
#     if car_model is None:
#         return JSONResponse(
#             status_code=503,
#             content={
#                 "status": "unhealthy", 
#                 "message": "Modelo não carregado",
#                 "model_loaded": False
#             }
#         )
    
#     return {
#         "status": "healthy",
#         "model_loaded": True,
#         "device": str(car_model.device),
#         "model_path": str(car_model.model_path)
#     }

# @app.get("/info")
# async def model_info():
#     """Retorna informações sobre o modelo carregado"""
#     if car_model is None:
#         raise HTTPException(status_code=503, detail="Modelo não carregado")
    
#     return {
#         "model_info": car_model.model_info,
#         "total_classes": len(car_model.idx_to_class),
#         "device": str(car_model.device),
#         "sample_classes": list(car_model.class_to_idx.keys())[:10]
#     }

@app.post("/predict")
async def predict_car(
    file: UploadFile = File(...),
    top_k: int = 5
):
    """
    Classifica uma imagem de carro e retorna marca e modelo
    
    Args:
        file: Arquivo de imagem (JPG, PNG, JPEG)
        top_k: Número de predições top para retornar (padrão: 5, máximo: 10)
    
    Returns:
        JSON com predições de marca e modelo
    """
    if car_model is None:
        raise HTTPException(status_code=503, detail="Modelo não carregado")
    
    # Validar tipo de arquivo
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Arquivo deve ser uma imagem (JPG, PNG, JPEG)")
    
    # Validar top_k
    if top_k < 1 or top_k > 10:
        raise HTTPException(status_code=400, detail="top_k deve estar entre 1 e 10")
    
    try:
        # Ler e processar imagem
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Fazer predição
        result = car_model.predict(image, top_k)
        
        # Adicionar metadados
        result['file_info'] = {
            'filename': file.filename,
            'content_type': file.content_type,
            'size_bytes': len(image_data)
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Erro ao processar imagem: {e}")
        raise HTTPException(status_code=500, detail=f"Erro ao processar imagem: {str(e)}")

if __name__ == "__main__":
    # Obter configurações do .env
    host = os.getenv('API_HOST', '0.0.0.0')
    port = int(os.getenv('API_PORT', '8000'))
    
    uvicorn.run(app, host=host, port=port)