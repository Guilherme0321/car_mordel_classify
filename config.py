"""
Configurações da Car Classification API
======================================
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()

# Configurações do modelo
DRIVE_MODEL_URL = os.getenv('DRIVE_MODEL_URL', '')
MODEL_PATH = os.getenv("MODEL_PATH", "models/best_model_efficientnet_b3_acc_84.81.pth")
MODEL_TYPE = "efficientnet_b3"

# Configurações da API
HOST = os.getenv("API_HOST", os.getenv("HOST", "0.0.0.0"))
PORT = int(os.getenv("API_PORT", os.getenv("PORT", "8000")))
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# Configurações de processamento
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MAX_TOP_K = 10

# Configurações de logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Validações
def validate_config():
    """Valida as configurações"""
    errors = []
    
    # Verificar configuração do Google Drive
    if not DRIVE_MODEL_URL:
        errors.append("DRIVE_MODEL_URL não está configurada no arquivo .env")
    elif 'SEU_FILE_ID_AQUI' in DRIVE_MODEL_URL:
        errors.append("DRIVE_MODEL_URL precisa ser atualizada com o ID real do arquivo no Google Drive")
    
    # Verificar porta
    if not (1000 <= PORT <= 65535):
        errors.append(f"Porta inválida: {PORT}")
    
    return errors

if __name__ == "__main__":
    # Teste das configurações
    print("=== Configurações da API ===")
    print(f"DRIVE_MODEL_URL: {DRIVE_MODEL_URL}")
    print(f"MODEL_PATH: {MODEL_PATH}")
    print(f"API_HOST: {HOST}")
    print(f"API_PORT: {PORT}")
    print()
    
    errors = validate_config()
    if errors:
        print("❌ Erros de configuração:")
        for error in errors:
            print(f"   - {error}")
    else:
        print("✅ Configurações válidas!")
        print(f"   Modelo: {MODEL_PATH}")
        print(f"   Host: {HOST}:{PORT}")
        print(f"   Debug: {DEBUG}")