"""
Configurações da Car Classification API
======================================
"""

import os
from pathlib import Path

# Configurações do modelo
MODEL_PATH = os.getenv("MODEL_PATH", "models/best_model_efficientnet_b3_acc_84.81.pth")
MODEL_TYPE = "efficientnet_b3"

# Configurações da API
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))
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
    
    # Verificar se o modelo existe
    if not Path(MODEL_PATH).exists():
        errors.append(f"Modelo não encontrado: {MODEL_PATH}")
    
    # Verificar porta
    if not (1000 <= PORT <= 65535):
        errors.append(f"Porta inválida: {PORT}")
    
    return errors

if __name__ == "__main__":
    # Teste das configurações
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