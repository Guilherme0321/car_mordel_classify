"""
Script para verificar e testar as configurações da API
"""
import sys
import os
from pathlib import Path

# Adicionar o diretório raiz ao path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

# Carregar .env
load_dotenv()

def check_env_file():
    """Verifica se o arquivo .env existe"""
    env_path = Path('.env')
    if not env_path.exists():
        print("❌ Arquivo .env não encontrado!")
        print("   Copie o arquivo .env.example para .env:")
        print("   cp .env.example .env")
        return False
    print("✅ Arquivo .env encontrado")
    return True

def check_drive_url():
    """Verifica a configuração da URL do Google Drive"""
    drive_url = os.getenv('DRIVE_MODEL_URL', '')
    
    if not drive_url:
        print("❌ DRIVE_MODEL_URL não configurada no .env")
        return False
    
    if 'SEU_FILE_ID_AQUI' in drive_url:
        print("⚠️  DRIVE_MODEL_URL precisa ser atualizada com o ID real do Google Drive")
        print("   Siga as instruções no README.md para obter o ID do arquivo")
        return False
    
    print(f"✅ DRIVE_MODEL_URL configurada")
    print(f"   URL: {drive_url}")
    return True

def check_model_path():
    """Verifica o caminho do modelo"""
    model_path = Path(os.getenv('MODEL_PATH', 'models/best_model_efficientnet_b3_acc_84.81.pth'))
    
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"✅ Modelo encontrado localmente")
        print(f"   Caminho: {model_path}")
        print(f"   Tamanho: {size_mb:.2f} MB")
        return True
    else:
        print("ℹ️  Modelo não encontrado localmente (será baixado do Google Drive)")
        print(f"   Será salvo em: {model_path}")
        return True

def check_api_config():
    """Verifica configurações da API"""
    host = os.getenv('API_HOST', '0.0.0.0')
    port = os.getenv('API_PORT', '8000')
    
    print(f"✅ Configurações da API")
    print(f"   Host: {host}")
    print(f"   Port: {port}")
    return True

def main():
    print("=" * 60)
    print("🔍 Verificação de Configuração - Car Classification API")
    print("=" * 60)
    print()
    
    checks = [
        ("Arquivo .env", check_env_file),
        ("URL do Google Drive", check_drive_url),
        ("Caminho do Modelo", check_model_path),
        ("Configurações da API", check_api_config),
    ]
    
    all_passed = True
    
    for name, check_func in checks:
        print(f"\n📋 Verificando: {name}")
        print("-" * 60)
        passed = check_func()
        if not passed:
            all_passed = False
        print()
    
    print("=" * 60)
    if all_passed:
        print("✅ Todas as verificações passaram!")
        print("   Você pode executar a API com: python main.py")
    else:
        print("❌ Algumas verificações falharam!")
        print("   Corrija os problemas acima antes de executar a API")
    print("=" * 60)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
