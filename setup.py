"""
Script para configurar a API - Copia modelo e verifica dependências
=================================================================

Este script prepara o ambiente da API copiando o modelo treinado
e verificando se tudo está configurado corretamente.
"""

import shutil
from pathlib import Path
import sys

def setup_api():
    """Configura a API copiando arquivos necessários"""
    print("🔧 Configurando Car Classification API...")
    
    # Caminhos
    current_dir = Path(__file__).parent
    parent_dir = current_dir.parent
    model_source = parent_dir / "weight" / "best_model_efficientnet_b3_acc_84.81.pth"
    model_dest = current_dir / "models" / "best_model_efficientnet_b3_acc_84.81.pth"
    
    # Verificar se modelo fonte existe
    if not model_source.exists():
        print(f"❌ Modelo não encontrado: {model_source}")
        print("   Certifique-se de que o modelo foi treinado e está na pasta weight/")
        return False
    
    # Criar pasta models se não existir
    model_dest.parent.mkdir(exist_ok=True)
    
    # Copiar modelo
    if not model_dest.exists():
        print(f"📄 Copiando modelo para API...")
        print(f"   De: {model_source}")
        print(f"   Para: {model_dest}")
        
        try:
            shutil.copy2(model_source, model_dest)
            print("✅ Modelo copiado com sucesso!")
        except Exception as e:
            print(f"❌ Erro ao copiar modelo: {e}")
            return False
    else:
        print("✅ Modelo já existe na API!")
    
    # Verificar tamanho do arquivo
    size_mb = model_dest.stat().st_size / (1024 * 1024)
    print(f"📊 Tamanho do modelo: {size_mb:.1f} MB")
    
    # Verificar dependências
    print("\n🔍 Verificando dependências...")
    try:
        import fastapi
        import uvicorn
        import torch
        import torchvision
        import PIL
        print("✅ Todas as dependências estão instaladas!")
    except ImportError as e:
        print(f"❌ Dependência faltando: {e}")
        print("   Execute: pip install -r requirements.txt")
        return False
    
    # Verificar GPU
    import torch
    if torch.cuda.is_available():
        print(f"🚀 GPU disponível: {torch.cuda.get_device_name()}")
    else:
        print("💻 Usando CPU (GPU não disponível)")
    
    print("\n✅ Setup concluído! A API está pronta para uso.")
    print("\nPara executar:")
    print("   python main.py")
    print("\nPara testar:")
    print("   python tests/test_api.py")
    
    return True

def clean_api():
    """Remove arquivos gerados"""
    print("🧹 Limpando arquivos da API...")
    
    current_dir = Path(__file__).parent
    model_file = current_dir / "models" / "best_model_efficientnet_b3_acc_84.81.pth"
    
    if model_file.exists():
        model_file.unlink()
        print("✅ Modelo removido!")
    
    # Remover __pycache__
    for pycache in current_dir.rglob("__pycache__"):
        shutil.rmtree(pycache)
        print(f"✅ Removido: {pycache}")
    
    print("✅ Limpeza concluída!")

def main():
    """Função principal"""
    if len(sys.argv) > 1 and sys.argv[1] == "clean":
        clean_api()
    else:
        success = setup_api()
        if not success:
            sys.exit(1)

if __name__ == "__main__":
    main()