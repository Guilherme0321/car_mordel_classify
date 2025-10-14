"""
Teste da Car Classification API
=============================

Script para testar a API de classificação de carros.

Uso:
    python tests/test_api.py
    python tests/test_api.py --image caminho/para/imagem.jpg
"""

import requests
import json
import argparse
from pathlib import Path
import sys

class APITester:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
    
    def test_health(self):
        """Testa endpoint de saúde"""
        print("🔍 Testando /health...")
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Status: {data.get('status')}")
                print(f"✅ Modelo carregado: {data.get('model_loaded')}")
                return True
            else:
                print(f"❌ Erro: {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Erro de conexão: {e}")
            return False
    
    def test_info(self):
        """Testa endpoint de informações"""
        print("\n🔍 Testando /info...")
        try:
            response = requests.get(f"{self.base_url}/info", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                info = data['model_info']
                print(f"✅ Modelo: {info['model_type']}")
                print(f"✅ Classes: {data['total_classes']}")
                print(f"✅ Acurácia: {info['best_val_acc']:.2f}%")
                return True
            else:
                print(f"❌ Erro: {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Erro de conexão: {e}")
            return False
    
    def test_prediction(self, image_path):
        """Testa predição com imagem"""
        if not Path(image_path).exists():
            print(f"❌ Imagem não encontrada: {image_path}")
            return False
        
        print(f"\n🔍 Testando /predict com: {image_path}")
        
        try:
            with open(image_path, 'rb') as f:
                files = {'file': (Path(image_path).name, f, 'image/jpeg')}
                params = {'top_k': 3}
                
                response = requests.post(
                    f"{self.base_url}/predict",
                    files=files,
                    params=params,
                    timeout=30
                )
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('success'):
                    print("✅ Predição realizada!")
                    print(f"📁 Arquivo: {data['file_info']['filename']}")
                    print("🎯 Top 3 predições:")
                    
                    for pred in data['predictions']:
                        print(f"   {pred['rank']}. {pred['brand']} {pred['model']} - {pred['confidence_percent']}%")
                    
                    return True
                else:
                    print(f"❌ Erro na predição: {data.get('error')}")
                    return False
            else:
                print(f"❌ Erro HTTP: {response.status_code}")
                try:
                    error = response.json()
                    print(f"   Detalhes: {error.get('detail')}")
                except:
                    pass
                return False
                
        except Exception as e:
            print(f"❌ Erro: {e}")
            return False
    
    def run_tests(self, image_path=None):
        """Executa todos os testes"""
        print("🧪 TESTE DA CAR CLASSIFICATION API")
        print("=" * 40)
        
        # Testes básicos
        tests = [
            ("Health Check", self.test_health),
            ("Model Info", self.test_info)
        ]
        
        results = {}
        
        for test_name, test_func in tests:
            results[test_name] = test_func()
        
        # Teste de predição se imagem fornecida
        if image_path:
            results["Prediction"] = self.test_prediction(image_path)
        else:
            # Procurar imagem de exemplo
            sample_image = self.find_sample_image()
            if sample_image:
                print(f"\n📸 Usando imagem de exemplo: {sample_image}")
                results["Prediction"] = self.test_prediction(sample_image)
            else:
                print("\n⚠️  Nenhuma imagem de teste fornecida")
                results["Prediction"] = None
        
        # Resumo
        print("\n" + "=" * 40)
        print("📊 RESUMO:")
        
        for test_name, result in results.items():
            if result is True:
                print(f"✅ {test_name}: PASSOU")
            elif result is False:
                print(f"❌ {test_name}: FALHOU")
            else:
                print(f"⏭️  {test_name}: PULADO")
        
        passed = sum(1 for r in results.values() if r is True)
        total = sum(1 for r in results.values() if r is not None)
        
        print(f"\n🎯 Total: {passed}/{total} testes passaram")
        
        return passed == total
    
    def find_sample_image(self):
        """Procura imagem de exemplo"""
        # Procurar no diretório pai
        parent_dir = Path("..").resolve()
        
        for pattern in ["*.jpg", "*.jpeg", "*.png"]:
            for path in parent_dir.rglob(pattern):
                if path.is_file() and "Dataset_Reorganizado" in str(path):
                    return str(path)
        
        return None

def main():
    parser = argparse.ArgumentParser(description="Testa a Car Classification API")
    parser.add_argument("--url", default="http://localhost:8000", help="URL da API")
    parser.add_argument("--image", help="Caminho para imagem de teste")
    
    args = parser.parse_args()
    
    tester = APITester(args.url)
    success = tester.run_tests(args.image)
    
    if not success:
        print("\n❌ Alguns testes falharam!")
        print("Certifique-se de que a API está rodando: python main.py")
        sys.exit(1)
    else:
        print("\n🎉 Todos os testes passaram!")

if __name__ == "__main__":
    main()