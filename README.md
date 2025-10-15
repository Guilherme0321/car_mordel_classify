# 🚗 Car Classification API

API REST para classificação automática de marca e modelo de carros usando Deep Learning com FastAPI.

## 📋 Características

- **Modelo**: EfficientNet-B3 treinado com 84.81% de acurácia
- **Classes**: 742 combinações de marca_modelo (ex: BMW_X5, Audi_A4, Mercedes-Benz_C-Class)
- **Framework**: FastAPI com documentação automática
- **Performance**: Inferência rápida com suporte a GPU/CPU
- **Fácil deploy**: Container Docker incluído

## 🚀 Instalação

### 1. Clone o repositório
```bash
git clone https://github.com/Guilherme0321/car-classification-api.git
cd car-classification-api
```

### 2. Instale as dependências
```bash
pip install -r requirements.txt
```

### 3. Configure as variáveis de ambiente
Copie o arquivo `.env.example` para `.env`:
```bash
cp .env.example .env
```

Edite o arquivo `.env` e configure a URL do modelo no Google Drive:
```env
DRIVE_MODEL_URL=https://drive.google.com/uc?id=SEU_FILE_ID_AQUI
MODEL_PATH=models/best_model_efficientnet_b3_acc_84.81.pth
API_PORT=8000
API_HOST=0.0.0.0
```

**Como obter o ID do arquivo do Google Drive:**
1. Faça upload do modelo treinado para o Google Drive
2. Clique com o botão direito no arquivo → "Obter link"
3. Altere a permissão para "Qualquer pessoa com o link"
4. Copie o ID do arquivo da URL (a parte após `/d/` e antes de `/view`)
   - Exemplo: `https://drive.google.com/file/d/1ABC123xyz456/view`
   - O ID é: `1ABC123xyz456`
5. Cole no `.env`: `DRIVE_MODEL_URL=https://drive.google.com/uc?id=1ABC123xyz456`

### 4. Execute a API
```bash
python main.py
```

O modelo será baixado automaticamente do Google Drive na primeira execução.

A API estará disponível em: `http://localhost:8000`

## 📡 Endpoints

### 🏠 Root
- **GET** `/` - Informações básicas da API

### 🏥 Health Check  
- **GET** `/health` - Verifica se a API e modelo estão funcionando

### ℹ️ Model Info
- **GET** `/info` - Informações detalhadas do modelo

### 🎯 Predict (Principal)
- **POST** `/predict` - Classifica imagem de carro

## 🔧 Como usar

### Exemplo com curl
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@meu_carro.jpg" \
     -F "top_k=5"
```

### Exemplo com Python
```python
import requests

# Fazer predição
with open('carro.jpg', 'rb') as f:
    files = {'file': f}
    params = {'top_k': 3}
    
    response = requests.post(
        'http://localhost:8000/predict',
        files=files,
        params=params
    )
    
    result = response.json()
    
    if result['success']:
        for pred in result['predictions']:
            print(f"{pred['brand']} {pred['model']}: {pred['confidence_percent']}%")
```

### Exemplo de resposta
```json
{
  "success": true,
  "predictions": [
    {
      "rank": 1,
      "brand": "BMW",
      "model": "X5",
      "brand_model": "BMW_X5",
      "confidence": 0.8521,
      "confidence_percent": 85.21
    },
    {
      "rank": 2,
      "brand": "BMW", 
      "model": "X3",
      "brand_model": "BMW_X3",
      "confidence": 0.0892,
      "confidence_percent": 8.92
    }
  ],
  "model_info": {
    "accuracy": 84.81,
    "total_classes": 742
  },
  "file_info": {
    "filename": "carro.jpg",
    "content_type": "image/jpeg",
    "size_bytes": 245760
  }
}
```

## 🧪 Testes

Execute o script de teste para verificar se tudo está funcionando:

```bash
python tests/test_api.py
```

Para testar com uma imagem específica:
```bash
python tests/test_api.py --image caminho/para/imagem.jpg
```

## 📊 Estrutura do Projeto

```
car-classification-api/
├── main.py                 # API principal
├── requirements.txt        # Dependências
├── README.md              # Este arquivo
├── models/                # Modelos treinados
│   └── best_model_efficientnet_b3_acc_84.81.pth
├── tests/                 # Scripts de teste
│   └── test_api.py
└── docs/                  # Documentação adicional
    └── API_USAGE.md
```

## 🐳 Docker (Opcional)

### Construir imagem
```bash
docker build -t car-classification-api .
```

### Executar container
```bash
docker run -p 8000:8000 car-classification-api
```

## 🔍 Documentação da API

Com a API rodando, acesse:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## ⚙️ Configurações

### Variáveis de ambiente
- `MODEL_PATH`: Caminho para o modelo (padrão: `models/best_model_efficientnet_b3_acc_84.81.pth`)
- `HOST`: Host da API (padrão: `0.0.0.0`)
- `PORT`: Porta da API (padrão: `8000`)

### Requisitos de sistema
- **Python**: 3.8+
- **RAM**: 2GB mínimo (4GB recomendado)
- **GPU**: Opcional (CUDA compatível)
- **Espaço**: 500MB para modelo

## 📈 Performance

- **Inferência**: ~45ms por imagem (GPU) / ~150ms (CPU)
- **Throughput**: ~22 imagens/segundo (GPU)
- **Modelos suportados**: EfficientNet-B3, ResNet152, ConvNeXt

## 🛠️ Desenvolvimento

### Executar em modo desenvolvimento
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Executar testes
```bash
python -m pytest tests/ -v
```

## 🤝 Contribuição

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/nova-feature`)
3. Commit suas mudanças (`git commit -am 'Adiciona nova feature'`)
4. Push para a branch (`git push origin feature/nova-feature`)
5. Abra um Pull Request

## 📝 Changelog

### v1.0.0
- ✅ API básica com FastAPI
- ✅ Classificação marca + modelo
- ✅ Suporte a GPU/CPU
- ✅ Documentação Swagger
- ✅ Testes automatizados
- ✅ Containerização Docker

## 🐛 Issues Conhecidos

- Modelo pode ter dificuldade com imagens muito escuras ou desfocadas
- Requer pelo menos 1GB de RAM livre
- GPU memory pode ser limitada em GPUs antigas

## 📄 Licença

Este projeto é desenvolvido para fins acadêmicos e está disponível sob licença MIT.

## 👨‍💻 Autor

**Guilherme0321**
- GitHub: [@Guilherme0321](https://github.com/Guilherme0321)
- Projeto: Trabalho da Faculdade - Sexto Período - TI6

## 🙏 Agradecimentos

- Dataset utilizado para treinamento
- Comunidade PyTorch
- FastAPI team