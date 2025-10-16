# 🚗 Car Classification API v2.0 - Arquitetura Modular

API inteligente para classificação de marca e modelo de carros usando **arquitetura two-stage modular**.

## 📁 Estrutura do Projeto

```
car_classification_api/
├── main.py                      # Ponto de entrada da aplicação
├── requirements.txt             # Dependências Python
├── .env                         # Variáveis de ambiente (não comitar!)
├── .env.example                 # Template de configuração
├── README.md                    # Documentação principal
│
├── app/                         # Pacote principal da aplicação
│   ├── __init__.py
│   │
│   ├── core/                    # 🔧 Módulos principais
│   │   ├── __init__.py
│   │   ├── config.py           # Configurações centralizadas (Settings)
│   │   └── logging.py          # Sistema de logging
│   │
│   ├── services/                # 🤖 Lógica de classificação
│   │   ├── __init__.py
│   │   ├── brand_classifier.py      # Classificador de marcas (Estágio 1)
│   │   └── car_model_classifier.py  # Classificador de modelos (Estágio 2)
│   │
│   ├── api/                     # 🌐 Endpoints REST
│   │   ├── __init__.py
│   │   ├── health.py           # GET /, /health, /info
│   │   ├── predict.py          # POST /predict, /predict/brand
│   │   └── models.py           # GET /models, DELETE /models/{brand}
│   │
│   ├── utils/                   # 🛠️ Utilitários
│   │   ├── __init__.py
│   │   ├── downloader.py       # Download do Google Drive
│   │   ├── file_utils.py       # Manipulação de arquivos
│   │   └── image_processor.py  # Processamento de imagens
│   │
│   └── schemas/                 # 📋 Modelos Pydantic
│       ├── __init__.py
│       └── responses.py        # Schemas de request/response
│
├── models/                      # 🧠 Modelos de ML
│   ├── mark_efficientnet_b3_acc_97.46.pth  # Modelo de marcas
│   └── car_models/                          # Modelos por marca (download sob demanda)
│       ├── BMW_efficientnet_b3.pth
│       ├── Audi_efficientnet_b3.pth
│       └── ...
│
├── logs/                        # 📝 Arquivos de log
│   └── api.log
│
└── tests/                       # 🧪 Testes
    └── test_api.py
```

## 🏗️ Arquitetura

### **Separação de Responsabilidades**

#### 1. **Core (`app/core/`)**
- `config.py`: Classe `Settings` com todas as configurações do `.env`
- `logging.py`: Setup centralizado de logging (console + arquivo)

#### 2. **Services (`app/services/`)**
- `brand_classifier.py`: 
  - Carrega modelo de marcas
  - Prediz as top N marcas de um carro
  - Sempre mantido em memória (modelo leve)
  
- `car_model_classifier.py`:
  - Gerencia modelos específicos por marca
  - Download sob demanda do Google Drive
  - Cache em memória e disco
  - Limpeza automática opcional

#### 3. **API (`app/api/`)**
- `health.py`: Endpoints de status e informações
- `predict.py`: Endpoints de predição (marca e modelo)
- `models.py`: Endpoints para gerenciar modelos carregados

#### 4. **Utils (`app/utils/`)**
- `downloader.py`: Classe `ModelDownloader` para Google Drive
- `file_utils.py`: Classe `FileUtils` para operações de arquivo
- `image_processor.py`: Classe `ImageProcessor` para validação e processamento

#### 5. **Schemas (`app/schemas/`)**
- `responses.py`: Modelos Pydantic para validação de dados

## 🚀 Como Executar

### 1. Instalar Dependências
```bash
pip install -r requirements.txt
```

### 2. Configurar `.env`
```bash
# Windows
Copy-Item .env.example .env

# Linux/Mac
cp .env.example .env
```

Edite o `.env` com suas configurações:
```env
# Modelo de marcas (obrigatório)
DRIVE_BRAND_MODEL_URL=https://drive.google.com/uc?id=SEU_ID_MARCA

# Mapeamento de modelos por marca
DRIVE_BRAND_MODELS_IDS=BMW:1ABC,Audi:1DEF,Mercedes-Benz:1GHI

# Diretórios
MODELS_DIR=models/car_models
BRAND_MODEL_PATH=models/mark_efficientnet_b3_acc_97.46.pth

# API
API_PORT=8000
API_HOST=0.0.0.0

# Limpeza automática
AUTO_CLEANUP_MODELS=false
```

### 3. Executar API
```bash
python main.py
```

Acesse:
- **API**: http://localhost:8000
- **Documentação**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 📡 Endpoints

### Saúde e Informações
```http
GET /                # Informações da API
GET /health          # Status dos classificadores
GET /info            # Informações detalhadas
```

### Predição
```http
POST /predict/brand  # Apenas marca (rápido)
POST /predict        # Completo: marca + modelo
```

### Gerenciamento de Modelos
```http
GET    /models           # Listar modelos carregados
DELETE /models/{brand}   # Remover modelo específico
DELETE /models           # Remover todos os modelos
```

## 🔧 Fluxo de Funcionamento

### **Inicialização da API** (`main.py`)
```python
1. Setup logging
2. Carregar BrandClassifier (modelo de marcas)
3. Inicializar CarModelClassifier (gerenciador de modelos)
4. Injetar classificadores nas rotas
5. Iniciar servidor FastAPI
```

### **Predição Completa** (`POST /predict`)
```python
1. Validar imagem (tipo, tamanho)
2. Carregar imagem com ImageProcessor
3. BrandClassifier.predict() → Top 3 marcas
4. Selecionar marca principal
5. CarModelClassifier.predict():
   a. Verificar se modelo está em cache
   b. Se não, baixar do Google Drive
   c. Carregar modelo em memória
   d. Fazer predição
6. (Opcional) Limpar modelo do disco
7. Retornar resultado
```

### **Download Sob Demanda**
```python
# Primeira predição de BMW
1. Usuário envia imagem
2. Marca prevista: BMW
3. Modelo BMW não existe localmente
4. Download automático do Google Drive
5. Modelo salvo em models/car_models/BMW_efficientnet_b3.pth
6. Predição realizada
7. Modelo mantido para próximas predições

# Próximas predições de BMW
1. Usuário envia imagem
2. Marca prevista: BMW
3. Modelo BMW já existe localmente
4. Carregamento direto (rápido)
5. Predição realizada
```

## 💡 Benefícios da Arquitetura Modular

### ✅ **Manutenibilidade**
- Cada módulo tem uma responsabilidade única
- Fácil localizar e corrigir bugs
- Código organizado e legível

### ✅ **Testabilidade**
- Cada componente pode ser testado isoladamente
- Mocks e injeção de dependência facilitados
- Testes unitários por módulo

### ✅ **Escalabilidade**
- Adicionar novos endpoints é simples (novo arquivo em `api/`)
- Adicionar novos classificadores (ex: cor, ano) é modular
- Fácil adicionar cache, filas, workers

### ✅ **Reutilização**
- Utilitários podem ser usados em qualquer parte
- Services podem ser importados em scripts externos
- Fácil criar CLIs ou outros frontends

### ✅ **Configuração Centralizada**
- Todas as configs em um único lugar (`core/config.py`)
- Fácil mudança entre ambientes (dev/prod)
- Validação de configurações no startup

## 🧪 Exemplo de Uso

### Python
```python
import requests

# Predição completa
with open("carro.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/predict",
        files={"file": f},
        params={"top_k": 5, "cleanup_after": False}
    )
    
result = response.json()
print(f"Marca: {result['brand_prediction']['selected_brand']}")
print(f"Modelo: {result['model_prediction']['predictions'][0]['model']}")
```

### cURL
```bash
# Predição de marca apenas
curl -X POST "http://localhost:8000/predict/brand?top_k=3" \
  -F "file=@carro.jpg"

# Predição completa com limpeza
curl -X POST "http://localhost:8000/predict?cleanup_after=true" \
  -F "file=@carro.jpg"

# Listar modelos carregados
curl http://localhost:8000/models

# Limpar modelo específico
curl -X DELETE http://localhost:8000/models/BMW
```

## 🔐 Boas Práticas Implementadas

### **1. Separação de Camadas**
- **Apresentação**: API endpoints (`app/api/`)
- **Lógica de Negócio**: Services (`app/services/`)
- **Utilitários**: Utils (`app/utils/`)
- **Configuração**: Core (`app/core/`)

### **2. Injeção de Dependência**
- Classificadores injetados nas rotas
- Facilita testes e mocks
- Desacoplamento entre módulos

### **3. Tratamento de Erros**
- Try-except em todos os pontos críticos
- Logging detalhado de erros
- Respostas HTTP apropriadas

### **4. Validação de Dados**
- Pydantic schemas para request/response
- Validação de imagens (tipo, tamanho)
- Validação de configurações no startup

### **5. Logging Estruturado**
- Logs em console e arquivo
- Níveis apropriados (INFO, WARNING, ERROR)
- Timestamps e contexto

### **6. Configuração por Ambiente**
- Todas as configs no `.env`
- Valores padrão sensatos
- Validação de configurações obrigatórias

## 📊 Comparação: Antes vs Depois

### **❌ Antes (main.py monolítico)**
```
main.py (1000+ linhas)
├── Imports
├── Classes misturadas
├── Endpoints inline
├── Configurações hardcoded
├── Difícil testar
└── Difícil manter
```

### **✅ Depois (Arquitetura modular)**
```
Estrutura clara e organizada
├── Responsabilidades separadas
├── Cada arquivo < 300 linhas
├── Fácil navegar
├── Fácil testar
├── Fácil escalar
└── Código profissional
```

## 🛠️ Desenvolvimento

### Adicionar Novo Endpoint
1. Crie função em `app/api/predict.py` (ou novo arquivo)
2. Use decoradores FastAPI
3. Importe e use os services necessários
4. Retorne com schemas Pydantic

### Adicionar Novo Classificador
1. Crie arquivo em `app/services/`
2. Implemente classe com métodos `__init__`, `predict`
3. Use utils para download e processamento
4. Injete no `main.py`

### Adicionar Testes
```python
# tests/test_brand_classifier.py
from app.services import BrandClassifier

def test_brand_prediction():
    classifier = BrandClassifier()
    # ... seu teste
```

## 📝 Próximos Passos

- [ ] Adicionar testes unitários
- [ ] Adicionar testes de integração
- [ ] Implementar cache Redis
- [ ] Adicionar rate limiting
- [ ] Dockerizar aplicação
- [ ] CI/CD pipeline
- [ ] Métricas e monitoring (Prometheus)
- [ ] Documentação adicional (Swagger UI personalizado)

## 🤝 Contribuindo

1. Fork o projeto
2. Crie uma branch (`git checkout -b feature/nova-funcionalidade`)
3. Commit suas mudanças (`git commit -m 'Adiciona nova funcionalidade'`)
4. Push para a branch (`git push origin feature/nova-funcionalidade`)
5. Abra um Pull Request

## 📄 Licença

MIT

---

**Desenvolvido por**: Guilherme0321  
**Versão**: 2.0.0  
**Repositório**: https://github.com/Guilherme0321/car-classification-api
