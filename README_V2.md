# 🚗 Car Classification API v2.0 - Two-Stage Model

API inteligente para classificação de marca e modelo de carros usando abordagem em **duas etapas**:

1. **Estágio 1**: Predição da marca (modelo leve ~50MB)
2. **Estágio 2**: Predição do modelo específico (baixado sob demanda por marca)

## 🎯 Principais Benefícios

✅ **Economia de Espaço**: Baixa apenas os modelos das marcas necessárias  
✅ **Performance**: Modelo de marcas sempre em memória (rápido)  
✅ **Flexibilidade**: Opção de limpar modelos após uso  
✅ **Escalabilidade**: Suporta centenas de marcas sem ocupar GB de espaço  
✅ **CORS Habilitado**: Pronto para integração frontend  

## 📦 Instalação

### 1. Instalar Dependências
```bash
pip install -r requirements.txt
```

### 2. Configurar .env

Copie o template:
```bash
Copy-Item .env.example .env
```

Configure as URLs do Google Drive no `.env`:

```env
# Modelo de marcas (obrigatório - sempre carregado)
DRIVE_BRAND_MODEL_URL=https://drive.google.com/uc?id=SEU_ID_MARCA

# Mapeamento dos modelos por marca (baixados sob demanda)
# Formato: MARCA:FILE_ID,MARCA:FILE_ID,...
DRIVE_BRAND_MODELS_IDS=BMW:1ABC123,Audi:1DEF456,Mercedes-Benz:1GHI789

# Diretórios
MODELS_DIR=models/car_models
BRAND_MODEL_PATH=models/mark_efficientnet_b3_acc_97.46.pth

# Limpeza automática (true/false)
AUTO_CLEANUP_MODELS=false

# API
API_PORT=8000
API_HOST=0.0.0.0
```

### 3. Estrutura do Google Drive

Organize seus modelos assim:

```
Google Drive:
├── mark_efficientnet_b3_acc_97.46.pth  (Modelo de marcas)
└── models/ (Pasta com modelos por marca)
    ├── BMW_efficientnet_b3.pth
    ├── Audi_efficientnet_b3.pth
    ├── Mercedes_Benz_efficientnet_b3.pth
    └── ...
```

**Para cada arquivo:**
1. Compartilhe publicamente
2. Copie o ID do arquivo
3. Adicione ao `.env`

### 4. Executar API

```bash
python main.py
```

Acesse: http://localhost:8000/docs

## 🔌 Endpoints da API

### 📊 GET `/` - Informações
Retorna informações da API e endpoints disponíveis.

### 🏥 GET `/health` - Status
Verifica se os classificadores estão carregados.

```json
{
  "status": "healthy",
  "brand_classifier": "loaded",
  "model_classifier": "loaded",
  "device": "cuda",
  "auto_cleanup": false
}
```

### 📋 GET `/models` - Modelos Carregados
Lista modelos em memória e no disco.

```json
{
  "total_in_memory": 2,
  "total_on_disk": 5,
  "models": [
    {
      "brand": "BMW",
      "status": "loaded",
      "in_memory": true
    },
    {
      "brand": "Audi",
      "status": "on_disk",
      "in_memory": false,
      "size_mb": 87.5
    }
  ]
}
```

### 🎯 POST `/predict/brand` - Predição de Marca
Apenas prediz a marca (rápido).

```bash
curl -X POST "http://localhost:8000/predict/brand?top_k=3" \
  -F "file=@carro.jpg"
```

**Resposta:**
```json
{
  "success": true,
  "stage": "brand_only",
  "predictions": [
    {
      "rank": 1,
      "brand": "BMW",
      "confidence": 0.9234,
      "confidence_percent": 92.34
    },
    {
      "rank": 2,
      "brand": "Audi",
      "confidence": 0.0456,
      "confidence_percent": 4.56
    }
  ]
}
```

### 🚀 POST `/predict` - Classificação Completa
Prediz marca + modelo (em duas etapas).

**Parâmetros:**
- `file`: Imagem (obrigatório)
- `top_k`: Número de modelos (padrão: 5)
- `cleanup_after`: Remover modelo após uso (padrão: false)
- `brand_hint`: Marca específica para pular predição (opcional)

```bash
# Classificação completa
curl -X POST "http://localhost:8000/predict?top_k=5" \
  -F "file=@bmw.jpg"

# Com limpeza após uso
curl -X POST "http://localhost:8000/predict?top_k=5&cleanup_after=true" \
  -F "file=@bmw.jpg"

# Se você já sabe a marca
curl -X POST "http://localhost:8000/predict?brand_hint=BMW" \
  -F "file=@bmw.jpg"
```

**Resposta:**
```json
{
  "success": true,
  "stage": "complete",
  "brand_prediction": {
    "top_brands": [
      {
        "rank": 1,
        "brand": "BMW",
        "confidence_percent": 95.67
      }
    ],
    "selected_brand": "BMW"
  },
  "model_prediction": {
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
        "confidence": 0.0923,
        "confidence_percent": 9.23
      }
    ],
    "model_info": {
      "brand": "BMW",
      "accuracy": 89.45,
      "total_classes": 87
    }
  },
  "file_info": {
    "filename": "bmw.jpg",
    "content_type": "image/jpeg",
    "size_bytes": 245678
  },
  "cleanup_performed": false
}
```

### 🗑️ DELETE `/models/{brand}` - Limpar Modelo
Remove modelo de uma marca do disco e cache.

```bash
curl -X DELETE "http://localhost:8000/models/BMW"
```

**Resposta:**
```json
{
  "success": true,
  "message": "Modelo da marca 'BMW' removido",
  "brand": "BMW"
}
```

### 🗑️ DELETE `/models` - Limpar Todos
Remove todos os modelos de marcas.

```bash
curl -X DELETE "http://localhost:8000/models"
```

**Resposta:**
```json
{
  "success": true,
  "message": "Todos os modelos foram removidos",
  "removed_from_cache": 3,
  "removed_from_disk": 5
}
```

## 💻 Exemplo com Python

```python
import requests
from pathlib import Path

API_URL = "http://localhost:8000"

def predict_brand_only(image_path):
    """Prediz apenas a marca"""
    with open(image_path, 'rb') as f:
        response = requests.post(
            f"{API_URL}/predict/brand",
            files={'file': f},
            params={'top_k': 3}
        )
    return response.json()

def predict_complete(image_path, cleanup_after=False):
    """Classificação completa com opção de limpeza"""
    with open(image_path, 'rb') as f:
        response = requests.post(
            f"{API_URL}/predict",
            files={'file': f},
            params={
                'top_k': 5,
                'cleanup_after': cleanup_after
            }
        )
    return response.json()

def get_loaded_models():
    """Lista modelos carregados"""
    response = requests.get(f"{API_URL}/models")
    return response.json()

def cleanup_brand(brand):
    """Remove modelo de uma marca"""
    response = requests.delete(f"{API_URL}/models/{brand}")
    return response.json()

# Uso
if __name__ == "__main__":
    # 1. Predição de marca apenas
    print("Predizendo marca...")
    brand_result = predict_brand_only("carro.jpg")
    print(f"Marca: {brand_result['predictions'][0]['brand']}")
    
    # 2. Classificação completa
    print("\nClassificação completa...")
    result = predict_complete("carro.jpg", cleanup_after=False)
    
    if result['success']:
        brand = result['brand_prediction']['selected_brand']
        model = result['model_prediction']['predictions'][0]['model']
        confidence = result['model_prediction']['predictions'][0]['confidence_percent']
        
        print(f"Marca: {brand}")
        print(f"Modelo: {model}")
        print(f"Confiança: {confidence}%")
    
    # 3. Ver modelos carregados
    print("\nModelos carregados:")
    models_info = get_loaded_models()
    print(f"Em memória: {models_info['total_in_memory']}")
    print(f"No disco: {models_info['total_on_disk']}")
    
    # 4. Limpar modelo
    print("\nLimpando modelo BMW...")
    cleanup_result = cleanup_brand("BMW")
    print(cleanup_result['message'])
```

## 🔧 Configuração Avançada

### Limpeza Automática

Para economizar espaço automaticamente:

```env
AUTO_CLEANUP_MODELS=true
```

Com isso ativado, os modelos são removidos após cada predição.

### Obter IDs do Google Drive

**Script PowerShell para ajudar:**

```powershell
# Listar arquivos compartilhados
function Get-DriveFileId {
    param($url)
    if ($url -match '/d/([^/]+)') {
        return $matches[1]
    }
}

# Exemplo
$url = "https://drive.google.com/file/d/1ABC123xyz/view"
$id = Get-DriveFileId $url
Write-Host "ID: $id"
Write-Host "URL para .env: https://drive.google.com/uc?id=$id"
```

### Mapeamento de Modelos

Formato no `.env`:

```env
DRIVE_BRAND_MODELS_IDS=BMW:1ABC,Audi:1DEF,Mercedes-Benz:1GHI,Toyota:1JKL
```

**Dica**: Use um script para gerar automaticamente:

```python
# generate_ids.py
drive_files = {
    "BMW": "1ABC123xyz",
    "Audi": "1DEF456abc",
    "Mercedes-Benz": "1GHI789def",
    "Toyota": "1JKL012ghi"
}

ids_string = ",".join([f"{k}:{v}" for k, v in drive_files.items()])
print(f"DRIVE_BRAND_MODELS_IDS={ids_string}")
```

## 📊 Fluxo de Trabalho

```
1. Usuário envia imagem
   ↓
2. API prediz marca (modelo leve já carregado)
   ↓
3. API verifica se modelo da marca está disponível
   ├─ Sim: Usa modelo em cache
   └─ Não: Baixa do Google Drive
   ↓
4. API prediz modelo específico
   ↓
5. Retorna resultado
   ↓
6. (Opcional) Remove modelo do disco
```

## 🎯 Casos de Uso

### 1. Ambiente de Produção (economizar espaço)
```env
AUTO_CLEANUP_MODELS=true
```
Cada predição limpa o modelo após uso.

### 2. Desenvolvimento (manter cache)
```env
AUTO_CLEANUP_MODELS=false
```
Modelos permanecem para uso posterior.

### 3. API Pública (controle manual)
```bash
# Limpar periodicamente via cron/agendador
curl -X DELETE "http://localhost:8000/models"
```

## 🐛 Troubleshooting

### Modelo não baixa
- Verifique se o arquivo está compartilhado publicamente
- Confirme o ID no `.env`
- Teste manualmente: `gdown https://drive.google.com/uc?id=SEU_ID`

### Erro "Brand model not found"
- Verifique o `DRIVE_BRAND_MODELS_IDS`
- Formato correto: `MARCA:ID,MARCA:ID`
- Nome da marca deve ser exato

### Memória insuficiente
- Ative `AUTO_CLEANUP_MODELS=true`
- Ou use endpoint DELETE após predições

### Performance lenta
- Primeira predição de cada marca é mais lenta (download)
- Predições subsequentes são rápidas (cache)
- Considere pré-carregar marcas comuns na inicialização

## 📈 Performance

- **Predição de marca**: ~100-200ms
- **Primeira predição de modelo**: ~5-30s (download + inferência)
- **Predições subsequentes**: ~200-500ms (apenas inferência)
- **Tamanho modelo marcas**: ~50MB
- **Tamanho modelo por marca**: ~80-100MB

## 🔐 Segurança

✅ CORS configurado (ajuste `allow_origins` para produção)  
✅ Validação de tipos de arquivo  
✅ Limite de tamanho de imagem  
✅ Tratamento de erros robusto  
✅ `.env` não commitado (use `.gitignore`)  

## 📝 Changelog

### v2.0.0
- ✨ Nova arquitetura two-stage
- ✨ Download sob demanda de modelos
- ✨ Sistema de cache inteligente
- ✨ Endpoints de limpeza
- ✨ Suporte a `brand_hint`
- ✨ Limpeza automática opcional

### v1.0.0
- 🎉 Versão inicial com modelo único

---

**Desenvolvido por**: Guilherme0321  
**Repositório**: https://github.com/Guilherme0321/car-classification-api  
**Licença**: MIT
