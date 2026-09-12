# Classificador Dino

Classificador de imagens de dinossauros (6 espécies) com backend próprio em
Python/FastAPI, servindo um modelo de transfer learning (MobileNetV2) treinado
em PyTorch.

**Demo ao vivo**: https://classificador-dino-cnn.onrender.com
(instância gratuita — se estiver inativa, a primeira requisição pode levar
uns 30-60s pra "acordar")

---

## Origem

Este projeto começou como uma atividade de curso (CNN + PyTorch), com uma
API Node.js/Express que só chamava um worker Python via subprocess pra rodar
a inferência — Python não era o backend, era só um script chamado por fora.
O modelo treinado na época também era fraco (CNN rasa treinada do zero, com
pouco dado).

A versão atual é uma reescrita deliberada: backend Python de verdade
(FastAPI, sem Node.js), modelo via transfer learning em vez de CNN do zero, e
deploy real. O guia original da atividade fica preservado em
[`README_atividade.md`](README_atividade.md) como registro histórico.

---

## Stack

- **Backend**: Python + FastAPI
- **Modelo**: MobileNetV2 pré-treinado (ImageNet), backbone congelado,
  fine-tuning só da camada final — escolhido em vez de treinar do zero
  porque o dataset é pequeno (~60 imagens/classe)
- **Treino**: `torchvision` + augmentation (rotação, flip, cor, recorte) e
  randomização de fundo via segmentação (`rembg`/U2Net), pra reduzir o risco
  do modelo aprender "tipo de fundo" em vez de anatomia
- **Frontend**: HTML/CSS/JS puro (sem framework), servido pelo próprio FastAPI
- **Deploy**: Render (Web Service via Docker, free tier)

## Resultado do treino

Medido no conjunto de teste (30 imagens, 5 por classe, nunca vistas no
treino): **100% de acurácia**, sem confusão entre nenhuma das 6 classes.

Artefatos completos (histórico de loss/acurácia por época, matriz de
confusão, classification report) ficam versionados em
[`training/results/`](training/results/) — evidência de que o treino
realmente rodou, não só o código dele.

**Limitação honesta**: o dataset é uma mistura de ilustrações, renders 3D e
arte gerada por IA — não fotos reais. O modelo tende a funcionar bem em
imagens desse mesmo estilo, mas isso não foi validado em fotos reais
(brinquedos, museus, fósseis), que podem confundir o modelo mais do que arte
digital. O conjunto de teste usado para medir a acurácia acima compartilha
esse mesmo viés de estilo do treino.

## Classes reconhecidas

`ankylosaurus`, `brachiosaurus`, `dimorphodon`, `gallimimus`, `triceratops`,
`tyrannosaurus`

---

## Rodando localmente

### Pré-requisitos

- Python 3.11+
- Dataset próprio estruturado em `dataset/dinossauros/{train,test}/<classe>/`
  (não incluído no repo — ver seção Dataset)

### Ambiente

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1   # Windows PowerShell
pip install -r requirements.txt
```

### Treinar o modelo

```bash
python -m training.train --epocas 15
```

Salva o modelo em `models_saved/model.pth` e os artefatos de avaliação em
`training/results/`.

### Rodar a API

```bash
uvicorn app.main:app --reload
```

Abre em `http://localhost:8000` — a própria API serve o frontend na raiz.

### Testar o endpoint diretamente

```bash
curl -X POST http://localhost:8000/predict -F "image=@caminho/para/imagem.jpg"
```

Resposta:

```json
{
  "predicted_class": "triceratops",
  "confidence": 0.93,
  "top_predictions": [
    { "class": "triceratops", "confidence": 0.93 },
    { "class": "ankylosaurus", "confidence": 0.07 }
  ]
}
```

## Dataset

Não incluído no repositório (é grande e mistura imagens copiadas do Google
Fotos/Imagens com imagens geradas por IA, sem separação por origem ainda).
Estrutura esperada pra rodar o treino:

```
dataset/dinossauros/
├── train/<classe>/*.jpg
└── test/<classe>/*.jpg
```
