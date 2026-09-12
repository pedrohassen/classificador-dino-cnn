"""Carregamento do modelo treinado e função de inferência.

Independente do `training/` de propósito — a API só precisa saber montar a
mesma arquitetura e aplicar a mesma transformação de avaliação usadas no
treino, sem depender das bibliotecas de treino (rembg, scikit-learn etc.).
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

logger = logging.getLogger(__name__)

CLASSES = [
    "ankylosaurus",
    "brachiosaurus",
    "dimorphodon",
    "gallimimus",
    "triceratops",
    "tyrannosaurus",
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
TAMANHO_IMAGEM = 224

CAMINHO_MODELO_PADRAO = Path(__file__).resolve().parent.parent / "models_saved" / "model.pth"

_transformacao_inferencia = transforms.Compose(
    [
        transforms.Resize((TAMANHO_IMAGEM, TAMANHO_IMAGEM)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)


def _montar_arquitetura(num_classes: int) -> nn.Module:
    modelo = models.mobilenet_v2(weights=None)
    num_features = modelo.classifier[1].in_features
    modelo.classifier[1] = nn.Linear(num_features, num_classes)
    return modelo


class ClassificadorDinos:
    """Carrega o modelo uma única vez e expõe a função de previsão."""

    def __init__(self, caminho_modelo: Path = CAMINHO_MODELO_PADRAO):
        if not caminho_modelo.exists():
            raise FileNotFoundError(
                f"Modelo não encontrado em {caminho_modelo}. Rode o treino "
                "(training/train.py) antes de subir a API."
            )

        self.dispositivo = torch.device("cpu")
        self.modelo = _montar_arquitetura(num_classes=len(CLASSES))
        estado = torch.load(caminho_modelo, map_location=self.dispositivo)
        self.modelo.load_state_dict(estado)
        self.modelo.to(self.dispositivo)
        self.modelo.eval()
        logger.info("Modelo carregado de %s", caminho_modelo)

    @torch.no_grad()
    def prever(self, imagem: Image.Image, top_k: int = 3) -> list[tuple[str, float]]:
        """Retorna as `top_k` classes mais prováveis, ordenadas por confiança (0-1)."""
        tensor = _transformacao_inferencia(imagem.convert("RGB")).unsqueeze(0).to(self.dispositivo)
        saida = self.modelo(tensor)
        probabilidades = torch.softmax(saida, dim=1).squeeze(0)

        top_k = min(top_k, len(CLASSES))
        valores, indices = torch.topk(probabilidades, top_k)

        return [(CLASSES[indice], valor.item()) for valor, indice in zip(valores, indices)]
