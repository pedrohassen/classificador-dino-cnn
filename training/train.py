"""Treino do classificador de dinossauros via transfer learning.

Carrega um MobileNetV2 pré-treinado (ImageNet), congela o backbone e faz
fine-tuning só da camada final, adaptada pro nosso número de classes. Ao
final, salva o modelo (`models_saved/model.pth`, gitignored) e os artefatos
de avaliação (`training/results/`, versionados) que comprovam que o treino
rodou de verdade: log de loss/acurácia por época, matriz de confusão e
classification report medidos no conjunto de teste.

MobileNetV2 em vez de ResNet18: bem menos FLOPs (~0.3G vs ~1.8G), o que importa
aqui porque o treino roda em CPU (ver CLAUDE.md/TAREFAS.md).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from torch import nn, optim
from torchvision import models
from sklearn.metrics import classification_report, confusion_matrix

from training.dataset import CLASSES, criar_dataloaders

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RAIZ_PROJETO = Path(__file__).resolve().parent.parent
CAMINHO_MODELO = RAIZ_PROJETO / "models_saved" / "model.pth"
PASTA_RESULTADOS = RAIZ_PROJETO / "training" / "results"


def montar_modelo(num_classes: int) -> nn.Module:
    """MobileNetV2 pré-treinado, backbone congelado, classificador final trocado."""
    modelo = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

    for parametro in modelo.parameters():
        parametro.requires_grad = False

    num_features = modelo.classifier[1].in_features
    modelo.classifier[1] = nn.Linear(num_features, num_classes)
    # só os parâmetros da camada nova entram no otimizador (fine-tuning restrito a ela)

    return modelo


def treinar_uma_epoca(modelo, loader_treino, criterio, otimizador, dispositivo) -> tuple[float, float]:
    modelo.train()
    perda_total = 0.0
    acertos = 0
    total = 0

    for imagens, rotulos in loader_treino:
        imagens, rotulos = imagens.to(dispositivo), rotulos.to(dispositivo)

        otimizador.zero_grad()
        saidas = modelo(imagens)
        perda = criterio(saidas, rotulos)
        perda.backward()
        otimizador.step()

        perda_total += perda.item() * imagens.size(0)
        acertos += (saidas.argmax(dim=1) == rotulos).sum().item()
        total += imagens.size(0)

    return perda_total / total, acertos / total


@torch.no_grad()
def avaliar(modelo, loader_teste, criterio, dispositivo) -> tuple[float, float, list[int], list[int]]:
    modelo.eval()
    perda_total = 0.0
    acertos = 0
    total = 0
    rotulos_previstos: list[int] = []
    rotulos_reais: list[int] = []

    for imagens, rotulos in loader_teste:
        imagens, rotulos = imagens.to(dispositivo), rotulos.to(dispositivo)

        saidas = modelo(imagens)
        perda = criterio(saidas, rotulos)

        previsoes = saidas.argmax(dim=1)
        perda_total += perda.item() * imagens.size(0)
        acertos += (previsoes == rotulos).sum().item()
        total += imagens.size(0)

        rotulos_previstos.extend(previsoes.cpu().tolist())
        rotulos_reais.extend(rotulos.cpu().tolist())

    return perda_total / total, acertos / total, rotulos_reais, rotulos_previstos


def salvar_matriz_confusao(rotulos_reais, rotulos_previstos, caminho_saida: Path) -> None:
    import matplotlib.pyplot as plt

    matriz = confusion_matrix(rotulos_reais, rotulos_previstos, labels=range(len(CLASSES)))

    fig, eixo = plt.subplots(figsize=(7, 6))
    imagem = eixo.imshow(matriz, cmap="Blues")
    eixo.set_xticks(range(len(CLASSES)))
    eixo.set_yticks(range(len(CLASSES)))
    eixo.set_xticklabels(CLASSES, rotation=45, ha="right")
    eixo.set_yticklabels(CLASSES)
    eixo.set_xlabel("Previsto")
    eixo.set_ylabel("Real")
    eixo.set_title("Matriz de confusão — conjunto de teste")

    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            eixo.text(j, i, str(matriz[i, j]), ha="center", va="center")

    fig.colorbar(imagem)
    fig.tight_layout()
    fig.savefig(caminho_saida)
    plt.close(fig)


def treinar(
    raiz_dataset: str = "dataset/dinossauros",
    epocas: int = 15,
    tamanho_lote: int = 16,
    taxa_aprendizado: float = 1e-3,
    prob_fundo_aleatorio: float = 0.3,
) -> None:
    dispositivo = torch.device("cpu")
    logger.info("Dispositivo: %s", dispositivo)

    loader_treino, loader_teste = criar_dataloaders(
        raiz_dataset, tamanho_lote=tamanho_lote, prob_fundo_aleatorio=prob_fundo_aleatorio
    )

    modelo = montar_modelo(num_classes=len(CLASSES)).to(dispositivo)
    criterio = nn.CrossEntropyLoss()
    otimizador = optim.Adam(modelo.classifier[1].parameters(), lr=taxa_aprendizado)

    historico = []
    for epoca in range(1, epocas + 1):
        perda_treino, acc_treino = treinar_uma_epoca(modelo, loader_treino, criterio, otimizador, dispositivo)
        perda_teste, acc_teste, _, _ = avaliar(modelo, loader_teste, criterio, dispositivo)

        logger.info(
            "Época %d/%d — treino: perda=%.4f acc=%.4f | teste: perda=%.4f acc=%.4f",
            epoca, epocas, perda_treino, acc_treino, perda_teste, acc_teste,
        )
        historico.append(
            {
                "epoca": epoca,
                "perda_treino": perda_treino,
                "acc_treino": acc_treino,
                "perda_teste": perda_teste,
                "acc_teste": acc_teste,
            }
        )

    # Avaliação final detalhada (matriz de confusão + classification report)
    _, acc_final, rotulos_reais, rotulos_previstos = avaliar(modelo, loader_teste, criterio, dispositivo)
    relatorio = classification_report(
        rotulos_reais, rotulos_previstos, target_names=CLASSES, digits=3, zero_division=0
    )
    logger.info("Acurácia final no teste: %.4f", acc_final)
    logger.info("Classification report:\n%s", relatorio)

    CAMINHO_MODELO.parent.mkdir(parents=True, exist_ok=True)
    torch.save(modelo.state_dict(), CAMINHO_MODELO)
    logger.info("Modelo salvo em %s", CAMINHO_MODELO)

    PASTA_RESULTADOS.mkdir(parents=True, exist_ok=True)
    with open(PASTA_RESULTADOS / "historico_treino.json", "w", encoding="utf-8") as arquivo:
        json.dump(historico, arquivo, ensure_ascii=False, indent=2)
    with open(PASTA_RESULTADOS / "classification_report.txt", "w", encoding="utf-8") as arquivo:
        arquivo.write(f"Acurácia final no teste: {acc_final:.4f}\n\n{relatorio}")
    salvar_matriz_confusao(rotulos_reais, rotulos_previstos, PASTA_RESULTADOS / "matriz_confusao.png")
    logger.info("Artefatos de avaliação salvos em %s", PASTA_RESULTADOS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treina o classificador de dinossauros")
    parser.add_argument("--raiz-dataset", default="dataset/dinossauros")
    parser.add_argument("--epocas", type=int, default=15)
    parser.add_argument("--tamanho-lote", type=int, default=16)
    parser.add_argument("--taxa-aprendizado", type=float, default=1e-3)
    parser.add_argument("--prob-fundo-aleatorio", type=float, default=0.3)
    argumentos = parser.parse_args()

    treinar(
        raiz_dataset=argumentos.raiz_dataset,
        epocas=argumentos.epocas,
        tamanho_lote=argumentos.tamanho_lote,
        taxa_aprendizado=argumentos.taxa_aprendizado,
        prob_fundo_aleatorio=argumentos.prob_fundo_aleatorio,
    )
