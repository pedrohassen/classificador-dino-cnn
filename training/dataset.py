"""Dataset e DataLoaders para o classificador de dinossauros.

Além da augmentation padrão (rotação, flip, cor, recorte), aplica com uma
certa probabilidade uma randomização de fundo: remove o fundo original da
imagem (via `rembg`/U2Net) e cola o dinossauro recortado sobre um fundo de cor
aleatória. Isso mitiga o risco do modelo aprender "tipo de fundo" como atalho
pra classificar, em vez da anatomia (ver CLAUDE.md, seção Dataset).
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

logger = logging.getLogger(__name__)

CLASSES = [
    "ankylosaurus",
    "brachiosaurus",
    "dimorphodon",
    "gallimimus",
    "triceratops",
    "tyrannosaurus",
]

EXTENSOES_VALIDAS = {".jpg", ".jpeg", ".png", ".webp"}

# Estatísticas do ImageNet — backbone pré-treinado espera entrada normalizada assim.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
TAMANHO_IMAGEM = 224


def _gerar_fundo_aleatorio(tamanho: tuple[int, int]) -> Image.Image:
    cor = tuple(random.randint(0, 255) for _ in range(3))
    return Image.new("RGB", tamanho, cor)


_sessao_rembg = None


def _obter_sessao_rembg():
    """Sessão do rembg usando o modelo `u2net` (~176MB) explicitamente — o
    padrão da lib (`bria-rmbg`) pesa ~1GB, desproporcional pro que precisamos
    aqui (silhueta de dinossauro em fundo simples, não segmentação de precisão)."""
    global _sessao_rembg
    if _sessao_rembg is None:
        from rembg import new_session

        _sessao_rembg = new_session("u2net")
    return _sessao_rembg


def _obter_recorte_sem_fundo(caminho_imagem: Path, cache_dir: Path) -> Image.Image:
    """Retorna o recorte RGBA (dino sem fundo) da imagem, cacheado em disco pra
    não rodar a segmentação de novo a cada época de treino."""
    caminho_cache = cache_dir / f"{caminho_imagem.stem}.png"
    if caminho_cache.exists():
        return Image.open(caminho_cache).convert("RGBA")

    from rembg import remove  # import tardio: só carrega o modelo de segmentação se for usado

    cache_dir.mkdir(parents=True, exist_ok=True)
    imagem_original = Image.open(caminho_imagem).convert("RGB")
    recorte = remove(imagem_original, session=_obter_sessao_rembg())
    recorte.save(caminho_cache)
    return recorte


class DatasetDinos(Dataset):
    """Carrega imagens de `raiz/<classe>/*.jpg` (estrutura tipo ImageFolder)."""

    def __init__(
        self,
        raiz: str | Path,
        transform=None,
        prob_fundo_aleatorio: float = 0.0,
        cache_recortes: str | Path | None = None,
    ):
        self.raiz = Path(raiz)
        self.transform = transform
        self.prob_fundo_aleatorio = prob_fundo_aleatorio
        self.cache_recortes = Path(cache_recortes) if cache_recortes else self.raiz.parent / "_recortes_cache"

        self.amostras: list[tuple[Path, int]] = []
        for indice_classe, nome_classe in enumerate(CLASSES):
            pasta_classe = self.raiz / nome_classe
            if not pasta_classe.is_dir():
                continue
            for caminho in sorted(pasta_classe.iterdir()):
                if caminho.suffix.lower() in EXTENSOES_VALIDAS:
                    self.amostras.append((caminho, indice_classe))

        if not self.amostras:
            raise RuntimeError(f"Nenhuma imagem encontrada em {self.raiz}")

        logger.info("Dataset carregado de %s: %d imagens", self.raiz, len(self.amostras))

    def __len__(self) -> int:
        return len(self.amostras)

    def __getitem__(self, indice: int):
        caminho, rotulo = self.amostras[indice]

        if self.prob_fundo_aleatorio > 0 and random.random() < self.prob_fundo_aleatorio:
            imagem = self._compor_com_fundo_aleatorio(caminho)
        else:
            imagem = Image.open(caminho).convert("RGB")

        if self.transform:
            imagem = self.transform(imagem)

        return imagem, rotulo

    def _compor_com_fundo_aleatorio(self, caminho: Path) -> Image.Image:
        cache_dir = self.cache_recortes / self.raiz.name / caminho.parent.name
        recorte = _obter_recorte_sem_fundo(caminho, cache_dir)
        fundo = _gerar_fundo_aleatorio(recorte.size)
        fundo.paste(recorte, (0, 0), recorte)
        return fundo.convert("RGB")


def transformacoes_treino() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.RandomResizedCrop(TAMANHO_IMAGEM, scale=(0.8, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def transformacoes_avaliacao() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((TAMANHO_IMAGEM, TAMANHO_IMAGEM)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def criar_dataloaders(
    raiz_dataset: str | Path,
    tamanho_lote: int = 16,
    prob_fundo_aleatorio: float = 0.3,
    num_trabalhadores: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Monta os DataLoaders de treino (com augmentation) e teste (sem)."""
    raiz_dataset = Path(raiz_dataset)

    dataset_treino = DatasetDinos(
        raiz_dataset / "train",
        transform=transformacoes_treino(),
        prob_fundo_aleatorio=prob_fundo_aleatorio,
    )
    dataset_teste = DatasetDinos(
        raiz_dataset / "test",
        transform=transformacoes_avaliacao(),
        prob_fundo_aleatorio=0.0,
    )

    loader_treino = DataLoader(
        dataset_treino, batch_size=tamanho_lote, shuffle=True, num_workers=num_trabalhadores
    )
    loader_teste = DataLoader(
        dataset_teste, batch_size=tamanho_lote, shuffle=False, num_workers=num_trabalhadores
    )

    return loader_treino, loader_teste
