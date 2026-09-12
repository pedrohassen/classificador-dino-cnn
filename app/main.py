"""API FastAPI do classificador de dinossauros."""

from __future__ import annotations

import io
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from app.model import ClassificadorDinos
from app.schemas import PrevisaoClasse, RespostaPredicao

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RAIZ_PROJETO = Path(__file__).resolve().parent.parent
PASTA_FRONTEND = RAIZ_PROJETO / "frontend"

classificador: ClassificadorDinos | None = None


@asynccontextmanager
async def ciclo_de_vida(app: FastAPI):
    global classificador
    classificador = ClassificadorDinos()
    yield


app = FastAPI(title="Classificador de Dinossauros", lifespan=ciclo_de_vida)


@app.post("/predict", response_model=RespostaPredicao)
async def prever(image: UploadFile = File(...)) -> RespostaPredicao:
    if classificador is None:
        raise HTTPException(status_code=503, detail="Modelo ainda não carregado")

    conteudo = await image.read()
    try:
        imagem = Image.open(io.BytesIO(conteudo))
        imagem.load()
    except UnidentifiedImageError as erro:
        raise HTTPException(status_code=400, detail="Arquivo enviado não é uma imagem válida") from erro

    previsoes = classificador.prever(imagem, top_k=3)
    classe_top, confianca_top = previsoes[0]

    return RespostaPredicao(
        predicted_class=classe_top,
        confidence=confianca_top,
        top_predictions=[
            PrevisaoClasse(**{"class": classe, "confidence": confianca}) for classe, confianca in previsoes
        ],
    )


if PASTA_FRONTEND.is_dir():
    app.mount("/", StaticFiles(directory=PASTA_FRONTEND, html=True), name="frontend")
