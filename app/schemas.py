"""Modelos de resposta (Pydantic) da API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PrevisaoClasse(BaseModel):
    classe: str = Field(alias="class")
    confianca: float = Field(alias="confidence")

    model_config = {"populate_by_name": True}


class RespostaPredicao(BaseModel):
    predicted_class: str
    confidence: float
    top_predictions: list[PrevisaoClasse]
