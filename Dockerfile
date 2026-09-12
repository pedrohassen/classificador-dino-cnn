FROM python:3.11-slim

# Hugging Face Spaces (Docker SDK) espera um usuário não-root com uid 1000
RUN useradd -m -u 1000 usuario
USER usuario
ENV PATH="/home/usuario/.local/bin:$PATH"

WORKDIR /app

COPY --chown=usuario requirements-api.txt requirements-api.txt
# --extra-index-url garante o build CPU-only do torch/torchvision (bem menor
# que o build com CUDA, que o índice padrão do PyPI traria por engano aqui)
RUN pip install --no-cache-dir --user \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements-api.txt

COPY --chown=usuario app/ app/
COPY --chown=usuario frontend/ frontend/
COPY --chown=usuario models_saved/ models_saved/

EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
