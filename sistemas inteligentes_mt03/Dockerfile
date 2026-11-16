# Paso 6: imagen para desplegar la API + pagina web en cualquier servidor o nube.
#   docker build -t polaridad-docente .
#   docker run -p 8000:8000 -v polaridad_monitoreo:/app/monitoreo_datos polaridad-docente
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/hf_cache

WORKDIR /app

# torch solo CPU (la version con GPU pesa varios GB mas y no se usa)
RUN pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu

COPY requirements_modelo.txt requirements_app.txt ./
RUN pip install -r requirements_app.txt

# Descargar el modelo de embeddings DURANTE la construccion: el contenedor
# arranca sin internet (HF_HUB_OFFLINE=1). Debe ser la MISMA revision que pide
# el Pipeline (REVISION_EMBEDDINGS en pipeline_feedback.py; test_api.py lo
# comprueba): si se descargara otra, el contenedor no encontraria la del modelo.
ARG REVISION_EMBEDDINGS=e8f8c211226b894fcb81acc59f3b34ba3efd5f42
RUN python -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', revision='${REVISION_EMBEDDINGS}')"

# Solo lo necesario para servir (nada de train/test ni scripts de entrenamiento)
COPY api.py pipeline_feedback.py monitoreo.py carga_segura.py ./
# El modelo viaja con su huella sha256 (modelo_final_metadatos.json): la API no lo
# deserializa si no coinciden. Para fijarla fuera de la imagen: -e MODELO_SHA256=...
COPY modelo_final.joblib modelo_final_metadatos.json ./
COPY web/ ./web/
COPY monitoreo/referencia.json ./monitoreo/referencia.json

# Usuario sin privilegios; el registro de monitoreo va a un volumen
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app/monitoreo_datos \
    && chown -R app:app /app/monitoreo_datos
USER app

ENV HF_HUB_OFFLINE=1 \
    PORT=8000 \
    DOCS=0 \
    RATE_LIMIT=60 \
    RUTA_REGISTRO=/app/monitoreo_datos/predicciones.jsonl

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/salud', timeout=8)"

# 1 proceso: cada proceso carga su propia copia del modelo (~300-600 MB).
# Sin --proxy-headers: el contenedor se expone directo (docker run -p), sin proxy
# delante. Con --forwarded-allow-ips='*' cualquiera podia enviar X-Forwarded-For
# con una IP falsa en cada peticion y saltarse el limite por IP.
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT} --workers 1 --no-server-header"]
