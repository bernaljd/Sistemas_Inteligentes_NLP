# Paso 6: imagen para desplegar la API + pagina web (Docker local).
#   docker build -t polaridad-docente .
#   docker run -p 8000:8000 -v polaridad_monitoreo:/app/monitoreo_datos polaridad-docente
# Python 3.12 = misma version mayor.menor con que se guardo el modelo (3.12.5).
# compatibilidad.py exige que coincida al arrancar; el parche (3.12.x) no cambia pickle.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/hf_cache

WORKDIR /app

# torch solo CPU (la version con GPU pesa varios GB mas y no se usa). La version
# debe ser la de requirements_modelo.txt (test_compatibilidad.py lo comprueba).
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

# Dependencias coherentes entre si (falla el build si algun paquete exige otra version)
RUN pip check

# Solo lo necesario para servir (nada de train/test ni scripts de entrenamiento)
COPY api.py pipeline_feedback.py aspectos.py monitoreo.py carga_segura.py compatibilidad.py      versiones_modelo.py ./
# Punto 6: TODAS las versiones registradas viajan en la imagen (pesan ~0.3 MB cada
# una), asi se puede volver a una anterior sin reconstruir: -e MODELO_VERSION=1.1.0.
# La API carga la activa de modelos/registro.json y no deserializa nada cuya
# huella sha256 no coincida (puntos 3 y 6). Huella fija desde fuera: -e MODELO_SHA256=...
COPY modelos/ ./modelos/
# El build falla si algun archivo de modelo no coincide con su huella registrada
RUN python versiones_modelo.py verificar
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
