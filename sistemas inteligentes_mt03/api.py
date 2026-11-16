"""
Pasos 4-6: Servicio de prediccion (FastAPI) + pagina web + seguridad + monitoreo.

Carga modelo_final.joblib UNA vez al arrancar y expone:
  GET  /                 pagina web para docentes (carpeta web/)
  GET  /salud            estado del servicio y del modelo
  GET  /modelo           metadatos (metricas, parametros, versiones, datos)
  POST /predecir         un comentario      {"texto": "..."}
  POST /predecir/lote    varios comentarios {"textos": ["...", "..."]}
  POST /predecir/csv     archivo CSV (multipart) + ?columna=feedback
  GET  /monitoreo        reporte de deriva de los datos (paso 6)

Seguridad (paso 6):
  - Limite de peticiones por IP en /predecir* (RATE_LIMIT por minuto) -> 429.
  - Limite de tamano del cuerpo (6 MB, contando los bytes reales: tambien con
    Transfer-Encoding chunked) -> 413, textos <= 5000 caracteres, lotes <= 1000.
  - La IP del limite es la de la conexion: no se confia en X-Forwarded-For
    (el Dockerfile no usa --proxy-headers; ver DESPLIEGUE.md si se pone un proxy).
  - Cabeceras: Content-Security-Policy, X-Content-Type-Options, X-Frame-Options,
    Referrer-Policy, Permissions-Policy.
  - Sin CORS: la pagina se sirve desde el mismo origen (CORS_ORIGINS para abrirlo).
  - Errores internos sin detalles (no se filtran rutas ni trazas).
  - /docs se puede apagar en produccion (DOCS=0); /monitoreo se puede proteger con
    ADMIN_TOKEN (cabecera X-Admin-Token).
  - No se guarda el texto de los comentarios en el registro de monitoreo.

Variables de entorno (todas opcionales):
  RATE_LIMIT=60  DOCS=1  MONITOREO=1  ADMIN_TOKEN=  CORS_ORIGINS=  FRAME_ANCESTORS='none'
  RUTA_REGISTRO=monitoreo/predicciones.jsonl

Ejecutar (en esta carpeta):
    uvicorn api:app --host 127.0.0.1 --port 8000
"""

import logging
import math
import os
import secrets
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, File, Header, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import monitoreo
from pipeline_feedback import CLASES, leer_csv_bytes, predecir

CARPETA = Path(__file__).resolve().parent
RUTA_MODELO = CARPETA / "modelo_final.joblib"
RUTA_REFERENCIA = CARPETA / "monitoreo" / "referencia.json"
RUTA_REGISTRO = Path(os.environ.get("RUTA_REGISTRO", CARPETA / "monitoreo" / "predicciones.jsonl"))

MAX_CARACTERES = 5000        # por comentario
MAX_TEXTOS_LOTE = 1000       # por peticion
MAX_BYTES_CSV = 5 * 1024 * 1024
MAX_BYTES_PETICION = 6 * 1024 * 1024

RATE_LIMIT = int(os.environ.get("RATE_LIMIT", "60"))      # peticiones / minuto / IP
DOCS = os.environ.get("DOCS", "1") == "1"
MONITOREO = os.environ.get("MONITOREO", "1") == "1"
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "")
CORS_ORIGINS = [o for o in os.environ.get("CORS_ORIGINS", "").split(",") if o]
# Quien puede mostrar la pagina dentro de un iframe. 'none' = nadie (por defecto).
# Hugging Face Spaces muestra la app en un iframe: FRAME_ANCESTORS="https://huggingface.co"
FRAME_ANCESTORS = os.environ.get("FRAME_ANCESTORS", "'none'").strip() or "'none'"

log = logging.getLogger("api")
estado_servicio = {}


@asynccontextmanager
async def lifespan(app):
    t0 = time.perf_counter()
    paquete = joblib.load(RUTA_MODELO)  # archivo propio y de confianza
    estado_servicio["modelo"] = paquete["pipeline"]
    estado_servicio["metadatos"] = paquete["metadatos"]
    estado_servicio["vocab"] = monitoreo.vocabulario(paquete["pipeline"])
    if RUTA_REFERENCIA.exists():
        import json
        estado_servicio["referencia"] = json.loads(RUTA_REFERENCIA.read_text(encoding="utf-8"))
    # Calentamiento: carga el modelo de embeddings ahora y no en la 1a peticion
    predecir(estado_servicio["modelo"], ["calentamiento del modelo"])
    estado_servicio["segundos_carga"] = round(time.perf_counter() - t0, 2)
    yield
    estado_servicio.clear()


app = FastAPI(
    title="API de polaridad de comentarios docentes",
    description="Clasifica comentarios de estudiantes en negativo / neutral / positivo.",
    version="1.1.0",
    lifespan=lifespan,
    docs_url="/docs" if DOCS else None,
    redoc_url="/redoc" if DOCS else None,
    openapi_url="/openapi.json" if DOCS else None,
)

if CORS_ORIGINS:
    from fastapi.middleware.cors import CORSMiddleware
    app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_methods=["GET", "POST"],
                       allow_headers=["Content-Type"])


# ---------------------------------------------------------------------------
# Seguridad: limite de peticiones, tamano maximo y cabeceras
# ---------------------------------------------------------------------------

class LimitadorPeticiones:
    """Ventana deslizante de 60 s por IP (en memoria: vale para 1 proceso)."""

    def __init__(self, maximo, ventana=60.0):
        self.maximo, self.ventana = maximo, ventana
        self.historial = defaultdict(deque)
        self.candado = threading.Lock()

    def permitir(self, clave):
        ahora = time.monotonic()
        with self.candado:
            h = self.historial[clave]
            while h and ahora - h[0] > self.ventana:
                h.popleft()
            if len(h) >= self.maximo:
                return False, int(self.ventana - (ahora - h[0])) + 1
            h.append(ahora)
            if len(self.historial) > 10000:  # evita crecer sin limite
                for k in [k for k, v in self.historial.items() if not v]:
                    del self.historial[k]
            return True, 0


limitador = LimitadorPeticiones(RATE_LIMIT)


class LimiteCuerpo:
    """Middleware ASGI: corta la peticion cuando el cuerpo REAL supera el maximo.

    Revisar solo Content-Length no basta: con Transfer-Encoding: chunked no hay
    Content-Length y el cuerpo puede ser de cualquier tamano. Aqui se cuentan los
    bytes a medida que llegan y se lanza HTTPException(413), que FastAPI deja
    pasar al leer el cuerpo y responde como JSON.
    """

    def __init__(self, app, maximo):
        self.app, self.maximo = app, maximo

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        recibido = 0

        async def recibir_con_limite():
            nonlocal recibido
            mensaje = await receive()
            if mensaje["type"] == "http.request":
                recibido += len(mensaje.get("body", b""))
                if recibido > self.maximo:
                    raise HTTPException(413, "Peticion demasiado grande")
            return mensaje

        await self.app(scope, recibir_con_limite, send)


app.add_middleware(LimiteCuerpo, maximo=MAX_BYTES_PETICION)

CSP = ("default-src 'self'; script-src 'self'; "
       "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
       "font-src 'self' https://fonts.gstatic.com; img-src 'self' data:; "
       f"connect-src 'self'; frame-ancestors {FRAME_ANCESTORS}; base-uri 'self'; form-action 'self'")


@app.middleware("http")
async def seguridad(request: Request, call_next):
    ruta = request.url.path
    largo = request.headers.get("content-length")
    if largo and largo.isdigit() and int(largo) > MAX_BYTES_PETICION:
        return JSONResponse({"detail": "Peticion demasiado grande"}, status_code=413)
    if ruta.startswith("/predecir") and RATE_LIMIT > 0:
        ip = request.client.host if request.client else "desconocida"
        ok, espera = limitador.permitir(ip)
        if not ok:
            return JSONResponse({"detail": f"Demasiadas peticiones. Intenta de nuevo en {espera} s."},
                                status_code=429, headers={"Retry-After": str(espera)})
    t0 = time.perf_counter()
    respuesta = await call_next(request)
    respuesta.headers["X-Tiempo-ms"] = f"{(time.perf_counter() - t0) * 1000:.1f}"
    respuesta.headers["X-Content-Type-Options"] = "nosniff"
    respuesta.headers["Referrer-Policy"] = "no-referrer"
    respuesta.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    if not ruta.startswith(("/docs", "/redoc")):  # Swagger usa scripts de un CDN
        respuesta.headers["Content-Security-Policy"] = CSP
        if FRAME_ANCESTORS == "'none'":  # X-Frame-Options no admite listas de origenes
            respuesta.headers["X-Frame-Options"] = "DENY"
    return respuesta


@app.exception_handler(Exception)
async def error_interno(request: Request, exc: Exception):
    log.exception("Error no controlado en %s", request.url.path)
    return JSONResponse({"detail": "Error interno del servidor"}, status_code=500)


# ---------------------------------------------------------------------------
# Pagina web
# ---------------------------------------------------------------------------

CARPETA_WEB = CARPETA / "web"
app.mount("/web", StaticFiles(directory=CARPETA_WEB), name="web")


@app.get("/", include_in_schema=False)
def pagina():
    return FileResponse(CARPETA_WEB / "index.html")


# ---------------------------------------------------------------------------
# Esquemas
# ---------------------------------------------------------------------------

class Comentario(BaseModel):
    texto: str = Field(..., max_length=MAX_CARACTERES,
                       examples=["El profesor explica muy bien, pero los parciales son muy dificiles."])


class Lote(BaseModel):
    textos: list[str] = Field(..., max_length=MAX_TEXTOS_LOTE,
                              examples=[["Excelente curso", "No aprendi nada", ""]])


class Probabilidades(BaseModel):
    negativo: float | None
    neutral: float | None
    positivo: float | None


class Prediccion(BaseModel):
    texto: str
    estado: str = Field(description="ok | ambiguo | texto_vacio")
    prediccion: str | None
    confianza: float | None
    probabilidades: Probabilidades


class RespuestaLote(BaseModel):
    n: int
    resumen: dict[str, int]
    milisegundos: float
    resultados: list[Prediccion]


# ---------------------------------------------------------------------------
# Logica comun
# ---------------------------------------------------------------------------

def _num(v):
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else round(float(v), 4)


def _clasificar(textos):
    if "modelo" not in estado_servicio:
        raise HTTPException(503, "Modelo no cargado")
    largos = [i for i, t in enumerate(textos) if len(t) > MAX_CARACTERES]
    if largos:
        raise HTTPException(422, f"Textos con mas de {MAX_CARACTERES} caracteres en posiciones {largos[:10]}")
    t0 = time.perf_counter()
    df = predecir(estado_servicio["modelo"], textos)
    ms = round((time.perf_counter() - t0) * 1000, 1)
    if MONITOREO and textos:
        try:
            monitoreo.registrar(RUTA_REGISTRO, monitoreo.caracteristicas(textos, df, estado_servicio["vocab"], ms))
        except OSError:
            log.exception("No se pudo escribir el registro de monitoreo")  # no se cae la prediccion
    return [Prediccion(
        texto=f["texto"],
        estado=f["estado"],
        prediccion=None if pd.isna(f["prediccion"]) else f["prediccion"],
        confianza=_num(f["confianza"]),
        probabilidades=Probabilidades(**{c: _num(f[f"prob_{c}"]) for c in CLASES}),
    ) for _, f in df.iterrows()]


def _respuesta_lote(textos):
    t0 = time.perf_counter()
    resultados = _clasificar(textos)
    resumen = {}
    for r in resultados:
        clave = r.prediccion if r.estado == "ok" else r.estado
        resumen[clave] = resumen.get(clave, 0) + 1
    return RespuestaLote(n=len(resultados), resumen=resumen,
                         milisegundos=round((time.perf_counter() - t0) * 1000, 1),
                         resultados=resultados)


# ---------------------------------------------------------------------------
# Endpoints (def normales: FastAPI los ejecuta en un hilo aparte y no bloquean)
# ---------------------------------------------------------------------------

@app.get("/salud")
def salud():
    m = estado_servicio.get("metadatos", {})
    return {"estado": "ok" if "modelo" in estado_servicio else "sin_modelo",
            "modelo": m.get("nombre"), "version_modelo": m.get("version_modelo"),
            "clases": CLASES, "segundos_carga": estado_servicio.get("segundos_carga")}


@app.get("/modelo")
def modelo():
    return estado_servicio["metadatos"]


@app.post("/predecir", response_model=Prediccion)
def predecir_uno(comentario: Comentario):
    return _clasificar([comentario.texto])[0]


@app.post("/predecir/lote", response_model=RespuestaLote)
def predecir_lote(lote: Lote):
    return _respuesta_lote(lote.textos)


@app.post("/predecir/csv", response_model=RespuestaLote)
def predecir_csv(archivo: UploadFile = File(...),
                 columna: str = Query("feedback", max_length=200, description="Columna con los comentarios")):
    contenido = archivo.file.read(MAX_BYTES_CSV + 1)
    if len(contenido) > MAX_BYTES_CSV:
        raise HTTPException(413, f"El archivo supera {MAX_BYTES_CSV // 1024 // 1024} MB")
    try:
        df = leer_csv_bytes(contenido)
    except ValueError as e:
        raise HTTPException(422, str(e))
    if columna not in df.columns:
        raise HTTPException(422, f"No existe la columna '{columna}'. Columnas: {list(df.columns)[:50]}")
    if len(df) > MAX_TEXTOS_LOTE:
        raise HTTPException(413, f"Maximo {MAX_TEXTOS_LOTE} filas por archivo (tiene {len(df)})")
    textos = ["" if pd.isna(t) else str(t) for t in df[columna]]
    return _respuesta_lote(textos)


@app.get("/monitoreo")
def ver_monitoreo(ultimos: int = Query(500, ge=1, le=100000),
                  x_admin_token: str | None = Header(None)):
    if ADMIN_TOKEN and not secrets.compare_digest(x_admin_token or "", ADMIN_TOKEN):
        raise HTTPException(401, "Token de administrador invalido")
    if "referencia" not in estado_servicio:
        raise HTTPException(503, "Falta monitoreo/referencia.json (ejecuta paso10_deriva.py)")
    registros = monitoreo.leer_registros(RUTA_REGISTRO, ultimos)
    return monitoreo.reporte_deriva(registros, estado_servicio["referencia"])
