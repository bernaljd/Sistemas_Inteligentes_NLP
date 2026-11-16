"""
Pasos 4-6: Servicio de prediccion (FastAPI) + pagina web + seguridad + monitoreo.

Carga la version activa de los modelos (modelos/registro.json) UNA vez al
arrancar y expone:
  GET  /                 pagina web para docentes (carpeta web/)
  GET  /salud            estado del servicio y del modelo
  GET  /modelo           metadatos (metricas, parametros, versiones, datos)
  GET  /modelo/aspectos  metadatos del modelo de 9 aspectos (punto 5)
  GET  /modelo/versiones versiones registradas, activas e historial (punto 6)
  POST /predecir         un comentario      {"texto": "..."}
  POST /predecir/lote    varios comentarios {"textos": ["...", "..."]}
  POST /predecir/csv     archivo CSV (multipart) + ?columna=feedback
  GET  /monitoreo        reporte de deriva de los datos (paso 6)

Seguridad (paso 6):
  - Limite de peticiones por IP en /predecir* (RATE_LIMIT por minuto) -> 429.
  - Limite de tamano del cuerpo (6 MB, contando los bytes reales: tambien con
    Transfer-Encoding chunked) -> 413, textos <= 5000 caracteres, lotes <= 1000.
  - La IP del limite es la de la conexion: no se confia en X-Forwarded-For
    (el Dockerfile no usa --proxy-headers; ver docs/DESPLIEGUE.md si se pone un proxy).
  - Cabeceras: Content-Security-Policy, X-Content-Type-Options, X-Frame-Options,
    Referrer-Policy, Permissions-Policy.
  - Sin CORS: la pagina se sirve desde el mismo origen (CORS_ORIGINS para abrirlo).
  - Errores internos sin detalles (no se filtran rutas ni trazas).
  - /docs se puede apagar en produccion (DOCS=0); /monitoreo se puede proteger con
    ADMIN_TOKEN (cabecera X-Admin-Token).
  - No se guarda el texto de los comentarios en el registro de monitoreo.
  - Deserializacion segura (imagen 3, punto 3): el .joblib solo se carga si su
    sha256 coincide con el registrado (carga_segura.py); si no, no arranca.
  - Compatibilidad de versiones (imagen 3, punto 4): Python, scikit-learn y joblib
    deben ser los del modelo (compatibilidad.py); si no, no arranca.

Salida estructurada (imagen 3, punto 5): cada prediccion trae "aspectos", una
lista con los 9 aspectos (aspectos.py), cada uno con mencionado, probabilidad,
sentimiento y probabilidades_sentimiento. Siempre los 9 y en el mismo orden.

Control de versiones (imagen 3, punto 6): versiones_modelo.py registra cada
modelo entrenado como version inmutable en modelos/, y promueve o revierte cual
carga la API; /salud dice que version se cargo y de donde.

Variables de entorno (todas opcionales):
  RATE_LIMIT=60  DOCS=1  MONITOREO=1  ADMIN_TOKEN=  CORS_ORIGINS=  FRAME_ANCESTORS='none'
  RUTA_REGISTRO=monitoreo/predicciones.jsonl
  MODELO_VERSION=  (version de polaridad a cargar; si falta, la activa de modelos/registro.json)
  ASPECTOS_VERSION=  (igual, para el modelo de aspectos)
  MODELO_SHA256=  (huella esperada del modelo; si falta, la de modelos/registro.json)
  ASPECTOS_SHA256=  (igual, para el modelo de aspectos)
  VERSIONES_ESTRICTAS=0  (1 = cualquier version distinta a la del modelo impide arrancar)

Ejecutar (en esta carpeta):
    uvicorn api:app --host 127.0.0.1 --port 8000
"""

import json
import logging
import math
import os
import secrets
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, Header, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import monitoreo
from aspectos import CLAVES as CLAVES_ASPECTOS
from pipeline_feedback import CLASES, leer_csv_bytes, predecir
from versiones_modelo import (CARPETA_MODELOS, FAMILIAS, ErrorVersiones, calcular_firma, cargar_verificado,
                              comparar_firma, firma_guardada, resolver, resumen)

CARPETA = Path(__file__).resolve().parent
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


class ModeloNoDisponible(RuntimeError):
    """El servicio no arranca: falta el modelo o no esta entrenado."""


def _cargar_version(familia, ruta, carpeta_permitida, carpeta_modelos):
    """Resuelve que .joblib cargar y lo carga verificado (puntos 3, 4 y 6).

    ruta=None: la version de MODELO_VERSION / ASPECTOS_VERSION o la activa de
    modelos/registro.json (punto 6); la huella esperada sale del registro y,
    tras cargar, se comprueba la firma de comportamiento de esa version.
    ruta dada: ese archivo, con la huella de sus metadatos (pruebas).
    Devuelve (paquete, compatibilidad, integridad, version) o None si es el
    modelo de aspectos y no hay ninguno.
    """
    cfg = FAMILIAS[familia]
    if ruta is not None:
        if not Path(ruta).exists():
            if familia == "aspectos":
                return None
            raise ModeloNoDisponible(f"No existe {Path(ruta).name}. El servicio no entrena: "
                                     "genera el modelo con paso8_guardar_modelo.py.")
        paquete, compat, integ = cargar_verificado(ruta, carpeta_permitida, cfg["variable_sha256"])
        return paquete, compat, integ, {"version": paquete["metadatos"].get("version_modelo"),
                                        "fuente": f"archivo {Path(ruta).name}", "firma": None}
    try:
        info = resolver(familia, carpeta_modelos=carpeta_modelos)
    except ErrorVersiones as e:
        if familia == "aspectos" and not os.environ.get(cfg["variable_version"]):
            return None
        raise ModeloNoDisponible(str(e)) from e
    if not info["ruta"].exists():
        raise ModeloNoDisponible(f"Falta {info['ruta'].name} "
                                 f"({familia} {info['version']}).")
    paquete, compat, integ = cargar_verificado(info["ruta"], carpeta_modelos, cfg["variable_sha256"],
                                               sha256=info["sha256"])
    # La firma tambien sirve de calentamiento (carga los embeddings ahora y no en la 1a peticion)
    firma = comparar_firma(firma_guardada(familia, info["version"], carpeta_modelos),
                           calcular_firma(familia, paquete))
    if not firma["coincide"]:
        raise ModeloNoDisponible(
            f"{familia} {info['version']}: el codigo instalado no reproduce las predicciones de esta "
            f"version ({firma['motivo']}). Usa el codigo del commit {info['entrada'].get('commit')} "
            "o vuelve a otra version (versiones_modelo.py revertir).")
    return paquete, compat, integ, {"version": info["version"], "fuente": info["fuente"], "firma": firma}


def cargar_servicio(ruta_modelo=None, carpeta_permitida=CARPETA, ruta_aspectos=None,
                    carpeta_modelos=CARPETA_MODELOS):
    """Carga el modelo YA ENTRENADO una sola vez, al arrancar (punto 2: eficiencia).

    El servicio NUNCA entrena: el entrenamiento es un paso aparte y offline
    (paso8_guardar_modelo.py). Si el .joblib falta o trae un Pipeline sin
    ajustar, el arranque falla con un mensaje claro en vez de reentrenar
    (entrenar en cada arranque tardaria minutos y daria un modelo distinto
    al evaluado en el paso 7).

    Punto 3 (seguridad): el .joblib se carga con cargar_joblib_confiable(): solo
    desde la carpeta de modelos y solo si su sha256 coincide con el registrado
    (MODELO_SHA256 o modelos/registro.json). Si no, ModeloNoConfiable y el
    servicio no arranca: el archivo nunca se deserializa.

    Punto 4 (versiones): ANTES de deserializar se comparan las versiones
    registradas en los metadatos (leer JSON no ejecuta codigo) con las
    instaladas (compatibilidad.py). Si Python (mayor.menor), scikit-learn o
    joblib difieren, VersionesIncompatibles y el servicio no arranca; el resto
    se informa en /salud. Al cargar, un InconsistentVersionWarning de
    scikit-learn tambien se trata como incompatibilidad.

    Punto 5 (aspectos): el modelo de aspectos se carga con las mismas
    verificaciones (huella en ASPECTOS_SHA256 o el registro). Es opcional: sin
    el, el servicio funciona y "aspectos" sale null.

    Punto 6 (versiones del modelo): se carga la version activa de
    modelos/registro.json, o la fijada con MODELO_VERSION / ASPECTOS_VERSION
    (reversion de emergencia sin tocar el registro). Si el codigo instalado no
    reproduce la firma de esa version, no arranca.
    """
    from sklearn.exceptions import NotFittedError
    from sklearn.utils.validation import check_is_fitted

    t0 = time.perf_counter()
    carpeta_modelos = Path(carpeta_modelos)
    paquete, compat, integridad, version = _cargar_version("polaridad", ruta_modelo, carpeta_permitida,
                                                          carpeta_modelos)
    modelo = paquete["pipeline"]
    try:
        check_is_fitted(modelo)
    except NotFittedError as e:
        raise ModeloNoDisponible("El modelo contiene un Pipeline sin entrenar: "
                                 "regeneralo con paso8_guardar_modelo.py.") from e
    estado = {"modelo": modelo, "metadatos": paquete["metadatos"],
              "vocab": monitoreo.vocabulario(modelo),
              "integridad": integridad, "compatibilidad": compat,
              "versiones_cargadas": {"polaridad": version}}
    if ruta_modelo is None:
        estado["registro_versiones"] = resumen(carpeta_modelos)
    if RUTA_REFERENCIA.exists():
        estado["referencia"] = json.loads(RUTA_REFERENCIA.read_text(encoding="utf-8"))
    # Calentamiento: carga el modelo de embeddings ahora y no en la 1a peticion
    predecir(modelo, ["calentamiento del modelo"])

    # Punto 5: modelo de aspectos (mismas verificaciones de huella y versiones)
    aspectos = _cargar_version("aspectos", ruta_aspectos, carpeta_permitida, carpeta_modelos)
    if aspectos:
        paq_asp, compat_asp, integ_asp, version_asp = aspectos
        estado["aspectos"] = {"modelo": paq_asp["modelo"], "metadatos": paq_asp["metadatos"],
                              "integridad": integ_asp, "compatibilidad": compat_asp}
        estado["versiones_cargadas"]["aspectos"] = version_asp
        paq_asp["modelo"].predecir(["calentamiento del modelo de aspectos"])
    else:
        log.warning("No hay modelo de aspectos: las respuestas no traeran aspectos (ejecuta paso11_aspectos.py)")
    estado["segundos_carga"] = round(time.perf_counter() - t0, 2)
    return estado


@asynccontextmanager
async def lifespan(app):
    estado_servicio.update(cargar_servicio())
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


class PrediccionAspecto(BaseModel):
    """Punto 5: una prediccion por aspecto (siempre los 9, en el mismo orden)."""
    aspecto: str = Field(description="clave: " + ", ".join(CLAVES_ASPECTOS))
    nombre: str
    mencionado: bool = Field(description="el comentario habla de este aspecto (probabilidad >= 0.5)")
    probabilidad: float | None = Field(description="probabilidad de que se mencione; null si el texto esta vacio")
    sentimiento: str | None = Field(description="negativo | neutral | positivo; null si no se menciona")
    probabilidades_sentimiento: Probabilidades | None


class Prediccion(BaseModel):
    texto: str
    estado: str = Field(description="ok | ambiguo | texto_vacio")
    prediccion: str | None
    confianza: float | None
    probabilidades: Probabilidades
    aspectos: list[PrediccionAspecto] | None = Field(
        None, description="9 aspectos con mencion y sentimiento; null si no hay modelo de aspectos")


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
    asp = estado_servicio.get("aspectos")
    por_texto = asp["modelo"].predecir(textos) if asp and textos else [None] * len(textos)
    return [Prediccion(
        texto=f["texto"],
        estado=f["estado"],
        prediccion=None if pd.isna(f["prediccion"]) else f["prediccion"],
        confianza=_num(f["confianza"]),
        probabilidades=Probabilidades(**{c: _num(f[f"prob_{c}"]) for c in CLASES}),
        aspectos=None if a is None else [PrediccionAspecto(**x) for x in a],
    ) for (_, f), a in zip(df.iterrows(), por_texto)]


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
            "entrenado_el": m.get("fecha"),  # se entreno antes (paso 8), no al arrancar
            "integridad": estado_servicio.get("integridad"),  # sha256 verificado al cargar
            "compatibilidad": {k: estado_servicio["compatibilidad"][k] for k in ("estado", "diferencias")}
            if "compatibilidad" in estado_servicio else None,  # versiones modelo vs. instaladas
            "clases": CLASES, "segundos_carga": estado_servicio.get("segundos_carga"),
            "aspectos": _resumen_aspectos(),
            "versiones_cargadas": estado_servicio.get("versiones_cargadas")}  # punto 6


def _resumen_aspectos():
    asp = estado_servicio.get("aspectos")
    if not asp:
        return None
    m = asp["metadatos"]
    return {"modelo": m["nombre"], "version_modelo": m["version_modelo"], "entrenado_el": m["fecha"],
            "aspectos": CLAVES_ASPECTOS, "integridad": asp["integridad"],
            "compatibilidad": {k: asp["compatibilidad"][k] for k in ("estado", "diferencias")}}


@app.get("/modelo")
def modelo():
    return estado_servicio["metadatos"]


@app.get("/modelo/versiones")
def modelo_versiones():
    """Punto 6: versiones registradas de cada modelo, la activa e historial de cambios."""
    if "registro_versiones" not in estado_servicio:
        raise HTTPException(404, "El servicio no se cargo desde modelos/registro.json")
    return {"cargadas": estado_servicio["versiones_cargadas"], **estado_servicio["registro_versiones"]}


@app.get("/modelo/aspectos")
def modelo_aspectos():
    """Metadatos del modelo de aspectos: los 9 aspectos, formato de salida y metricas."""
    if "aspectos" not in estado_servicio:
        raise HTTPException(404, "No hay modelo de aspectos cargado")
    return estado_servicio["aspectos"]["metadatos"]


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
