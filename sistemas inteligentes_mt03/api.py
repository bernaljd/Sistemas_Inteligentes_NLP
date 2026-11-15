"""
Paso 4: Servicio de prediccion (FastAPI).

Carga modelo_final.joblib UNA vez al arrancar y expone:
  GET  /                 pagina web para docentes (paso 5, carpeta web/)
  GET  /salud           estado del servicio y del modelo
  GET  /modelo           metadatos (metricas, parametros, versiones, datos)
  POST /predecir         un comentario      {"texto": "..."}
  POST /predecir/lote    varios comentarios {"textos": ["...", "..."]}
  POST /predecir/csv     archivo CSV (multipart) + ?columna=feedback

Cada prediccion devuelve: estado (ok / ambiguo / texto_vacio), clase predicha,
confianza y probabilidad por clase. Usa pipeline_feedback.predecir(), que valida
la entrada (textos vacios, tipos invalidos) y marca las predicciones ambiguas.

Ejecutar (en esta carpeta):
    uvicorn api:app --host 127.0.0.1 --port 8000
Pagina web:                http://127.0.0.1:8000
Documentacion interactiva: http://127.0.0.1:8000/docs
"""

import math
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from pipeline_feedback import CLASES, leer_csv_bytes, predecir

CARPETA = Path(__file__).resolve().parent
RUTA_MODELO = CARPETA / "modelo_final.joblib"

MAX_CARACTERES = 5000        # por comentario
MAX_TEXTOS_LOTE = 1000       # por peticion
MAX_BYTES_CSV = 5 * 1024 * 1024

estado_servicio = {}


@asynccontextmanager
async def lifespan(app):
    t0 = time.perf_counter()
    paquete = joblib.load(RUTA_MODELO)  # archivo propio y de confianza
    estado_servicio["modelo"] = paquete["pipeline"]
    estado_servicio["metadatos"] = paquete["metadatos"]
    # Calentamiento: carga el modelo de embeddings ahora y no en la 1a peticion
    predecir(estado_servicio["modelo"], ["calentamiento del modelo"])
    estado_servicio["segundos_carga"] = round(time.perf_counter() - t0, 2)
    yield
    estado_servicio.clear()


app = FastAPI(
    title="API de polaridad de comentarios docentes",
    description="Clasifica comentarios de estudiantes en negativo / neutral / positivo.",
    version="1.0.0",
    lifespan=lifespan,
)

# Pagina web (paso 5): web/index.html en "/" y sus archivos en "/web/..."
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
    df = predecir(estado_servicio["modelo"], textos)
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
                 columna: str = Query("feedback", description="Columna con los comentarios")):
    contenido = archivo.file.read(MAX_BYTES_CSV + 1)
    if len(contenido) > MAX_BYTES_CSV:
        raise HTTPException(413, f"El archivo supera {MAX_BYTES_CSV // 1024 // 1024} MB")
    df = leer_csv(contenido)
    if columna not in df.columns:
        raise HTTPException(422, f"No existe la columna '{columna}'. Columnas: {list(df.columns)}")
    if len(df) > MAX_TEXTOS_LOTE:
        raise HTTPException(413, f"Maximo {MAX_TEXTOS_LOTE} filas por archivo (tiene {len(df)})")
    textos = ["" if pd.isna(t) else str(t) for t in df[columna]]
    return _respuesta_lote(textos)


def leer_csv(contenido: bytes) -> pd.DataFrame:
    try:
        return leer_csv_bytes(contenido)
    except ValueError as e:
        raise HTTPException(422, str(e))
