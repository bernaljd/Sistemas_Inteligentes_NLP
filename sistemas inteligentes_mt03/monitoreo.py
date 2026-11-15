"""
Paso 10 (6 de la guia): monitoreo de calidad y de cambios en los datos (deriva).

Idea: en produccion no hay etiquetas reales, asi que no se puede medir el
macro-F1 en vivo. Lo que SI se puede vigilar es si los comentarios que llegan
se parecen a los datos con que se evaluo el modelo. Si cambian (otro tipo de
textos, otro vocabulario, otra distribucion de clases), las metricas del paso 7
dejan de ser validas y conviene reetiquetar y reentrenar.

Que se registra por cada comentario (monitoreo/predicciones.jsonl):
  fecha, n_caracteres, n_palabras (tras la limpieza), oov (fraccion de palabras
  que el TF-IDF no conoce), estado, prediccion, confianza, ms (latencia del lote)
  y un hash del texto. NO se guarda el texto: los comentarios pueden contener
  datos personales de estudiantes y docentes.

Referencia (monitoreo/referencia.json): las mismas variables calculadas sobre
test.csv (datos que el modelo no vio al entrenar = comportamiento esperado).

Indicadores de deriva:
  - PSI (Population Stability Index) de n_palabras, oov y confianza.
      < 0.10 estable | 0.10-0.25 cambio moderado | > 0.25 cambio importante
  - PSI de la distribucion de clases predichas.
  - Tasa de ambiguos y de textos vacios frente a la referencia.
"""

import hashlib
import json
import threading
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from pipeline_feedback import CLASES, limpiar_texto

UMBRAL_PSI_MODERADO = 0.10
UMBRAL_PSI_IMPORTANTE = 0.25
N_BINS = 5            # la referencia tiene 123 comentarios: 5 grupos de ~25
MIN_REGISTROS = 30    # por debajo, el reporte avisa que hay pocos datos
VARIABLES = ["n_palabras", "oov", "confianza"]

_candado = threading.Lock()


def vocabulario(modelo):
    """Palabras sueltas (unigramas) que conoce el TF-IDF del pipeline."""
    tfidf = modelo.named_steps["features"].transformer_list[0][1].named_steps["vectorizador"]
    return {t for t in tfidf.vocabulary_ if " " not in t}


def caracteristicas(textos, resultados, vocab, ms=None):
    """Una fila por comentario con las variables que se vigilan (sin el texto)."""
    filas = []
    for t, (_, r) in zip(textos, resultados.iterrows()):
        palabras = limpiar_texto(t).split()
        filas.append({
            "n_caracteres": len(t),
            "n_palabras": len(palabras),
            "oov": round(sum(p not in vocab for p in palabras) / len(palabras), 4) if palabras else None,
            "estado": r["estado"],
            "prediccion": r["prediccion"] if isinstance(r["prediccion"], str) else None,
            "confianza": None if r["confianza"] != r["confianza"] else round(float(r["confianza"]), 4),
            "ms": ms,
            "hash": hashlib.sha256(t.encode("utf-8")).hexdigest()[:16],
        })
    return filas


# ---------------------------------------------------------------------------
# Registro
# ---------------------------------------------------------------------------

def registrar(ruta, filas):
    ahora = datetime.now(timezone.utc).isoformat(timespec="seconds")
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with _candado, open(ruta, "a", encoding="utf-8") as f:
        for fila in filas:
            f.write(json.dumps({"fecha": ahora, **fila}, ensure_ascii=False) + "\n")


def leer_registros(ruta, ultimos=None):
    ruta = Path(ruta)
    if not ruta.exists():
        return []
    with _candado:
        lineas = ruta.read_text(encoding="utf-8").splitlines()
    if ultimos:
        lineas = lineas[-ultimos:]
    return [json.loads(l) for l in lineas if l.strip()]


# ---------------------------------------------------------------------------
# Referencia y reporte
# ---------------------------------------------------------------------------

def _valores(filas, var):
    return np.array([f[var] for f in filas if f.get(var) is not None and f["estado"] != "texto_vacio"],
                    dtype=float)


def _proporciones(filas):
    validas = [f for f in filas if f["estado"] != "texto_vacio"]
    n = max(len(validas), 1)
    return {c: sum(f["prediccion"] == c for f in validas) / n for c in CLASES}


def _tasas(filas):
    n = max(len(filas), 1)
    return {"ambiguo": sum(f["estado"] == "ambiguo" for f in filas) / n,
            "texto_vacio": sum(f["estado"] == "texto_vacio" for f in filas) / n}


def construir_referencia(filas, descripcion):
    ref = {"descripcion": descripcion, "n": len(filas), "creada": datetime.now().isoformat(timespec="seconds"),
           "variables": {}, "clases": _proporciones(filas), "tasas": _tasas(filas)}
    for var in VARIABLES:
        v = _valores(filas, var)
        # cortes por cuantiles de la referencia: cada grupo tiene ~la misma cantidad
        cortes = np.unique(np.quantile(v, np.linspace(0, 1, N_BINS + 1))[1:-1]).tolist()
        ref["variables"][var] = {"cortes": cortes, "proporciones": _histograma(v, cortes),
                                 "media": float(v.mean()), "p50": float(np.median(v))}
    return ref


def _histograma(v, cortes):
    idx = np.searchsorted(cortes, v, side="right")
    return (np.bincount(idx, minlength=len(cortes) + 1) / max(len(v), 1)).tolist()


def psi(esperado, actual, eps=1e-4):
    e = np.clip(np.asarray(esperado, float), eps, None)
    a = np.clip(np.asarray(actual, float), eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def _nivel(valor):
    if valor > UMBRAL_PSI_IMPORTANTE:
        return "importante"
    return "moderado" if valor > UMBRAL_PSI_MODERADO else "estable"


def reporte_deriva(filas, ref):
    """Compara los registros con la referencia y devuelve un diccionario."""
    rep = {"n": len(filas), "referencia": ref["descripcion"], "n_referencia": ref["n"],
           "indicadores": {}, "alertas": []}
    if len(filas) < MIN_REGISTROS:
        rep["estado"] = "pocos_datos"
        rep["alertas"].append(f"Solo {len(filas)} registros; se necesitan al menos {MIN_REGISTROS}.")
        return rep

    for var in VARIABLES:
        r = ref["variables"][var]
        v = _valores(filas, var)
        if len(v) == 0:
            continue
        valor = psi(r["proporciones"], _histograma(v, r["cortes"]))
        rep["indicadores"][var] = {"psi": round(valor, 3), "nivel": _nivel(valor),
                                   "media_referencia": round(r["media"], 3), "media_actual": round(float(v.mean()), 3)}
    actual = _proporciones(filas)
    valor = psi([ref["clases"][c] for c in CLASES], [actual[c] for c in CLASES])
    rep["indicadores"]["clases_predichas"] = {
        "psi": round(valor, 3), "nivel": _nivel(valor),
        "referencia": {c: round(ref["clases"][c], 3) for c in CLASES},
        "actual": {c: round(actual[c], 3) for c in CLASES}}
    tasas = _tasas(filas)
    for k in ["ambiguo", "texto_vacio"]:
        # alerta si la tasa duplica la de referencia y pasa del 10%
        alto = tasas[k] > max(2 * ref["tasas"][k], 0.10)
        rep["indicadores"][f"tasa_{k}"] = {"referencia": round(ref["tasas"][k], 3), "actual": round(tasas[k], 3),
                                           "nivel": "importante" if alto else "estable"}

    latencias = [f["ms"] for f in filas if f.get("ms") is not None]
    if latencias:
        rep["latencia_ms_lote"] = {"p50": round(float(np.percentile(latencias, 50)), 1),
                                   "p95": round(float(np.percentile(latencias, 95)), 1)}
    repetidos = len(filas) - len({f["hash"] for f in filas})
    rep["comentarios_repetidos"] = repetidos

    explicacion = {
        "n_palabras": "la longitud de los comentarios cambio",
        "oov": "aparecen muchas palabras que el modelo no conoce (otro tema, idioma o jerga)",
        "confianza": "el modelo esta menos (o mas) seguro que en la evaluacion",
        "clases_predichas": "cambio la proporcion de positivos/neutrales/negativos predichos",
        "tasa_ambiguo": "hay muchos mas comentarios ambiguos que en la evaluacion",
        "tasa_texto_vacio": "llegan muchos comentarios vacios (revisar el formulario o el CSV)",
    }
    for k, ind in rep["indicadores"].items():
        if ind["nivel"] != "estable":
            rep["alertas"].append(f"[{ind['nivel']}] {k}: {explicacion[k]}")
    niveles = [i["nivel"] for i in rep["indicadores"].values()]
    rep["estado"] = "importante" if "importante" in niveles else ("moderado" if "moderado" in niveles else "estable")
    rep["recomendacion"] = {
        "estable": "Sin cambios relevantes: las metricas del paso 7 siguen siendo una buena guia.",
        "moderado": "Revisar una muestra de comentarios recientes y vigilar la tendencia.",
        "importante": "Los datos cambiaron: etiquetar una muestra nueva, medir el desempeno real "
                      "y reentrenar si baja.",
    }[rep["estado"]]
    return rep
