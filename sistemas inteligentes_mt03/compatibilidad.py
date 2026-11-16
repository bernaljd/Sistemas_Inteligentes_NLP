"""
Punto 4 (imagen 3): compatibilidad de versiones entre entrenamiento y produccion.

modelo_final.joblib guarda objetos de scikit-learn serializados. scikit-learn
NO garantiza que un modelo guardado con una version se comporte igual (ni que
cargue) con otra. Por eso el paso 8 registra las versiones exactas en los
metadatos y en requirements_modelo.txt, y este modulo las compara con las
instaladas ANTES de servir.

Politica:
  - CRITICAS (si difieren, el servicio NO arranca):
      python (mayor.menor, p. ej. 3.12), scikit-learn, joblib
    -> son las que deserializan y ejecutan el Pipeline.
  - EXACTAS (si difieren, se avisa en /salud y en el log, y se puede volver
    critico con VERSIONES_ESTRICTAS=1):
      numpy, scipy, pandas, sentence-transformers, torch, transformers
    -> cambian calculos numericos o los embeddings; el paso 9 mide su efecto.
  - Las etiquetas locales no cuentan: torch 2.15.0+cpu == 2.15.0.
  - El parche de Python (3.12.5 vs 3.12.16) no cuenta: mismo formato pickle.
"""

import os
import platform
from importlib import metadata

CRITICAS = ["scikit-learn", "joblib"]
EXACTAS = ["numpy", "scipy", "pandas", "sentence-transformers", "torch", "transformers"]


class VersionesIncompatibles(RuntimeError):
    """Las versiones instaladas no son compatibles con las del modelo."""


def _sin_local(v):
    return str(v).split("+")[0] if v else v


def instalada(paquete):
    if paquete == "python":
        return platform.python_version()
    try:
        return metadata.version(paquete)
    except metadata.PackageNotFoundError:
        return None


def _mayor_menor(v):
    return ".".join(str(v).split(".")[:2])


def comparar(registradas, estricto=None):
    """Compara las versiones registradas en los metadatos con las instaladas.

    Devuelve {"estado": "ok" | "avisos" | "incompatible", "diferencias": [...],
              "versiones": {paquete: {"registrada", "instalada", "nivel"}}}.
    """
    if estricto is None:
        estricto = os.environ.get("VERSIONES_ESTRICTAS", "0") == "1"
    filas, diferencias = {}, []

    def revisar(paquete, registrada, actual, iguales, nivel):
        ok = actual is not None and iguales
        filas[paquete] = {"registrada": registrada, "instalada": actual, "nivel": nivel, "ok": ok}
        if not ok:
            diferencias.append({"paquete": paquete, "registrada": registrada, "instalada": actual,
                                "nivel": "critico" if nivel == "critica" or estricto else "aviso"})

    py = instalada("python")
    revisar("python", registradas.get("python"), py,
            _mayor_menor(py) == _mayor_menor(registradas.get("python", "")), "critica")
    for paquete, nivel in [(p, "critica") for p in CRITICAS] + [(p, "exacta") for p in EXACTAS]:
        if paquete not in registradas:
            continue
        actual = instalada(paquete)
        revisar(paquete, registradas[paquete], actual,
                _sin_local(actual) == _sin_local(registradas[paquete]), nivel)

    if any(d["nivel"] == "critico" for d in diferencias):
        estado = "incompatible"
    else:
        estado = "avisos" if diferencias else "ok"
    return {"estado": estado, "estricto": estricto, "diferencias": diferencias, "versiones": filas}


def exigir_compatibles(registradas, estricto=None):
    """Como comparar(), pero lanza VersionesIncompatibles si hay diferencias criticas."""
    reporte = comparar(registradas, estricto)
    if reporte["estado"] == "incompatible":
        detalle = "; ".join(f"{d['paquete']}: modelo {d['registrada']}, instalada {d['instalada']}"
                            for d in reporte["diferencias"] if d["nivel"] == "critico")
        raise VersionesIncompatibles(
            f"Versiones incompatibles con el modelo ({detalle}). Instala requirements_modelo.txt "
            "(pip install -r requirements_modelo.txt) o reentrena con el paso 8 en este entorno.")
    return reporte


def leer_requirements(ruta):
    """{paquete: version} de las lineas 'paquete==version' (sigue -r recursivamente)."""
    from pathlib import Path
    ruta = Path(ruta)
    versiones = {}
    for linea in ruta.read_text(encoding="utf-8").splitlines():
        linea = linea.split("#")[0].strip()
        if linea.startswith("-r "):
            versiones.update(leer_requirements(ruta.parent / linea[3:].strip()))
        elif "==" in linea:
            paquete, version = linea.split("==", 1)
            versiones[paquete.strip().lower()] = version.strip()
    return versiones
