"""
Paso 10a: Tiempos de inferencia y uso de memoria del servicio.

Mide la API completa (validacion, modelo, serializacion JSON y registro de
monitoreo) con TestClient, en este computador y en CPU:
  1. Arranque en frio: cargar el modelo + calentamiento.
  2. Latencia de un comentario: 300 peticiones -> media, p50, p95, p99.
  3. Lotes de 1, 10, 50, 100, 500 y 1000 comentarios: ms por lote y comentarios/s.
  4. Endpoint CSV con 1000 filas.
  5. Memoria del proceso (RSS) y tamano del modelo.
Compara con objetivos de servicio definidos de antemano (OBJETIVOS).

Salidas: resultados_rendimiento.csv, paso10_rendimiento_salida.txt
"""

import os
import platform
import sys
import tempfile
import time
from pathlib import Path

CARPETA = Path(__file__).resolve().parent
os.environ["RATE_LIMIT"] = "0"                                  # sin limite durante la medicion
os.environ["RUTA_REGISTRO"] = str(Path(tempfile.mkdtemp()) / "registro.jsonl")  # no ensucia el real

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import psutil  # noqa: E402

# Objetivos fijados ANTES de medir (uso interactivo de una pagina web)
OBJETIVOS = {
    "p95 un comentario (ms)": 500,
    "lote de 100 (s)": 5,
    "CSV de 1000 filas (s)": 60,
    "arranque (s)": 120,
    "memoria RSS (MB)": 2048,
}


class Duplicar:
    def __init__(self, *f):
        self.f = f

    def write(self, t):
        for x in self.f:
            x.write(t)

    def flush(self):
        for x in self.f:
            x.flush()


_salida = open(CARPETA / "paso10_rendimiento_salida.txt", "w", encoding="utf-8")
sys.stdout = Duplicar(sys.__stdout__, _salida)
proceso = psutil.Process()
mb = lambda: proceso.memory_info().rss / 1024 ** 2  # noqa: E731

print(f"CPU: {platform.processor()} | nucleos logicos: {psutil.cpu_count()} | "
      f"RAM: {psutil.virtual_memory().total / 1024 ** 3:.1f} GB | Python {platform.python_version()}")
print(f"Tamano de modelo_final.joblib: {(CARPETA / 'modelo_final.joblib').stat().st_size / 1024:.0f} KB "
      "(+ ~470 MB del modelo MiniLM en la cache de Hugging Face)")
mem0 = mb()

# 1. Arranque en frio ---------------------------------------------------------
t0 = time.perf_counter()
from fastapi.testclient import TestClient  # noqa: E402

from api import app  # noqa: E402

cliente = TestClient(app)
cliente.__enter__()
arranque = time.perf_counter() - t0
mem_cargado = mb()
print(f"\n1. Arranque (importar + cargar + calentar): {arranque:.1f} s | memoria {mem0:.0f} -> {mem_cargado:.0f} MB")

test = pd.read_csv(CARPETA / "test.csv", sep=";")
textos = test["feedback"].tolist()
filas = []

# 2. Latencia de un comentario ------------------------------------------------
for t in textos[:10]:  # calentamiento extra
    cliente.post("/predecir", json={"texto": t})
lat = []
for i in range(300):
    t = textos[i % len(textos)]
    t0 = time.perf_counter()
    r = cliente.post("/predecir", json={"texto": t})
    lat.append((time.perf_counter() - t0) * 1000)
    assert r.status_code == 200
lat = np.array(lat)
print(f"\n2. Un comentario (300 peticiones): media {lat.mean():.0f} ms | p50 {np.percentile(lat, 50):.0f} | "
      f"p95 {np.percentile(lat, 95):.0f} | p99 {np.percentile(lat, 99):.0f} | max {lat.max():.0f} ms")
filas.append({"prueba": "un comentario", "n": 1, "media_ms": lat.mean(), "p50_ms": np.percentile(lat, 50),
              "p95_ms": np.percentile(lat, 95), "p99_ms": np.percentile(lat, 99)})

# Largo del texto
largo = " ".join(textos) [:5000]
t_largo = []
for _ in range(20):
    t0 = time.perf_counter()
    cliente.post("/predecir", json={"texto": largo})
    t_largo.append((time.perf_counter() - t0) * 1000)
print(f"   Comentario de 5000 caracteres (maximo permitido): mediana {np.median(t_largo):.0f} ms")
filas.append({"prueba": "comentario de 5000 caracteres", "n": 1, "media_ms": np.mean(t_largo),
              "p50_ms": np.median(t_largo)})

# 3. Lotes -------------------------------------------------------------------
print("\n3. Lotes:")
lote100 = None
for n in [1, 10, 50, 100, 500, 1000]:
    lote = (textos * (n // len(textos) + 1))[:n]
    tiempos = []
    for _ in range(3 if n <= 100 else 1):
        t0 = time.perf_counter()
        r = cliente.post("/predecir/lote", json={"textos": lote})
        tiempos.append((time.perf_counter() - t0) * 1000)
        assert r.status_code == 200 and r.json()["n"] == n
    ms = float(np.median(tiempos))
    lote100 = ms if n == 100 else lote100
    print(f"   {n:>5} comentarios: {ms:>8.0f} ms  ({ms / n:6.1f} ms/comentario, {n / ms * 1000:6.1f} comentarios/s)")
    filas.append({"prueba": "lote", "n": n, "media_ms": ms, "ms_por_comentario": ms / n,
                  "comentarios_por_s": n / ms * 1000})

# 4. CSV ---------------------------------------------------------------------
csv = "id;feedback\n" + "\n".join(f"{i};{(textos * 9)[i]}" for i in range(1000))
t0 = time.perf_counter()
r = cliente.post("/predecir/csv", files={"archivo": ("c.csv", csv.encode("utf-8"), "text/csv")})
t_csv = time.perf_counter() - t0
assert r.status_code == 200, r.text
print(f"\n4. CSV de 1000 filas: {t_csv:.1f} s")
filas.append({"prueba": "csv", "n": 1000, "media_ms": t_csv * 1000})

# 5. Memoria -----------------------------------------------------------------
mem_final = mb()
print(f"\n5. Memoria RSS: tras cargar {mem_cargado:.0f} MB | tras todas las pruebas {mem_final:.0f} MB")

# Objetivos ------------------------------------------------------------------
medido = {
    "p95 un comentario (ms)": np.percentile(lat, 95),
    "lote de 100 (s)": lote100 / 1000,
    "CSV de 1000 filas (s)": t_csv,
    "arranque (s)": arranque,
    "memoria RSS (MB)": mem_final,
}
print("\n" + "=" * 70 + "\nOBJETIVOS DE SERVICIO\n" + "=" * 70)
for k, lim in OBJETIVOS.items():
    print(f"  {k:<26} medido {medido[k]:>8.1f}   objetivo <= {lim:<6} {'CUMPLE' if medido[k] <= lim else 'NO CUMPLE'}")
    filas.append({"prueba": f"objetivo: {k}", "medido": medido[k], "objetivo": lim, "cumple": medido[k] <= lim})

pd.DataFrame(filas).round(2).to_csv(CARPETA / "resultados_rendimiento.csv", index=False, sep=";")
print("\nGuardados: resultados_rendimiento.csv, paso10_rendimiento_salida.txt")
cliente.__exit__(None, None, None)
sys.stdout = sys.__stdout__
_salida.close()
