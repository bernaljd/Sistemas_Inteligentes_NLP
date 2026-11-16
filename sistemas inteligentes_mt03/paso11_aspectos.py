"""
Paso 11 (imagen 3, punto 5): modelo de 9 aspectos con salida estructurada.

Misma disciplina que los pasos 5-8:
  1. DATOS: los 22 nombres de aspecto de la columna 'aspectos' se agrupan en
     9 aspectos (aspectos.ASPECTOS). Mismo split train/test del paso 3.
  2. VALIDACION CRUZADA (5 particiones, SOLO train) para elegir el diseno:
       presencia : regresion logistica por aspecto (TF-IDF + embeddings)
                   vs. linea base aleatoria con la prevalencia de train.
       sentimiento (sobre los pares reales comentario-aspecto):
         - "copiar polaridad": sentimiento del aspecto = polaridad REAL del
           comentario (linea base optimista: usa la etiqueta verdadera).
         - "mayoritaria": siempre la clase mas frecuente de train.
         - "comentario": regresion sobre [embedding del comentario, aspecto].
         - "fragmento+comentario": [fragmento relevante, comentario, aspecto].
  3. TEST (una sola vez): presencia P/R/F1 por aspecto, sentimiento y la
     exactitud de la salida estructurada completa.
  4. GUARDADO: modelo_aspectos.joblib + modelo_aspectos_metadatos.json
     (sha256 y versiones: los cargan carga_segura.py y compatibilidad.py).

Uso:  python paso11_aspectos.py
Salidas: modelo_aspectos.joblib, modelo_aspectos_metadatos.json,
         resultados_aspectos.csv, paso11_salida.txt
"""

import hashlib
import json
import platform
import sys
from datetime import datetime
from importlib import metadata
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.model_selection import KFold

from aspectos import ASPECTOS, CLAVES, ModeloAspectos, etiquetas
from pipeline_feedback import CLASES, MODELO_EMBEDDINGS, REVISION_EMBEDDINGS

CARPETA = Path(__file__).resolve().parent
SEMILLA = 42
VERSION_MODELO = "1.0.0"
DISENO = "fragmento+comentario"


class Duplicar:
    def __init__(self, *f):
        self.f = f

    def write(self, t):
        for x in self.f:
            x.write(t)

    def flush(self):
        for x in self.f:
            x.flush()


_salida = open(CARPETA / "paso11_salida.txt", "w", encoding="utf-8")
sys.stdout = Duplicar(sys.__stdout__, _salida)


def titulo(t):
    print("\n" + "=" * 80 + f"\n{t}\n" + "=" * 80)


def sha256(ruta):
    return hashlib.sha256(Path(ruta).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# 1. Datos
# ---------------------------------------------------------------------------

titulo("1. DATOS: 22 nombres de aspecto -> 9 aspectos")
train = pd.read_csv(CARPETA / "train.csv", sep=";")
test = pd.read_csv(CARPETA / "test.csv", sep=";")
for clave, (nombre, originales) in ASPECTOS.items():
    print(f"  {clave:<20} {nombre:<28} <- {', '.join(originales)}")

et_train = [etiquetas(a if isinstance(a, str) else "[]") for a in train["aspectos"]]
et_test = [etiquetas(a if isinstance(a, str) else "[]") for a in test["aspectos"]]
Y_train = np.array([[c in e for c in CLAVES] for e in et_train], dtype=int)
Y_test = np.array([[c in e for c in CLAVES] for e in et_test], dtype=int)
tabla = pd.DataFrame({"train": Y_train.sum(0), "test": Y_test.sum(0)}, index=CLAVES)
print("\nComentarios que mencionan cada aspecto:")
print(tabla.to_string())
print(f"Sin ningun aspecto: train {int((Y_train.sum(1) == 0).sum())}, test {int((Y_test.sum(1) == 0).sum())}")
X_train, X_test = train["feedback"].tolist(), test["feedback"].tolist()


def pares(et):
    return [(i, c, s) for i, e in enumerate(et) for c, s in e.items()]


pares_train = pares(et_train)
pol_train = train["polaridad_basica"].to_numpy()
coincide = np.mean([pol_train[i] == s for i, _, s in pares_train])
print(f"Pares comentario-aspecto en train: {len(pares_train)} | sentimiento == polaridad del comentario: "
      f"{coincide:.0%}")


# ---------------------------------------------------------------------------
# 2. Validacion cruzada (solo train)
# ---------------------------------------------------------------------------

titulo("2. VALIDACION CRUZADA (5 particiones, solo train)")
cv = KFold(n_splits=5, shuffle=True, random_state=SEMILLA)
rng = np.random.default_rng(SEMILLA)
P_cv = np.zeros(Y_train.shape)
P_azar = np.zeros(Y_train.shape)
sent_real = np.array([s for _, _, s in pares_train])
sent_pred = {k: np.empty(len(pares_train), dtype=object)
             for k in ["copiar polaridad", "mayoritaria", "comentario", "fragmento+comentario"]}
pi = np.array([i for i, _, _ in pares_train])

for k, (a, b) in enumerate(cv.split(X_train), 1):
    en_b = np.isin(pi, b)
    modelos = {}
    for diseno in ["comentario", "fragmento+comentario"]:
        m = ModeloAspectos(entrada_sentimiento=diseno).fit([X_train[i] for i in a], [et_train[i] for i in a])
        modelos[diseno] = m
        idx = np.where(en_b)[0]
        sent_pred[diseno][idx] = m.predecir_sentimiento([X_train[pi[n]] for n in idx],
                                                        [pares_train[n][1] for n in idx])
    m = modelos[DISENO]
    E = m._emb([X_train[i] for i in b])
    F = m._features([X_train[i] for i in b], E)
    P_cv[b] = np.column_stack([p.predict_proba(F)[:, 1] for p in m.presencia_])
    prevalencia = Y_train[a].mean(0)
    P_azar[b] = rng.random((len(b), len(CLAVES))) < prevalencia
    sent_pred["copiar polaridad"][en_b] = pol_train[pi[en_b]]
    mayor = pd.Series([s for i, _, s in pares_train if i in set(a)]).mode()[0]
    sent_pred["mayoritaria"][en_b] = mayor
    print(f"  particion {k} lista")

filas = []
f1_cv = [f1_score(Y_train[:, j], P_cv[:, j] >= 0.5) for j in range(len(CLAVES))]
f1_azar = [f1_score(Y_train[:, j], P_azar[:, j]) for j in range(len(CLAVES))]
print("\nPresencia (F1 por aspecto, umbral 0.5):")
for j, c in enumerate(CLAVES):
    print(f"  {c:<20} modelo {f1_cv[j]:.2f} | azar {f1_azar[j]:.2f}")
    filas.append({"evaluacion": "cv", "componente": "presencia", "aspecto": c, "f1": f1_cv[j],
                  "f1_linea_base_azar": f1_azar[j]})
print(f"  {'MACRO':<20} modelo {np.mean(f1_cv):.3f} | azar {np.mean(f1_azar):.3f}")

print("\nSentimiento sobre los pares reales (macro-F1):")
for nombre, pred in sent_pred.items():
    f = f1_score(sent_real, pred.astype(str), labels=CLASES, average="macro", zero_division=0)
    print(f"  {nombre:<22} macro-F1 = {f:.3f}  accuracy = {(pred == sent_real).mean():.3f}")
    filas.append({"evaluacion": "cv", "componente": f"sentimiento ({nombre})", "aspecto": "todos", "f1": f})
mejor = max(["comentario", "fragmento+comentario"],
            key=lambda d: f1_score(sent_real, sent_pred[d].astype(str), average="macro"))
print(f"\nDiseno elegido por CV: {mejor}")
assert mejor == DISENO, f"La CV prefiere '{mejor}': actualizar DISENO y el docstring de aspectos.py"


# ---------------------------------------------------------------------------
# 3. Test (una sola vez)
# ---------------------------------------------------------------------------

titulo("3. EVALUACION EN TEST (una sola vez)")
modelo = ModeloAspectos(entrada_sentimiento=DISENO).fit(X_train, et_train)
salida_test = modelo.predecir(X_test)
P_test = np.array([[a["probabilidad"] if a["probabilidad"] is not None else 0.0 for a in s] for s in salida_test])
M_test = np.array([[a["mencionado"] for a in s] for s in salida_test], dtype=int)
p, r, f, sop = precision_recall_fscore_support(Y_test, M_test, zero_division=0)
print("Presencia por aspecto:")
for j, c in enumerate(CLAVES):
    print(f"  {c:<20} P = {p[j]:.2f}  R = {r[j]:.2f}  F1 = {f[j]:.2f}  (n = {sop[j]})")
    filas.append({"evaluacion": "test", "componente": "presencia", "aspecto": c, "precision": p[j],
                  "recall": r[j], "f1": f[j], "soporte": int(sop[j])})
macro_presencia = float(np.mean(f))
print(f"  {'MACRO':<20} F1 = {macro_presencia:.3f}")

pares_test = pares(et_test)
st_real = np.array([s for _, _, s in pares_test])
st_pred = modelo.predecir_sentimiento([X_test[i] for i, _, _ in pares_test], [c for _, c, _ in pares_test])
macro_sent = f1_score(st_real, st_pred, labels=CLASES, average="macro")
st_pol = test["polaridad_basica"].to_numpy()[[i for i, _, _ in pares_test]]
macro_pol = f1_score(st_real, st_pol, labels=CLASES, average="macro")
print(f"\nSentimiento en los {len(pares_test)} pares reales de test: macro-F1 = {macro_sent:.3f} "
      f"(copiar la polaridad real del comentario: {macro_pol:.3f})")
filas.append({"evaluacion": "test", "componente": f"sentimiento ({DISENO})", "aspecto": "todos", "f1": macro_sent})
filas.append({"evaluacion": "test", "componente": "sentimiento (copiar polaridad)", "aspecto": "todos",
              "f1": macro_pol})

# Salida estructurada completa. Metrica principal: P/R/F1 de las TUPLAS
# (comentario, aspecto, sentimiento): una tupla predicha es correcta si el
# comentario si habla de ese aspecto CON ese sentimiento. No predecir nada da 0.
# (La exactitud por celda se reporta junto a su linea base trivial: como casi
# todas las celdas son "no mencionado", no detectar nunca nada ya acierta ~85 %.)
reales = {(i, c, s) for i, c, s in pares_test}
predichas = {(i, a["aspecto"], a["sentimiento"]) for i, s in enumerate(salida_test) for a in s if a["mencionado"]}
tp = len(reales & predichas)
tuplas_p = tp / len(predichas) if predichas else 0.0
tuplas_r = tp / len(reales)
tuplas_f1 = 2 * tuplas_p * tuplas_r / (tuplas_p + tuplas_r) if tp else 0.0
celdas = len(salida_test) * len(CLAVES)
correctas = sum((et_test[i].get(c) is None and not a["mencionado"]) or
                (a["mencionado"] and et_test[i].get(c) == a["sentimiento"])
                for i, s in enumerate(salida_test) for c, a in zip(CLAVES, s))
exactitud_celdas = correctas / celdas
base_celdas = 1 - len(reales) / celdas
print(f"Salida estructurada, tuplas (comentario, aspecto, sentimiento): P = {tuplas_p:.3f}  R = {tuplas_r:.3f}  "
      f"F1 = {tuplas_f1:.3f}  ({tp} correctas de {len(predichas)} predichas y {len(reales)} reales)")
print(f"Exactitud por celda: {exactitud_celdas:.1%} (linea base 'nunca mencionado': {base_celdas:.1%}; "
      "no es buena medida aqui)")
filas.append({"evaluacion": "test", "componente": "salida estructurada: tuplas (aspecto, sentimiento)",
              "aspecto": "todos", "precision": tuplas_p, "recall": tuplas_r, "f1": tuplas_f1})
filas.append({"evaluacion": "test", "componente": "salida estructurada: exactitud por celda", "aspecto": "todos",
              "f1": exactitud_celdas, "f1_linea_base_azar": base_celdas})
pd.DataFrame(filas).round(4).to_csv(CARPETA / "resultados_aspectos.csv", index=False, sep=";")

ejemplo = "El profesor sabe mucho y explica muy bien, pero los parciales son injustos y nunca responde los correos."
print(f"\nEjemplo: {ejemplo}")
for a in modelo.predecir([ejemplo])[0]:
    if a["mencionado"]:
        print(f"  {a['aspecto']:<20} p = {a['probabilidad']:.2f}  sentimiento = {a['sentimiento']}")


# ---------------------------------------------------------------------------
# 4. Guardado (con huella y versiones para carga_segura.py y compatibilidad.py)
# ---------------------------------------------------------------------------

titulo("4. GUARDADO")
paquetes = ["scikit-learn", "joblib", "numpy", "pandas", "scipy", "sentence-transformers", "torch", "transformers"]
versiones = {"python": platform.python_version(), "sistema": platform.platform()}
for paq in paquetes:
    versiones[paq] = metadata.version(paq)
metadatos = {
    "nombre": f"Aspectos (9) - presencia + sentimiento ({DISENO})",
    "version_modelo": VERSION_MODELO,
    "fecha": datetime.now().isoformat(timespec="seconds"),
    "tarea": "Para cada comentario, una prediccion por cada uno de los 9 aspectos: mencionado y sentimiento",
    "aspectos": {c: {"nombre": n, "originales": o} for c, (n, o) in ASPECTOS.items()},
    "formato_salida": {"aspecto": "clave", "nombre": "texto", "mencionado": "bool",
                       "probabilidad": "float | null", "sentimiento": "negativo | neutral | positivo | null",
                       "probabilidades_sentimiento": "{clase: float} | null"},
    "metricas": {
        "presencia_macro_f1_cv": float(np.mean(f1_cv)),
        "presencia_macro_f1_test": macro_presencia,
        "presencia_f1_test": {c: float(f[j]) for j, c in enumerate(CLAVES)},
        "sentimiento_macro_f1_test": float(macro_sent),
        "sentimiento_copiar_polaridad_macro_f1_test": float(macro_pol),
        "salida_estructurada_tuplas_test": {"precision": tuplas_p, "recall": tuplas_r, "f1": tuplas_f1},
        "salida_estructurada_exactitud_celdas_test": exactitud_celdas,
        "linea_base_nunca_mencionado_celdas_test": base_celdas,
    },
    "parametros": {"umbral_mencion": modelo.umbral, "C_presencia": modelo.C_presencia,
                   "C_sentimiento": modelo.C_sentimiento, "entrada_sentimiento": DISENO,
                   "embeddings": {"modelo": MODELO_EMBEDDINGS, "revision": REVISION_EMBEDDINGS}},
    "datos": {"train.csv": {"sha256": sha256(CARPETA / "train.csv"), "filas": len(train)},
              "test.csv": {"sha256": sha256(CARPETA / "test.csv"), "filas": len(test)}},
    "entrenado_con": "train.csv; test.csv solo para la evaluacion final",
    "versiones": versiones,
    "dependencia_codigo": "aspectos.py y pipeline_feedback.py deben estar importables al cargar el .joblib",
}
joblib.dump({"modelo": modelo, "metadatos": metadatos}, CARPETA / "modelo_aspectos.joblib", compress=3)
metadatos["sha256_modelo_aspectos.joblib"] = sha256(CARPETA / "modelo_aspectos.joblib")
with open(CARPETA / "modelo_aspectos_metadatos.json", "w", encoding="utf-8") as fh:
    json.dump(metadatos, fh, ensure_ascii=False, indent=2)
tam = (CARPETA / "modelo_aspectos.joblib").stat().st_size / 1024
print(f"Guardados: modelo_aspectos.joblib ({tam:.0f} KB), modelo_aspectos_metadatos.json,")
print("           resultados_aspectos.csv, paso11_salida.txt")
sys.stdout = sys.__stdout__
_salida.close()
