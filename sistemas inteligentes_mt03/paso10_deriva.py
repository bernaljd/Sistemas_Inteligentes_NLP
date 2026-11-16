"""
Paso 10b: Referencia de monitoreo y prueba de deteccion de deriva.

1. REFERENCIA (monitoreo/referencia.json): como se comporta el modelo con
   comentarios que NO vio. Se usan predicciones fuera de muestra de train
   (validacion cruzada 5-fold, mismas particiones e hiperparametros del paso 6):
   en cada particion el TF-IDF y la regresion se ajustan sin el comentario, asi
   la confianza y el % de palabras desconocidas (oov) son realistas.
   (Las predicciones del modelo final sobre train no sirven: las memoriza, F1 = 1.0.)

2. ESCENARIOS: se pasan por el modelo final distintos lotes y se mide si el
   monitor detecta el cambio:
     A. test.csv (comentarios reales no vistos)       -> deberia salir ESTABLE
     B. comentarios recortados a 4 palabras           -> cambio de longitud
     C. comentarios en ingles                          -> vocabulario desconocido
     D. solo comentarios negativos                     -> cambio en las clases
     E. 40% de comentarios vacios                      -> problema de captura

Salidas: monitoreo/referencia.json, resultados_deriva.csv, paso10_deriva_salida.txt
"""

import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

import monitoreo
from carga_segura import cargar_joblib_confiable
from pipeline_feedback import UMBRAL_CONFIANZA, limpiar_texto, predecir

CARPETA = Path(__file__).resolve().parent
ETIQUETA, SEMILLA = "polaridad_basica", 42


class Duplicar:
    def __init__(self, *f):
        self.f = f

    def write(self, t):
        for x in self.f:
            x.write(t)

    def flush(self):
        for x in self.f:
            x.flush()


_salida = open(CARPETA / "paso10_deriva_salida.txt", "w", encoding="utf-8")
sys.stdout = Duplicar(sys.__stdout__, _salida)

paquete = cargar_joblib_confiable(CARPETA / "modelo_final.joblib")  # verifica sha256 antes de cargar
modelo, meta = paquete["pipeline"], paquete["metadatos"]
p = meta["parametros"]
train = pd.read_csv(CARPETA / "train.csv", sep=";")
test = pd.read_csv(CARPETA / "test.csv", sep=";")


# ---------------------------------------------------------------------------
# 1. Referencia con predicciones fuera de muestra (sin recalcular embeddings)
# ---------------------------------------------------------------------------

print("=" * 80 + "\n1. REFERENCIA (validacion cruzada 5-fold sobre train)\n" + "=" * 80)
X_emb = np.load(CARPETA / "embeddings_train.npy")
cols = [f"emb_{i}" for i in range(X_emb.shape[1])]
X = pd.concat([train[["texto_limpio"]], pd.DataFrame(X_emb, columns=cols)], axis=1)
y = train[ETIQUETA]

filas_ref = [None] * len(train)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)
for k, (i_tr, i_va) in enumerate(cv.split(X, y), 1):
    m = Pipeline([
        ("prep", ColumnTransformer([
            ("tfidf", TfidfVectorizer(ngram_range=tuple(p["tfidf"]["ngram_range"]), min_df=p["tfidf"]["min_df"],
                                      sublinear_tf=True), "texto_limpio"),
            ("emb", "passthrough", cols)])),
        ("clf", LogisticRegression(C=p["clasificador"]["C"], class_weight="balanced", max_iter=5000)),
    ]).fit(X.iloc[i_tr], y.iloc[i_tr])
    vocab = {t for t in m.named_steps["prep"].named_transformers_["tfidf"].vocabulary_ if " " not in t}
    probas = m.predict_proba(X.iloc[i_va])
    clases = list(m.classes_)
    for i, pr in zip(i_va, probas):
        palabras = train["texto_limpio"].iloc[i].split()
        j = int(np.argmax(pr))
        filas_ref[i] = {
            "n_palabras": len(palabras),
            "oov": sum(w not in vocab for w in palabras) / len(palabras),
            "estado": "ok" if pr[j] >= UMBRAL_CONFIANZA else "ambiguo",
            "prediccion": clases[j], "confianza": float(pr[j]),
        }
    print(f"  particion {k}: {len(i_va)} comentarios")

ref = monitoreo.construir_referencia(filas_ref, "train.csv, predicciones fuera de muestra (CV 5-fold)")
(CARPETA / "monitoreo").mkdir(exist_ok=True)
(CARPETA / "monitoreo" / "referencia.json").write_text(json.dumps(ref, indent=2, ensure_ascii=False),
                                                       encoding="utf-8")
for var, r in ref["variables"].items():
    print(f"  {var:<11} media {r['media']:.3f}  cortes {[round(c, 3) for c in r['cortes']]}")
print(f"  clases predichas {({c: round(v, 3) for c, v in ref['clases'].items()})}")
print(f"  tasas {({c: round(v, 3) for c, v in ref['tasas'].items()})}")
print("  Guardada: monitoreo/referencia.json")


# ---------------------------------------------------------------------------
# 2. Escenarios
# ---------------------------------------------------------------------------

rng = random.Random(SEMILLA)
textos_test = test["feedback"].tolist()

sujetos = ["The professor", "The teacher", "This course", "The class", "The lecturer", "The instructor"]
predicados = ["explains everything very clearly", "was really boring and confusing",
              "gave us useful feedback on the projects", "never answered our questions",
              "made the exams too difficult", "is well organized and engaging",
              "helped me understand distributed systems", "should improve the grading criteria"]
cierres = ["", " I would recommend it.", " Overall it was fine.", " I learned a lot.", " Not worth it."]
ingles = [f"{rng.choice(sujetos)} {rng.choice(predicados)}.{rng.choice(cierres)}" for _ in range(120)]

negativos = test.loc[test[ETIQUETA] == "negativo", "feedback"].tolist()
con_vacios = [("" if rng.random() < 0.4 else t) for t in textos_test]

escenarios = {
    "A. test.csv (sin cambios)": textos_test,
    "B. recortados a 4 palabras": [" ".join(t.split()[:4]) for t in textos_test],
    "C. comentarios en ingles": ingles,
    "D. solo negativos": negativos,
    "E. 40% vacios": con_vacios,
}

vocab_final = monitoreo.vocabulario(modelo)
filas_tabla = []
print("\n" + "=" * 80 + "\n2. ESCENARIOS\n" + "=" * 80)
for nombre, textos in escenarios.items():
    df = predecir(modelo, textos)
    filas = monitoreo.caracteristicas(textos, df, vocab_final)
    rep = monitoreo.reporte_deriva(filas, ref)
    ind = rep["indicadores"]
    print(f"\n{nombre}  (n = {rep['n']})  ->  estado: {rep['estado'].upper()}")
    for k, v in ind.items():
        detalle = f"PSI {v['psi']:.3f}" if "psi" in v else f"{v['referencia']:.2f} -> {v['actual']:.2f}"
        if "media_actual" in v:
            detalle += f"  (media {v['media_referencia']:.2f} -> {v['media_actual']:.2f})"
        print(f"    {k:<18} {v['nivel']:<10} {detalle}")
    filas_tabla.append({"escenario": nombre, "n": rep["n"], "estado": rep["estado"],
                        **{f"psi_{k}": v.get("psi") for k, v in ind.items() if "psi" in v},
                        **{f"{k}_actual": v["actual"] for k, v in ind.items() if k.startswith("tasa_")},
                        "alertas": " | ".join(rep["alertas"])})

tabla = pd.DataFrame(filas_tabla)
tabla.to_csv(CARPETA / "resultados_deriva.csv", index=False, sep=";")

print("\n" + "=" * 80 + "\nRESUMEN\n" + "=" * 80)
print(tabla[["escenario", "estado"] + [c for c in tabla if c.startswith("psi_")]].round(3).to_string(index=False))
esperado = {"A": "estable"}
ok_a = tabla.iloc[0]["estado"] == "estable"
ok_resto = (tabla.iloc[1:]["estado"] != "estable").all()
print(f"\n  Escenario sin cambios marcado como estable: {'SI' if ok_a else 'NO'}")
print(f"  Escenarios con cambios detectados: {(tabla.iloc[1:]['estado'] != 'estable').sum()}/{len(tabla) - 1}")
print("\nGuardados: monitoreo/referencia.json, resultados_deriva.csv, paso10_deriva_salida.txt")
sys.stdout = sys.__stdout__
_salida.close()
