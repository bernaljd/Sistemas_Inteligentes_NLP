"""
Paso 7: Evaluacion final y comparacion sobre test.

REGLAS (se cumplen en este script):
  - test.csv se usa UNA sola vez y solo para evaluar. No se reajustan
    hiperparametros, no se cambia de modelo ni de preprocesamiento.
  - El modelo principal es el de modelo_paso6.joblib (LogReg + TF-IDF+Emb).
    Los otros candidatos y los Dummy son solo comparacion.
  - Las cifras de validacion cruzada (CV) se leen de resultados_paso5.csv y
    resultados_paso6.csv; los mejores parametros de los candidatos tambien.

Que hace:
  1. Carga train/test y verifica que no haya fuga (ids y textos disjuntos).
  2. Calcula los embeddings de test con el MISMO modelo y parametros del paso 4
     (feedback, normalize_embeddings=True, 384 dim.). Antes, recalcula los de
     train y los compara con embeddings_train.npy para comprobar que el modelo
     de embeddings de este computador produce el mismo espacio vectorial.
  3. Carga el modelo principal (si no carga, lo reentrena con train y avisa).
  4. Reentrena con train los otros dos candidatos (mejores parametros del paso 6)
     y los dos Dummy.
  5. Evalua todos en test: accuracy, precision/recall/F1 por clase, macro-F1,
     e intervalo de confianza del 95% por bootstrap (1000 remuestreos).
  6. Compara macro-F1 de test con el de CV.
  7. Guarda la matriz de confusion (conteos y normalizada por fila) en PNG,
     el analisis de errores y un registro con versiones (CL12).

Uso (en la carpeta donde estan train.csv, test.csv, etc.):
    python paso7_evaluacion.py
o indicando la carpeta:
    python paso7_evaluacion.py --carpeta "C:\\ruta\\a\\sistemas inteligentes"

Salidas (en la misma carpeta):
  - resultados_paso7.csv          : una fila por modelo (sep=";").
  - errores_test.csv              : comentarios mal clasificados por el modelo principal (sep=";").
  - predicciones_test.csv         : prediccion de cada modelo para los 123 comentarios (sep=";").
  - matriz_confusion_paso7.png    : matriz de confusion del modelo principal.
  - embeddings_test.npy           : embeddings de test (para reproducir).
  - registro_modelo_paso7.json    : metricas, parametros, huellas de los datos y versiones.
  - paso7_salida.txt              : copia de todo lo que se imprime en consola.
"""

import argparse
import ast
import hashlib
import json
import platform
import sys
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

import matplotlib
matplotlib.use("Agg")  # sin ventanas: solo guarda el PNG
import matplotlib.pyplot as plt

ETIQUETA = "polaridad_basica"
SEMILLA = 42
CLASES = ["negativo", "neutral", "positivo"]
N_BOOTSTRAP = 1000
MODELO_EMBEDDINGS = "paraphrase-multilingual-MiniLM-L12-v2"  # el mismo del paso 4
NOMBRE_PRINCIPAL = "LogReg + TF-IDF+Emb"

parser = argparse.ArgumentParser(description="Paso 7: evaluacion final sobre test")
parser.add_argument("--carpeta", default=str(Path(__file__).resolve().parent),
                    help="Carpeta con train.csv, test.csv, embeddings_train.npy, modelo_paso6.joblib, ...")
CARPETA = Path(parser.parse_args().carpeta)


# ---------------------------------------------------------------------------
# Todo lo que se imprime tambien queda en paso7_salida.txt
# ---------------------------------------------------------------------------

class Duplicar:
    def __init__(self, *flujos):
        self.flujos = flujos

    def write(self, texto):
        for f in self.flujos:
            f.write(texto)

    def flush(self):
        for f in self.flujos:
            f.flush()


_salida = open(CARPETA / "paso7_salida.txt", "w", encoding="utf-8")
sys.stdout = Duplicar(sys.__stdout__, _salida)


def titulo(texto):
    print("\n" + "=" * 90)
    print(texto)
    print("=" * 90)


def huella(ruta):
    return hashlib.sha256(Path(ruta).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# 0. Versiones y huellas de los datos (CL12: registrar versiones y datos)
# ---------------------------------------------------------------------------

titulo("0. VERSIONES Y DATOS")
import sentence_transformers  # noqa: E402  (se importa aqui para registrar su version)

versiones = {
    "python": platform.python_version(),
    "sistema": platform.platform(),
    "scikit-learn": sklearn.__version__,
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "joblib": joblib.__version__,
    "sentence-transformers": sentence_transformers.__version__,
}
try:
    import torch
    versiones["torch"] = torch.__version__
except ImportError:
    pass
for k, v in versiones.items():
    print(f"  {k:<22} {v}")

huellas = {f: huella(CARPETA / f) for f in
           ["train.csv", "test.csv", "embeddings_train.npy", "modelo_paso6.joblib"]}
for f, h in huellas.items():
    print(f"  sha256 {f:<22} {h[:16]}...")


# ---------------------------------------------------------------------------
# 1. Datos
# ---------------------------------------------------------------------------

titulo("1. DATOS")
train = pd.read_csv(CARPETA / "train.csv", sep=";")
test = pd.read_csv(CARPETA / "test.csv", sep=";")
X_emb_train = np.load(CARPETA / "embeddings_train.npy")
res5 = pd.read_csv(CARPETA / "resultados_paso5.csv", sep=";")
res6 = pd.read_csv(CARPETA / "resultados_paso6.csv", sep=";")

assert len(train) == len(X_emb_train), "embeddings_train.npy no coincide con train.csv"
assert not set(train["id"]) & set(test["id"]), "Hay ids repetidos entre train y test"
assert not set(train["feedback"]) & set(test["feedback"]), "Hay comentarios repetidos entre train y test"
for nombre, df in [("train", train), ("test", test)]:
    faltan = df[[ETIQUETA, "texto_limpio", "feedback"]].isna().sum().sum()
    assert faltan == 0, f"{nombre}.csv tiene {faltan} valores vacios en etiqueta/texto"
    assert set(df[ETIQUETA]) == set(CLASES), f"Clases inesperadas en {nombre}: {set(df[ETIQUETA])}"

y_train, y_test = train[ETIQUETA], test[ETIQUETA]
print(f"Train: {len(train)} | Test: {len(test)}")
print("Clases en test:", y_test.value_counts().reindex(CLASES).to_dict())
print("Contraste en test:", test["contraste"].value_counts().to_dict())

cols_emb = [f"emb_{i}" for i in range(X_emb_train.shape[1])]


# ---------------------------------------------------------------------------
# 2. Embeddings de test (mismo modelo y mismos parametros que en el paso 4)
# ---------------------------------------------------------------------------

titulo("2. EMBEDDINGS")
from sentence_transformers import SentenceTransformer  # noqa: E402

print(f"Cargando {MODELO_EMBEDDINGS} (la primera vez lo descarga)...")
modelo_st = SentenceTransformer(MODELO_EMBEDDINGS)


def embeddings(textos):
    # Igual que vectorizacion.py: texto original (feedback), batch 32, normalizados
    return modelo_st.encode(list(textos), batch_size=32, show_progress_bar=True,
                            normalize_embeddings=True)


# Comprobacion: los embeddings de train recalculados aqui deben coincidir con
# los guardados en el paso 4. Si no coinciden, el modelo de embeddings de este
# computador no es el mismo y los de test no serian comparables.
X_emb_train_recalc = embeddings(train["feedback"])
cos = np.sum(X_emb_train_recalc * X_emb_train, axis=1)  # ambos tienen norma 1
print(f"Similitud coseno train recalculado vs. guardado: min = {cos.min():.6f}, media = {cos.mean():.6f}")
if cos.min() < 0.999:
    raise SystemExit("ALTO: los embeddings recalculados no coinciden con embeddings_train.npy. "
                     "Revisa la version de sentence-transformers / del modelo antes de evaluar.")

X_emb_test = embeddings(test["feedback"])
assert X_emb_test.shape == (len(test), X_emb_train.shape[1]), f"Forma inesperada: {X_emb_test.shape}"
np.save(CARPETA / "embeddings_test.npy", X_emb_test)
print(f"Embeddings de test: {X_emb_test.shape} -> guardados en embeddings_test.npy")


def tabla_combo(df, X):
    # Mismo formato que en ajuste.py: texto_limpio + emb_0 ... emb_383
    return pd.concat([df[["texto_limpio"]].reset_index(drop=True),
                      pd.DataFrame(X, columns=cols_emb)], axis=1)


X_combo_train = tabla_combo(train, X_emb_train)
X_combo_test = tabla_combo(test, X_emb_test)


# ---------------------------------------------------------------------------
# 3. Modelos: mismas estructuras que en ajuste.py + mejores parametros del paso 6
# ---------------------------------------------------------------------------

def prep_combo():
    return ColumnTransformer([
        ("tfidf", TfidfVectorizer(sublinear_tf=True), "texto_limpio"),
        ("emb", "passthrough", cols_emb),
    ])


estructuras = {
    "LogReg + TF-IDF+Emb": lambda: Pipeline([
        ("prep", prep_combo()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=5000))]),
    "SVM + TF-IDF+Emb": lambda: Pipeline([
        ("prep", prep_combo()),
        ("clf", LinearSVC(class_weight="balanced", max_iter=20000, random_state=SEMILLA))]),
    "SVM + Embeddings": lambda: Pipeline([
        ("clf", LinearSVC(max_iter=20000, random_state=SEMILLA))]),
}
mejores = {f["candidato"]: ast.literal_eval(f["mejores_parametros"]) for _, f in res6.iterrows()}
cv6 = res6.set_index("candidato")


def construir(nombre):
    return estructuras[nombre]().set_params(**mejores[nombre])


titulo("3. MODELO PRINCIPAL")
print(f"Parametros del paso 6 para {NOMBRE_PRINCIPAL}: {mejores[NOMBRE_PRINCIPAL]}")
principal, origen_principal = None, ""
try:
    with warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter("always")
        paquete = joblib.load(CARPETA / "modelo_paso6.joblib")  # archivo propio y de confianza
    for a in avisos:
        print(f"  AVISO al cargar: {a.category.__name__}: {a.message}")
    principal = paquete["modelo"]
    assert paquete["nombre"] == NOMBRE_PRINCIPAL, f"El joblib contiene '{paquete['nombre']}'"
    assert paquete["modelo_embeddings"] == MODELO_EMBEDDINGS
    # El modelo cargado debe tener los hiperparametros indicados
    p = principal.get_params()
    esperado = {"clf__C": 10, "clf__class_weight": "balanced", "prep__tfidf__ngram_range": (1, 2),
                "prep__tfidf__min_df": 2, "prep__tfidf__sublinear_tf": True}
    distintos = {k: p[k] for k in esperado if p[k] != esperado[k]}
    assert not distintos, f"Hiperparametros distintos a los esperados: {distintos}"
    # Debe dar las mismas probabilidades que un reentrenamiento con los mismos datos y parametros
    control = construir(NOMBRE_PRINCIPAL).fit(X_combo_train, y_train)
    dif = np.abs(principal.predict_proba(X_combo_train) - control.predict_proba(X_combo_train)).max()
    print(f"  Diferencia max. de probabilidades (cargado vs. reentrenado, en train): {dif:.2e}")
    assert dif < 1e-3, "El modelo cargado no se comporta como el del paso 6"
    origen_principal = "cargado de modelo_paso6.joblib"
    print("  OK: modelo cargado de modelo_paso6.joblib")
except Exception as e:  # incluye fallos por diferencia de version de scikit-learn
    print(f"\n  AVISO IMPORTANTE: no se pudo usar modelo_paso6.joblib ({type(e).__name__}: {e}).")
    print("  Se REENTRENA con train.csv, los hiperparametros del paso 6 y los embeddings")
    print("  de train recalculados con el mismo modelo (regla 5).")
    principal = construir(NOMBRE_PRINCIPAL).fit(tabla_combo(train, X_emb_train_recalc), y_train)
    origen_principal = "REENTRENADO (modelo_paso6.joblib no se pudo usar)"

f1_train_principal = f1_score(y_train, principal.predict(X_combo_train), average="macro")
print(f"  Macro-F1 del modelo principal sobre train (memorizacion): {f1_train_principal:.3f}")

titulo("4. MODELOS DE COMPARACION (reentrenados con train)")
modelos = {NOMBRE_PRINCIPAL: (principal, X_combo_test, "principal")}
for nombre in ["SVM + TF-IDF+Emb", "SVM + Embeddings"]:
    X_tr, X_te = (X_emb_train, X_emb_test) if nombre == "SVM + Embeddings" else (X_combo_train, X_combo_test)
    modelos[nombre] = (construir(nombre).fit(X_tr, y_train), X_te, "candidato (comparacion)")
    print(f"  {nombre}: {mejores[nombre]}")

# Los Dummy no miran el texto: se ajustan solo con la distribucion de clases de train
X_vacio_train, X_vacio_test = np.zeros((len(train), 1)), np.zeros((len(test), 1))
for nombre, estrategia in [("Dummy (clase mayoritaria)", "most_frequent"),
                           ("Dummy (aleatorio estratificado)", "stratified")]:
    d = DummyClassifier(strategy=estrategia, random_state=SEMILLA).fit(X_vacio_train, y_train)
    modelos[nombre] = (d, X_vacio_test, "linea base")
    print(f"  {nombre}")


# ---------------------------------------------------------------------------
# 5. Evaluacion en test (unica vez)
# ---------------------------------------------------------------------------

titulo("5. EVALUACION EN TEST (n = %d)" % len(test))
predicciones = {n: np.asarray(m.predict(X)) for n, (m, X, _) in modelos.items()}
y_true = y_test.to_numpy()

# Mismos remuestreos para todos los modelos (comparacion pareada)
rng = np.random.default_rng(SEMILLA)
remuestreos = rng.integers(0, len(y_true), size=(N_BOOTSTRAP, len(y_true)))


def f1_bootstrap(pred):
    return np.array([f1_score(y_true[i], pred[i], labels=CLASES, average="macro", zero_division=0)
                     for i in remuestreos])


boot = {n: f1_bootstrap(p) for n, p in predicciones.items()}


def cv_de(nombre):
    if nombre in cv6.index:
        return cv6.loc[nombre, "macro_f1_media"], cv6.loc[nombre, "macro_f1_desv"], "paso 6"
    f = res5[res5["modelo"] == nombre].iloc[0]  # los Dummy no dependen de la representacion
    return f["macro_f1_media"], f["macro_f1_desv"], "paso 5"


filas = []
for nombre, pred in predicciones.items():
    p, r, f, s = precision_recall_fscore_support(y_true, pred, labels=CLASES, zero_division=0)
    mp, mr, mf, _ = precision_recall_fscore_support(y_true, pred, labels=CLASES, average="macro",
                                                    zero_division=0)
    cv_media, cv_desv, cv_fuente = cv_de(nombre)
    ic_inf, ic_sup = np.percentile(boot[nombre], [2.5, 97.5])
    fila = {
        "modelo": nombre,
        "rol": modelos[nombre][2],
        "macro_f1_cv": cv_media,
        "macro_f1_cv_desv": cv_desv,
        "fuente_cv": cv_fuente,
        "macro_f1_test": mf,
        "macro_f1_test_ic95_inf": ic_inf,
        "macro_f1_test_ic95_sup": ic_sup,
        "diferencia_test_menos_cv": mf - cv_media,
        "accuracy_test": accuracy_score(y_true, pred),
        "macro_precision_test": mp,
        "macro_recall_test": mr,
    }
    for i, c in enumerate(CLASES):
        fila[f"precision_{c}"] = p[i]
        fila[f"recall_{c}"] = r[i]
        fila[f"f1_{c}"] = f[i]
        fila[f"soporte_{c}"] = int(s[i])
    if nombre != NOMBRE_PRINCIPAL:
        # IC 95% de la diferencia (principal - este modelo), con los mismos remuestreos
        dif = boot[NOMBRE_PRINCIPAL] - boot[nombre]
        fila["dif_principal_menos_este_ic95_inf"], fila["dif_principal_menos_este_ic95_sup"] = \
            np.percentile(dif, [2.5, 97.5])
    filas.append(fila)

resultados = pd.DataFrame(filas)
resultados.round(4).to_csv(CARPETA / "resultados_paso7.csv", index=False, sep=";")

pd.set_option("display.width", 250, "display.max_columns", 50)
print(resultados[["modelo", "rol", "macro_f1_cv", "macro_f1_test", "macro_f1_test_ic95_inf",
                  "macro_f1_test_ic95_sup", "accuracy_test"]].round(3).to_string(index=False))

print("\nPrecision / recall / F1 por clase en test:")
for _, f in resultados.iterrows():
    print(f"  {f['modelo']}")
    for c in CLASES:
        print(f"    {c:<9} P = {f[f'precision_{c}']:.3f}  R = {f[f'recall_{c}']:.3f}  "
              f"F1 = {f[f'f1_{c}']:.3f}  (n = {f[f'soporte_{c}']})")

print("\nIC 95% de la diferencia de macro-F1 (principal - otro), bootstrap pareado:")
for _, f in resultados[resultados["modelo"] != NOMBRE_PRINCIPAL].iterrows():
    inf, sup = f["dif_principal_menos_este_ic95_inf"], f["dif_principal_menos_este_ic95_sup"]
    concl = "incluye 0: diferencia NO concluyente" if inf <= 0 <= sup else "no incluye 0"
    print(f"  vs {f['modelo']:<33} [{inf:+.3f}, {sup:+.3f}]  {concl}")


# ---------------------------------------------------------------------------
# 6. Test vs. validacion cruzada (modelo principal)
# ---------------------------------------------------------------------------

titulo("6. TEST vs. VALIDACION CRUZADA (modelo principal)")
fp = resultados.set_index("modelo").loc[NOMBRE_PRINCIPAL]
print(f"  Macro-F1 CV (paso 6):  {fp['macro_f1_cv']:.3f} +/- {fp['macro_f1_cv_desv']:.3f}")
print(f"  Macro-F1 test:         {fp['macro_f1_test']:.3f}  "
      f"IC 95% [{fp['macro_f1_test_ic95_inf']:.3f}, {fp['macro_f1_test_ic95_sup']:.3f}]")
print(f"  Diferencia test - CV:  {fp['diferencia_test_menos_cv']:+.3f}")
if fp["macro_f1_test"] < fp["macro_f1_cv"]:
    print("  El rendimiento en test es MENOR que en validacion cruzada (se reporta tal cual).")
else:
    print("  El rendimiento en test es igual o mayor que en validacion cruzada.")
dentro = fp["macro_f1_test_ic95_inf"] <= fp["macro_f1_cv"] <= fp["macro_f1_test_ic95_sup"]
print(f"  El valor de CV {'SI' if dentro else 'NO'} cae dentro del IC 95% de test.")
print(f"  Macro-F1 en train: {f1_train_principal:.3f} (memoriza el entrenamiento).")


# ---------------------------------------------------------------------------
# 7. Matriz de confusion del modelo principal
# ---------------------------------------------------------------------------

titulo("7. MATRIZ DE CONFUSION (modelo principal)")
pred_p = predicciones[NOMBRE_PRINCIPAL]
mc = confusion_matrix(y_true, pred_p, labels=CLASES)
mc_norm = mc / mc.sum(axis=1, keepdims=True)
print("Conteos (filas = real, columnas = predicho):")
print(pd.DataFrame(mc, index=CLASES, columns=CLASES).to_string())
print("\nNormalizada por fila (recall en la diagonal):")
print(pd.DataFrame(mc_norm, index=CLASES, columns=CLASES).round(3).to_string())

fig, ejes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
for ax, datos, subtitulo, formato in [
        (ejes[0], mc, "Conteos", lambda v: f"{v:d}"),
        (ejes[1], mc_norm, "Normalizada por fila", lambda v: f"{v:.2f}")]:
    escala = datos / datos.max() if subtitulo == "Conteos" else datos
    ax.imshow(escala, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(CLASES)):
        for j in range(len(CLASES)):
            ax.text(j, i, formato(datos[i, j]), ha="center", va="center", fontsize=13,
                    color="white" if escala[i, j] > 0.55 else "#1f2937")
    ax.set_xticks(range(len(CLASES)), CLASES)
    ax.set_yticks(range(len(CLASES)), CLASES)
    ax.set_xlabel("Clase predicha")
    ax.set_ylabel("Clase real")
    ax.set_title(subtitulo, fontsize=12)
    for borde in ax.spines.values():
        borde.set_visible(False)
    ax.tick_params(length=0)
fig.suptitle(f"Regresión logística (TF-IDF + embeddings) en test (n = {len(test)})", fontsize=13)
fig.savefig(CARPETA / "matriz_confusion_paso7.png", dpi=200)
plt.close(fig)
print("\nGuardada: matriz_confusion_paso7.png")


# ---------------------------------------------------------------------------
# 8. Errores del modelo principal
# ---------------------------------------------------------------------------

titulo("8. ERRORES DEL MODELO PRINCIPAL")
probas = principal.predict_proba(X_combo_test)
assert list(principal.classes_) == CLASES


def categoria(real, pred):
    if real == pred:
        return ""
    return "positivo<->negativo" if {real, pred} == {"positivo", "negativo"} else "neutral<->polar"


detalle = test[["id", "feedback", ETIQUETA, "contraste", "archivo_origen"]].reset_index(drop=True).copy()
detalle = detalle.rename(columns={ETIQUETA: "polaridad_real"})
detalle["prediccion"] = pred_p
detalle["acierto"] = detalle["polaridad_real"] == detalle["prediccion"]
detalle["categoria_error"] = [categoria(r, p) for r, p in zip(detalle["polaridad_real"], pred_p)]
detalle["direccion"] = np.where(detalle["acierto"], "",
                                detalle["polaridad_real"] + "->" + detalle["prediccion"])
for i, c in enumerate(CLASES):
    detalle[f"prob_{c}"] = probas[:, i].round(3)
detalle["prob_prediccion"] = probas.max(axis=1).round(3)

errores = detalle[~detalle["acierto"]][
    ["id", "feedback", "polaridad_real", "prediccion", "contraste", "categoria_error",
     "direccion", "archivo_origen", "prob_negativo", "prob_neutral", "prob_positivo", "prob_prediccion"]]
errores.to_csv(CARPETA / "errores_test.csv", index=False, sep=";")

todas = test[["id", "polaridad_basica", "contraste", "archivo_origen"]].reset_index(drop=True).copy()
for n, p in predicciones.items():
    todas[f"pred_{n}"] = p
todas.to_csv(CARPETA / "predicciones_test.csv", index=False, sep=";")

print(f"Errores: {len(errores)} de {len(test)} ({len(errores) / len(test):.1%})")
print("\nPor tipo de error:")
print(errores["categoria_error"].value_counts().to_string() if len(errores) else "  (sin errores)")
print("\nPor direccion (real->predicho):")
print(errores["direccion"].value_counts().to_string() if len(errores) else "  (sin errores)")

print("\nTasa de error segun contraste:")
por_contraste = detalle.groupby("contraste").agg(n=("acierto", "size"), errores=("acierto", lambda s: (~s).sum()))
por_contraste["tasa_error"] = (por_contraste["errores"] / por_contraste["n"]).round(3)
print(por_contraste.to_string())

print("\nTasa de error segun archivo_origen:")
por_archivo = detalle.groupby("archivo_origen").agg(n=("acierto", "size"), errores=("acierto", lambda s: (~s).sum()))
por_archivo["tasa_error"] = (por_archivo["errores"] / por_archivo["n"]).round(3)
print(por_archivo.to_string())


# ---------------------------------------------------------------------------
# 9. Registro del modelo (CL12: metricas, parametros, datos y versiones)
# ---------------------------------------------------------------------------

registro = {
    "fecha": datetime.now().isoformat(timespec="seconds"),
    "modelo_principal": NOMBRE_PRINCIPAL,
    "origen_modelo": origen_principal,
    "parametros": {k: str(principal.get_params()[k]) for k in
                   ["clf__C", "clf__class_weight", "clf__max_iter", "prep__tfidf__ngram_range",
                    "prep__tfidf__min_df", "prep__tfidf__sublinear_tf"]},
    "modelo_embeddings": MODELO_EMBEDDINGS,
    "entrada": "texto_limpio (TF-IDF) + embeddings de feedback (emb_0..emb_383)",
    "macro_f1_cv": float(fp["macro_f1_cv"]),
    "macro_f1_cv_desv": float(fp["macro_f1_cv_desv"]),
    "macro_f1_train": float(f1_train_principal),
    "macro_f1_test": float(fp["macro_f1_test"]),
    "macro_f1_test_ic95": [float(fp["macro_f1_test_ic95_inf"]), float(fp["macro_f1_test_ic95_sup"])],
    "accuracy_test": float(fp["accuracy_test"]),
    "n_train": len(train),
    "n_test": len(test),
    "bootstrap": {"remuestreos": N_BOOTSTRAP, "metodo": "percentil", "semilla": SEMILLA},
    "sha256_datos": huellas,
    "versiones": versiones,
}
with open(CARPETA / "registro_modelo_paso7.json", "w", encoding="utf-8") as f:
    json.dump(registro, f, ensure_ascii=False, indent=2)

titulo("LISTO")
print("Guardados: resultados_paso7.csv, errores_test.csv, predicciones_test.csv,")
print("           matriz_confusion_paso7.png, embeddings_test.npy,")
print("           registro_modelo_paso7.json, paso7_salida.txt")
sys.stdout = sys.__stdout__
_salida.close()
