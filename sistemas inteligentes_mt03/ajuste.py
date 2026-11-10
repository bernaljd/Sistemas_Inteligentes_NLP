"""
Paso 6: Ajuste de hiperparametros (solo sobre train.csv; test.csv no se toca).

Candidatos (los 3 mejores del paso 5):
  - Linear SVM con TF-IDF + Embeddings
  - Linear SVM con Embeddings
  - Logistic Regression con TF-IDF + Embeddings

GridSearchCV con validacion cruzada estratificada de 5 particiones, seleccion por macro-F1.

Requiere haber ejecutado antes vectorizacion.py (genera embeddings_train.npy).

Salidas:
  - resultados_paso6.csv          : mejor resultado de cada candidato.
  - resultados_paso6_detalle.csv  : todas las combinaciones probadas (para interpretarlas).
  - modelo_paso6.joblib           : mejor modelo reentrenado con todo train, listo para el paso 7.
"""
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

ETIQUETA, SEMILLA = "polaridad_basica", 42
MODELO_EMBEDDINGS = "paraphrase-multilingual-MiniLM-L12-v2"  # el mismo del paso 4

train = pd.read_csv("train.csv", sep=";")
y = train[ETIQUETA]
X_emb = np.load("embeddings_train.npy")
assert len(train) == len(X_emb), "embeddings_train.npy no coincide con train.csv"

cols_emb = [f"emb_{i}" for i in range(X_emb.shape[1])]
X_combo = pd.concat([train[["texto_limpio"]], pd.DataFrame(X_emb, columns=cols_emb)], axis=1)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)


def prep_combo():
    return ColumnTransformer([
        ("tfidf", TfidfVectorizer(sublinear_tf=True), "texto_limpio"),
        ("emb", "passthrough", cols_emb),
    ])


C_svm = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
tfidf_grid = {"prep__tfidf__ngram_range": [(1, 1), (1, 2)], "prep__tfidf__min_df": [1, 2, 3]}

# "entrada" indica que datos necesita cada modelo, para preparar test igual en el paso 7
busquedas = {
    "SVM + TF-IDF+Emb": (
        GridSearchCV(
            Pipeline([("prep", prep_combo()),
                      ("clf", LinearSVC(class_weight="balanced", max_iter=20000, random_state=SEMILLA))]),
            {**tfidf_grid, "clf__C": C_svm},
            scoring="f1_macro", cv=cv, n_jobs=-1, return_train_score=True),
        X_combo, "texto_limpio + embeddings"),
    "SVM + Embeddings": (
        GridSearchCV(
            Pipeline([("clf", LinearSVC(max_iter=20000, random_state=SEMILLA))]),
            {"clf__C": C_svm, "clf__class_weight": ["balanced", None]},
            scoring="f1_macro", cv=cv, n_jobs=-1, return_train_score=True),
        X_emb, "embeddings"),
    "LogReg + TF-IDF+Emb": (
        GridSearchCV(
            Pipeline([("prep", prep_combo()),
                      ("clf", LogisticRegression(class_weight="balanced", max_iter=5000))]),
            {**tfidf_grid, "clf__C": [0.1, 0.3, 1, 3, 10, 30, 100]},
            scoring="f1_macro", cv=cv, n_jobs=-1, return_train_score=True),
        X_combo, "texto_limpio + embeddings"),
}

filas, detalle, ajustados = [], [], {}
for nombre, (busqueda, X, entrada) in busquedas.items():
    t0 = time.time()
    busqueda.fit(X, y)
    i, r = busqueda.best_index_, busqueda.cv_results_
    ajustados[nombre] = (busqueda.best_estimator_, entrada)
    filas.append({
        "candidato": nombre,
        "macro_f1_media": r["mean_test_score"][i],
        "macro_f1_desv": r["std_test_score"][i],
        "macro_f1_train": r["mean_train_score"][i],
        "combinaciones_probadas": len(r["params"]),
        "mejores_parametros": str(busqueda.best_params_),
        "segundos": time.time() - t0,
    })
    d = pd.DataFrame({
        "candidato": nombre,
        "parametros": [str(p) for p in r["params"]],
        "macro_f1_media": r["mean_test_score"],
        "macro_f1_desv": r["std_test_score"],
        "macro_f1_train": r["mean_train_score"],
        "ranking": r["rank_test_score"],
    })
    detalle.append(d)
    print(f"listo: {nombre}")

res = pd.DataFrame(filas).sort_values("macro_f1_media", ascending=False).round(3)
res.to_csv("resultados_paso6.csv", index=False, sep=";")
detalle = pd.concat(detalle).sort_values(["candidato", "ranking"]).round(4)
detalle.to_csv("resultados_paso6_detalle.csv", index=False, sep=";")

pd.set_option("display.max_colwidth", 120, "display.width", 250)
print("\n" + "=" * 100)
print("RESULTADOS (validacion cruzada 5 particiones sobre train)")
print("=" * 100)
print(res.drop(columns="mejores_parametros").to_string(index=False))
for _, f in res.iterrows():
    print(f"\n{f['candidato']}: {f['mejores_parametros']}")

# Si las 5 mejores combinaciones de un candidato dan casi lo mismo,
# el resultado es estable y no depende de un parametro exacto.
print("\nTop 5 combinaciones de cada candidato:")
for nombre, d in detalle.groupby("candidato", sort=False):
    print(f"\n  {nombre}")
    for _, f in d.head(5).iterrows():
        print(f"    {f['macro_f1_media']:.3f} +/- {f['macro_f1_desv']:.3f}  {f['parametros']}")

mejor = res.iloc[0]["candidato"]
modelo, entrada = ajustados[mejor]
# GridSearchCV ya reentreno el mejor modelo con todo train (refit=True).
joblib.dump({
    "nombre": mejor,
    "modelo": modelo,
    "entrada": entrada,                       # que datos espera el modelo
    "columnas_embeddings": cols_emb,          # nombres de columnas si usa TF-IDF + embeddings
    "modelo_embeddings": MODELO_EMBEDDINGS,   # con que modelo calcular embeddings de test
    "macro_f1_cv": res.iloc[0]["macro_f1_media"],
}, "modelo_paso6.joblib")
print(f"\nMejor candidato: {mejor} -> guardado en modelo_paso6.joblib")
print("Guardados: resultados_paso6.csv, resultados_paso6_detalle.csv")
