"""
Paso 5: Entrenamiento inicial y linea base.

Compara DummyClassifier (referencia) contra Logistic Regression, Linear SVM,
Naive Bayes y Random Forest, con tres representaciones del texto:
  1. TF-IDF (palabras, 1-2 gramas)
  2. Embeddings MiniLM (calculados en el paso 4)
  3. TF-IDF + Embeddings combinados (el paso 4 mostro que se complementan)

Todo se evalua con validacion cruzada estratificada de 5 particiones SOLO
sobre train.csv. test.csv no se toca hasta el paso 7.
Los hiperparametros son los de por defecto; se ajustan en el paso 6.

Requiere haber ejecutado antes vectorizacion.py (genera embeddings_train.npy).

Salidas:
  - resultados_paso5.csv : metricas de cada combinacion modelo + representacion.
"""

import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.naive_bayes import ComplementNB, GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC

ETIQUETA = "polaridad_basica"
SEMILLA = 42

train = pd.read_csv("train.csv", sep=";")
y = train[ETIQUETA]
X_emb = np.load("embeddings_train.npy")
assert len(train) == len(X_emb), "embeddings_train.npy no coincide con train.csv: vuelve a correr vectorizacion.py"

print(f"Train: {len(train)} comentarios")
print(y.value_counts().to_string(), "\n")

# Tabla con el texto y las 384 columnas del embedding, para la representacion combinada
cols_emb = [f"emb_{i}" for i in range(X_emb.shape[1])]
X_combo = pd.concat([train[["texto_limpio"]], pd.DataFrame(X_emb, columns=cols_emb)], axis=1)


# ---------------------------------------------------------------------------
# Representaciones
# ---------------------------------------------------------------------------

def tfidf():
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)


def tfidf_mas_embeddings(escalar_embeddings=False):
    # TF-IDF y los embeddings tienen norma 1, asi que pesan parecido al unirlos.
    # escalar_embeddings=True los pasa a [0, 1] porque Naive Bayes no acepta negativos.
    emb = MinMaxScaler() if escalar_embeddings else "passthrough"
    return ColumnTransformer([("tfidf", tfidf(), "texto_limpio"), ("emb", emb, cols_emb)])


# ---------------------------------------------------------------------------
# Modelos
# ---------------------------------------------------------------------------

def clasificadores():
    # class_weight="balanced" compensa que haya menos comentarios positivos
    return {
        "Dummy (clase mayoritaria)": DummyClassifier(strategy="most_frequent"),
        "Dummy (aleatorio estratificado)": DummyClassifier(strategy="stratified", random_state=SEMILLA),
        "Logistic Regression": LogisticRegression(max_iter=5000, class_weight="balanced"),
        "Linear SVM": LinearSVC(class_weight="balanced", random_state=SEMILLA, max_iter=10000),
        "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                                random_state=SEMILLA, n_jobs=-1),
    }


experimentos = []  # (representacion, modelo, estimador, X)

for nombre, clf in clasificadores().items():
    experimentos.append(("TF-IDF", nombre, make_pipeline(tfidf(), clf), train["texto_limpio"]))
# ComplementNB: variante de Naive Bayes para texto con clases desbalanceadas.
# MultinomialNB daba macro-F1 de 0.46 porque casi siempre predecia "neutral".
experimentos.append(("TF-IDF", "Naive Bayes", make_pipeline(tfidf(), ComplementNB()), train["texto_limpio"]))

for nombre, clf in clasificadores().items():
    experimentos.append(("Embeddings", nombre, clf, X_emb))
# Los embeddings tienen valores negativos: se usa GaussianNB
experimentos.append(("Embeddings", "Naive Bayes", GaussianNB(), X_emb))

for nombre, clf in clasificadores().items():
    experimentos.append(("TF-IDF + Embeddings", nombre, make_pipeline(tfidf_mas_embeddings(), clf), X_combo))
experimentos.append(("TF-IDF + Embeddings", "Naive Bayes",
                     make_pipeline(tfidf_mas_embeddings(escalar_embeddings=True), ComplementNB()), X_combo))


# ---------------------------------------------------------------------------
# Validacion cruzada
# ---------------------------------------------------------------------------

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)
filas = []

for rep, nombre, modelo, X in experimentos:
    inicio = time.time()
    r = cross_validate(modelo, X, y, cv=cv, scoring=["f1_macro", "accuracy"], return_train_score=True)
    filas.append({
        "representacion": rep,
        "modelo": nombre,
        "macro_f1_media": r["test_f1_macro"].mean(),
        "macro_f1_desv": r["test_f1_macro"].std(),
        "accuracy_media": r["test_accuracy"].mean(),
        "accuracy_desv": r["test_accuracy"].std(),
        # Si train es mucho mayor que validacion, el modelo esta memorizando (sobreajuste)
        "macro_f1_train": r["train_f1_macro"].mean(),
        "segundos": time.time() - inicio,
    })
    print(f"  listo: {rep:<20} {nombre}")

res = pd.DataFrame(filas).sort_values("macro_f1_media", ascending=False).round(3)
res.to_csv("resultados_paso5.csv", index=False, sep=";")

print("\n" + "=" * 100)
print("RESULTADOS (validacion cruzada 5 particiones sobre train, ordenados por macro-F1)")
print("=" * 100)
print(res.to_string(index=False))

print("\nMacro-F1 por modelo y representacion:")
tabla = res.pivot(index="modelo", columns="representacion", values="macro_f1_media")
tabla = tabla[["TF-IDF", "Embeddings", "TF-IDF + Embeddings"]]
print(tabla.sort_values("TF-IDF + Embeddings", ascending=False).to_string())


# ---------------------------------------------------------------------------
# Detalle del mejor modelo vs. la linea base
# ---------------------------------------------------------------------------

mejor = res.iloc[0]
dummy = res[res["modelo"].str.startswith("Dummy")].iloc[0]
print("\n" + "=" * 100)
print(f"MEJOR: {mejor['modelo']} con {mejor['representacion']}")
print(f"  macro-F1 = {mejor['macro_f1_media']:.3f} +/- {mejor['macro_f1_desv']:.3f}")
print(f"  Mejor Dummy ({dummy['modelo']}): macro-F1 = {dummy['macro_f1_media']:.3f}")
print(f"  Mejora sobre la linea base: +{mejor['macro_f1_media'] - dummy['macro_f1_media']:.3f}")
print("=" * 100)

_, _, modelo_mejor, X_mejor = next(e for e in experimentos
                                   if e[0] == mejor["representacion"] and e[1] == mejor["modelo"])
pred = cross_val_predict(modelo_mejor, X_mejor, y, cv=cv)
clases = sorted(y.unique())
print(classification_report(y, pred, digits=3))
print("Matriz de confusion (filas = real, columnas = predicho):")
print(pd.DataFrame(confusion_matrix(y, pred, labels=clases), index=clases, columns=clases))

print("\nGuardado: resultados_paso5.csv")
