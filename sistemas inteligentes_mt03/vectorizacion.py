"""
Paso 4: Representacion vectorial del texto.

Compara TF-IDF vs. embeddings semanticos usando SOLO train.csv.
test.csv no se toca aqui (se reserva para el paso 7).

Como se compara sin usar test:
  - Validacion cruzada estratificada de 5 particiones dentro de train.
  - El mismo clasificador simple (Regresion Logistica) para todas las
    representaciones, asi la diferencia se debe a la representacion y no al modelo.
  - TF-IDF va dentro de un Pipeline: en cada particion el vocabulario y los
    pesos IDF se aprenden solo con la parte de entrenamiento (sin fuga de informacion).
  - Los embeddings vienen de un modelo ya preentrenado: no aprenden nada de
    nuestros datos, por eso se pueden calcular una vez para todo train.

Salidas:
  - resultados_vectorizacion.csv : macro-F1 y accuracy de cada representacion.
  - errores_vectorizacion.csv    : prediccion de cada metodo por comentario,
                                   para el analisis de errores linguisticos.
  - embeddings_train.npy         : embeddings guardados (evita recalcularlos).
"""

import os

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.pipeline import make_pipeline

ETIQUETA = "polaridad_basica"
MODELO_EMBEDDINGS = "paraphrase-multilingual-MiniLM-L12-v2"  # entiende espanol
SEMILLA = 42

train = pd.read_csv("train.csv", sep=";")
y = train[ETIQUETA]
print(f"Train: {len(train)} comentarios")
print(y.value_counts().to_string(), "\n")


# ---------------------------------------------------------------------------
# 1. Representaciones
# ---------------------------------------------------------------------------

def nuevo_clasificador():
    # class_weight="balanced" compensa que haya menos comentarios positivos
    return LogisticRegression(max_iter=2000, class_weight="balanced")


# TF-IDF por palabras (unigramas y bigramas, p. ej. "no encontre", "muy clara").
# Usa texto_limpio porque ya esta normalizado y conserva las negaciones.
tfidf_palabras = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
    nuevo_clasificador(),
)

# TF-IDF por caracteres: aguanta mejor errores de escritura y variaciones
# de una misma palabra (explicacion / explicaciones).
tfidf_caracteres = make_pipeline(
    TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True),
    nuevo_clasificador(),
)


def obtener_embeddings(textos, archivo_cache=None):
    """Convierte cada texto en un vector de 384 numeros que resume su significado."""
    if archivo_cache and os.path.exists(archivo_cache):
        X = np.load(archivo_cache)
        if len(X) == len(textos):
            print(f"Embeddings cargados de {archivo_cache}")
            return X

    from sentence_transformers import SentenceTransformer

    print(f"Calculando embeddings con {MODELO_EMBEDDINGS} (la primera vez descarga el modelo)...")
    modelo = SentenceTransformer(MODELO_EMBEDDINGS)
    X = modelo.encode(list(textos), batch_size=32, show_progress_bar=True,
                      normalize_embeddings=True)
    if archivo_cache:
        np.save(archivo_cache, X)
    return X


# Los embeddings usan el texto original (feedback): estos modelos se entrenaron
# con frases completas y aprovechan conectores como "sin embargo" o "aunque".
X_embeddings = obtener_embeddings(train["feedback"], "embeddings_train.npy")
print(f"Forma de los embeddings: {X_embeddings.shape}\n")

representaciones = {
    "TF-IDF palabras (1-2 gramas)": (tfidf_palabras, train["texto_limpio"]),
    "TF-IDF caracteres (3-5 gramas)": (tfidf_caracteres, train["texto_limpio"]),
    f"Embeddings ({MODELO_EMBEDDINGS})": (nuevo_clasificador(), X_embeddings),
}


# ---------------------------------------------------------------------------
# 2. Comparacion con validacion cruzada (solo train)
# ---------------------------------------------------------------------------

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)
filas = []
predicciones = {}

for nombre, (modelo, X) in representaciones.items():
    puntajes = cross_validate(modelo, X, y, cv=cv, scoring=["f1_macro", "accuracy"])
    predicciones[nombre] = cross_val_predict(modelo, X, y, cv=cv)
    filas.append({
        "representacion": nombre,
        "macro_f1_media": puntajes["test_f1_macro"].mean(),
        "macro_f1_desv": puntajes["test_f1_macro"].std(),
        "accuracy_media": puntajes["test_accuracy"].mean(),
        "accuracy_desv": puntajes["test_accuracy"].std(),
    })

resultados = pd.DataFrame(filas).sort_values("macro_f1_media", ascending=False)
resultados.to_csv("resultados_vectorizacion.csv", index=False, sep=";")

print("=" * 70)
print("COMPARACION (validacion cruzada 5 particiones sobre train)")
print("=" * 70)
for _, r in resultados.iterrows():
    print(f"{r['representacion']:<55} macro-F1 = {r['macro_f1_media']:.3f} +/- {r['macro_f1_desv']:.3f}"
          f" | accuracy = {r['accuracy_media']:.3f}")

clases = sorted(y.unique())
for nombre, pred in predicciones.items():
    print(f"\n--- {nombre} ---")
    print(classification_report(y, pred, digits=3, zero_division=0))
    print("Matriz de confusion (filas = real, columnas = predicho):")
    print(pd.DataFrame(confusion_matrix(y, pred, labels=clases), index=clases, columns=clases))


# ---------------------------------------------------------------------------
# 3. Analisis de errores
# ---------------------------------------------------------------------------

mejor_tfidf = max((n for n in predicciones if n.startswith("TF-IDF")),
                  key=lambda n: resultados.set_index("representacion").loc[n, "macro_f1_media"])
nombre_emb = next(n for n in predicciones if n.startswith("Embeddings"))

errores = train[["id", "feedback", "texto_limpio", ETIQUETA, "contraste"]].copy()
errores["pred_tfidf"] = predicciones[mejor_tfidf]
errores["pred_embeddings"] = predicciones[nombre_emb]
errores["acierta_tfidf"] = errores["pred_tfidf"] == y
errores["acierta_embeddings"] = errores["pred_embeddings"] == y
errores.to_csv("errores_vectorizacion.csv", index=False, sep=";")

print("\n" + "=" * 70)
print(f"ANALISIS DE ERRORES: {mejor_tfidf}  vs  Embeddings")
print("=" * 70)

# Comentarios con contraste ("buen profesor, pero...") suelen ser los mas dificiles
print("\nAciertos segun si el comentario tiene contraste:")
print(errores.groupby("contraste")[["acierta_tfidf", "acierta_embeddings"]].mean().round(3))

solo_emb = errores[~errores["acierta_tfidf"] & errores["acierta_embeddings"]]
solo_tfidf = errores[errores["acierta_tfidf"] & ~errores["acierta_embeddings"]]
ambos_fallan = errores[~errores["acierta_tfidf"] & ~errores["acierta_embeddings"]]
print(f"\nSolo embeddings acierta: {len(solo_emb)}")
print(f"Solo TF-IDF acierta:     {len(solo_tfidf)}")
print(f"Ambos fallan:            {len(ambos_fallan)}")


def mostrar(titulo, df, n=3):
    print(f"\n{titulo}")
    for _, r in df.head(n).iterrows():
        print(f"  [id {r['id']}] real={r[ETIQUETA]} | tfidf={r['pred_tfidf']} | emb={r['pred_embeddings']}")
        print(f"    {r['feedback'][:200]}")


mostrar("Ejemplos donde solo embeddings acierta:", solo_emb)
mostrar("Ejemplos donde solo TF-IDF acierta:", solo_tfidf)
mostrar("Ejemplos donde ambos fallan:", ambos_fallan)

print("\nGuardados: resultados_vectorizacion.csv, errores_vectorizacion.csv, embeddings_train.npy")
