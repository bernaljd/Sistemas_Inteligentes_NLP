"""
Componentes del Pipeline final (pasos 8 y 9).

Este modulo DEBE estar disponible (misma carpeta o instalado) cuando se carga
modelo_final.joblib: joblib/pickle guarda las clases propias por referencia
(nombre del modulo + nombre de la clase), no su codigo.

Contiene:
  - limpiar_texto / LimpiadorTexto : reproduce EXACTAMENTE la columna texto_limpio
    de feedback_preprocesado.csv (verificado: 614 de 614 coincidencias).
  - EmbeddingsMiniLM               : texto original -> embedding de 384 dim.
    (mismo modelo y parametros que el paso 4).
  - construir_pipeline             : Pipeline completo texto crudo -> polaridad.
  - predecir                       : uso recomendado en produccion; valida la
    entrada y marca textos vacios y predicciones ambiguas.
"""

import re
import unicodedata

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

MODELO_EMBEDDINGS = "paraphrase-multilingual-MiniLM-L12-v2"
CLASES = ["negativo", "neutral", "positivo"]

# Palabras que el preprocesado original elimino. Se dedujeron de los datos: son
# las unicas palabras de 'feedback' que nunca aparecen en 'texto_limpio'.
# Se conservan negaciones e intensificadores ("no", "muy", "poco", "mas", "sin").
STOPWORDS = frozenset({
    "a", "al", "como", "con", "de", "del", "el", "en", "es", "esta", "estas",
    "este", "estos", "fue", "ha", "he", "la", "las", "lo", "los", "me", "mi",
    "mis", "o", "para", "por", "que", "se", "su", "sus", "un", "una", "unas", "y",
})

_PALABRA = re.compile(r"[a-zñ0-9]+")


def _a_texto(t):
    """None / NaN -> "" ; str -> str ; cualquier otra cosa -> TypeError."""
    if t is None or (isinstance(t, float) and np.isnan(t)):
        return ""
    if not isinstance(t, str):
        raise TypeError(f"Se esperaba texto (str) y se recibio {type(t).__name__}: {t!r}")
    return t


def limpiar_texto(texto):
    """Minusculas, sin tildes, sin puntuacion y sin stopwords."""
    t = unicodedata.normalize("NFKD", _a_texto(texto).lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return " ".join(w for w in _PALABRA.findall(t) if w not in STOPWORDS)


class LimpiadorTexto(BaseEstimator, TransformerMixin):
    """Paso de preprocesamiento: feedback original -> texto_limpio."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [limpiar_texto(t) for t in X]


class EmbeddingsMiniLM(BaseEstimator, TransformerMixin):
    """Texto original -> embedding normalizado (sentence-transformers).

    El modelo preentrenado NO se guarda dentro del .joblib (pesa ~470 MB):
    se carga la primera vez que se usa, desde la cache de Hugging Face o
    descargandolo. Por eso el entorno de destino necesita sentence-transformers.
    """

    def __init__(self, nombre_modelo=MODELO_EMBEDDINGS, batch_size=32):
        self.nombre_modelo = nombre_modelo
        self.batch_size = batch_size

    def _modelo(self):
        if getattr(self, "_st", None) is None:
            from sentence_transformers import SentenceTransformer
            self._st = SentenceTransformer(self.nombre_modelo)
        return self._st

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        textos = [_a_texto(t) for t in X]
        return self._modelo().encode(textos, batch_size=self.batch_size,
                                     show_progress_bar=False, normalize_embeddings=True)

    def __getstate__(self):
        estado = self.__dict__.copy()
        estado.pop("_st", None)  # no serializar el modelo preentrenado
        return estado


def construir_pipeline(C=10, ngram_range=(1, 2), min_df=2):
    """Misma arquitectura e hiperparametros que el modelo elegido en el paso 6
    (LogReg + TF-IDF+Emb), pero recibiendo el comentario original sin preparar."""
    return Pipeline([
        ("features", FeatureUnion([
            ("tfidf", Pipeline([
                ("limpieza", LimpiadorTexto()),
                ("vectorizador", TfidfVectorizer(ngram_range=ngram_range, min_df=min_df,
                                                 sublinear_tf=True)),
            ])),
            ("embeddings", EmbeddingsMiniLM()),
        ])),
        ("clf", LogisticRegression(C=C, class_weight="balanced", max_iter=5000)),
    ])


# ---------------------------------------------------------------------------
# Prediccion segura
# ---------------------------------------------------------------------------

def leer_csv_bytes(contenido):
    """Lee un CSV subido por el usuario (API e interfaz).

    - Codificacion: utf-8 (con o sin BOM) o latin-1 (Excel en espanol).
    - Separador: ';' o ',' segun la fila de encabezados (los comentarios suelen
      tener comas dentro, por eso no se adivina con el contenido).
    Lanza ValueError si no se puede leer.
    """
    import io
    for codificacion in ["utf-8-sig", "latin-1"]:
        try:
            texto = contenido.decode(codificacion)
            break
        except UnicodeDecodeError:
            continue
    if not texto.strip():
        raise ValueError("El archivo esta vacio")
    encabezado = texto.splitlines()[0]
    sep = ";" if encabezado.count(";") >= encabezado.count(",") and ";" in encabezado else ","
    try:
        return pd.read_csv(io.StringIO(texto), sep=sep)
    except Exception as e:
        raise ValueError(f"No se pudo leer el CSV: {e}") from e


UMBRAL_CONFIANZA = 0.5  # con 3 clases, < 0.5 = ninguna clase tiene mayoria
MIN_PALABRAS_UTILES = 1


def predecir(modelo, textos, umbral=UMBRAL_CONFIANZA):
    """Predice la polaridad y devuelve un DataFrame con una fila por texto.

    Columnas: texto, estado, prediccion, confianza, prob_<clase>.
      estado = "ok"           -> prediccion confiable
               "ambiguo"      -> la clase mas probable tiene prob < umbral
               "texto_vacio"  -> vacio, None o sin palabras utiles tras la limpieza
                                 (no se predice: prediccion = None)
    Lanza TypeError si algun elemento no es texto.
    """
    if isinstance(textos, str):
        textos = [textos]          # un solo comentario
    textos = [_a_texto(t) for t in textos]

    filas = [{"texto": t, "estado": "texto_vacio", "prediccion": None, "confianza": np.nan,
              **{f"prob_{c}": np.nan for c in CLASES}} for t in textos]
    validos = [i for i, t in enumerate(textos)
               if t.strip() and len(limpiar_texto(t).split()) >= MIN_PALABRAS_UTILES]
    if validos:
        probas = modelo.predict_proba([textos[i] for i in validos])
        clases = list(modelo.classes_)
        for i, p in zip(validos, probas):
            k = int(np.argmax(p))
            filas[i].update({
                "prediccion": clases[k],
                "confianza": float(p[k]),
                "estado": "ok" if p[k] >= umbral else "ambiguo",
                **{f"prob_{c}": float(p[clases.index(c)]) for c in CLASES},
            })
    return pd.DataFrame(filas)
