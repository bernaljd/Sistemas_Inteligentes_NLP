"""
Punto 5 (imagen 3): salida estructurada con UNA prediccion por aspecto.

Los datos traen, por comentario, una lista de aspectos con su sentimiento
(columna 'aspectos'). Hay 22 nombres distintos; se agrupan en 9 ASPECTOS
(ASPECTOS / GRUPOS abajo). Para cada comentario el modelo devuelve SIEMPRE los
9 aspectos, en el mismo orden, con este formato:

    {"aspecto": "evaluacion", "nombre": "Evaluacion",
     "mencionado": true, "probabilidad": 0.81,
     "sentimiento": "negativo",
     "probabilidades_sentimiento": {"negativo": 0.7, "neutral": 0.2, "positivo": 0.1}}

  - mencionado / probabilidad: el comentario habla de ese aspecto (umbral 0.5).
  - sentimiento: solo si se menciona (si no, null). Un comentario puede ser
    positivo con un aspecto y negativo con otro: "sabe mucho, PERO los parciales
    son injustos" -> metodologia positivo, evaluacion negativo.

Modelo (elegido con validacion cruzada SOLO sobre train, ver paso11_aspectos.py):
  1. Presencia: una regresion logistica por aspecto sobre TF-IDF + embeddings
     del comentario (las mismas representaciones del modelo de polaridad).
  2. Fragmento: el comentario se corta en fragmentos ("." ";" "pero" "aunque"
     "sin embargo"...) y, para cada aspecto mencionado, se elige el fragmento
     con mayor probabilidad de hablar de el (selector por aspecto, embeddings).
  3. Sentimiento: una regresion logistica sobre [embedding del fragmento,
     embedding del comentario, aspecto]. En CV supero a copiar la polaridad
     del comentario (macro-F1 0.68 vs 0.50): el sentimiento de un aspecto solo
     coincide con la polaridad del comentario en ~49 % de los casos.
"""

import json
import re

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from pipeline_feedback import CLASES, EmbeddingsMiniLM, LimpiadorTexto, _a_texto, limpiar_texto

# 9 aspectos: clave -> (nombre para mostrar, nombres originales en los datos)
ASPECTOS = {
    "contenidos": ("Contenidos", ["contenidos", "contenido y pertinencia",
                                  "cantidad de temas y repaso al inicio del curso"]),
    "evaluacion": ("Evaluación", ["parciales", "dificultad de las preguntas en los parciales",
                                  "diseno de la evaluacion y criterios"]),
    "retroalimentacion": ("Retroalimentación", ["retroalimentacion de los examenes",
                                                "retroalimentacion recibida", "comentarios del profesor"]),
    "metodologia": ("Metodología", ["metodologia del profesor", "actividades y metodologia",
                                    "claridad de exposicion"]),
    "material": ("Material de clase", ["material de clase", "diseno de las diapositivas"]),
    "organizacion": ("Organización y horario", ["organizacion del curso y su planeacion", "horario"]),
    "carga": ("Carga de trabajo", ["carga_academica", "ritmo y carga de trabajo"]),
    "recursos_digitales": ("Recursos digitales", ["disponibilidad de los recursos en el aula digital",
                                                  "soporte tecnico y plataforma"]),
    "comunicacion": ("Comunicación e interacción", ["respuesta oportuna a los correos",
                                                    "clima de aula e interaccion"]),
}
CLAVES = list(ASPECTOS)
GRUPO_DE = {original: clave for clave, (_, originales) in ASPECTOS.items() for original in originales}
UMBRAL_MENCION = 0.5

# Cache de embeddings por texto (el mismo comentario o fragmento se codifica una vez).
# Acotada: al pasar de MAX_CACHE se vacia. No se guarda dentro del .joblib.
_CACHE_EMB = {}
MAX_CACHE = 20000

_CORTE = re.compile(r"[.;!?\n]+|,?\s+\b(?:pero|aunque|sin embargo|no obstante|aun asi|aún así|"
                    r"mientras que|en cambio)\b", re.I)


def fragmentos(texto):
    """Corta un comentario en fragmentos (frases y partes unidas por 'pero', 'aunque'...)."""
    partes = [p.strip(" ,") for p in _CORTE.split(texto) if p and len(p.strip(" ,").split()) >= 3]
    return partes or [texto]


def etiquetas(lista_aspectos):
    """Lista original [{"aspecto", "sentimiento"}, ...] -> {clave: sentimiento} de los 9 aspectos.

    Si un comentario tiene varios aspectos originales del mismo grupo con
    sentimientos distintos, gana el mas frecuente; en empate, "neutral" (mixto).
    """
    if isinstance(lista_aspectos, str):
        lista_aspectos = json.loads(lista_aspectos) if lista_aspectos.strip() else []
    por_grupo = {}
    for a in lista_aspectos or []:
        por_grupo.setdefault(GRUPO_DE[a["aspecto"]], []).append(a["sentimiento"])
    salida = {}
    for clave, sents in por_grupo.items():
        conteo = {s: sents.count(s) for s in set(sents)}
        maximo = max(conteo.values())
        ganadores = [s for s, n in conteo.items() if n == maximo]
        salida[clave] = ganadores[0] if len(ganadores) == 1 else "neutral"
    return salida


class ModeloAspectos(BaseEstimator):
    """Presencia + sentimiento de los 9 aspectos (ver docstring del modulo)."""

    def __init__(self, C_presencia=1.0, C_sentimiento=1.0, umbral=UMBRAL_MENCION,
                 entrada_sentimiento="fragmento+comentario"):
        self.C_presencia = C_presencia
        self.C_sentimiento = C_sentimiento
        self.umbral = umbral
        # "fragmento+comentario" (elegido) | "comentario" (comparacion en paso 11)
        self.entrada_sentimiento = entrada_sentimiento

    # --------------------------- representaciones ---------------------------
    def _emb(self, textos):
        """Embeddings con cache. Usa una copia local: si otro hilo vacia la cache
        a mitad de la llamada, esta llamada no pierde nada."""
        textos = list(textos)
        if not textos:
            return np.zeros((0, 384))
        clave = (self.embeddings_.nombre_modelo, self.embeddings_.revision)
        locales = {t: _CACHE_EMB.get((clave, t)) for t in dict.fromkeys(textos)}
        nuevos = [t for t, v in locales.items() if v is None]
        if nuevos:
            locales.update(zip(nuevos, self.embeddings_.transform(nuevos)))
            if len(_CACHE_EMB) + len(nuevos) > MAX_CACHE:
                _CACHE_EMB.clear()
            for t in nuevos:
                _CACHE_EMB[(clave, t)] = locales[t]
        return np.vstack([locales[t] for t in textos])

    def _features(self, textos, E):
        return hstack([self.tfidf_.transform(textos), csr_matrix(E)]).tocsr()

    def _sent_features(self, frag_E, coment_E, j):
        aspecto = np.eye(len(CLAVES))[j]
        if self.entrada_sentimiento == "comentario":
            return np.hstack([coment_E, aspecto])
        return np.hstack([frag_E, coment_E, aspecto])

    def _fragmento_relevante(self, frags_E, j):
        return int(np.argmax(self.selectores_[j].predict_proba(frags_E)[:, 1]))

    def _emb_fragmentos(self, textos):
        """Embeddings de los fragmentos de varios textos en UNA llamada al modelo."""
        frags = [fragmentos(t) for t in textos]
        todos = self._emb([f for fs in frags for f in fs])
        cortes = np.cumsum([0] + [len(fs) for fs in frags])
        return [todos[a:b] for a, b in zip(cortes[:-1], cortes[1:])]

    # --------------------------------- fit ----------------------------------
    def fit(self, textos, aspectos_por_texto):
        """textos: comentarios originales; aspectos_por_texto: {clave: sentimiento} por comentario."""
        textos = [_a_texto(t) for t in textos]
        Y = np.array([[c in a for c in CLAVES] for a in aspectos_por_texto], dtype=int)
        self.embeddings_ = EmbeddingsMiniLM()
        self.tfidf_ = Pipeline([("limpieza", LimpiadorTexto()),
                                ("vectorizador", TfidfVectorizer(ngram_range=(1, 2), min_df=2,
                                                                 sublinear_tf=True))]).fit(textos)
        E = self._emb(textos)
        F = self._features(textos, E)

        def lr(C):
            return LogisticRegression(C=C, class_weight="balanced", max_iter=5000)

        self.presencia_ = [lr(self.C_presencia).fit(F, Y[:, j]) for j in range(len(CLAVES))]
        self.selectores_ = [lr(1.0).fit(E, Y[:, j]) for j in range(len(CLAVES))]

        con_aspectos = [i for i, a in enumerate(aspectos_por_texto) if a]
        frags_E = dict(zip(con_aspectos, self._emb_fragmentos([textos[i] for i in con_aspectos])))
        filas, ys = [], []
        for i in con_aspectos:
            FE = frags_E[i]
            for clave, sent in aspectos_por_texto[i].items():
                j = CLAVES.index(clave)
                filas.append(self._sent_features(FE[self._fragmento_relevante(FE, j)], E[i], j))
                ys.append(sent)
        self.sentimiento_ = lr(self.C_sentimiento).fit(np.vstack(filas), ys)
        self.clases_sentimiento_ = list(self.sentimiento_.classes_)
        assert self.clases_sentimiento_ == CLASES
        return self

    # ------------------------------- predecir -------------------------------
    def predecir(self, textos):
        """Lista (una por texto) de 9 diccionarios con el formato del modulo.

        Textos vacios o sin palabras utiles: los 9 aspectos con mencionado=False
        y probabilidad None (no se predice nada).
        """
        textos = [_a_texto(t) for t in ([textos] if isinstance(textos, str) else textos)]
        salida = [[self._vacio(c) for c in CLAVES] for _ in textos]
        validos = [i for i, t in enumerate(textos) if t.strip() and limpiar_texto(t).split()]
        if not validos:
            return salida
        tv = [textos[i] for i in validos]
        E = self._emb(tv)
        F = self._features(tv, E)
        P = np.column_stack([m.predict_proba(F)[:, 1] for m in self.presencia_])
        con_mencion = [n for n in range(len(tv)) if (P[n] >= self.umbral).any()]
        frags_E = dict(zip(con_mencion, self._emb_fragmentos([tv[n] for n in con_mencion])))
        for n, i in enumerate(validos):
            mencionados = [j for j in range(len(CLAVES)) if P[n, j] >= self.umbral]
            FE = frags_E.get(n)
            for j, clave in enumerate(CLAVES):
                r = salida[i][j]
                r["probabilidad"] = round(float(P[n, j]), 4)
                if j in mencionados:
                    X = self._sent_features(FE[self._fragmento_relevante(FE, j)], E[n], j)[None, :]
                    ps = self.sentimiento_.predict_proba(X)[0]
                    r.update(mencionado=True, sentimiento=self.clases_sentimiento_[int(np.argmax(ps))],
                             probabilidades_sentimiento={c: round(float(p), 4)
                                                         for c, p in zip(self.clases_sentimiento_, ps)})
        return salida

    def predecir_sentimiento(self, textos, claves):
        """Sentimiento de pares (texto, aspecto) aunque el modelo no detecte el aspecto.
        Sirve para evaluar el componente de sentimiento por separado."""
        textos = [_a_texto(t) for t in textos]
        E = self._emb(textos)
        frags_E = self._emb_fragmentos(textos)
        filas = []
        for n, clave in enumerate(claves):
            j = CLAVES.index(clave)
            filas.append(self._sent_features(frags_E[n][self._fragmento_relevante(frags_E[n], j)], E[n], j))
        return self.sentimiento_.predict(np.vstack(filas)) if filas else np.array([])

    @staticmethod
    def _vacio(clave):
        return {"aspecto": clave, "nombre": ASPECTOS[clave][0], "mencionado": False, "probabilidad": None,
                "sentimiento": None, "probabilidades_sentimiento": None}
