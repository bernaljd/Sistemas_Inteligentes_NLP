"""
Paso 8: Seleccionar el mejor modelo y guardarlo con joblib.

1. SELECCION (solo con resultados ya calculados, sin reajustar nada):
   - Criterio principal: macro-F1 en validacion cruzada (paso 6). Elegir por
     test seria fuga de informacion: test solo CONFIRMA la eleccion.
   - Confirmacion en test (paso 7): el elegido supera claramente a los Dummy y
     su diferencia con los otros candidatos no es concluyente (IC 95% incluye 0).
   - Criterio practico: la regresion logistica da probabilidades (predict_proba),
     necesarias para detectar comentarios ambiguos; LinearSVC no.

2. GUARDADO: un Pipeline COMPLETO que recibe el comentario original:
       texto -> entrada -> [limpieza -> TF-IDF] + [embeddings MiniLM] -> LogisticRegression
   (el paso "entrada" acepta un string suelto y convierte None/NaN en ""; el
   modelo de embeddings queda fijado a una revision exacta de Hugging Face).
   modelo_paso6.joblib necesitaba recibir texto_limpio y los 384 embeddings ya
   calculados; este no necesita ningun preprocesamiento externo.
   Se entrena solo con train.csv (igual que en el paso 6), asi las metricas de
   test del paso 7 siguen siendo validas para este archivo, y se VERIFICA que da
   las mismas probabilidades que modelo_paso6.joblib.

Salidas (punto 6: en modelos/polaridad/<VERSION_MODELO>/, registrada como candidata):
  - modelo_final.joblib             : {"pipeline", "metadatos"}
  - modelo_final_metadatos.json     : metricas, parametros, datos y versiones (legible)
  - firma.json                      : predicciones de referencia (versiones_modelo.py)
  - requirements_modelo.txt         : versiones exactas para cargar el modelo
  - paso8_salida.txt                : copia de la consola
"""

import ast
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

from carga_segura import cargar_joblib_confiable, huella_en_registro
from versiones_modelo import leer_registro, registrar, ruta_version
from pipeline_feedback import (CLASES, MODELO_EMBEDDINGS, REVISION_EMBEDDINGS, STOPWORDS,
                               UMBRAL_CONFIANZA, construir_pipeline, limpiar_texto)

CARPETA = Path(__file__).resolve().parent
ETIQUETA = "polaridad_basica"
VERSION_MODELO = "1.1.0"  # 1.1.0: paso "entrada" + revision de embeddings fijada


# Punto 6: cada version es inmutable. Si esta ya esta registrada no se reentrena
# encima: hay que subir VERSION_MODELO (y luego promoverla con versiones_modelo.py).
if VERSION_MODELO in leer_registro()["familias"]["polaridad"]["versiones"]:
    sys.exit(f"La version {VERSION_MODELO} de 'polaridad' ya esta registrada en modelos/registro.json: "
             "las versiones no se sobrescriben. Sube VERSION_MODELO para guardar una nueva.")
DESTINO = ruta_version("polaridad", VERSION_MODELO)  # modelos/polaridad/<version>/modelo_final.joblib
DESTINO.parent.mkdir(parents=True, exist_ok=True)


class Duplicar:
    def __init__(self, *flujos):
        self.flujos = flujos

    def write(self, texto):
        for f in self.flujos:
            f.write(texto)

    def flush(self):
        for f in self.flujos:
            f.flush()


_salida = open(CARPETA / "paso8_salida.txt", "w", encoding="utf-8")
sys.stdout = Duplicar(sys.__stdout__, _salida)


def titulo(t):
    print("\n" + "=" * 80 + f"\n{t}\n" + "=" * 80)


def sha256(ruta):
    return hashlib.sha256(Path(ruta).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# 1. Seleccion del mejor modelo
# ---------------------------------------------------------------------------

titulo("1. SELECCION DEL MEJOR MODELO")
res6 = pd.read_csv(CARPETA / "resultados_paso6.csv", sep=";")
res7 = pd.read_csv(CARPETA / "resultados_paso7.csv", sep=";")

tabla = res7[["modelo", "rol", "macro_f1_cv", "macro_f1_cv_desv", "macro_f1_test",
              "macro_f1_test_ic95_inf", "macro_f1_test_ic95_sup", "accuracy_test",
              "dif_principal_menos_este_ic95_inf", "dif_principal_menos_este_ic95_sup"]]
pd.set_option("display.width", 250, "display.max_columns", 20)
print(tabla.round(3).to_string(index=False))

candidatos = res6.sort_values("macro_f1_media", ascending=False)
elegido = candidatos.iloc[0]["candidato"]
params = ast.literal_eval(candidatos.iloc[0]["mejores_parametros"])
f7 = res7.set_index("modelo").loc[elegido]

print(f"\nElegido por macro-F1 de validacion cruzada: {elegido}")
print(f"  CV:   {f7['macro_f1_cv']:.3f} +/- {f7['macro_f1_cv_desv']:.3f}")
print(f"  Test: {f7['macro_f1_test']:.3f}  IC 95% [{f7['macro_f1_test_ic95_inf']:.3f}, "
      f"{f7['macro_f1_test_ic95_sup']:.3f}]  accuracy {f7['accuracy_test']:.3f}")

justificacion = []
for _, f in res7[res7["modelo"] != elegido].iterrows():
    inf, sup = f["dif_principal_menos_este_ic95_inf"], f["dif_principal_menos_este_ic95_sup"]
    veredicto = "no concluyente (IC incluye 0)" if inf <= 0 <= sup else \
        ("el elegido es mejor" if inf > 0 else "el elegido es peor")
    linea = f"vs {f['modelo']}: diferencia IC 95% [{inf:+.3f}, {sup:+.3f}] -> {veredicto}"
    justificacion.append(linea)
    print("  " + linea)
    # Si algun modelo fuera significativamente mejor en test, hay que revisarlo a mano
    assert not sup < 0, f"{f['modelo']} es significativamente mejor en test: revisar la seleccion"
print("  Ademas, la regresion logistica da probabilidades -> permite marcar comentarios ambiguos.")
justificacion.append("Regresion logistica: entrega probabilidades (predict_proba) para detectar ambiguos.")


# ---------------------------------------------------------------------------
# 2. Datos (con su version: huella sha256, tamano y distribucion de clases)
# ---------------------------------------------------------------------------

titulo("2. DATOS")
train = pd.read_csv(CARPETA / "train.csv", sep=";")
test = pd.read_csv(CARPETA / "test.csv", sep=";")
X_train, y_train = train["feedback"].tolist(), train[ETIQUETA].to_numpy()
X_test, y_test = test["feedback"].tolist(), test[ETIQUETA].to_numpy()

# La limpieza del pipeline debe reproducir texto_limpio exactamente
for nombre, df in [("train", train), ("test", test)]:
    iguales = sum(limpiar_texto(f) == l for f, l in zip(df["feedback"], df["texto_limpio"]))
    print(f"  Limpieza reproduce texto_limpio en {nombre}: {iguales}/{len(df)}")
    assert iguales == len(df), "La limpieza del pipeline no coincide con el preprocesado"

datos = {}
for archivo, df in [("train.csv", train), ("test.csv", test),
                    ("feedback_preprocesado.csv", None)]:
    info = {"sha256": sha256(CARPETA / archivo)}
    if df is not None:
        info["filas"] = len(df)
        info["clases"] = {c: int((df[ETIQUETA] == c).sum()) for c in CLASES}
    datos[archivo] = info
    print(f"  {archivo:<26} sha256 {info['sha256'][:16]}...  {info.get('clases', '')}")


# ---------------------------------------------------------------------------
# 3. Entrenar el Pipeline completo (solo train)
# ---------------------------------------------------------------------------

titulo("3. ENTRENAMIENTO DEL PIPELINE COMPLETO (solo train)")
pipeline = construir_pipeline(C=params["clf__C"], ngram_range=params["prep__tfidf__ngram_range"],
                              min_df=params["prep__tfidf__min_df"])
print(pipeline)
print("Calculando embeddings y entrenando...")
pipeline.fit(X_train, y_train)
assert list(pipeline.classes_) == CLASES


# ---------------------------------------------------------------------------
# 4. Verificacion: mismo comportamiento que modelo_paso6.joblib
# ---------------------------------------------------------------------------

titulo("4. VERIFICACION CONTRA modelo_paso6.joblib")
emb_pipe = pipeline.named_steps["features"].transformer_list[1][1].transform(X_train)
print(f"  Modelo de embeddings: {MODELO_EMBEDDINGS} @ {REVISION_EMBEDDINGS[:12]}")
cos = np.sum(emb_pipe * np.load(CARPETA / "embeddings_train.npy"), axis=1)
print(f"  Embeddings train vs embeddings_train.npy: coseno min = {cos.min():.6f}")
assert cos.min() > 0.999

# Se carga solo si su sha256 coincide con el registrado por el paso 7
anterior = cargar_joblib_confiable(
    CARPETA / "modelo_paso6.joblib",
    sha256=huella_en_registro(CARPETA / "registro_modelo_paso7.json", "modelo_paso6.joblib"))["modelo"]
cols = [f"emb_{i}" for i in range(384)]
X_test_combo = pd.concat([test[["texto_limpio"]].reset_index(drop=True),
                          pd.DataFrame(np.load(CARPETA / "embeddings_test.npy"), columns=cols)], axis=1)
p_ant = anterior.predict_proba(X_test_combo)
p_new = pipeline.predict_proba(X_test)
dif = np.abs(p_ant - p_new).max()
pred_new = pipeline.predict(X_test)
pred_paso7 = pd.read_csv(CARPETA / "predicciones_test.csv", sep=";")[f"pred_{elegido}"].to_numpy()
print(f"  Diferencia max. de probabilidades en test: {dif:.2e}")
print(f"  Predicciones iguales a las del paso 7: {(pred_new == pred_paso7).sum()}/{len(test)}")
assert dif < 1e-3 and (pred_new == pred_paso7).all(), "El pipeline no reproduce el modelo del paso 6"

from sklearn.metrics import accuracy_score, f1_score  # noqa: E402
macro_f1 = f1_score(y_test, pred_new, average="macro")
print(f"  Macro-F1 test = {macro_f1:.4f} (paso 7: {f7['macro_f1_test']:.4f})")


# ---------------------------------------------------------------------------
# 5. Guardar
# ---------------------------------------------------------------------------

titulo("5. GUARDADO")
paquetes = ["scikit-learn", "joblib", "numpy", "pandas", "scipy",
            "sentence-transformers", "torch", "transformers"]
versiones = {"python": platform.python_version(), "sistema": platform.platform()}
for p in paquetes:
    try:
        versiones[p] = metadata.version(p)
    except metadata.PackageNotFoundError:
        pass
for k, v in versiones.items():
    print(f"  {k:<22} {v}")

metadatos = {
    "nombre": elegido,
    "version_modelo": VERSION_MODELO,
    "fecha": datetime.now().isoformat(timespec="seconds"),
    "tarea": f"Clasificar la polaridad ({', '.join(CLASES)}) de comentarios de estudiantes",
    "entrada": "lista de comentarios en texto original (columna 'feedback'); sin preprocesar",
    "salida": CLASES,
    "seleccion": {
        "criterio": "mayor macro-F1 en validacion cruzada estratificada 5-fold (paso 6); "
                    "test solo para confirmar",
        "justificacion": justificacion,
    },
    "metricas": {
        "macro_f1_cv": float(f7["macro_f1_cv"]),
        "macro_f1_cv_desv": float(f7["macro_f1_cv_desv"]),
        "macro_f1_test": float(macro_f1),
        "macro_f1_test_ic95": [float(f7["macro_f1_test_ic95_inf"]), float(f7["macro_f1_test_ic95_sup"])],
        "accuracy_test": float(accuracy_score(y_test, pred_new)),
        "por_clase_test": {c: {m: float(f7[f"{m}_{c}"]) for m in ["precision", "recall", "f1"]}
                           for c in CLASES},
    },
    "parametros": {
        "pasos_pipeline": ["entrada", "features (tfidf: limpieza + vectorizador | embeddings)", "clf"],
        "preprocesamiento": "minusculas, sin tildes, sin puntuacion, sin stopwords (pipeline_feedback.py)",
        "stopwords": sorted(STOPWORDS),
        "tfidf": {"ngram_range": list(params["prep__tfidf__ngram_range"]),
                  "min_df": params["prep__tfidf__min_df"], "sublinear_tf": True},
        "embeddings": {"modelo": MODELO_EMBEDDINGS, "revision": REVISION_EMBEDDINGS,
                       "dimension": 384, "normalizados": True, "texto": "feedback original"},
        "clasificador": {"tipo": "LogisticRegression", "C": params["clf__C"],
                         "class_weight": "balanced", "max_iter": 5000},
        "umbral_ambiguo": UMBRAL_CONFIANZA,
    },
    "datos": datos,
    "entrenado_con": "train.csv (491 comentarios); test.csv no se uso para entrenar",
    "versiones": versiones,
    "dependencia_codigo": "pipeline_feedback.py debe estar importable al cargar el .joblib",
}

joblib.dump({"pipeline": pipeline, "metadatos": metadatos}, DESTINO, compress=3)
metadatos["sha256_modelo_final.joblib"] = sha256(DESTINO)
with open(DESTINO.with_name("modelo_final_metadatos.json"), "w", encoding="utf-8") as f:
    json.dump(metadatos, f, ensure_ascii=False, indent=2)

with open(CARPETA / "requirements_modelo.txt", "w", encoding="utf-8") as f:
    f.write(f"# Versiones con las que se guardo modelo_final.joblib (Python {versiones['python']})\n")
    f.write("# scikit-learn debe ser EXACTAMENTE esta version para cargar el modelo sin riesgos\n")
    for p in paquetes:
        if p in versiones:
            # sin etiqueta local (+cpu, +cu121): la variante de torch la elige el Dockerfile
            f.write(f"{p}=={versiones[p].split('+')[0]}\n")

tam = DESTINO.stat().st_size / 1024
print(f"\nGuardados en {DESTINO.parent.relative_to(CARPETA).as_posix()}/: modelo_final.joblib ({tam:.0f} KB), "
      "modelo_final_metadatos.json")
print("           requirements_modelo.txt, paso8_salida.txt")

# Punto 6: queda registrada como CANDIDATA; la API sigue con la activa hasta promoverla
res = registrar("polaridad", DESTINO, motivo=f"entrenada con paso8_guardar_modelo.py ({elegido})")
print(f"Registrada polaridad {res['version']} en modelos/registro.json (activa: {res['activa']}).")
if res["activa"] != res["version"]:
    print(f"Para ponerla en servicio: python versiones_modelo.py promover polaridad {res['version']} "
          '--motivo "..."')
sys.stdout = sys.__stdout__
_salida.close()
