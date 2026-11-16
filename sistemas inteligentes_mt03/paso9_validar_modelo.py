"""
Paso 9: Validar la carga del modelo y sus predicciones.

"Entorno limpio": el script copia a una carpeta temporal VACIA solo lo minimo
  - modelo_final.joblib, su huella (modelo_final_metadatos.json),
    pipeline_feedback.py, carga_segura.py y este script
  - casos_referencia.csv (comentarios de test + prediccion esperada del paso 7)
y se vuelve a ejecutar alli en un proceso nuevo con `python -I` (aislado: sin
variables PYTHON*, sin la carpeta del proyecto en sys.path). Asi se comprueba
que el .joblib no depende de train.csv, embeddings_*.npy ni de otros scripts.

Pruebas:
  A. Carga          : huella sha256 = registrada y un archivo alterado se rechaza
                      sin deserializarlo (carga_segura.py), sin errores ni avisos de
                      version, versiones instaladas = registradas para Python y los
                      8 paquetes del modelo (compatibilidad.py, modo estricto).
  B. Coherencia     : mismas predicciones que en el paso 7 (123/123), mismo macro-F1,
                      determinismo, lote vs. uno a uno, probabilidades suman 1.
  C. Textos vacios  : "", espacios, None, NaN, solo signos, solo stopwords.
  D. Errores        : tipos invalidos, string suelto, lista vacia.
  E. Ambiguos       : comentarios mixtos y casos claros; tasa de error en test
                      segun la confianza (descriptivo, el umbral NO se ajusta).

Uso:  python paso9_validar_modelo.py
Salidas: paso9_salida.txt, validacion_casos.csv
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

CODIGO = ["pipeline_feedback.py", "carga_segura.py", "compatibilidad.py"]
MODELO = ["modelo_final.joblib", "modelo_final_metadatos.json"]  # de modelos/polaridad/<version>/


# ---------------------------------------------------------------------------
# Proceso padre: prepara la carpeta limpia y lanza el hijo
# ---------------------------------------------------------------------------

def lanzar():
    import pandas as pd

    from versiones_modelo import resolver

    proyecto = Path(__file__).resolve().parent
    # Punto 6: se valida la version activa, o la indicada con --version X.Y.Z
    version = sys.argv[sys.argv.index("--version") + 1] if "--version" in sys.argv else None
    info = resolver("polaridad", version)
    print(f"Version validada: polaridad {info['version']} ({info['fuente']})")
    with tempfile.TemporaryDirectory(prefix="entorno_limpio_") as tmp:
        tmp = Path(tmp)
        for f in CODIGO:
            shutil.copy2(proyecto / f, tmp / f)
        for f in MODELO:
            shutil.copy2(info["ruta"].with_name(f), tmp / f)
        shutil.copy2(Path(__file__), tmp / Path(__file__).name)

        test = pd.read_csv(proyecto / "datos" / "test.csv", sep=";")
        pred7 = pd.read_csv(proyecto / "resultados" / "predicciones_test.csv", sep=";")
        assert (test["id"].to_numpy() == pred7["id"].to_numpy()).all()
        pd.DataFrame({"feedback": test["feedback"], "real": test["polaridad_basica"],
                      "esperada": pred7["pred_LogReg + TF-IDF+Emb"]}
                     ).to_csv(tmp / "casos_referencia.csv", index=False, sep=";")

        print(f"Carpeta limpia: {tmp}")
        print(f"Contenido: {sorted(p.name for p in tmp.iterdir())}\n")
        proc = subprocess.run([sys.executable, "-I", Path(__file__).name, "--hijo"],
                              cwd=tmp, capture_output=True, text=True, encoding="utf-8",
                              env={**_env_minimo(), "PYTHONIOENCODING": "utf-8"})
        salida = proc.stdout + (("\n[stderr]\n" + proc.stderr) if proc.returncode else "")
        print(salida)
        (proyecto / "resultados" / "paso9_salida.txt").write_text(salida, encoding="utf-8")
        if (tmp / "validacion_casos.csv").exists():
            shutil.copy2(tmp / "validacion_casos.csv", proyecto / "resultados" / "validacion_casos.csv")
    print("Guardados: paso9_salida.txt, validacion_casos.csv")
    sys.exit(proc.returncode)


def _env_minimo():
    # Solo lo necesario para encontrar Python, la cache de Hugging Face y el sistema
    import os
    claves = ["PATH", "SYSTEMROOT", "TEMP", "TMP", "USERPROFILE", "HOME", "HF_HOME",
              "LOCALAPPDATA", "APPDATA", "HOMEDRIVE", "HOMEPATH"]
    return {k: os.environ[k] for k in claves if k in os.environ}


# ---------------------------------------------------------------------------
# Proceso hijo: pruebas en la carpeta limpia
# ---------------------------------------------------------------------------

def validar():
    import hashlib
    import warnings

    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.metrics import f1_score

    resultados = []

    def check(grupo, nombre, ok, detalle=""):
        resultados.append((grupo, nombre, bool(ok)))
        print(f"  [{'OK ' if ok else 'FALLA'}] {nombre}" + (f"  -> {detalle}" if detalle else ""))

    def titulo(t):
        print("\n" + "=" * 80 + f"\n{t}\n" + "=" * 80)

    print("Python", sys.version.split()[0], "| carpeta:", Path.cwd())
    print("sys.path contiene la carpeta del proyecto:",
          any("sistemas inteligentes_mt03" in p for p in sys.path))

    # ---------------- A. Carga ----------------
    titulo("A. CARGA DEL MODELO")
    # carga_segura.py se importa desde su archivo, SIN poner la carpeta en sys.path,
    # para que pipeline_feedback.py siga sin ser importable en la prueba 1.
    import importlib.util
    spec = importlib.util.spec_from_file_location("carga_segura", Path("carga_segura.py").resolve())
    cs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cs)
    spec = importlib.util.spec_from_file_location("compatibilidad", Path("compatibilidad.py").resolve())
    compat = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compat)

    # 0) Seguridad en la deserializacion (imagen 3, punto 3): huella sha256
    registrada, fuente = cs.huella_esperada(Path("modelo_final.joblib").resolve())
    real = hashlib.sha256(Path("modelo_final.joblib").read_bytes()).hexdigest()
    check("A", "Huella sha256 = registrada", real == registrada, f"{real[:16]}... ({fuente})")
    datos = bytearray(Path("modelo_final.joblib").read_bytes())
    datos[len(datos) // 2] ^= 0xFF
    Path("alterado.joblib").write_bytes(bytes(datos))
    try:
        cs.cargar_joblib_confiable("alterado.joblib", sha256=registrada)
        check("A", "Archivo alterado -> rechazado sin deserializar", False, "se cargo")
    except cs.ModeloNoConfiable as e:
        check("A", "Archivo alterado -> rechazado sin deserializar", True, str(e)[:60] + "...")

    # 1) Sin el modulo pipeline_feedback importable, la carga debe fallar de forma clara
    try:
        cs.cargar_joblib_confiable("modelo_final.joblib")
        check("A", "Sin pipeline_feedback.py importable -> error claro", False, "cargo igual")
    except ModuleNotFoundError as e:
        check("A", "Sin pipeline_feedback.py importable -> error claro", True, str(e))
    # 2) Con el modulo en sys.path (como debe hacerse al desplegar)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    with warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter("always")
        paquete = cs.cargar_joblib_confiable("modelo_final.joblib")  # verifica sha256 antes
    avisos = [a for a in avisos if "sklearn" in str(a.category.__module__) or
              "Version" in a.category.__name__]
    modelo, meta = paquete["pipeline"], paquete["metadatos"]
    check("A", "joblib.load sin avisos de version", not avisos, "; ".join(str(a.message) for a in avisos))
    print(f"     Modelo: {meta['nombre']} v{meta['version_modelo']} ({meta['fecha']})")
    # Punto 4: misma politica que la API (compatibilidad.py). Criticas: Python
    # mayor.menor, scikit-learn y joblib; exactas: numpy, scipy, pandas, torch...
    reporte = compat.comparar(meta["versiones"], estricto=True)
    for paquete, v in reporte["versiones"].items():
        check("A", f"{paquete} instalado == registrado ({v['nivel']})", v["ok"],
              f"instalado {v['instalada']}, registrado {v['registrada']}")
    check("A", "Clases del modelo", list(modelo.classes_) == meta["salida"], str(list(modelo.classes_)))
    # Pipeline unificado: todas las transformaciones viajan dentro del mismo objeto
    pasos = list(modelo.named_steps)
    check("A", "Pipeline unico: entrada -> features -> clf", pasos == ["entrada", "features", "clf"], str(pasos))
    emb = modelo.named_steps["features"].transformer_list[1][1]
    check("A", "Embeddings con revision fijada", emb.revision == meta["parametros"]["embeddings"]["revision"],
          f"{emb.nombre_modelo} @ {emb.revision[:12]}")

    from pipeline_feedback import CLASES, predecir  # noqa: E402

    # ---------------- B. Coherencia ----------------
    titulo("B. COHERENCIA DE LAS PREDICCIONES (123 comentarios de test)")
    ref = pd.read_csv("casos_referencia.csv", sep=";")
    textos = ref["feedback"].tolist()
    pred = modelo.predict(textos)
    iguales = int((pred == ref["esperada"].to_numpy()).sum())
    check("B", "Predicciones = paso 7", iguales == len(ref), f"{iguales}/{len(ref)}")
    f1 = f1_score(ref["real"], pred, average="macro")
    check("B", "Macro-F1 = registrado", abs(f1 - meta["metricas"]["macro_f1_test"]) < 1e-9,
          f"{f1:.4f} vs {meta['metricas']['macro_f1_test']:.4f}")
    probas = modelo.predict_proba(textos)
    check("B", "Probabilidades suman 1", np.allclose(probas.sum(axis=1), 1))
    check("B", "Determinista (2a ejecucion igual)", np.allclose(probas, modelo.predict_proba(textos)))
    uno_a_uno = np.vstack([modelo.predict_proba([t]) for t in textos[:15]])
    check("B", "Lote = uno a uno", np.allclose(uno_a_uno, probas[:15], atol=1e-5),
          f"dif max {np.abs(uno_a_uno - probas[:15]).max():.1e}")
    invertido = modelo.predict_proba(textos[::-1])[::-1]
    check("B", "No depende del orden", np.allclose(invertido, probas, atol=1e-5))
    df_seguro = predecir(modelo, textos)
    check("B", "predecir() coincide con pipeline.predict",
          (df_seguro["prediccion"].to_numpy() == pred).all())

    # ---------------- C. Textos vacios ----------------
    titulo("C. TEXTOS VACIOS")
    vacios = ["", "   ", "\n\t", None, float("nan"), "!!! ... ???", "el de la que y"]
    r = predecir(modelo, vacios)
    for t, (_, f) in zip(vacios, r.iterrows()):
        check("C", f"{t!r:<18} -> estado={f['estado']}",
              f["estado"] == "texto_vacio" and pd.isna(f["prediccion"]))
    # Sin la capa de validacion, el pipeline igual predice algo: por eso se usa predecir()
    crudo = modelo.predict(["", "   "])
    print(f"     (pipeline.predict sin validar devuelve {list(crudo)} para textos vacios:"
          f" prediccion sin sentido -> usar predecir())")
    mezcla = predecir(modelo, ["", "Excelente curso, aprendi muchisimo.", None])
    check("C", "Vacios mezclados con validos no afectan a los validos",
          list(mezcla["estado"].iloc[[0, 2]]) == ["texto_vacio"] * 2 and pd.notna(mezcla["prediccion"].iloc[1]))

    # ---------------- D. Errores de entrada ----------------
    titulo("D. ERRORES DE ENTRADA")

    def lanza(func, tipo):
        try:
            func()
            return False, "no lanzo excepcion"
        except tipo as e:
            return True, f"{type(e).__name__}: {e}"
        except Exception as e:  # excepcion distinta a la esperada
            return False, f"{type(e).__name__}: {e}"

    for nombre, entrada in [("numero", [123]), ("diccionario", [{"texto": "hola"}]),
                            ("lista anidada", [["bueno"]])]:
        ok, det = lanza(lambda e=entrada: predecir(modelo, e), TypeError)
        check("D", f"{nombre} -> TypeError claro", ok, det)
    r = predecir(modelo, "El profesor explica muy bien y con paciencia.")
    check("D", "String suelto (no lista) se acepta", len(r) == 1 and pd.notna(r["prediccion"].iloc[0]),
          f"-> {r['prediccion'].iloc[0]}")
    # El paso "entrada" del Pipeline trata un string suelto como UN comentario
    # (en la version 1.0.0 devolvia una prediccion por caracter).
    try:
        n = len(modelo.predict("texto suelto"))
        det = f"{n} prediccion(es)"
    except Exception as e:
        n, det = -1, f"{type(e).__name__}: {str(e)[:70]}"
    check("D", "pipeline.predict con string suelto -> 1 prediccion", n == 1, det)
    ok, det = lanza(lambda: modelo.predict([123]), TypeError)
    check("D", "pipeline.predict con numero -> TypeError claro", ok, det)
    r = predecir(modelo, [])
    check("D", "Lista vacia -> DataFrame vacio", len(r) == 0)
    r = predecir(modelo, ["ExCeLeNtE  PROFESOR!!!  explica MUY bien 😀😀", "pésimo, no aprendí nada"])
    check("D", "Mayusculas, emojis y tildes no rompen", r["prediccion"].notna().all(),
          f"{list(r['prediccion'])}")

    # ---------------- E. Claros y ambiguos ----------------
    titulo("E. COMENTARIOS CLAROS Y AMBIGUOS")
    claros = [
        ("Excelente profesor, las clases son muy claras y aprendi muchisimo.", "positivo"),
        ("El curso me encanto, muy bien organizado y util.", "positivo"),
        ("Pesimo curso, el profesor no explica y las evaluaciones son injustas.", "negativo"),
        ("Muy desorganizado, no aprendi nada y nunca respondia dudas.", "negativo"),
    ]
    ambiguos = [
        "El profesor sabe mucho, pero los parciales son injustos.",
        "Algunas clases fueron buenas y otras muy aburridas.",
        "El curso es normal.",
        "No estuvo mal, aunque podria mejorar.",
        "Interesante pero demasiado teorico, no se si lo recomendaria.",
        "ok",
    ]
    # Los aciertos del modelo ya se midieron en el paso 7 (accuracy 0.667): aqui un
    # error en un caso claro se REPORTA, pero no invalida la carga del modelo.
    r_claros = predecir(modelo, [t for t, _ in claros])
    aciertos_claros = 0
    print("  Comentarios claros (informativo):")
    for (t, esperado), (_, f) in zip(claros, r_claros.iterrows()):
        acierta = f["prediccion"] == esperado
        aciertos_claros += acierta
        print(f"    [{'acierta' if acierta else 'ERROR DEL MODELO'}] {t[:55]:<55} -> "
              f"{f['prediccion']} ({f['confianza']:.2f})" + ("" if acierta else f", esperado {esperado}"))
    check("E", "Casos claros: mayoria correcta", aciertos_claros > len(claros) / 2,
          f"{aciertos_claros}/{len(claros)}")
    r_amb = predecir(modelo, ambiguos)
    print("\n  Comentarios mixtos (informativo):")
    for _, f in r_amb.iterrows():
        probs = " ".join(f"{c[:3]}={f[f'prob_{c}']:.2f}" for c in CLASES)
        print(f"    {f['texto'][:55]:<55} -> {f['prediccion']:<9} [{f['estado']:<7}] {probs}")
    conf_claros, conf_amb = r_claros["confianza"].mean(), r_amb["confianza"].mean()
    check("E", "Confianza media: claros > mixtos", conf_claros > conf_amb,
          f"{conf_claros:.2f} vs {conf_amb:.2f}")

    print(f"\n  En test, segun el umbral fijo de {meta['parametros']['umbral_ambiguo']} (no ajustado con test):")
    df_seguro["real"] = ref["real"]
    for estado, g in df_seguro.groupby("estado"):
        print(f"    {estado:<8} n = {len(g):>3}  accuracy = {(g['prediccion'] == g['real']).mean():.3f}")

    # ---------------- Resumen ----------------
    casos = pd.concat([predecir(modelo, vacios[:3] + ["!!! ... ???", "el de la que y"]).assign(grupo="C vacio"),
                       r_claros.assign(grupo="E claro"), r_amb.assign(grupo="E mixto")])
    casos.round(3).to_csv("validacion_casos.csv", index=False, sep=";")

    titulo("RESUMEN")
    tabla = pd.DataFrame(resultados, columns=["grupo", "prueba", "ok"])
    for g, d in tabla.groupby("grupo"):
        print(f"  {g}: {d['ok'].sum()}/{len(d)} pruebas OK")
    fallas = tabla[~tabla["ok"]]
    print(f"\n  TOTAL: {tabla['ok'].sum()}/{len(tabla)}" +
          ("" if fallas.empty else f"  | FALLAS: {list(fallas['prueba'])}"))
    sys.exit(0 if fallas.empty else 1)


if __name__ == "__main__":
    validar() if "--hijo" in sys.argv else lanzar()
