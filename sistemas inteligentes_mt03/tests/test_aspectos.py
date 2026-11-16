"""
Pruebas del punto 5 (imagen 3): salida estructurada con una prediccion por aspecto.

Ejecutar (en la carpeta del proyecto):  python -m pytest tests/test_aspectos.py -v   (o: python tests/test_aspectos.py)
"""

import json
import os
import shutil
import tempfile
from pathlib import Path

os.environ.setdefault("RUTA_REGISTRO", str(Path(tempfile.mkdtemp()) / "registro_aspectos.jsonl"))

from proyecto import CARPETA, DATOS  # noqa: E402  # primero: agrega el proyecto al sys.path
import pandas as pd  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import api  # noqa: E402
from aspectos import ASPECTOS, CLAVES, GRUPO_DE, etiquetas, fragmentos  # noqa: E402
from carga_segura import ModeloNoConfiable, cargar_joblib_confiable  # noqa: E402
from versiones_modelo import ruta_activa  # noqa: E402

CAMPOS = {"aspecto", "nombre", "mencionado", "probabilidad", "sentimiento", "probabilidades_sentimiento"}
_modelo = None
_cliente = None


def modelo():
    global _modelo
    if _modelo is None:
        _modelo = cargar_joblib_confiable(ruta_activa("aspectos"), variable_sha256="ASPECTOS_SHA256")
    return _modelo


def cliente():
    global _cliente
    if _cliente is None:
        _cliente = TestClient(api.app)
        _cliente.__enter__()
    return _cliente


def _formato_valido(salida):
    assert [a["aspecto"] for a in salida] == CLAVES, "deben venir los 9 aspectos en orden fijo"
    for a in salida:
        assert set(a) == CAMPOS
        assert a["nombre"] == ASPECTOS[a["aspecto"]][0]
        if a["mencionado"]:
            assert a["probabilidad"] >= 0.5 and a["sentimiento"] in ("negativo", "neutral", "positivo")
            ps = a["probabilidades_sentimiento"]
            assert abs(sum(ps.values()) - 1) < 1e-3 and max(ps, key=ps.get) == a["sentimiento"]
        else:
            assert a["sentimiento"] is None and a["probabilidades_sentimiento"] is None
            assert a["probabilidad"] is None or a["probabilidad"] < 0.5


# ---------------------------------------------------------------------------
# Datos y reglas
# ---------------------------------------------------------------------------

def test_los_22_aspectos_de_los_datos_caen_en_9_grupos():
    df = pd.read_csv(DATOS / "feedback_preprocesado.csv", sep=";")
    nombres = {a["aspecto"] for x in df["aspectos"].dropna() for a in json.loads(x)}
    assert len(nombres) == 22 and nombres <= set(GRUPO_DE), nombres - set(GRUPO_DE)
    assert len(ASPECTOS) == 9 and set(GRUPO_DE.values()) == set(CLAVES)


def test_etiquetas_conflicto_en_un_grupo():
    # dos aspectos originales del mismo grupo con sentimientos distintos -> empate -> neutral
    e = etiquetas([{"aspecto": "parciales", "sentimiento": "negativo"},
                   {"aspecto": "diseno de la evaluacion y criterios", "sentimiento": "positivo"},
                   {"aspecto": "horario", "sentimiento": "negativo"}])
    assert e == {"evaluacion": "neutral", "organizacion": "negativo"}
    assert etiquetas("[]") == {} and etiquetas("") == {}


def test_fragmentos_separa_contrastes():
    f = fragmentos("El profesor explica muy bien, pero los parciales son injustos. Los correos tardan mucho.")
    assert f == ["El profesor explica muy bien", "los parciales son injustos", "Los correos tardan mucho"]
    assert fragmentos("ok") == ["ok"]


# ---------------------------------------------------------------------------
# Formato de salida del modelo
# ---------------------------------------------------------------------------

def test_formato_estructurado_nueve_aspectos():
    textos = ["El profesor sabe mucho, pero los parciales son injustos y el horario es terrible.",
              "Excelente curso, el material de clase y las diapositivas son muy buenos."]
    for salida in modelo()["modelo"].predecir(textos):
        _formato_valido(salida)
    assert any(a["mencionado"] for s in modelo()["modelo"].predecir(textos) for a in s)


def test_textos_vacios_no_predicen_aspectos():
    for salida in modelo()["modelo"].predecir(["", "   ", None, "!!!", "el de la que"]):
        _formato_valido(salida)
        assert all(not a["mencionado"] and a["probabilidad"] is None for a in salida)


def test_lote_igual_a_uno_a_uno_y_determinista():
    textos = pd.read_csv(DATOS / "test.csv", sep=";")["feedback"].tolist()[:8]
    m = modelo()["modelo"]
    lote = m.predecir(textos)
    assert lote == m.predecir(textos)
    for t, esperado in zip(textos, lote):
        uno = m.predecir([t])[0]
        assert [a["mencionado"] for a in uno] == [a["mencionado"] for a in esperado]
        assert [a["sentimiento"] for a in uno] == [a["sentimiento"] for a in esperado]


def test_metricas_registradas_se_reproducen():
    # El modelo guardado reproduce en test la F1 de presencia registrada por el paso 11
    from sklearn.metrics import f1_score
    test = pd.read_csv(DATOS / "test.csv", sep=";")
    salida = modelo()["modelo"].predecir(test["feedback"].tolist())
    reales = [etiquetas(a if isinstance(a, str) else "[]") for a in test["aspectos"]]
    f1 = [f1_score([c in r for r in reales], [s[j]["mencionado"] for s in salida], zero_division=0)
          for j, c in enumerate(CLAVES)]
    registrada = modelo()["metadatos"]["metricas"]["presencia_macro_f1_test"]
    assert abs(sum(f1) / len(f1) - registrada) < 1e-9, (sum(f1) / len(f1), registrada)


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

def test_api_devuelve_aspectos_en_cada_prediccion():
    d = cliente().post("/predecir", json={"texto": "Los parciales son muy dificiles y el horario es malo."}).json()
    _formato_valido(d["aspectos"])
    d = cliente().post("/predecir/lote", json={"textos": ["Buen material de clase", ""]}).json()
    for r in d["resultados"]:
        _formato_valido(r["aspectos"])
    assert not any(a["mencionado"] for a in d["resultados"][1]["aspectos"])
    csv = "feedback\nEl profesor responde rapido los correos\n"
    r = cliente().post("/predecir/csv", files={"archivo": ("c.csv", csv.encode(), "text/csv")}).json()
    _formato_valido(r["resultados"][0]["aspectos"])


def test_api_salud_y_metadatos_de_aspectos():
    s = cliente().get("/salud").json()["aspectos"]
    assert s["aspectos"] == CLAVES and s["compatibilidad"]["estado"] == "ok"
    assert s["integridad"]["verificado_con"] == "modelos/registro.json"  # punto 6
    m = cliente().get("/modelo/aspectos").json()
    assert list(m["aspectos"]) == CLAVES and "presencia_macro_f1_test" in m["metricas"]


def test_servicio_sin_modelo_de_aspectos_funciona():
    estado = api.cargar_servicio(ruta_aspectos=CARPETA / "no_existe.joblib")
    assert "aspectos" not in estado and "modelo" in estado


def test_huella_de_aspectos_independiente_de_la_de_polaridad():
    tmp = Path(tempfile.mkdtemp(prefix="aspectos_"))
    for familia in ("polaridad", "aspectos"):
        ruta = ruta_activa(familia)
        for f in (ruta, ruta.with_name(f"{ruta.stem}_metadatos.json")):
            shutil.copy2(f, tmp / f.name)
    anterior = {k: os.environ.get(k) for k in ("MODELO_SHA256", "ASPECTOS_SHA256")}
    try:
        # la huella de polaridad fijada por variable no debe aplicarse al modelo de aspectos
        meta = json.loads((tmp / "modelo_final_metadatos.json").read_text(encoding="utf-8"))
        os.environ["MODELO_SHA256"] = meta["sha256_modelo_final.joblib"]
        os.environ.pop("ASPECTOS_SHA256", None)
        estado = api.cargar_servicio(tmp / "modelo_final.joblib", carpeta_permitida=tmp,
                                     ruta_aspectos=tmp / "modelo_aspectos.joblib")
        assert estado["aspectos"]["integridad"]["verificado_con"] == "modelo_aspectos_metadatos.json"
        # una huella de aspectos equivocada impide arrancar
        os.environ["ASPECTOS_SHA256"] = "0" * 64
        try:
            api.cargar_servicio(tmp / "modelo_final.joblib", carpeta_permitida=tmp,
                                ruta_aspectos=tmp / "modelo_aspectos.joblib")
            raise AssertionError("arranco con una huella de aspectos equivocada")
        except ModeloNoConfiable as e:
            assert "ASPECTOS_SHA256" in str(e)
    finally:
        for k, v in anterior.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v


if __name__ == "__main__":
    pruebas = [(n, f) for n, f in dict(globals()).items() if n.startswith("test_")]
    fallas = 0
    for nombre, f in pruebas:
        try:
            f()
            print(f"[OK]    {nombre}")
        except Exception as e:
            fallas += 1
            print(f"[FALLA] {nombre}: {type(e).__name__}: {e}")
    print(f"\n{len(pruebas) - fallas}/{len(pruebas)} pruebas OK")
