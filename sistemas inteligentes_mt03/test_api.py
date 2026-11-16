"""
Pruebas del servicio de prediccion (pasos 4-6: funcionalidad, seguridad, monitoreo).

Ejecutar:  python -m pytest test_api.py -v
           (o sin pytest: python test_api.py)
No hace falta tener uvicorn corriendo: TestClient levanta la app en memoria.
"""

import os
import tempfile
from pathlib import Path

# El registro de monitoreo de las pruebas va a un archivo temporal, no al real
os.environ["RUTA_REGISTRO"] = str(Path(tempfile.mkdtemp()) / "registro_pruebas.jsonl")

import pandas as pd  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import api  # noqa: E402
from api import MAX_TEXTOS_LOTE, app  # noqa: E402

_cliente = None


def cliente():
    global _cliente
    if _cliente is None:
        _cliente = TestClient(app)
        _cliente.__enter__()  # ejecuta el lifespan: carga el modelo una vez
    return _cliente


def test_salud():
    r = cliente().get("/salud")
    assert r.status_code == 200
    assert r.json()["estado"] == "ok"
    assert r.json()["clases"] == ["negativo", "neutral", "positivo"]
    # punto 4: versiones instaladas = las del modelo
    assert r.json()["compatibilidad"] == {"estado": "ok", "diferencias": []}


def test_pagina_web_y_archivos():
    c = cliente()
    r = c.get("/")
    assert r.status_code == 200 and "Polaridad de comentarios para docentes" in r.text
    for archivo, tipo in [("estilos.css", "text/css"), ("app.js", "javascript"), ("profesores.svg", "image/svg")]:
        r = c.get(f"/web/{archivo}")
        assert r.status_code == 200 and tipo in r.headers["content-type"], archivo


def test_metadatos():
    r = cliente().get("/modelo")
    assert r.status_code == 200 and "metricas" in r.json()


def test_un_comentario_devuelve_clase_y_probabilidades():
    r = cliente().post("/predecir", json={"texto": "Excelente profesor, explica muy claro."})
    assert r.status_code == 200
    d = r.json()
    assert d["prediccion"] == "positivo" and d["estado"] == "ok"
    probs = d["probabilidades"]
    assert set(probs) == {"negativo", "neutral", "positivo"}
    assert abs(sum(probs.values()) - 1) < 1e-3
    assert d["confianza"] == max(probs.values())


def test_texto_vacio_no_predice():
    for texto in ["", "   ", "!!!"]:
        d = cliente().post("/predecir", json={"texto": texto}).json()
        assert d["estado"] == "texto_vacio" and d["prediccion"] is None
        assert d["probabilidades"]["negativo"] is None


def test_entradas_invalidas_dan_422():
    c = cliente()
    assert c.post("/predecir", json={}).status_code == 422
    assert c.post("/predecir", json={"texto": None}).status_code == 422
    assert c.post("/predecir", json={"texto": 123}).status_code == 422
    assert c.post("/predecir", json={"texto": "a" * 5001}).status_code == 422
    assert c.post("/predecir", content="no es json",
                  headers={"Content-Type": "application/json"}).status_code == 422
    assert c.post("/predecir/lote", json={"textos": ["ok"] * (MAX_TEXTOS_LOTE + 1)}).status_code == 422


def test_lote_coincide_con_predicciones_del_paso7():
    test = pd.read_csv("test.csv", sep=";")
    esperado = pd.read_csv("predicciones_test.csv", sep=";")["pred_LogReg + TF-IDF+Emb"]
    d = cliente().post("/predecir/lote", json={"textos": test["feedback"].tolist()}).json()
    assert d["n"] == len(test)
    assert [r["prediccion"] for r in d["resultados"]] == esperado.tolist()
    assert sum(d["resumen"].values()) == len(test)


def test_lote_vacio():
    d = cliente().post("/predecir/lote", json={"textos": []}).json()
    assert d["n"] == 0 and d["resultados"] == []


def test_csv_punto_y_coma_con_comas_en_el_texto():
    csv = "id;feedback\n1;Muy buen curso, aprendi mucho, gracias\n2;\n3;Pesimo, no explica nada\n"
    r = cliente().post("/predecir/csv", files={"archivo": ("c.csv", csv.encode("utf-8"), "text/csv")})
    assert r.status_code == 200, r.text
    estados = [x["estado"] for x in r.json()["resultados"]]
    assert len(estados) == 3 and estados[1] == "texto_vacio"


def test_csv_coma_latin1_y_otra_columna():
    csv = 'comentario,nota\n"El profesor es pésimo, nunca responde",2\n'
    r = cliente().post("/predecir/csv?columna=comentario",
                       files={"archivo": ("c.csv", csv.encode("latin-1"), "text/csv")})
    assert r.status_code == 200, r.text
    assert r.json()["resultados"][0]["texto"].startswith("El profesor es pésimo")


def test_csv_columna_inexistente_y_archivo_vacio():
    c = cliente()
    r = c.post("/predecir/csv", files={"archivo": ("c.csv", b"texto\nhola\n", "text/csv")})
    assert r.status_code == 422 and "feedback" in r.json()["detail"]
    r = c.post("/predecir/csv", files={"archivo": ("c.csv", b"", "text/csv")})
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# Eficiencia: el servicio no entrena al arrancar (imagen 3, punto 2)
# ---------------------------------------------------------------------------

def test_arranque_no_entrena():
    from unittest import mock

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import FeatureUnion, Pipeline

    import pipeline_feedback as pf

    def prohibido(*a, **k):
        raise AssertionError("El servicio intento ENTRENAR durante el arranque")

    metodos = [(c, m) for c in (Pipeline, FeatureUnion, TfidfVectorizer, LogisticRegression,
                                pf.EntradaTexto, pf.LimpiadorTexto, pf.EmbeddingsMiniLM)
               for m in ("fit", "fit_transform") if hasattr(c, m)]
    parches = [mock.patch.object(c, m, prohibido) for c, m in metodos]
    for p in parches:
        p.start()
    try:
        estado = api.cargar_servicio()  # carga + calentamiento, igual que el lifespan
        r = pf.predecir(estado["modelo"], ["Excelente curso, aprendi mucho."])
        assert r["estado"].iloc[0] == "ok"
    finally:
        for p in parches:
            p.stop()
    # el modelo servido es el entrenado en el paso 8, sin cambios
    assert estado["metadatos"]["fecha"] == cliente().get("/salud").json()["entrenado_el"]


def test_sin_modelo_falla_claro_y_no_entrena(tmp_path=None):
    import joblib

    from pipeline_feedback import construir_pipeline
    tmp = Path(tmp_path or tempfile.mkdtemp())
    for ruta, mensaje in [(tmp / "no_existe.joblib", "No existe"),
                          (tmp / "sin_entrenar.joblib", "sin entrenar")]:
        if "sin_entrenar" in ruta.name:
            # archivo de confianza (huella registrada) pero sin entrenar
            import json

            from carga_segura import ruta_metadatos, sha256_archivo
            # con las mismas versiones del modelo real (compatibles con este entorno)
            versiones = api.estado_servicio["metadatos"]["versiones"] if api.estado_servicio else \
                json.loads(ruta_metadatos(api.RUTA_MODELO).read_text(encoding="utf-8"))["versiones"]
            joblib.dump({"pipeline": construir_pipeline(), "metadatos": {"versiones": versiones}}, ruta)
            ruta_metadatos(ruta).write_text(json.dumps({f"sha256_{ruta.name}": sha256_archivo(ruta),
                                                        "versiones": versiones}))
        try:
            api.cargar_servicio(ruta, carpeta_permitida=tmp)
            raise AssertionError(f"{ruta.name}: el servicio arranco igual")
        except api.ModeloNoDisponible as e:
            assert mensaje in str(e) and "paso8_guardar_modelo.py" in str(e)


# ---------------------------------------------------------------------------
# Seguridad (paso 6)
# ---------------------------------------------------------------------------

def test_cabeceras_de_seguridad():
    r = cliente().get("/")
    h = r.headers
    assert h["X-Content-Type-Options"] == "nosniff"
    assert h["X-Frame-Options"] == "DENY"
    assert "script-src 'self'" in h["Content-Security-Policy"]
    assert "frame-ancestors 'none'" in h["Content-Security-Policy"]
    assert h["Referrer-Policy"] == "no-referrer"
    assert "X-Tiempo-ms" in h
    assert "access-control-allow-origin" not in {k.lower() for k in h}  # sin CORS abierto


def test_html_malicioso_se_devuelve_como_dato():
    ataque = '<script>alert("x")</script><img src=x onerror=alert(1)>'
    r = cliente().post("/predecir", json={"texto": ataque})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/json")
    assert r.json()["texto"] == ataque  # se devuelve tal cual; la pagina lo escapa con esc()
    assert "esc(r.texto)" in Path("web/app.js").read_text(encoding="utf-8")


def test_csv_descargado_neutraliza_formulas():
    # La descarga de la pagina escapa comentarios que Excel ejecutaria como formula
    js = Path("web/app.js").read_text(encoding="utf-8")
    assert "function sinFormula" in js and "celda = (v) => `\"${sinFormula(v)" in js
    # y la API devuelve el texto sin modificar (el escape es solo al exportar)
    ataque = '=HYPERLINK("http://ejemplo.invalid","clic")'
    assert cliente().post("/predecir", json={"texto": ataque}).json()["texto"] == ataque


def test_peticion_demasiado_grande_413():
    r = cliente().post("/predecir/lote", content=b"x" * (7 * 1024 * 1024),
                       headers={"Content-Type": "application/json"})
    assert r.status_code == 413


def test_peticion_grande_en_chunks_413():
    # Sin Content-Length (Transfer-Encoding: chunked): antes pasaba con 200
    cuerpo = b'{"textos": ["x"]}' + b" " * (7 * 1024 * 1024)
    trozos = (cuerpo[i:i + 65536] for i in range(0, len(cuerpo), 65536))
    r = cliente().post("/predecir/lote", content=trozos, headers={"Content-Type": "application/json"})
    assert r.status_code == 413, r.status_code
    # y un CSV grande enviado igual tampoco pasa
    csv = b"feedback\n" + b"buen curso\n" * (700 * 1024)
    r = cliente().post("/predecir/csv", files={"archivo": ("c.csv", csv, "text/csv")})
    assert r.status_code == 413


def test_dockerfile_no_confia_en_x_forwarded_for():
    # Con --forwarded-allow-ips='*' una IP falsa en X-Forwarded-For saltaba el limite
    cmd = [l for l in Path("Dockerfile").read_text(encoding="utf-8").splitlines() if l.startswith("CMD")][0]
    assert "--proxy-headers" not in cmd and "forwarded-allow-ips" not in cmd


def test_dockerfile_descarga_la_misma_revision_de_embeddings():
    # El contenedor no tiene internet: debe traer exactamente la revision del Pipeline
    from pipeline_feedback import MODELO_EMBEDDINGS, REVISION_EMBEDDINGS
    docker = Path("Dockerfile").read_text(encoding="utf-8")
    assert f"ARG REVISION_EMBEDDINGS={REVISION_EMBEDDINGS}" in docker
    assert f"SentenceTransformer('{MODELO_EMBEDDINGS}', revision='${{REVISION_EMBEDDINGS}}')" in docker


def test_limite_de_peticiones_429():
    original = api.limitador
    api.limitador = api.LimitadorPeticiones(maximo=3)
    try:
        codigos = [cliente().post("/predecir", json={"texto": "Buen curso"}).status_code for _ in range(5)]
        assert codigos[:3] == [200] * 3 and codigos[3:] == [429, 429], codigos
        r = cliente().post("/predecir", json={"texto": "Buen curso"})
        assert int(r.headers["Retry-After"]) > 0
        assert cliente().get("/salud").status_code == 200  # el limite solo aplica a /predecir*
    finally:
        api.limitador = original


def test_error_interno_no_filtra_detalles():
    original = api.estado_servicio["modelo"]
    class Roto:
        def predict_proba(self, X):
            raise RuntimeError("C:/ruta/secreta/modelo.py linea 42")
    api.estado_servicio["modelo"] = Roto()
    try:
        r = TestClient(app, raise_server_exceptions=False).post("/predecir", json={"texto": "hola curso"})
        assert r.status_code == 500
        assert "secreta" not in r.text and r.json()["detail"] == "Error interno del servidor"
    finally:
        api.estado_servicio["modelo"] = original


# ---------------------------------------------------------------------------
# Monitoreo (paso 6)
# ---------------------------------------------------------------------------

def test_monitoreo_registra_sin_texto_y_reporta():
    secreto = "El profesor Perez de Sistemas Operativos explica muy bien"
    cliente().post("/predecir", json={"texto": secreto})
    registro = Path(os.environ["RUTA_REGISTRO"]).read_text(encoding="utf-8")
    assert "Perez" not in registro and '"hash"' in registro  # privacidad: no se guarda el texto
    test = pd.read_csv("test.csv", sep=";")
    cliente().post("/predecir/lote", json={"textos": test["feedback"].tolist()})
    rep = cliente().get("/monitoreo?ultimos=123").json()
    assert rep["estado"] == "estable", rep["alertas"]
    assert {"n_palabras", "oov", "confianza", "clases_predichas"} <= set(rep["indicadores"])


def test_monitoreo_protegido_con_token():
    api.ADMIN_TOKEN = "clave-de-prueba"
    try:
        assert cliente().get("/monitoreo").status_code == 401
        assert cliente().get("/monitoreo", headers={"X-Admin-Token": "otra"}).status_code == 401
        assert cliente().get("/monitoreo", headers={"X-Admin-Token": "clave-de-prueba"}).status_code == 200
    finally:
        api.ADMIN_TOKEN = ""


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
