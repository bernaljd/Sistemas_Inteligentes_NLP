"""
Pruebas del punto 4 (imagen 3): compatibilidad de versiones entre
entrenamiento y produccion.

Ejecutar (en la carpeta del proyecto):  python -m pytest tests/test_compatibilidad.py -v   (o: python tests/test_compatibilidad.py)

Comprueba:
  - la politica de compatibilidad.py (criticas vs. exactas, parche de Python,
    etiquetas locales como +cpu, modo estricto);
  - que la API no deserializa el modelo si las versiones criticas difieren;
  - que TODAS las fuentes de versiones digan lo mismo: metadatos del modelo,
    requirements_modelo.txt, requirements*.txt y el Dockerfile.
"""

import json
import os
import re
import shutil
import tempfile
from pathlib import Path

import joblib

from proyecto import CARPETA  # noqa: E402  # primero: agrega el proyecto al sys.path
import compatibilidad as cp  # noqa: E402
from carga_segura import ruta_metadatos, sha256_archivo
from versiones_modelo import ruta_activa

META = json.loads(ruta_metadatos(ruta_activa("polaridad")).read_text(encoding="utf-8"))
REGISTRADAS = META["versiones"]


def _con(**cambios):
    """Versiones registradas del modelo con algunos paquetes cambiados."""
    v = dict(REGISTRADAS)
    v.update({k.replace("_", "-"): val for k, val in cambios.items()})
    return v


# ---------------------------------------------------------------------------
# Politica
# ---------------------------------------------------------------------------

def test_entorno_actual_es_compatible():
    r = cp.comparar(REGISTRADAS, estricto=True)
    assert r["estado"] == "ok", r["diferencias"]


def test_scikit_learn_distinto_impide_arrancar():
    r = cp.comparar(_con(scikit_learn="1.8.0"), estricto=False)
    assert r["estado"] == "incompatible"
    assert r["diferencias"] == [{"paquete": "scikit-learn", "registrada": "1.8.0",
                                 "instalada": cp.instalada("scikit-learn"), "nivel": "critico"}]
    try:
        cp.exigir_compatibles(_con(scikit_learn="1.8.0"), estricto=False)
        raise AssertionError("no lanzo VersionesIncompatibles")
    except cp.VersionesIncompatibles as e:
        assert "scikit-learn" in str(e) and "requirements_modelo.txt" in str(e)


def test_joblib_distinto_es_critico():
    assert cp.comparar(_con(joblib="1.4.2"), estricto=False)["estado"] == "incompatible"


def test_numpy_distinto_avisa_y_en_estricto_es_critico():
    r = cp.comparar(_con(numpy="2.0.0"), estricto=False)
    assert r["estado"] == "avisos" and r["diferencias"][0]["nivel"] == "aviso"
    assert cp.exigir_compatibles(_con(numpy="2.0.0"), estricto=False)["estado"] == "avisos"
    assert cp.comparar(_con(numpy="2.0.0"), estricto=True)["estado"] == "incompatible"


def test_python_solo_cuenta_mayor_menor():
    mm = ".".join(cp.instalada("python").split(".")[:2])
    assert cp.comparar(_con(python=f"{mm}.99"), estricto=True)["estado"] == "ok"   # otro parche
    assert cp.comparar(_con(python="3.11.9"), estricto=False)["estado"] == "incompatible"


def test_etiqueta_local_no_cuenta():
    torch = cp.instalada("torch").split("+")[0]
    assert cp.comparar(_con(torch=torch), estricto=True)["estado"] == "ok"
    assert cp.comparar(_con(torch=torch + "+cu121"), estricto=True)["estado"] == "ok"


def test_paquete_no_instalado_es_diferencia():
    r = cp.comparar(dict(REGISTRADAS, **{"scikit-learn": REGISTRADAS["scikit-learn"]}), estricto=False)
    assert r["estado"] == "ok"
    original = cp.instalada
    cp.instalada = lambda p: None if p == "scipy" else original(p)
    try:
        r = cp.comparar(REGISTRADAS, estricto=False)
        assert r["estado"] == "avisos" and r["diferencias"][0]["instalada"] is None
    finally:
        cp.instalada = original


def test_variable_versiones_estrictas():
    anterior = os.environ.get("VERSIONES_ESTRICTAS")
    try:
        os.environ["VERSIONES_ESTRICTAS"] = "1"
        assert cp.comparar(_con(numpy="2.0.0"))["estado"] == "incompatible"
        os.environ["VERSIONES_ESTRICTAS"] = "0"
        assert cp.comparar(_con(numpy="2.0.0"))["estado"] == "avisos"
    finally:
        os.environ.pop("VERSIONES_ESTRICTAS", None)
        if anterior is not None:
            os.environ["VERSIONES_ESTRICTAS"] = anterior


# ---------------------------------------------------------------------------
# La API comprueba las versiones ANTES de deserializar
# ---------------------------------------------------------------------------

class _Marca:
    def __init__(self, ruta):
        self.ruta = str(ruta)

    def __reduce__(self):
        return (os.mkdir, (self.ruta,))


def test_api_no_deserializa_con_versiones_incompatibles():
    import api

    tmp = Path(tempfile.mkdtemp(prefix="compat_"))
    marca = tmp / "SE_DESERIALIZO"
    ruta = tmp / "modelo_final.joblib"
    joblib.dump(_Marca(marca), ruta)  # si se deserializa, crea la carpeta marca
    ruta_metadatos(ruta).write_text(json.dumps({
        f"sha256_{ruta.name}": sha256_archivo(ruta),       # huella valida: pasaria el punto 3
        "versiones": _con(scikit_learn="1.8.0"),           # pero entrenado con otro sklearn
    }), encoding="utf-8")
    try:
        api.cargar_servicio(ruta, carpeta_permitida=tmp)
        raise AssertionError("el servicio arranco con scikit-learn distinto")
    except cp.VersionesIncompatibles as e:
        assert "scikit-learn" in str(e)
    assert not marca.exists(), "se deserializo el modelo antes de comprobar las versiones"
    shutil.rmtree(tmp, ignore_errors=True)


def test_api_sin_versiones_registradas_no_arranca():
    import api

    tmp = Path(tempfile.mkdtemp(prefix="compat_"))
    shutil.copy2(ruta_activa("polaridad"), tmp)
    meta = {k: v for k, v in META.items() if k != "versiones"}
    ruta_metadatos(tmp / "modelo_final.joblib").write_text(json.dumps(meta), encoding="utf-8")
    try:
        api.cargar_servicio(tmp / "modelo_final.joblib", carpeta_permitida=tmp)
        raise AssertionError("arranco sin versiones registradas")
    except cp.VersionesIncompatibles as e:
        assert "no registra las versiones" in str(e)
    shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Todas las fuentes de versiones dicen lo mismo
# ---------------------------------------------------------------------------

def test_requirements_modelo_igual_a_metadatos():
    req = cp.leer_requirements(CARPETA / "requirements_modelo.txt")
    for paquete, version in REGISTRADAS.items():
        if paquete in ("python", "sistema"):
            continue
        assert req.get(paquete) == version.split("+")[0], f"{paquete}: {req.get(paquete)} vs {version}"
    assert set(req) == {p for p in REGISTRADAS if p not in ("python", "sistema")}
    # la cabecera dice con que Python se guardo
    assert f"Python {REGISTRADAS['python']}" in (CARPETA / "requirements_modelo.txt").read_text(encoding="utf-8")


def test_todos_los_requirements_usan_las_versiones_del_modelo():
    modelo = cp.leer_requirements(CARPETA / "requirements_modelo.txt")
    for nombre in ["requirements.txt", "requirements_app.txt", "requirements_dev.txt"]:
        texto = (CARPETA / nombre).read_text(encoding="utf-8")
        assert re.search(r"^-r requirements_(modelo|app)\.txt", texto, re.M), f"{nombre} no incluye el modelo"
        req = cp.leer_requirements(CARPETA / nombre)
        for paquete, version in modelo.items():
            assert req[paquete] == version, f"{nombre}: {paquete} {req[paquete]} vs modelo {version}"
        # ninguna dependencia sin version fija
        for linea in texto.splitlines():
            linea = linea.split("#")[0].strip()
            if linea and not linea.startswith("-r "):
                assert "==" in linea, f"{nombre}: '{linea}' sin version fija"


def test_dockerfile_usa_las_versiones_del_modelo():
    docker = (CARPETA / "Dockerfile").read_text(encoding="utf-8")
    modelo = cp.leer_requirements(CARPETA / "requirements_modelo.txt")
    torch = re.search(r"pip install torch==([\w.]+)", docker).group(1)
    assert torch == modelo["torch"], f"Dockerfile torch {torch} vs requirements_modelo {modelo['torch']}"
    python = re.search(r"^FROM python:(\d+\.\d+)", docker, re.M).group(1)
    assert python == ".".join(REGISTRADAS["python"].split(".")[:2])
    assert "RUN pip check" in docker
    assert "compatibilidad.py" in docker and "!compatibilidad.py" in (CARPETA / ".dockerignore").read_text()


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
