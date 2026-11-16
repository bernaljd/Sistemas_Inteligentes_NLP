"""
Pruebas del punto 3 (imagen 3): seguridad en la deserializacion de joblib.

Ejecutar (en la carpeta del proyecto):  python -m pytest tests/test_carga_segura.py -v   (o: python tests/test_carga_segura.py)

La prueba central usa un .joblib MALICIOSO de verdad: al deserializarlo crea
una carpeta "marca" (inofensivo, dentro de una carpeta temporal). Si la marca
aparece, se ejecuto codigo del archivo. Se comprueba que:
  - joblib.load normal SI lo ejecuta (por eso el riesgo es real), y
  - cargar_joblib_confiable lo rechaza ANTES de deserializar (la marca no aparece).
"""

import json
import os
import re
import shutil
import tempfile
from pathlib import Path

import joblib

from proyecto import CARPETA, DATOS  # noqa: E402  # primero: agrega el proyecto al sys.path
import carga_segura  # noqa: E402
from carga_segura import (ModeloNoConfiable, cargar_joblib_confiable, ruta_metadatos,
                          sha256_archivo)
from versiones_modelo import ruta_activa

MODELO = ruta_activa("polaridad")  # modelos/polaridad/<version activa>/modelo_final.joblib


class Malicioso:
    """Al deserializarse llama a os.mkdir(marca): simula codigo arbitrario."""

    def __init__(self, marca):
        self.marca = str(marca)

    def __reduce__(self):
        return (os.mkdir, (self.marca,))


def _carpeta_con_modelo():
    """Copia el modelo real y sus metadatos (huella legitima) a una carpeta temporal."""
    tmp = Path(tempfile.mkdtemp(prefix="carga_segura_"))
    shutil.copy2(MODELO, tmp / MODELO.name)
    shutil.copy2(ruta_metadatos(MODELO), tmp / ruta_metadatos(MODELO).name)
    return tmp


def _rechaza(func, texto):
    try:
        func()
    except ModeloNoConfiable as e:
        assert texto in str(e), str(e)
        return
    raise AssertionError("se cargo un archivo que debia rechazarse")


def test_modelo_real_se_carga():
    paquete = cargar_joblib_confiable(MODELO)
    assert set(paquete) == {"pipeline", "metadatos"}
    meta = json.loads(ruta_metadatos(MODELO).read_text(encoding="utf-8"))
    assert meta[f"sha256_{MODELO.name}"] == sha256_archivo(MODELO)


def test_joblib_malicioso_no_se_ejecuta():
    tmp = _carpeta_con_modelo()
    marca = tmp / "CODIGO_EJECUTADO"
    # 1) El riesgo es real: joblib.load ejecuta el codigo del archivo
    joblib.dump(Malicioso(marca), tmp / "demo.joblib")
    joblib.load(tmp / "demo.joblib")
    assert marca.exists(), "el archivo de demostracion deberia ejecutar codigo con joblib.load"
    marca.rmdir()
    # 2) Un atacante reemplaza modelo_final.joblib; la huella registrada es la legitima
    joblib.dump(Malicioso(marca), tmp / MODELO.name)
    _rechaza(lambda: cargar_joblib_confiable(tmp / MODELO.name, carpeta_permitida=tmp), "no coincide")
    assert not marca.exists(), "cargar_joblib_confiable ejecuto codigo del archivo"


def test_archivo_alterado_un_byte():
    tmp = _carpeta_con_modelo()
    datos = bytearray((tmp / MODELO.name).read_bytes())
    datos[len(datos) // 2] ^= 0xFF
    (tmp / MODELO.name).write_bytes(bytes(datos))
    _rechaza(lambda: cargar_joblib_confiable(tmp / MODELO.name, carpeta_permitida=tmp), "no coincide")


def test_sin_huella_registrada_no_se_carga():
    tmp = _carpeta_con_modelo()
    ruta_metadatos(tmp / MODELO.name).unlink()
    _rechaza(lambda: cargar_joblib_confiable(tmp / MODELO.name, carpeta_permitida=tmp), "No hay huella")
    # metadatos sin la clave de la huella
    ruta_metadatos(tmp / MODELO.name).write_text(json.dumps({"nombre": "x"}), encoding="utf-8")
    _rechaza(lambda: cargar_joblib_confiable(tmp / MODELO.name, carpeta_permitida=tmp), "no tiene la clave")


def test_fuera_de_la_carpeta_permitida():
    tmp = _carpeta_con_modelo()  # huella correcta, pero fuera del proyecto
    _rechaza(lambda: cargar_joblib_confiable(tmp / MODELO.name), "fuera de la carpeta permitida")
    _rechaza(lambda: cargar_joblib_confiable(CARPETA / ".." / ".." / "x.joblib"), "No existe")


def test_solo_archivos_joblib():
    _rechaza(lambda: cargar_joblib_confiable(DATOS / "train.csv"), "Solo se cargan archivos .joblib")


def test_variable_de_entorno_tiene_prioridad():
    anterior = os.environ.get("MODELO_SHA256")
    try:
        os.environ["MODELO_SHA256"] = "0" * 64  # huella fijada en el despliegue que no coincide
        _rechaza(lambda: cargar_joblib_confiable(MODELO), "MODELO_SHA256")
        os.environ["MODELO_SHA256"] = "no-es-un-sha256"
        _rechaza(lambda: cargar_joblib_confiable(MODELO), "no es un sha256 valido")
        os.environ["MODELO_SHA256"] = sha256_archivo(MODELO).upper()  # mayusculas tambien valen
        assert "pipeline" in cargar_joblib_confiable(MODELO)
    finally:
        os.environ.pop("MODELO_SHA256", None)
        if anterior is not None:
            os.environ["MODELO_SHA256"] = anterior


def test_ningun_script_usa_joblib_load_directo():
    # Todo .joblib se carga con cargar_joblib_confiable (salvo demostraciones explicitas)
    permitidos = {"carga_segura.py", "test_carga_segura.py"}
    for py in [*CARPETA.glob("*.py"), *(CARPETA / "tests").glob("*.py")]:
        if py.name in permitidos:
            continue
        codigo = py.read_text(encoding="utf-8")
        assert not re.search(r"joblib\.load\(", codigo), f"{py.name} usa joblib.load sin verificar la huella"
    assert carga_segura.CARPETA == CARPETA


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
