"""
Pruebas del punto 6 (imagen 3): control y reversion de versiones de los modelos.

Las operaciones que cambian el registro se prueban sobre una COPIA de modelos/
en una carpeta temporal: el registro real no se toca.

Ejecutar:  python -m pytest test_versiones_modelo.py -v   (o: python test_versiones_modelo.py)
"""

import json
import os
import shutil
import tempfile
from pathlib import Path
from unittest import mock

os.environ.setdefault("RUTA_REGISTRO", str(Path(tempfile.mkdtemp()) / "registro_versiones.jsonl"))

import joblib  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import api  # noqa: E402
import pipeline_feedback as pf  # noqa: E402
import versiones_modelo as vm  # noqa: E402
from carga_segura import ModeloNoConfiable, cargar_joblib_confiable, ruta_metadatos, sha256_archivo  # noqa: E402
from compatibilidad import VersionesIncompatibles  # noqa: E402

CARPETA = Path(__file__).resolve().parent
_cliente = None


def cliente():
    global _cliente
    if _cliente is None:
        _cliente = TestClient(api.app)
        _cliente.__enter__()
    return _cliente


def _copia():
    """Copia de modelos/ para probar operaciones que escriben el registro."""
    tmp = Path(tempfile.mkdtemp(prefix="versiones_")) / "modelos"
    shutil.copytree(vm.CARPETA_MODELOS, tmp)
    return tmp


def _candidata(tmp, version="1.1.1", cambiar=None, versiones=None):
    """Un modelo 'recien entrenado' (el 1.1.0 con otra version) fuera de modelos/."""
    paq = cargar_joblib_confiable(vm.ruta_activa("polaridad"))
    paq["metadatos"] = {**paq["metadatos"], "version_modelo": version, "fecha": "2026-11-16T10:00:00"}
    if versiones:
        paq["metadatos"]["versiones"] = versiones
    if cambiar:
        cambiar(paq)
    ruta = tmp.parent / "candidatas" / version / "modelo_final.joblib"
    ruta.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(paq, ruta, compress=3)
    ruta_metadatos(ruta).write_text(json.dumps({**paq["metadatos"],
                                                "sha256_modelo_final.joblib": sha256_archivo(ruta)}),
                                    encoding="utf-8")
    return ruta


def _rechaza(funcion, excepcion, texto):
    try:
        funcion()
    except excepcion as e:
        assert texto in str(e), str(e)
        return
    raise AssertionError(f"no se rechazo (se esperaba {excepcion.__name__}: {texto})")


def _registro(tmp):
    return (tmp / vm.ARCHIVO_REGISTRO).read_bytes()


def _sin_entorno():
    return mock.patch.dict(os.environ, {k: "" for k in ("MODELO_VERSION", "ASPECTOS_VERSION",
                                                        "MODELO_SHA256", "ASPECTOS_SHA256")})


# ---------------------------------------------------------------------------
# Registro del repositorio
# ---------------------------------------------------------------------------

def test_registro_del_repositorio_es_coherente():
    assert vm.verificar_registro() == []
    registro = vm.leer_registro()
    for familia, cfg in vm.FAMILIAS.items():
        fam = registro["familias"][familia]
        ruta = vm.ruta_version(familia, fam["activa"])
        meta = json.loads(ruta_metadatos(ruta).read_text(encoding="utf-8"))
        assert meta["version_modelo"] == fam["activa"] and ruta.name == cfg["archivo"]
        assert ruta.relative_to(CARPETA).parts == ("modelos", familia, fam["activa"], cfg["archivo"])
    # ya no quedan modelos sueltos fuera de modelos/
    assert not list(CARPETA.glob("modelo_final*")) and not list(CARPETA.glob("modelo_aspectos*"))


def test_api_informa_la_version_cargada():
    s = cliente().get("/salud").json()["versiones_cargadas"]
    activas = {f: d["activa"] for f, d in vm.leer_registro()["familias"].items()}
    for familia in ("polaridad", "aspectos"):
        assert s[familia]["version"] == activas[familia]
        assert s[familia]["firma"]["coincide"] is True
    v = cliente().get("/modelo/versiones").json()
    assert v["cargadas"]["polaridad"]["version"] == v["familias"]["polaridad"]["activa"]
    assert v["historial"] and "versiones" not in v["familias"]["polaridad"]["versiones"]["1.1.0"]


# ---------------------------------------------------------------------------
# Ciclo de vida
# ---------------------------------------------------------------------------

def test_ciclo_registrar_promover_revertir():
    tmp = _copia()
    with _sin_entorno():
        # 1. registrar: queda como candidata, la activa no cambia
        res = vm.registrar("polaridad", _candidata(tmp), "prueba: modelo nuevo", tmp)
        assert res == {"familia": "polaridad", "version": "1.1.1", "ya_registrada": False, "activa": "1.1.0"}
        assert (tmp / "polaridad" / "1.1.1" / vm.ARCHIVO_FIRMA).is_file()
        assert vm.registrar("polaridad", tmp / "polaridad" / "1.1.1" / "modelo_final.joblib",
                            "otra vez", tmp)["ya_registrada"]
        # un numero de version no se reutiliza con otro contenido
        otra = _candidata(tmp, cambiar=lambda p: p["metadatos"].update(fecha="2026-11-17T00:00:00"))
        _rechaza(lambda: vm.registrar("polaridad", otra, "x", tmp), vm.ErrorVersiones, "no se sobrescriben")

        # 2. promover
        inf = vm.promover("polaridad", "1.1.1", "prueba: mejora", tmp)
        assert inf["firma"]["coincide"] and inf["compatibilidad"] == "ok"
        fam = vm.leer_registro(tmp)["familias"]["polaridad"]
        assert fam["activa"] == "1.1.1" and fam["anteriores"] == ["1.1.0"]
        estado = api.cargar_servicio(carpeta_modelos=tmp)  # la API carga la nueva
        assert estado["versiones_cargadas"]["polaridad"]["version"] == "1.1.1"
        assert estado["integridad"]["verificado_con"] == "modelos/registro.json"

        # 3. revertir
        inf = vm.revertir("polaridad", "prueba: falla en produccion", tmp)
        fam = vm.leer_registro(tmp)["familias"]["polaridad"]
        assert fam["activa"] == "1.1.0" and fam["anteriores"] == []
        assert fam["versiones"]["1.1.1"]["revertida"][0]["motivo"] == "prueba: falla en produccion"
        assert api.cargar_servicio(carpeta_modelos=tmp)["versiones_cargadas"]["polaridad"]["version"] == "1.1.0"
        _rechaza(lambda: vm.revertir("polaridad", "x", tmp), vm.ErrorVersiones, "no tiene una version anterior")

    acciones = [(h["accion"], h.get("hacia", h.get("version"))) for h in vm.leer_registro(tmp)["historial"]
                if h["familia"] == "polaridad"]
    assert acciones[-3:] == [("registrar", "1.1.1"), ("promover", "1.1.1"), ("revertir", "1.1.0")]
    assert vm.verificar_registro(tmp) == [] and not list(tmp.glob(".registro_*"))  # escritura atomica


def test_promover_exige_motivo_y_version_distinta():
    tmp = _copia()
    _rechaza(lambda: vm.promover("polaridad", "1.1.0", "", tmp), vm.ErrorVersiones, "motivo")
    _rechaza(lambda: vm.promover("polaridad", "1.1.0", "x", tmp), vm.ErrorVersiones, "ya es la version activa")
    _rechaza(lambda: vm.promover("polaridad", "2.0.0", "x", tmp), vm.ErrorVersiones, "no esta registrada")


# ---------------------------------------------------------------------------
# Cambios que se rechazan (el registro queda igual)
# ---------------------------------------------------------------------------

def test_no_promueve_un_archivo_alterado():
    tmp = _copia()
    with _sin_entorno():
        vm.registrar("polaridad", _candidata(tmp), "prueba", tmp)
        ruta = tmp / "polaridad" / "1.1.1" / "modelo_final.joblib"
        datos = bytearray(ruta.read_bytes())
        datos[len(datos) // 2] ^= 0xFF
        ruta.write_bytes(bytes(datos))
        antes = _registro(tmp)
        _rechaza(lambda: vm.promover("polaridad", "1.1.1", "x", tmp), ModeloNoConfiable, "no coincide")
        assert _registro(tmp) == antes
        assert any("1.1.1" in p for p in vm.verificar_registro(tmp))


def test_no_revierte_si_el_codigo_ya_no_reproduce_la_version():
    # El .joblib guarda pesos; la logica de las clases esta en pipeline_feedback.py.
    # Se simula un cambio de codigo posterior: la version vieja daria otras predicciones.
    tmp = _copia()
    with _sin_entorno():
        vm.registrar("polaridad", _candidata(tmp), "prueba", tmp)
        vm.promover("polaridad", "1.1.1", "prueba", tmp)
        antes = _registro(tmp)
        with mock.patch.object(pf.LimpiadorTexto, "transform", lambda self, X: ["" for _ in X]):
            _rechaza(lambda: vm.revertir("polaridad", "x", tmp), vm.ErrorVersiones, "no reproduce")
            assert _registro(tmp) == antes
            _rechaza(lambda: api.cargar_servicio(carpeta_modelos=tmp), api.ModeloNoDisponible, "no reproduce")
            # con revision explicita se permite y queda anotado
            inf = vm.revertir("polaridad", "revisado a mano", tmp, aceptar_firma_distinta=True)
            assert inf["firma"]["coincide"] is False
        assert vm.leer_registro(tmp)["historial"][-1]["firma"] is False


def test_no_registra_un_modelo_con_versiones_incompatibles():
    tmp = _copia()
    versiones = {**vm.leer_registro()["familias"]["polaridad"]["versiones"]["1.1.0"]["versiones"],
                 "scikit-learn": "1.8.0"}
    antes = _registro(tmp)
    _rechaza(lambda: vm.registrar("polaridad", _candidata(tmp, versiones=versiones), "x", tmp),
             VersionesIncompatibles, "scikit-learn")
    assert _registro(tmp) == antes and not (tmp / "polaridad" / "1.1.1").exists()  # sin restos


def test_registro_y_metadatos_deben_coincidir():
    tmp = _copia()
    registro = vm.leer_registro(tmp)
    registro["familias"]["polaridad"]["versiones"]["1.1.0"]["sha256"] = "0" * 64
    (tmp / vm.ARCHIVO_REGISTRO).write_text(json.dumps(registro), encoding="utf-8")
    _rechaza(lambda: vm.resolver("polaridad", carpeta_modelos=tmp), ModeloNoConfiable, "no coinciden")
    _rechaza(lambda: api.cargar_servicio(carpeta_modelos=tmp), ModeloNoConfiable, "no coinciden")


# ---------------------------------------------------------------------------
# Reversion de emergencia por variable de entorno (sin tocar el registro)
# ---------------------------------------------------------------------------

def test_version_fijada_por_variable_de_entorno():
    tmp = _copia()
    with _sin_entorno():
        vm.registrar("polaridad", _candidata(tmp), "prueba", tmp)
        vm.promover("polaridad", "1.1.1", "prueba", tmp)
        with mock.patch.dict(os.environ, {"MODELO_VERSION": "1.1.0"}):
            v = api.cargar_servicio(carpeta_modelos=tmp)["versiones_cargadas"]["polaridad"]
            assert v["version"] == "1.1.0" and v["fuente"] == "variable de entorno MODELO_VERSION"
        with mock.patch.dict(os.environ, {"MODELO_VERSION": "9.9.9"}):
            _rechaza(lambda: api.cargar_servicio(carpeta_modelos=tmp), api.ModeloNoDisponible, "no esta registrada")
        with mock.patch.dict(os.environ, {"ASPECTOS_VERSION": "9.9.9"}):
            _rechaza(lambda: api.cargar_servicio(carpeta_modelos=tmp), api.ModeloNoDisponible, "aspectos")
    assert vm.leer_registro(tmp)["familias"]["polaridad"]["activa"] == "1.1.1"


def test_linea_de_comandos():
    tmp = _copia()
    with _sin_entorno():
        assert vm.main(["--carpeta", str(tmp), "verificar"]) == 0
        assert vm.main(["--carpeta", str(tmp), "registrar", "polaridad", str(_candidata(tmp)),
                        "--motivo", "cli"]) == 0
        assert vm.main(["--carpeta", str(tmp), "promover", "polaridad", "1.1.1", "--motivo", "cli"]) == 0
        assert vm.main(["--carpeta", str(tmp), "revertir", "polaridad", "--motivo", "cli"]) == 0
        assert vm.main(["--carpeta", str(tmp), "revertir", "polaridad", "--motivo", "cli"]) == 2  # error claro
        assert vm.main(["--carpeta", str(tmp), "listar"]) == 0


def test_docker_lleva_todas_las_versiones_y_las_verifica():
    docker = (CARPETA / "Dockerfile").read_text(encoding="utf-8")
    ignorar = (CARPETA / ".dockerignore").read_text(encoding="utf-8")
    assert "COPY modelos/ ./modelos/" in docker and "versiones_modelo.py verificar" in docker
    assert "!modelos/" in ignorar and "versiones_modelo.py" in docker


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
