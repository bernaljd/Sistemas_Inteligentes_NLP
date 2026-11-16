"""
Punto 6 (imagen 3): control y reversion de versiones de los modelos.

Cada modelo entrenado es una VERSION inmutable guardada en su propia carpeta:

    modelos/
      registro.json                         <- que versiones hay y cual esta activa
      polaridad/1.1.0/modelo_final.joblib   (+ modelo_final_metadatos.json, firma.json)
      aspectos/1.0.0/modelo_aspectos.joblib (+ modelo_aspectos_metadatos.json, firma.json)

registro.json guarda, por familia (polaridad, aspectos):
  - "activa": la version que carga la API.
  - "anteriores": pila de versiones activas previas (para revertir).
  - "versiones": sha256, fecha de entrenamiento y de registro, commit de git,
    versiones de librerias y metricas de cada version.
y un "historial" con cada registro, promocion y reversion (fecha, desde, hacia, motivo).

Ciclo de vida:
  1. registrar  -> paso8 / paso11 guardan una version nueva y la registran como
                   CANDIDATA (no entra en servicio; solo la primera se activa sola).
                   Un numero de version no se reutiliza con otro contenido.
  2. promover   -> pasa a ser la activa, solo si supera la verificacion.
  3. revertir   -> vuelve a la version activa anterior (deshace la ultima promocion),
                   con la misma verificacion.

Verificacion (al registrar, promover y revertir):
  - huella sha256 del archivo = la del registro = la de sus metadatos (punto 3);
  - versiones de librerias compatibles con las instaladas (punto 4);
  - carga y prueba funcional (3 probabilidades / 9 aspectos por comentario);
  - firma de comportamiento: predicciones sobre TEXTOS_FIRMA guardadas al
    registrar (firma.json). Si al volver a una version antigua el codigo
    (pipeline_feedback.py, aspectos.py) ya no reproduce esas predicciones, la
    reversion se rechaza: el .joblib solo guarda los pesos, la logica de las
    clases vive en el codigo.

La API (api.py) carga la version activa. Para fijar otra sin tocar el registro
(reversion de emergencia en el contenedor): MODELO_VERSION=1.1.0 / ASPECTOS_VERSION=1.0.0.

Uso:
    python versiones_modelo.py listar
    python versiones_modelo.py verificar [--completo]
    python versiones_modelo.py registrar polaridad ruta/modelo_final.joblib --motivo "..."
    python versiones_modelo.py promover polaridad 1.2.0 --motivo "..."
    python versiones_modelo.py revertir polaridad --motivo "..."
"""

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import warnings
from datetime import datetime
from pathlib import Path

from carga_segura import (ModeloNoConfiable, cargar_joblib_confiable, huella_esperada, ruta_metadatos,
                          sha256_archivo)
from compatibilidad import VersionesIncompatibles, exigir_compatibles

CARPETA = Path(__file__).resolve().parent
CARPETA_MODELOS = CARPETA / "modelos"
ARCHIVO_REGISTRO = "registro.json"
ARCHIVO_FIRMA = "firma.json"

FAMILIAS = {
    "polaridad": {"archivo": "modelo_final.joblib", "variable_sha256": "MODELO_SHA256",
                  "variable_version": "MODELO_VERSION", "genera": "paso8_guardar_modelo.py"},
    "aspectos": {"archivo": "modelo_aspectos.joblib", "variable_sha256": "ASPECTOS_SHA256",
                 "variable_version": "ASPECTOS_VERSION", "genera": "paso11_aspectos.py"},
}

# Comentarios fijos (escritos para esto, no salen de test.csv) cuyas predicciones
# forman la "firma" de cada version.
TEXTOS_FIRMA = [
    "El profesor explica muy bien y siempre resuelve las dudas.",
    "Los parciales son injustos y no corresponden con lo visto en clase.",
    "El curso estuvo bien, nada especial.",
    "Las diapositivas son confusas y el horario es terrible.",
    "Responde rapido los correos y el material de clase es muy completo.",
    "Demasiada carga de trabajo para los creditos que tiene la materia.",
    "La plataforma del aula digital se cae seguido.",
    "Excelente metodologia, aprendi bastante, pero la retroalimentacion de los examenes llega tarde.",
    "",
    "ok",
]
TOLERANCIA_FIRMA = 1e-4  # diferencia maxima de probabilidad (Windows vs. Linux dan ~1e-7)

_VERSION = re.compile(r"\d+\.\d+\.\d+")
log = logging.getLogger("api")


class ErrorVersiones(RuntimeError):
    """Operacion de versiones invalida: el registro no se modifica."""


# ---------------------------------------------------------------------------
# Registro
# ---------------------------------------------------------------------------

def _cfg(familia):
    if familia not in FAMILIAS:
        raise ErrorVersiones(f"Familia desconocida '{familia}'. Opciones: {', '.join(FAMILIAS)}")
    return FAMILIAS[familia]


def _ahora():
    return datetime.now().isoformat(timespec="seconds")


def leer_registro(carpeta_modelos=CARPETA_MODELOS):
    ruta = Path(carpeta_modelos) / ARCHIVO_REGISTRO
    if not ruta.is_file():
        return {"formato": 1,
                "familias": {f: {"activa": None, "anteriores": [], "versiones": {}} for f in FAMILIAS},
                "historial": []}
    return json.loads(ruta.read_text(encoding="utf-8"))


def _guardar_registro(registro, carpeta_modelos):
    """Escritura atomica: si algo falla a mitad, registro.json queda como estaba."""
    carpeta = Path(carpeta_modelos)
    carpeta.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=carpeta, prefix=".registro_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(registro, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(tmp, carpeta / ARCHIVO_REGISTRO)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _familia(registro, familia):
    _cfg(familia)
    return registro["familias"].setdefault(familia, {"activa": None, "anteriores": [], "versiones": {}})


def ruta_version(familia, version, carpeta_modelos=CARPETA_MODELOS):
    return Path(carpeta_modelos) / familia / version / _cfg(familia)["archivo"]


def resolver(familia, version=None, carpeta_modelos=CARPETA_MODELOS, usar_entorno=True):
    """Que archivo cargar: la version pedida, la de MODELO_VERSION/ASPECTOS_VERSION
    o la activa del registro. Comprueba que la huella del registro y la de los
    metadatos coincidan (si no, alguien cambio uno de los dos a mano)."""
    cfg = _cfg(familia)
    fam = _familia(leer_registro(carpeta_modelos), familia)
    fuente = "argumento"
    if version is None:
        version = os.environ.get(cfg["variable_version"], "").strip() if usar_entorno else ""
        fuente = f"variable de entorno {cfg['variable_version']}" if version else "version activa del registro"
        version = version or fam.get("activa")
    if not version:
        raise ErrorVersiones(f"No hay version activa de '{familia}' en {ARCHIVO_REGISTRO} "
                             f"(genera una con {cfg['genera']}).")
    entrada = fam["versiones"].get(version)
    if entrada is None:
        raise ErrorVersiones(f"La version {version} de '{familia}' no esta registrada "
                             f"(registradas: {', '.join(sorted(fam['versiones'])) or 'ninguna'}).")
    ruta = ruta_version(familia, version, carpeta_modelos)
    meta = ruta_metadatos(ruta)
    huella_meta = (json.loads(meta.read_text(encoding="utf-8")).get(f"sha256_{ruta.name}")
                   if meta.is_file() else None)
    if huella_meta != entrada["sha256"]:
        raise ModeloNoConfiable(f"{familia} {version}: la huella de {ARCHIVO_REGISTRO} y la de "
                                f"{meta.name} no coinciden: no se carga.")
    return {"familia": familia, "version": version, "ruta": ruta, "sha256": entrada["sha256"],
            "fuente": fuente, "entrada": entrada}


def ruta_activa(familia, carpeta_modelos=CARPETA_MODELOS):
    return resolver(familia, carpeta_modelos=carpeta_modelos)["ruta"]


# ---------------------------------------------------------------------------
# Carga verificada (puntos 3 y 4) y firma de comportamiento
# ---------------------------------------------------------------------------

def cargar_verificado(ruta, carpeta_permitida=CARPETA, variable_sha256="MODELO_SHA256", sha256=None):
    """Carga un .joblib del servicio con las comprobaciones de los puntos 3 y 4.

    1. Versiones: compara las registradas en <nombre>_metadatos.json (leer JSON
       no ejecuta codigo) con las instaladas ANTES de deserializar.
    2. Huella sha256 y carga. Prioridad de la huella esperada: variable de
       entorno (variable_sha256; None la ignora) > sha256 (registro.json) >
       <nombre>_metadatos.json. Un InconsistentVersionWarning de scikit-learn
       al cargar tambien es incompatibilidad.
    3. Las versiones dentro del .joblib deben ser las del JSON.
    Devuelve (paquete, compatibilidad, integridad).
    """
    from sklearn.exceptions import InconsistentVersionWarning

    ruta = Path(ruta)
    meta_json = ruta_metadatos(ruta)
    registradas = (json.loads(meta_json.read_text(encoding="utf-8")).get("versiones")
                   if meta_json.is_file() else None)
    if not registradas:
        raise VersionesIncompatibles(f"{meta_json.name} no registra las versiones con que se entreno "
                                     "el modelo: no se puede garantizar la compatibilidad.")
    compat = exigir_compatibles(registradas)  # antes de deserializar
    for d in compat["diferencias"]:
        log.warning("%s: version distinta a la del modelo: %s (modelo %s, instalada %s)",
                    ruta.name, d["paquete"], d["registrada"], d["instalada"])

    variable = os.environ.get(variable_sha256, "").strip().lower() if variable_sha256 else ""
    if variable:
        esperado, fuente = variable, f"variable de entorno {variable_sha256}"
    elif sha256:
        esperado, fuente = sha256, f"modelos/{ARCHIVO_REGISTRO}"
    else:
        esperado, fuente = huella_esperada(ruta, variable_sha256 or "")
    with warnings.catch_warnings():
        warnings.simplefilter("error", InconsistentVersionWarning)  # aviso de sklearn -> error
        try:
            paquete = cargar_joblib_confiable(ruta, sha256=esperado, carpeta_permitida=carpeta_permitida,
                                              fuente=fuente)
        except InconsistentVersionWarning as e:
            raise VersionesIncompatibles(f"scikit-learn avisa de versiones distintas al cargar: {e}") from e
    if paquete["metadatos"].get("versiones") != registradas:
        raise VersionesIncompatibles(f"Las versiones dentro de {ruta.name} no coinciden "
                                     f"con {meta_json.name}: regeneralos juntos.")
    return paquete, compat, {"sha256": esperado, "verificado_con": fuente}


def calcular_firma(familia, paquete):
    """Predicciones de la version sobre TEXTOS_FIRMA: etiquetas y probabilidades."""
    etiquetas, valores = [], []
    if familia == "polaridad":
        modelo = paquete["pipeline"]
        proba = modelo.predict_proba(TEXTOS_FIRMA)
        for fila in proba:
            etiquetas.append(str(modelo.classes_[fila.argmax()]))
            valores.extend(round(float(v), 6) for v in fila)
    else:
        for salida in paquete["modelo"].predecir(TEXTOS_FIRMA):
            for a in salida:
                etiquetas.append(f"{a['aspecto']}:{a['sentimiento'] if a['mencionado'] else '-'}")
                valores.append(None if a["probabilidad"] is None else round(a["probabilidad"], 6))
                ps = a["probabilidades_sentimiento"] or {}
                valores.extend(round(ps[c], 6) if c in ps else None for c in ("negativo", "neutral", "positivo"))
    return {"textos": TEXTOS_FIRMA, "etiquetas": etiquetas, "valores": valores}


def comparar_firma(esperada, obtenida, tolerancia=TOLERANCIA_FIRMA):
    if esperada["textos"] != obtenida["textos"]:
        return {"coincide": False, "motivo": "la firma se hizo con otros textos", "diferencia_maxima": None}
    if esperada["etiquetas"] != obtenida["etiquetas"]:
        cambios = sum(a != b for a, b in zip(esperada["etiquetas"], obtenida["etiquetas"]))
        return {"coincide": False, "motivo": f"cambian {cambios} predicciones", "diferencia_maxima": None}
    pares = list(zip(esperada["valores"], obtenida["valores"]))
    if any((a is None) != (b is None) for a, b in pares):
        return {"coincide": False, "motivo": "cambia que valores existen", "diferencia_maxima": None}
    dif = max((abs(a - b) for a, b in pares if a is not None), default=0.0)
    return {"coincide": dif <= tolerancia,
            "motivo": "" if dif <= tolerancia else f"probabilidades distintas (hasta {dif:.2e})",
            "diferencia_maxima": dif}


def _prueba_funcional(familia, paquete):
    if familia == "polaridad":
        modelo = paquete["pipeline"]
        proba = modelo.predict_proba(TEXTOS_FIRMA[:3])
        ok = (list(modelo.classes_) == ["negativo", "neutral", "positivo"] and proba.shape == (3, 3)
              and abs(proba.sum(axis=1) - 1).max() < 1e-6)
    else:
        from aspectos import CLAVES
        salida = paquete["modelo"].predecir(TEXTOS_FIRMA[:3])
        ok = len(salida) == 3 and all([a["aspecto"] for a in s] == CLAVES for s in salida)
    if not ok:
        raise ErrorVersiones(f"La version no pasa la prueba funcional de '{familia}'.")


def firma_guardada(familia, version, carpeta_modelos=CARPETA_MODELOS):
    ruta = ruta_version(familia, version, carpeta_modelos).with_name(ARCHIVO_FIRMA)
    if not ruta.is_file():
        raise ErrorVersiones(f"{familia} {version} no tiene {ARCHIVO_FIRMA}: registrala de nuevo.")
    return json.loads(ruta.read_text(encoding="utf-8"))


def verificar_version(familia, version, carpeta_modelos=CARPETA_MODELOS, exigir_firma=True):
    """Huella + versiones + carga + prueba funcional + firma. Lanza si algo falla."""
    info = resolver(familia, version, carpeta_modelos, usar_entorno=False)
    paquete, compat, integridad = cargar_verificado(info["ruta"], carpeta_modelos, None, sha256=info["sha256"])
    _prueba_funcional(familia, paquete)
    firma = comparar_firma(firma_guardada(familia, version, carpeta_modelos), calcular_firma(familia, paquete))
    if exigir_firma and not firma["coincide"]:
        raise ErrorVersiones(
            f"{familia} {version}: el codigo actual no reproduce las predicciones que tenia al registrarse "
            f"({firma['motivo']}). Vuelve al commit {info['entrada'].get('commit') or '?'} junto con el "
            "modelo, o usa --aceptar-firma-distinta si revisaste el cambio.")
    return {"familia": familia, "version": version, "sha256": info["sha256"],
            "compatibilidad": compat["estado"], "firma": firma, "integridad": integridad}


# ---------------------------------------------------------------------------
# Operaciones
# ---------------------------------------------------------------------------

def _git(*args):
    try:
        r = subprocess.run(["git", *args], cwd=CARPETA, capture_output=True, text=True, timeout=10)
        return r.stdout.strip() if r.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _metricas(meta):
    return {k: round(v, 4) for k, v in meta.get("metricas", {}).items() if isinstance(v, (int, float))}


def registrar(familia, ruta_origen, motivo, carpeta_modelos=CARPETA_MODELOS):
    """Registra un modelo recien entrenado como version CANDIDATA.

    Copia el .joblib y sus metadatos a modelos/<familia>/<version>/ (si no estan
    ya ahi), lo verifica, guarda su firma y lo anota en el registro. Solo la
    primera version de una familia queda activa sola.
    """
    cfg = _cfg(familia)
    ruta_origen = Path(ruta_origen)
    if ruta_origen.name != cfg["archivo"]:
        raise ErrorVersiones(f"Para '{familia}' se espera {cfg['archivo']} (recibido {ruta_origen.name}).")
    meta = json.loads(ruta_metadatos(ruta_origen).read_text(encoding="utf-8"))
    version = str(meta.get("version_modelo", ""))
    if not _VERSION.fullmatch(version):
        raise ErrorVersiones(f"version_modelo '{version}' no tiene formato X.Y.Z.")
    huella = meta.get(f"sha256_{cfg['archivo']}")
    if huella != sha256_archivo(ruta_origen):
        raise ModeloNoConfiable(f"{ruta_origen.name} no coincide con la huella de sus metadatos.")

    registro = leer_registro(carpeta_modelos)
    fam = _familia(registro, familia)
    previa = fam["versiones"].get(version)
    if previa is not None:
        if previa["sha256"] == huella:
            return {"familia": familia, "version": version, "ya_registrada": True, "activa": fam["activa"]}
        raise ErrorVersiones(f"La version {version} de '{familia}' ya existe con otro contenido: las "
                             f"versiones no se sobrescriben. Sube VERSION_MODELO en {cfg['genera']}.")

    destino = ruta_version(familia, version, carpeta_modelos)
    copiada = ruta_origen.resolve() != destino.resolve()
    if copiada:
        if destino.parent.exists():
            shutil.rmtree(destino.parent)  # restos de un intento fallido (no estaba registrada)
        destino.parent.mkdir(parents=True)
        shutil.copy2(ruta_origen, destino)
        shutil.copy2(ruta_metadatos(ruta_origen), ruta_metadatos(destino))
    try:
        paquete, compat, _ = cargar_verificado(destino, carpeta_modelos, None, sha256=huella)
        _prueba_funcional(familia, paquete)
        firma = calcular_firma(familia, paquete)
    except BaseException:
        if copiada:
            shutil.rmtree(destino.parent, ignore_errors=True)
        raise
    destino.with_name(ARCHIVO_FIRMA).write_text(json.dumps(firma, ensure_ascii=False) + "\n", encoding="utf-8")

    fam["versiones"][version] = {
        "sha256": huella, "entrenado_el": meta.get("fecha"), "registrada_el": _ahora(),
        "commit": _git("rev-parse", "HEAD"),
        "codigo_sin_commit": bool(_git("status", "--porcelain", "--", "pipeline_feedback.py", "aspectos.py")),
        "genera": cfg["genera"], "versiones": meta.get("versiones"), "metricas": _metricas(meta),
        "compatibilidad_al_registrar": compat["estado"], "motivo": motivo,
    }
    registro["historial"].append({"fecha": _ahora(), "accion": "registrar", "familia": familia,
                                  "version": version, "motivo": motivo})
    if fam["activa"] is None:
        fam["activa"] = version
        registro["historial"].append({"fecha": _ahora(), "accion": "activar", "familia": familia,
                                      "desde": None, "hacia": version, "motivo": "primera version registrada"})
    _guardar_registro(registro, carpeta_modelos)
    return {"familia": familia, "version": version, "ya_registrada": False, "activa": fam["activa"]}


def promover(familia, version, motivo, carpeta_modelos=CARPETA_MODELOS, aceptar_firma_distinta=False):
    """Pone en servicio una version registrada, si supera la verificacion."""
    if not motivo:
        raise ErrorVersiones("Indica el motivo del cambio (--motivo).")
    activa = _familia(leer_registro(carpeta_modelos), familia)["activa"]
    if version == activa:
        raise ErrorVersiones(f"{familia} {version} ya es la version activa.")
    informe = verificar_version(familia, version, carpeta_modelos, exigir_firma=not aceptar_firma_distinta)
    registro = leer_registro(carpeta_modelos)
    fam = _familia(registro, familia)
    if fam["activa"]:
        fam["anteriores"].append(fam["activa"])
    registro["historial"].append({"fecha": _ahora(), "accion": "promover", "familia": familia,
                                  "desde": fam["activa"], "hacia": version, "motivo": motivo,
                                  "firma": informe["firma"]["coincide"]})
    fam["activa"] = version
    _guardar_registro(registro, carpeta_modelos)
    return informe


def revertir(familia, motivo, carpeta_modelos=CARPETA_MODELOS, aceptar_firma_distinta=False):
    """Deshace la ultima promocion: vuelve a la version activa anterior."""
    if not motivo:
        raise ErrorVersiones("Indica el motivo de la reversion (--motivo).")
    fam = _familia(leer_registro(carpeta_modelos), familia)
    if not fam["anteriores"]:
        raise ErrorVersiones(f"'{familia}' no tiene una version anterior a la que volver.")
    destino = fam["anteriores"][-1]
    informe = verificar_version(familia, destino, carpeta_modelos, exigir_firma=not aceptar_firma_distinta)
    registro = leer_registro(carpeta_modelos)
    fam = _familia(registro, familia)
    desde = fam["activa"]
    fam["anteriores"].pop()
    fam["activa"] = destino
    fam["versiones"][desde].setdefault("revertida", []).append({"fecha": _ahora(), "motivo": motivo})
    registro["historial"].append({"fecha": _ahora(), "accion": "revertir", "familia": familia,
                                  "desde": desde, "hacia": destino, "motivo": motivo,
                                  "firma": informe["firma"]["coincide"]})
    _guardar_registro(registro, carpeta_modelos)
    return informe


def verificar_registro(carpeta_modelos=CARPETA_MODELOS, completo=False):
    """Revisa todo el registro. Rapido: archivos y huellas. Completo: ademas carga
    cada version y compara su firma. Devuelve la lista de problemas (vacia = ok)."""
    registro = leer_registro(carpeta_modelos)
    problemas = []
    for familia in FAMILIAS:
        fam = _familia(registro, familia)
        if fam["activa"] and fam["activa"] not in fam["versiones"]:
            problemas.append(f"{familia}: la activa {fam['activa']} no esta registrada")
        problemas += [f"{familia}: la anterior {v} no esta registrada"
                      for v in fam["anteriores"] if v not in fam["versiones"]]
        for version, entrada in fam["versiones"].items():
            ruta = ruta_version(familia, version, carpeta_modelos)
            try:
                if not ruta.is_file():
                    raise ErrorVersiones(f"falta {ruta.name}")
                if sha256_archivo(ruta) != entrada["sha256"]:
                    raise ModeloNoConfiable("el archivo no coincide con su huella registrada")
                resolver(familia, version, carpeta_modelos, usar_entorno=False)
                firma_guardada(familia, version, carpeta_modelos)
                if completo:
                    verificar_version(familia, version, carpeta_modelos)
            except (ErrorVersiones, ModeloNoConfiable, VersionesIncompatibles, OSError) as e:
                problemas.append(f"{familia} {version}: {e}")
    return problemas


def resumen(carpeta_modelos=CARPETA_MODELOS, ultimos=20):
    """Vista del registro para /modelo/versiones (sin las versiones de librerias)."""
    registro = leer_registro(carpeta_modelos)
    familias = {}
    for familia in FAMILIAS:
        fam = _familia(registro, familia)
        familias[familia] = {
            "activa": fam["activa"], "anteriores": fam["anteriores"],
            "versiones": {v: {k: e.get(k) for k in ("sha256", "entrenado_el", "registrada_el", "commit",
                                                     "metricas", "motivo", "revertida")}
                          for v, e in fam["versiones"].items()}}
    return {"familias": familias, "historial": registro["historial"][-ultimos:]}


# ---------------------------------------------------------------------------
# Linea de comandos
# ---------------------------------------------------------------------------

def _listar(carpeta_modelos):
    r = resumen(carpeta_modelos, ultimos=10)
    for familia, fam in r["familias"].items():
        print(f"\n{familia}  (activa: {fam['activa'] or '-'}; anteriores: {fam['anteriores'] or '-'})")
        for v, e in sorted(fam["versiones"].items(), key=lambda x: [int(n) for n in x[0].split(".")]):
            marca = "*" if v == fam["activa"] else " "
            metricas = ", ".join(f"{k}={x}" for k, x in list(e["metricas"].items())[:3])
            print(f"  {marca} {v:<8} {e['sha256'][:12]}  entrenado {e['entrenado_el']}  {metricas}")
            if e.get("revertida"):
                print(f"             revertida: {e['revertida'][-1]['motivo']}")
    print("\nHistorial reciente:")
    for h in r["historial"]:
        cambio = h.get("version") or f"{h.get('desde') or '-'} -> {h.get('hacia')}"
        print(f"  {h['fecha']}  {h['accion']:<9} {h['familia']:<9} {cambio}  ({h.get('motivo', '')})")


def main(argv=None):
    p = argparse.ArgumentParser(description="Control y reversion de versiones de los modelos (punto 6).")
    p.add_argument("--carpeta", default=str(CARPETA_MODELOS), help=argparse.SUPPRESS)
    sub = p.add_subparsers(dest="orden", required=True)
    sub.add_parser("listar", help="versiones, la activa y el historial")
    v = sub.add_parser("verificar", help="huellas de todas las versiones (--completo: carga y firma)")
    v.add_argument("--completo", action="store_true")
    r = sub.add_parser("registrar", help="registra un modelo nuevo como candidata")
    r.add_argument("familia", choices=list(FAMILIAS))
    r.add_argument("ruta")
    r.add_argument("--motivo", required=True)
    for nombre, ayuda in (("promover", "pone en servicio una version"),
                          ("revertir", "vuelve a la version activa anterior")):
        s = sub.add_parser(nombre, help=ayuda)
        s.add_argument("familia", choices=list(FAMILIAS))
        if nombre == "promover":
            s.add_argument("version")
        s.add_argument("--motivo", required=True)
        s.add_argument("--aceptar-firma-distinta", action="store_true",
                       help="permite el cambio aunque el codigo actual no reproduzca sus predicciones")
    a = p.parse_args(argv)
    carpeta = Path(a.carpeta)
    try:
        if a.orden == "listar":
            _listar(carpeta)
        elif a.orden == "verificar":
            problemas = verificar_registro(carpeta, completo=a.completo)
            print("\n".join(f"  [FALLA] {x}" for x in problemas) or
                  f"  Registro correcto ({'carga y firma' if a.completo else 'huellas'} de todas las versiones).")
            return 1 if problemas else 0
        elif a.orden == "registrar":
            res = registrar(a.familia, a.ruta, a.motivo, carpeta)
            estado = "ya estaba registrada" if res["ya_registrada"] else "registrada"
            print(f"{a.familia} {res['version']} {estado}. Activa: {res['activa']}.")
            if res["activa"] != res["version"]:
                print(f"Para ponerla en servicio: python versiones_modelo.py promover {a.familia} "
                      f"{res['version']} --motivo \"...\"")
        else:
            if a.orden == "promover":
                inf = promover(a.familia, a.version, a.motivo, carpeta, a.aceptar_firma_distinta)
            else:
                inf = revertir(a.familia, a.motivo, carpeta, a.aceptar_firma_distinta)
            f = inf["firma"]
            print(f"{a.familia}: activa ahora {inf['version']} (sha256 {inf['sha256'][:12]}..., "
                  f"versiones {inf['compatibilidad']}, firma {'igual' if f['coincide'] else 'DISTINTA'}).")
            print("Reinicia la API (o reconstruye la imagen Docker) para que cargue esta version.")
    except (ErrorVersiones, ModeloNoConfiable, VersionesIncompatibles) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
