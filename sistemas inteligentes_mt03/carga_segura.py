"""
Punto 3 (imagen 3): seguridad en la deserializacion de archivos joblib.

joblib usa pickle: al cargar un .joblib se puede ejecutar CUALQUIER codigo que
el archivo indique (por ejemplo, un objeto con __reduce__ que llame a
os.system). No existe un "joblib.load seguro"; la unica defensa es cargar
solamente archivos de confianza. Este modulo lo hace cumplir:

  1. Ruta permitida: el archivo debe ser un .joblib normal (no un enlace
     simbolico) dentro de la carpeta del proyecto. Evita cargar algo de
     /tmp, de una subida de usuario o de una ruta manipulada (../../).
  2. Huella sha256 esperada: se compara con un valor registrado APARTE del
     .joblib, en un JSON (leer JSON no ejecuta codigo):
       - variable de entorno MODELO_SHA256 (fijarla en el despliegue), o
       - <nombre>_metadatos.json, clave "sha256_<nombre>.joblib" (paso 8).
     Si no hay huella registrada, NO se carga.
  3. Sin carrera entre verificar y cargar: el archivo se lee UNA vez en
     memoria, se calcula la huella de esos bytes y se deserializan esos mismos
     bytes (nadie puede cambiar el archivo entre la verificacion y la carga).

Si la huella no coincide se lanza ModeloNoConfiable ANTES de deserializar:
el codigo del archivo nunca llega a ejecutarse.

Uso:
    from carga_segura import cargar_joblib_confiable
    paquete = cargar_joblib_confiable("modelo_final.joblib")

Limite: esto garantiza que el archivo es el que se registro, no que quien lo
registro sea de confianza. El .joblib y su huella deben venir del propio
equipo (paso 8) y viajar por el repositorio, nunca de usuarios.
"""

import hashlib
import hmac
import io
import json
import os
import re
from pathlib import Path

import joblib

CARPETA = Path(__file__).resolve().parent
_HEX64 = re.compile(r"[0-9a-f]{64}")


class ModeloNoConfiable(RuntimeError):
    """El archivo no pasa las comprobaciones: no se deserializa."""


def sha256_bytes(datos):
    return hashlib.sha256(datos).hexdigest()


def sha256_archivo(ruta):
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def ruta_metadatos(ruta_modelo):
    """modelo_final.joblib -> modelo_final_metadatos.json (misma carpeta)."""
    ruta_modelo = Path(ruta_modelo)
    return ruta_modelo.with_name(f"{ruta_modelo.stem}_metadatos.json")


def huella_esperada(ruta_modelo):
    """Devuelve (sha256, fuente). Lanza ModeloNoConfiable si no esta registrada."""
    ruta_modelo = Path(ruta_modelo)
    variable = os.environ.get("MODELO_SHA256", "").strip().lower()
    if variable:
        return variable, "variable de entorno MODELO_SHA256"
    meta = ruta_metadatos(ruta_modelo)
    if not meta.is_file():
        raise ModeloNoConfiable(f"No hay huella registrada para {ruta_modelo.name}: falta {meta.name} "
                                "(lo genera paso8_guardar_modelo.py) y MODELO_SHA256 no esta definida.")
    clave = f"sha256_{ruta_modelo.name}"
    valor = json.loads(meta.read_text(encoding="utf-8")).get(clave)
    if not valor:
        raise ModeloNoConfiable(f"{meta.name} no tiene la clave '{clave}'.")
    return str(valor).strip().lower(), meta.name


def huella_en_registro(ruta_registro, nombre_archivo):
    """sha256 de un archivo guardado en un registro JSON con clave "sha256_datos"
    (registro_modelo_paso7.json guarda la de modelo_paso6.joblib)."""
    ruta_registro = Path(ruta_registro)
    if not ruta_registro.is_file():
        raise ModeloNoConfiable(f"No existe {ruta_registro.name}: no hay huella registrada para {nombre_archivo}.")
    valor = json.loads(ruta_registro.read_text(encoding="utf-8")).get("sha256_datos", {}).get(nombre_archivo)
    if not valor:
        raise ModeloNoConfiable(f"{ruta_registro.name} no registra la huella de {nombre_archivo}.")
    return valor


def cargar_joblib_confiable(ruta, sha256=None, carpeta_permitida=CARPETA):
    """Carga un .joblib solo si esta en la carpeta permitida y su sha256 coincide.

    sha256: huella esperada; si no se da, se busca con huella_esperada().
    carpeta_permitida: None desactiva la restriccion de carpeta (solo pruebas).
    """
    ruta = Path(ruta)
    if ruta.suffix != ".joblib":
        raise ModeloNoConfiable(f"Solo se cargan archivos .joblib (recibido: {ruta.name}).")
    if ruta.is_symlink():
        raise ModeloNoConfiable(f"{ruta.name} es un enlace simbolico: no se sigue.")
    if not ruta.is_file():
        raise ModeloNoConfiable(f"No existe el archivo {ruta}.")
    real = ruta.resolve(strict=True)
    if carpeta_permitida is not None and not real.is_relative_to(Path(carpeta_permitida).resolve()):
        raise ModeloNoConfiable(f"{real} esta fuera de la carpeta permitida ({carpeta_permitida}).")

    esperado, fuente = (sha256.strip().lower(), "argumento sha256") if sha256 else huella_esperada(real)
    if not _HEX64.fullmatch(esperado):
        raise ModeloNoConfiable(f"La huella esperada ({fuente}) no es un sha256 valido.")

    datos = real.read_bytes()  # una sola lectura: se verifica y se carga lo mismo
    obtenido = sha256_bytes(datos)
    if not hmac.compare_digest(obtenido, esperado):
        raise ModeloNoConfiable(
            f"La huella de {real.name} no coincide con la registrada en {fuente} "
            f"(esperada {esperado[:12]}..., obtenida {obtenido[:12]}...). "
            "El archivo pudo ser alterado: NO se deserializa.")
    return joblib.load(io.BytesIO(datos))  # verificado: archivo de confianza
