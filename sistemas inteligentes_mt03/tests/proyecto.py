"""Rutas del proyecto para las pruebas (que viven en tests/).

Importarlo antes que los modulos del proyecto: agrega la carpeta del proyecto
al sys.path, asi las pruebas funcionan con pytest y con python tests/test_x.py
desde cualquier carpeta.
"""
import sys
from pathlib import Path

CARPETA = Path(__file__).resolve().parents[1]
DATOS = CARPETA / "datos"
RESULTADOS = CARPETA / "resultados"

if str(CARPETA) not in sys.path:
    sys.path.insert(0, str(CARPETA))
