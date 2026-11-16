# Clasificación de la evaluación docente por parte de los estudiantes

Proyecto de Sistemas Inteligentes. A partir de comentarios de estudiantes en español:

- clasifica la **polaridad** de cada comentario (negativo / neutral / positivo);
- identifica de qué **aspecto del curso** habla (9 aspectos) y con qué sentimiento.

El servicio se despliega como una API (FastAPI) con una página web, en Docker local.

- Artículo: [`docs/articulo_clasificacion_evaluacion_docencia.pdf`](docs/articulo_clasificacion_evaluacion_docencia.pdf)
- Despliegue, seguridad, versiones y monitoreo: [`docs/DESPLIEGUE.md`](docs/DESPLIEGUE.md)

## Estructura

```
docs/                                artículo y documentación del despliegue
sistemas inteligentes_mt03/          proyecto (todos los comandos se ejecutan aquí)
  filtroDatos.py ... paso11_aspectos.py   un script por paso (tabla de abajo)
  pipeline_feedback.py               Pipeline completo: limpieza + TF-IDF + embeddings + clasificador
  aspectos.py                        modelo de 9 aspectos
  api.py, web/                       servicio de predicción y página web
  carga_segura.py                    carga de .joblib solo si su huella sha256 coincide
  compatibilidad.py                  versiones de librerías del modelo = las instaladas
  versiones_modelo.py                registrar, promover y revertir versiones del modelo
  monitoreo.py, monitoreo/           registro de uso y detección de deriva
  datos/                             feedback_preprocesado.csv, train.csv, test.csv, embeddings
  resultados/                        tablas, gráficas y salidas de consola de cada paso
  modelos/                           versiones de los modelos + registro.json
  tests/                             pruebas automáticas (pytest)
  Dockerfile, requirements*.txt
```

## Pasos y scripts

| Paso | Qué hace | Script | Salidas principales (en `resultados/` salvo indicación) |
|---|---|---|---|
| 1-2 | Exploración y preprocesamiento del CSV | — | `datos/feedback_preprocesado.csv` |
| 3 | División train (80 %) / test (20 %) estratificada | `filtroDatos.py` | `datos/train.csv`, `datos/test.csv` |
| 4 | TF-IDF vs. embeddings (validación cruzada solo en train) | `vectorizacion.py` | `resultados_vectorizacion.csv`, `datos/embeddings_train.npy` |
| 5 | Línea base (Dummy) y comparación de modelos | `entrenamiento.py` | `resultados_paso5.csv` |
| 6 | Ajuste de hiperparámetros (GridSearchCV, macro-F1) | `ajuste.py` | `resultados_paso6.csv`, `modelo_paso6.joblib` |
| 7 | Evaluación final en test (una sola vez) | `paso7.py` | `resultados_paso7.csv`, `matriz_confusion_paso7.png`, `errores_test.csv` |
| 8 | Seleccionar y guardar el Pipeline completo con joblib | `paso8_guardar_modelo.py` | `modelos/polaridad/<versión>/` |
| 9 | Validar la carga en un entorno limpio | `paso9_validar_modelo.py` | `paso9_salida.txt`, `validacion_casos.csv` |
| 10 | Rendimiento y monitoreo de deriva | `paso10_rendimiento.py`, `paso10_deriva.py` | `resultados_rendimiento.csv`, `monitoreo/referencia.json` |
| 11 | Modelo de 9 aspectos (salida estructurada) | `paso11_aspectos.py` | `modelos/aspectos/<versión>/`, `resultados_aspectos.csv` |

Los scripts usan rutas relativas a su propia carpeta: se pueden ejecutar desde cualquier lugar.

## Resultados

| Modelo | Métrica | Validación cruzada (train) | Test |
|---|---|---|---|
| Polaridad (LogReg + TF-IDF + MiniLM) | macro-F1 | 0.714 | 0.664 |
| Aspectos: detectar qué aspectos menciona | macro-F1 | 0.546 | 0.532 |
| Aspectos: sentimiento de cada aspecto | macro-F1 | 0.676 | 0.652 |

Las métricas detalladas, sus intervalos de confianza y los límites están en `resultados/` y en
[`docs/DESPLIEGUE.md`](docs/DESPLIEGUE.md).

## Cómo ejecutarlo

Requiere **Python 3.12**: el modelo se guardó con 3.12.5 y la API verifica que la versión coincida.

```powershell
cd "sistemas inteligentes_mt03"
python -m venv .venv; .venv\Scripts\activate
pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt -r requirements_dev.txt

python -m pytest                                  # todas las pruebas (carpeta tests/)
$env:HF_HUB_OFFLINE = "1"; python -m uvicorn api:app --port 8000   # http://127.0.0.1:8000
```

Con Docker (es el despliegue del proyecto):

```powershell
cd "sistemas inteligentes_mt03"
docker build -t polaridad-docente .
docker run -p 8000:8000 polaridad-docente           # http://127.0.0.1:8000
```

## Versiones del modelo

```powershell
python versiones_modelo.py listar                                          # versiones, activa e historial
python versiones_modelo.py promover polaridad 1.2.0 --motivo "..."         # poner una versión en servicio
python versiones_modelo.py revertir polaridad --motivo "..."               # volver a la anterior
```

El procedimiento completo de actualización y reversión está en
[`docs/DESPLIEGUE.md`](docs/DESPLIEGUE.md#control-y-reversión-de-versiones-del-modelo).
