# Paso 6 · Probar, desplegar y monitorear

Servicio: API FastAPI + página web (`/`) que clasifica comentarios de estudiantes
en negativo / neutral / positivo con `modelo_final.joblib` (LogReg + TF-IDF + MiniLM).

## 1. Verificación

| Aspecto | Cómo se verifica | Resultado |
|---|---|---|
| Funcionalidad | `python -m pytest test_api.py` (11 pruebas) | 11/11 OK |
| Seguridad | `python -m pytest test_api.py` (9 pruebas) + `pip-audit` | 9/9 OK · 0 vulnerabilidades conocidas* |
| Monitoreo | `python -m pytest test_api.py` (2 pruebas) + `python paso10_deriva.py` | 2/2 OK · deriva detectada en 4/4 escenarios |
| Tiempos | `python paso10_rendimiento.py --url ... --contenedor ...` | ver tabla abajo |

\* `pip-audit` encontró 10 vulnerabilidades en Starlette 0.52.1 (base de FastAPI). Se actualizó a
FastAPI 0.163.0 + Starlette 1.10.0 y la auditoría quedó limpia.

### Tiempos de inferencia (contenedor Docker local, CPU)

Medido por HTTP contra el contenedor real (`paso10_rendimiento.py --url`), que es el despliegue,
en un PC de 16 núcleos lógicos y 31.6 GB de RAM. Objetivos fijados antes de medir.

| Prueba | Medido | Objetivo |
|---|---|---|
| Un comentario (p50 / p95, 300 peticiones) | 35 ms / 52 ms | ≤ 500 ms ✔ |
| Comentario de 5000 caracteres (mediana) | 49 ms | — |
| Lote de 100 comentarios | 1.3 s (~13 ms por comentario) | ≤ 5 s ✔ |
| CSV de 1000 filas | 11.0 s | ≤ 60 s ✔ |
| Arranque (cargar + calentar, según `/salud`) | 5.3 s | ≤ 120 s ✔ |
| Memoria del contenedor (`docker stats`) | 672 MB tras cargar · 860 MB tras las pruebas | ≤ 2 GB ✔ |

Una medición anterior en un portátil con 7.7 GB de RAM y solo 0.5 GB libres (el sistema paginaba a
disco) dio p95 de 1076 ms, 15.8 s para el lote de 100 y 167 s de arranque: el servicio necesita
~1 GB de RAM libre. Casi todo el tiempo se va en los embeddings MiniLM; TF-IDF y la regresión
logística toman milisegundos.

### Eficiencia: el servicio no entrena

El entrenamiento es un paso aparte y offline (`paso8_guardar_modelo.py`). Al arrancar, la API
solo **carga** `modelo_final.joblib` una vez (`cargar_servicio()`), calienta el modelo de
embeddings y sirve; por eso arranca en ~5 s en vez de minutos, y siempre sirve exactamente el
modelo evaluado en el paso 7 (`/salud` muestra `entrenado_el`).

- Si falta el `.joblib` o trae un Pipeline sin entrenar, el servicio **no arranca** y dice que se
  ejecute el paso 8; nunca reentrena por su cuenta.
- `test_arranque_no_entrena` bloquea todos los `fit` del Pipeline durante la carga: si alguien
  agrega un entrenamiento al arranque, la prueba falla.

### Seguridad implementada

- **Límite de peticiones**: 60 por minuto por IP en `/predecir*` → `429` con `Retry-After`.
  La IP es la de la conexión: uvicorn corre sin `--proxy-headers`, así que una cabecera
  `X-Forwarded-For` falsa no sirve para saltarse el límite (verificado en el contenedor).
- **Límites de tamaño**: cuerpo ≤ 6 MB, CSV ≤ 5 MB / 1000 filas, texto ≤ 5000 caracteres, lote ≤ 1000.
  El cuerpo se mide contando los bytes reales, también con `Transfer-Encoding: chunked`.
- **Inyección CSV**: al descargar resultados, los comentarios que empiezan por `=`, `+`, `-` o `@`
  se escriben con `'` delante para que Excel no los ejecute como fórmula.
- **Cabeceras**: Content-Security-Policy (solo scripts propios), `X-Frame-Options: DENY`,
  `X-Content-Type-Options: nosniff`, `Referrer-Policy: no-referrer`, `Permissions-Policy`.
- **XSS**: la página escapa todo texto del usuario (`esc()`) antes de mostrarlo.
- **Sin CORS abierto**: la página y la API comparten origen (`CORS_ORIGINS` para habilitar otros).
- **Errores**: los errores internos responden `Error interno del servidor`, sin rutas ni trazas.
- **Producción**: `/docs` apagado (`DOCS=0`), `/monitoreo` protegido con `ADMIN_TOKEN`,
  contenedor con usuario sin privilegios, sin cabecera `Server`.
- **Privacidad**: el registro de monitoreo **no guarda el texto** de los comentarios (solo longitud,
  % de palabras desconocidas, clase, confianza y un hash).

### Seguridad en la deserialización del modelo

joblib usa pickle: **cargar un `.joblib` puede ejecutar cualquier código** que el archivo indique
(`test_carga_segura.py` lo demuestra con un archivo que crea una carpeta al cargarse). No existe un
`joblib.load` seguro, así que solo se cargan archivos de confianza, y `carga_segura.py` lo hace cumplir:

| Control | Qué evita |
|---|---|
| Solo `.joblib` normales dentro de la carpeta del proyecto (sin enlaces simbólicos ni `../`) | Cargar un archivo de `/tmp`, de una subida de usuario o de una ruta manipulada |
| Huella sha256 comparada **antes** de deserializar, contra `MODELO_SHA256` o `modelo_final_metadatos.json` | Un modelo reemplazado o alterado (basta 1 byte) nunca se ejecuta |
| Sin huella registrada → no se carga | Cargar "por si acaso" un archivo desconocido |
| Se lee el archivo una vez y se verifican y cargan los mismos bytes | Cambiar el archivo entre la verificación y la carga |
| En Docker el modelo pertenece a `root` y el servicio corre como `app` | Que el propio servicio pueda sobrescribir el modelo |

- Todos los scripts cargan con `cargar_joblib_confiable` (la API, `paso7`, `paso8`, `paso9` y
  `paso10_deriva`); `modelo_paso6.joblib` se verifica con la huella de `registro_modelo_paso7.json`.
  Una prueba falla si algún script vuelve a usar `joblib.load` directo.
- Si la huella no coincide, la API **no arranca**. Verificado en el contenedor: con
  `-e MODELO_SHA256=000…` el contenedor termina (exit 3) con `ModeloNoConfiable`, y con la huella
  correcta queda `healthy` y `/salud` muestra `integridad.sha256` y con qué se verificó.
- Para que la huella no dependa de un archivo dentro de la misma imagen, se puede fijar al
  ejecutar: `docker run -e MODELO_SHA256=<sha256 de modelo_final_metadatos.json> ...`.
- Límite: esto garantiza que el archivo es **el que se registró**, no que quien lo registró sea de
  confianza. El `.joblib` y su huella solo deben venir del equipo (paso 8) por el repositorio.

## 2. Despliegue

### Local (desarrollo)

```powershell
$env:HF_HUB_OFFLINE = "1"; python -m uvicorn api:app --port 8000
# http://127.0.0.1:8000
```

### Docker local (despliegue del proyecto)

```powershell
docker build -t polaridad-docente .
docker run -p 8000:8000 -e ADMIN_TOKEN=una-clave-larga -v polaridad_monitoreo:/app/monitoreo_datos polaridad-docente
# http://127.0.0.1:8000
```

La imagen trae torch solo CPU y el modelo MiniLM ya descargado en la **misma revisión** que pide el
Pipeline (`ARG REVISION_EMBEDDINGS`): arranca sin internet. Solo copia lo necesario para servir
(`.dockerignore`), nunca `train.csv` ni `test.csv`.

Verificado con la imagen construida (3.05 GB; la primera construcción tarda de 10 a 40 min según la
conexión): contenedor `healthy`, mismas 123/123 predicciones del paso 7, `/docs` apagado (404), sin
cabecera `Server`, cabeceras de seguridad presentes, `/monitoreo` exige token (401 sin él), proceso
corriendo como usuario `app` (sin privilegios), sin datos de entrenamiento dentro, registro de
monitoreo sin textos, `X-Forwarded-For` falso no salta el límite y cuerpo chunked de 7 MB → 413.

Si más adelante se pone un proxy con HTTPS delante (Nginx, servidor institucional), hay que volver
a activar `--proxy-headers` con `--forwarded-allow-ips=<IP del proxy>` (nunca `'*'`), para que el
límite use la IP real del usuario.

Variables de entorno:

| Variable | Por defecto (Docker) | Uso |
|---|---|---|
| `PORT` | 8000 | Puerto de escucha |
| `DOCS` | 0 | `1` muestra `/docs` |
| `RATE_LIMIT` | 60 | Peticiones/minuto/IP en `/predecir*` (`0` = sin límite) |
| `ADMIN_TOKEN` | vacío | Si se define, `/monitoreo` exige la cabecera `X-Admin-Token` |
| `CORS_ORIGINS` | vacío | Orígenes permitidos, separados por comas |
| `MONITOREO` | 1 | `0` desactiva el registro |
| `MODELO_SHA256` | vacío | Huella esperada del modelo; si falta, se usa la de `modelo_final_metadatos.json` |

## 3. Monitoreo de cambios en los datos

En producción no hay etiquetas reales, así que no se puede medir el macro-F1 en vivo. Lo que sí se
vigila es si los comentarios que llegan **se parecen a los datos con que se evaluó el modelo**:

| Indicador | Qué detecta |
|---|---|
| PSI de `n_palabras` | Comentarios mucho más cortos o largos |
| PSI de `oov` (% de palabras desconocidas) | Otro tema, idioma o jerga |
| PSI de `confianza` | El modelo está menos (o más) seguro que en la evaluación |
| PSI de clases predichas | Cambió la proporción positivo / neutral / negativo |
| Tasa de ambiguos y de vacíos | Comentarios difíciles o problemas en el formulario/CSV |

PSI < 0.10 estable · 0.10–0.25 moderado · > 0.25 importante.
La referencia (`monitoreo/referencia.json`) son predicciones fuera de muestra sobre train (CV 5-fold).

Consultar el estado:

```powershell
curl http://127.0.0.1:8000/monitoreo?ultimos=500
# con token: curl -H "X-Admin-Token: una-clave-larga" ...
```

Validación del monitor (`paso10_deriva.py`):

| Escenario | Estado detectado | Indicador que lo delata |
|---|---|---|
| A. test.csv (sin cambios) | estable | — |
| B. comentarios recortados a 4 palabras | importante | longitud, confianza, ambiguos |
| C. comentarios en inglés | importante | oov = 100% (¡la confianza sigue alta: 0.77!) |
| D. solo negativos | importante | clases predichas |
| E. 40% vacíos | importante | tasa de vacíos |

El escenario C muestra por qué no basta con vigilar la confianza: con textos en inglés el modelo
responde muy seguro aunque no conoce ninguna palabra. El indicador `oov` es el que lo detecta.

**Qué hacer ante una alerta**: si es *moderado*, revisar una muestra de comentarios recientes;
si es *importante*, etiquetar una muestra nueva (~100 comentarios), medir el desempeño real y,
si baja, reentrenar con los pasos 6–8 y regenerar la referencia con `paso10_deriva.py`.

## Archivos del paso 6

| Archivo | Qué es |
|---|---|
| `api.py` | API con seguridad y registro de monitoreo |
| `carga_segura.py` | Carga de `.joblib` solo si su sha256 coincide con el registrado |
| `test_carga_segura.py` | 8 pruebas de deserialización segura (incluye un `.joblib` malicioso) |
| `monitoreo.py` | Registro, referencia y reporte de deriva (PSI) |
| `paso10_rendimiento.py` | Medición de tiempos y memoria (en proceso o contra el contenedor con `--url`) |
| `paso10_deriva.py` | Crea la referencia y valida el monitor con 5 escenarios |
| `test_api.py` | 24 pruebas: funcionalidad, eficiencia, seguridad y monitoreo |
| `Dockerfile`, `.dockerignore` | Imagen de despliegue |
| `requirements_app.txt`, `requirements_dev.txt` | Dependencias de producción y de pruebas |
