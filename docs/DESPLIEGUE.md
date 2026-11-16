# Paso 6 · Probar, desplegar y monitorear

Servicio: API FastAPI + página web (`/`) que clasifica comentarios de estudiantes
en negativo / neutral / positivo con la versión activa de `modelos/polaridad/`
(`modelo_final.joblib`: LogReg + TF-IDF + MiniLM) y por aspecto con `modelos/aspectos/`.

Todos los comandos de este documento se ejecutan dentro de `sistemas inteligentes_mt03/`.

## 1. Verificación

| Aspecto | Cómo se verifica | Resultado |
|---|---|---|
| Funcionalidad | `python -m pytest tests/test_api.py` (11 pruebas) | 11/11 OK |
| Seguridad | `python -m pytest tests/test_api.py` (9 pruebas) + `pip-audit` | 9/9 OK · 0 vulnerabilidades conocidas* |
| Monitoreo | `python -m pytest tests/test_api.py` (2 pruebas) + `python paso10_deriva.py` | 2/2 OK · deriva detectada en 4/4 escenarios |
| Tiempos | `python paso10_rendimiento.py --url ... --contenedor ...` | ver tabla abajo |

\* `pip-audit` encontró 10 vulnerabilidades en Starlette 0.52.1 (base de FastAPI). Se actualizó a
FastAPI 0.163.0 + Starlette 1.10.0 y la auditoría quedó limpia.

### Tiempos de inferencia (contenedor Docker local, CPU)

Medido por HTTP contra el contenedor real, que es el despliegue, en un PC de 16 núcleos lógicos y
31.6 GB de RAM, con polaridad **y** los 9 aspectos (punto 5) en cada respuesta. Objetivos fijados
antes de medir.

| Prueba | Comentarios nuevos | Comentarios repetidos (`paso10_rendimiento.py --url`) | Objetivo |
|---|---|---|---|
| Un comentario (p50 / p95) | 53 ms / 73 ms | 24 ms / 71 ms | ≤ 500 ms ✔ |
| Lote de 100 comentarios | 1.9 s | 0.9 s | ≤ 5 s ✔ |
| CSV de 1000 filas | 16.1 s | 7.4 s | ≤ 60 s ✔ |
| Arranque (cargar + calentar los dos modelos, según `/salud`) | 7.7 s | | ≤ 120 s ✔ |
| Memoria del contenedor (`docker stats`) | ~1.0 GB tras cargar · 1.2 GB tras las pruebas | | ≤ 2 GB ✔ |

`paso10_rendimiento.py` repite los 123 comentarios de test, y el modelo de aspectos guarda en caché
los embeddings ya calculados (máximo 20 000 textos), así que esa columna es optimista. La columna
"comentarios nuevos" usa textos distintos en cada petición, y es la cifra realista. Sin aspectos
(antes del punto 5), un comentario tardaba p95 52 ms y el lote de 100, 1.3 s.

Una medición anterior en un portátil con 7.7 GB de RAM y solo 0.5 GB libres (el sistema paginaba a
disco) dio p95 de 1076 ms, 15.8 s para el lote de 100 y 167 s de arranque: el servicio necesita
~1 GB de RAM libre. Casi todo el tiempo se va en los embeddings MiniLM; TF-IDF y la regresión
logística toman milisegundos.

### Eficiencia: el servicio no entrena

El entrenamiento es un paso aparte y offline (`paso8_guardar_modelo.py`). Al arrancar, la API
solo **carga** la versión activa del modelo una vez (`cargar_servicio()`), calienta el modelo de
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
| Huella sha256 comparada **antes** de deserializar, contra `MODELO_SHA256` o `modelos/registro.json` (que debe coincidir con `modelo_final_metadatos.json`) | Un modelo reemplazado o alterado (basta 1 byte) nunca se ejecuta |
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
  ejecutar: `docker run -e MODELO_SHA256=<sha256 de la versión en modelos/registro.json> ...`.
- Límite: esto garantiza que el archivo es **el que se registró**, no que quien lo registró sea de
  confianza. El `.joblib` y su huella solo deben venir del equipo (paso 8) por el repositorio.

### Compatibilidad de versiones (entrenamiento = producción)

scikit-learn no garantiza que un modelo guardado con una versión se comporte igual con otra. El
paso 8 registra las versiones exactas (en los metadatos y en `requirements_modelo.txt`), y
`compatibilidad.py` las compara con las instaladas **antes de deserializar** el modelo:

| Nivel | Paquetes | Si difiere |
|---|---|---|
| Crítico | Python (mayor.menor: 3.12), `scikit-learn`, `joblib` | La API **no arranca** (`VersionesIncompatibles`) |
| Exacto | `numpy`, `scipy`, `pandas`, `sentence-transformers`, `torch`, `transformers` | Aviso en el log y en `/salud`; con `VERSIONES_ESTRICTAS=1` tampoco arranca |

- El parche de Python (3.12.5 al entrenar, 3.12.16 en Docker) y etiquetas como `+cpu` no cuentan.
- Un `InconsistentVersionWarning` de scikit-learn al cargar también se trata como incompatibilidad.
- `/salud` muestra `compatibilidad: {estado, diferencias}`.
- **Una sola fuente de versiones:** `requirements_modelo.txt` (lo genera el paso 8). `requirements.txt`
  (entrenamiento), `requirements_app.txt` (producción) y `requirements_dev.txt` (pruebas) lo incluyen
  con `-r`, y ninguna dependencia queda sin versión fija. El `Dockerfile` usa el mismo torch y
  Python 3.12, y ejecuta `pip check`. `test_compatibilidad.py` falla si alguna fuente se desalinea.
- Verificado en el contenedor: con las versiones correctas queda `healthy` y `compatibilidad: ok`;
  con metadatos que dicen scikit-learn 1.8.0, no arranca y no llega a deserializar el modelo.
- **Para actualizar una dependencia:** cambiarla en el entorno, reentrenar con el paso 8 (regenera
  el modelo, los metadatos y `requirements_modelo.txt`), validar con el paso 9 y reconstruir la imagen.
  Nunca actualizar scikit-learn en producción sin reentrenar. El modelo de aspectos (paso 11) pasa
  por la misma verificación con sus propios metadatos: si cambian las versiones, reentrenarlo también.

### Salida estructurada: una predicción por aspecto

Cada predicción (`/predecir`, `/predecir/lote`, `/predecir/csv`) trae, además de la polaridad, el
campo `aspectos`: **siempre los 9 aspectos, en el mismo orden**, cada uno con este formato:

```json
{"aspecto": "evaluacion", "nombre": "Evaluación", "mencionado": true, "probabilidad": 0.81,
 "sentimiento": "negativo",
 "probabilidades_sentimiento": {"negativo": 0.74, "neutral": 0.18, "positivo": 0.08}}
```

- `mencionado`: el comentario habla del aspecto (probabilidad ≥ 0.5). Si no, `sentimiento` y
  `probabilidades_sentimiento` son `null`. En textos vacíos los 9 salen con `probabilidad: null`.
- Los 9 aspectos agrupan los 22 nombres de la columna `aspectos` de los datos (ver `aspectos.py`):
  contenidos, evaluación, retroalimentación, metodología, material de clase, organización y horario,
  carga de trabajo, recursos digitales, comunicación e interacción.
- Modelo (`paso11_aspectos.py`, elegido con validación cruzada solo en train): presencia con una
  regresión logística por aspecto sobre TF-IDF + embeddings; sentimiento con el **fragmento** del
  comentario que habla del aspecto (cortando por ".", "pero", "aunque"...) + el comentario completo.
  El sentimiento de un aspecto solo coincide con la polaridad del comentario en ~50 % de los casos
  ("sabe mucho, pero los parciales son injustos"), por eso no basta con copiar la polaridad.

| Resultado | Validación cruzada (train) | Test |
|---|---|---|
| Presencia, macro-F1 de los 9 aspectos (azar: 0.15) | 0.546 | 0.532 |
| Sentimiento en los pares reales, macro-F1 | 0.676 | 0.652 |
| … copiando la polaridad **real** del comentario | 0.504 | 0.537 |
| Tuplas completas (aspecto + sentimiento) correctas: P / R / F1 | | 0.40 / 0.46 / 0.43 |

- **Límites (honestos):** la presencia es buena en organización/horario (F1 0.76), contenidos y
  evaluación, y débil en carga de trabajo y comunicación (F1 0.31, pocos ejemplos: 8 y 5 en test).
  Las etiquetas de aspectos son escasas (27-291 comentarios por aspecto en train). Úsese como apoyo
  para orientar la lectura, no como conteo exacto. (La "exactitud por celda", 84.7 %, no se usa como
  métrica: marcar siempre "no mencionado" ya daría 85.1 %.)
- `GET /modelo/aspectos` devuelve los metadatos y métricas; `/salud` muestra su versión, huella y
  compatibilidad. El modelo es un archivo aparte (`modelos/aspectos/1.0.0/modelo_aspectos.joblib`)
  con su propia huella (`ASPECTOS_SHA256` o el registro): el modelo de polaridad no cambió. Si no
  hay modelo de aspectos registrado, el servicio funciona igual y `aspectos` sale `null`.
- La página web muestra los aspectos mencionados con su sentimiento (en texto, no solo en color), y
  el CSV descargado tiene una columna `aspecto_<clave>` por aspecto.

### Control y reversión de versiones del modelo

Cada modelo entrenado es una **versión inmutable** con su propia carpeta, y un registro dice cuál
está en servicio (`versiones_modelo.py`):

```
modelos/
  registro.json                          versión activa, anteriores, datos de cada versión e historial
  polaridad/1.1.0/modelo_final.joblib    + modelo_final_metadatos.json + firma.json
  aspectos/1.0.0/modelo_aspectos.joblib  + modelo_aspectos_metadatos.json + firma.json
```

Por cada versión, el registro guarda la huella sha256, las fechas de entrenamiento y de registro, el
**commit de git** con que se generó, las versiones de librerías y las métricas. El `historial` anota
cada registro, promoción y reversión con fecha, versión de origen y de destino, y **motivo**
(obligatorio). Quién hizo el cambio queda en el autor del commit de git que modifica `registro.json`.

**Procedimiento de actualización**

```powershell
# 1. Entrenar: subir VERSION_MODELO en paso8_guardar_modelo.py (p. ej. 1.2.0) y ejecutarlo.
#    Guarda en modelos/polaridad/1.2.0/ y la registra como CANDIDATA: la API sigue igual.
python paso8_guardar_modelo.py
# 2. Validar la candidata en un entorno limpio
python paso9_validar_modelo.py --version 1.2.0
# 3. Ponerla en servicio (verifica huella, versiones, carga y firma antes de cambiar nada)
python versiones_modelo.py promover polaridad 1.2.0 --motivo "reentrenado con datos de 2027-1"
# 4. Commit de modelos/ y reconstruir la imagen
docker build -t polaridad-docente .
```

**Procedimiento de reversión**

```powershell
# Deshace la última promoción (vuelve a la versión activa anterior), con la misma verificación
python versiones_modelo.py revertir polaridad --motivo "más neutrales mal clasificados en producción"
docker build -t polaridad-docente .

# De emergencia, sin reconstruir ni tocar el registro: la imagen trae TODAS las versiones
docker run -p 8000:8000 -e MODELO_VERSION=1.1.0 polaridad-docente
```

`python versiones_modelo.py listar` muestra las versiones, la activa y el historial, y
`python versiones_modelo.py verificar --completo` comprueba todas. En la API, `/salud` dice qué
versión se cargó y de dónde (`versiones_cargadas`), y `GET /modelo/versiones` devuelve el registro.

Reglas que se hacen cumplir (y que se prueban):

| Regla | Qué evita |
|---|---|
| Un número de versión no se reutiliza con otro contenido; `paso8` y `paso11` se niegan a entrenar sobre una versión registrada | Que "la 1.1.0" sean dos modelos distintos según cuándo se mire |
| Registrar no activa: solo la primera versión de cada familia queda activa sola | Que un reentrenamiento entre en servicio sin revisión |
| Promover y revertir verifican la huella, las versiones de librerías, la carga y una prueba funcional **antes** de cambiar el registro | Volver a una versión alterada o que no carga en este entorno |
| **Firma de comportamiento**: al registrar se guardan las predicciones sobre 10 comentarios fijos (`firma.json`); al promover, revertir o arrancar la API, deben repetirse (tolerancia 1e-4) | El `.joblib` guarda pesos, pero la lógica de las clases está en `pipeline_feedback.py` y `aspectos.py`: si ese código cambió, una versión vieja daría otras predicciones sin ningún error |
| La huella del registro debe coincidir con la de los metadatos | Cambiar uno de los dos a mano |
| `registro.json` se escribe de forma atómica | Un registro a medio escribir si algo falla |
| El build de Docker ejecuta `versiones_modelo.py verificar` | Construir una imagen con un modelo que no corresponde a su huella |

Si la firma no coincide, el mensaje indica el commit con que se registró la versión: hay que
revertir el modelo **junto con** ese código. Si un cambio de código es intencional y se revisó, se
puede forzar con `--aceptar-firma-distinta`, y el historial lo anota (`"firma": false`).

Verificado con los scripts reales, sobre una copia del proyecto:
- con `VERSION_MODELO` todavía en 1.1.0, `paso8` se negó a reentrenar;
- con 1.1.1, guardó y registró la versión como candidata, y la activa siguió siendo la 1.1.0;
- se promovió la 1.1.1 y la API la cargó; se revirtió y la API volvió a cargar la 1.1.0;
- el historial quedó con registrar → promover → revertir.

En el contenedor Linux, la firma registrada en Windows coincide (diferencia máxima 1e-6),
`MODELO_VERSION=1.1.0` carga esa versión y `MODELO_VERSION=9.9.9` impide arrancar.

Límites:
- La versión 1.0.0 original de polaridad (commit `b545b07`) no está en el registro: no tenía huella
  y su pipeline no tiene el paso `entrada` del punto 1. Recuperarla exige volver a ese commit.
- La referencia de deriva (`monitoreo/referencia.json`) se calculó con la 1.1.0: al promover otra
  versión de polaridad hay que regenerarla con `paso10_deriva.py`.
- Cambiar de versión requiere reiniciar la API: no hay recarga en caliente, a propósito, para que
  un proceso nunca sirva con dos modelos a la vez.

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
| `MODELO_VERSION` | vacío | Versión de polaridad a cargar (reversión de emergencia); si falta, la activa de `modelos/registro.json` |
| `ASPECTOS_VERSION` | vacío | Lo mismo para el modelo de aspectos |
| `MODELO_SHA256` | vacío | Huella esperada del modelo; si falta, se usa la de `modelos/registro.json` |
| `ASPECTOS_SHA256` | vacío | Huella esperada del modelo de aspectos; si falta, la del registro |
| `VERSIONES_ESTRICTAS` | 0 | `1` = cualquier versión distinta a la del modelo impide arrancar |

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
| `tests/test_carga_segura.py` | 8 pruebas de deserialización segura (incluye un `.joblib` malicioso) |
| `compatibilidad.py` | Compara las versiones del modelo con las instaladas antes de cargarlo |
| `tests/test_compatibilidad.py` | 13 pruebas de versiones (política y que todas las fuentes coincidan) |
| `aspectos.py` | Los 9 aspectos y el modelo de presencia + sentimiento por aspecto |
| `paso11_aspectos.py` | Elige el diseño con validación cruzada, evalúa en test y guarda el modelo de aspectos |
| `modelos/` | Versiones de cada modelo (`.joblib`, metadatos y firma) y `registro.json` |
| `versiones_modelo.py` | Registrar, promover, revertir, listar y verificar versiones |
| `tests/test_versiones_modelo.py` | 11 pruebas del ciclo de versiones y de los cambios que se rechazan |
| `resultados/resultados_aspectos.csv`, `resultados/paso11_salida.txt` | Resultados por aspecto (CV y test) |
| `tests/test_aspectos.py` | 11 pruebas de la salida estructurada (formato, vacíos, API, huella, métricas) |
| `monitoreo.py` | Registro, referencia y reporte de deriva (PSI) |
| `paso10_rendimiento.py` | Medición de tiempos y memoria (en proceso o contra el contenedor con `--url`) |
| `paso10_deriva.py` | Crea la referencia y valida el monitor con 5 escenarios |
| `tests/test_api.py` | 24 pruebas: funcionalidad, eficiencia, seguridad y monitoreo |
| `Dockerfile`, `.dockerignore` | Imagen de despliegue |
| `requirements_modelo.txt` | Versiones exactas del modelo (única fuente; la genera el paso 8) |
| `requirements.txt`, `requirements_app.txt`, `requirements_dev.txt` | Entrenamiento, producción y pruebas (incluyen `requirements_modelo.txt`) |
