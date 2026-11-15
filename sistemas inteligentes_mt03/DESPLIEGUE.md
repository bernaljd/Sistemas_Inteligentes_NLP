# Paso 6 · Probar, desplegar y monitorear

Servicio: API FastAPI + página web (`/`) que clasifica comentarios de estudiantes
en negativo / neutral / positivo con `modelo_final.joblib` (LogReg + TF-IDF + MiniLM).

## 1. Verificación

| Aspecto | Cómo se verifica | Resultado |
|---|---|---|
| Funcionalidad | `python test_api.py` (11 pruebas) | 11/11 OK |
| Seguridad | `python test_api.py` (5 pruebas) + `pip-audit` | 5/5 OK · 0 vulnerabilidades conocidas* |
| Monitoreo | `python test_api.py` (2 pruebas) + `python paso10_deriva.py` | 2/2 OK · deriva detectada en 4/4 escenarios |
| Tiempos | `python paso10_rendimiento.py` | ver tabla abajo |

\* `pip-audit` encontró 10 vulnerabilidades en Starlette 0.52.1 (base de FastAPI). Se actualizó a
FastAPI 0.163.0 + Starlette 1.10.0 y la auditoría quedó limpia.

### Tiempos de inferencia (portátil de desarrollo, CPU, 7.7 GB RAM)

Medido con el equipo bajo presión de memoria (0.5 GB libres), por lo que son cifras pesimistas.

| Prueba | Medido | Objetivo |
|---|---|---|
| Un comentario, modelo aislado (p50 / p95) | ~110 ms / ~155 ms | ≤ 500 ms ✔ |
| Un comentario vía API con el equipo saturado (p95) | 1076 ms | ≤ 500 ms ✘ |
| Lote de 100 comentarios | 15.8 s (~80–160 ms por comentario) | ≤ 5 s ✘ |
| CSV de 1000 filas | 92.8 s | ≤ 60 s ✘ |
| Arranque (cargar + calentar) | 25–170 s | ≤ 120 s (variable) |
| Memoria del contenedor Docker (`docker stats`) | ~1 GB | ≤ 2 GB ✔ |
| Un comentario dentro del contenedor (p50 / p95) | 177 ms / 440 ms | ≤ 500 ms ✔ |
| Lote de test.csv (123) dentro del contenedor | 19.9 s | — |

(En Windows `psutil` mostró ~300 MB porque el sistema estaba paginando memoria a disco;
la cifra real es la del contenedor.)

Casi todo el tiempo se va en los embeddings MiniLM; TF-IDF y la regresión logística toman milisegundos.
Más hilos de torch no ayudan (probado con 1, 2, 4, 6 y 8 hilos). Para un uso interactivo (un docente
analizando comentarios) el tiempo es aceptable; para lotes grandes conviene un servidor con más RAM.

### Seguridad implementada

- **Límite de peticiones**: 60 por minuto por IP en `/predecir*` → `429` con `Retry-After`.
- **Límites de tamaño**: cuerpo ≤ 6 MB, CSV ≤ 5 MB / 1000 filas, texto ≤ 5000 caracteres, lote ≤ 1000.
- **Cabeceras**: Content-Security-Policy (solo scripts propios), `X-Frame-Options: DENY`,
  `X-Content-Type-Options: nosniff`, `Referrer-Policy: no-referrer`, `Permissions-Policy`.
- **XSS**: la página escapa todo texto del usuario (`esc()`) antes de mostrarlo.
- **Sin CORS abierto**: la página y la API comparten origen (`CORS_ORIGINS` para habilitar otros).
- **Errores**: los errores internos responden `Error interno del servidor`, sin rutas ni trazas.
- **Producción**: `/docs` apagado (`DOCS=0`), `/monitoreo` protegido con `ADMIN_TOKEN`,
  contenedor con usuario sin privilegios, sin cabecera `Server`.
- **Privacidad**: el registro de monitoreo **no guarda el texto** de los comentarios (solo longitud,
  % de palabras desconocidas, clase, confianza y un hash).

## 2. Despliegue

### Local (desarrollo)

```powershell
$env:HF_HUB_OFFLINE = "1"; python -m uvicorn api:app --port 8000
# http://127.0.0.1:8000
```

### Docker (cualquier servidor o nube)

```powershell
docker build -t polaridad-docente .
docker run -p 8000:8000 -e ADMIN_TOKEN=una-clave-larga -v polaridad_monitoreo:/app/monitoreo_datos polaridad-docente
```

La imagen trae torch solo CPU y el modelo MiniLM ya descargado: arranca sin internet. Solo copia lo
necesario para servir (`.dockerignore`), nunca `train.csv` ni `test.csv`.

Verificado con la imagen construida (3.05 GB, ~9 min de construcción): contenedor `healthy` en ~40 s,
mismas 123/123 predicciones del paso 7, `/docs` apagado (404), sin cabecera `Server`, cabeceras de
seguridad presentes, `/monitoreo` exige token (401 sin él), proceso corriendo como usuario `app`
(sin privilegios), sin datos de entrenamiento dentro y registro de monitoreo sin textos.

### Nube

| Plataforma | RAM gratuita | Notas |
|---|---|---|
| Hugging Face Spaces (Docker) | 16 GB | Desde 2026 crear un Space con Docker exige el plan PRO (9 USD/mes). |
| Azure for Students | Créditos de estudiante | Opción gratuita con correo institucional; admite contenedores de ~1 GB. |
| Render / Railway | 512 MB | No alcanza: el servicio usa ~1 GB. Se necesita un plan de ≥ 2 GB. |
| Servidor institucional | — | `docker run` detrás de Nginx con HTTPS. |

Variables de entorno:

| Variable | Por defecto (Docker) | Uso |
|---|---|---|
| `PORT` | 8000 | Puerto de escucha |
| `DOCS` | 0 | `1` muestra `/docs` |
| `RATE_LIMIT` | 60 | Peticiones/minuto/IP en `/predecir*` (`0` = sin límite) |
| `ADMIN_TOKEN` | vacío | Si se define, `/monitoreo` exige la cabecera `X-Admin-Token` |
| `CORS_ORIGINS` | vacío | Orígenes permitidos, separados por comas |
| `MONITOREO` | 1 | `0` desactiva el registro |

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
| `monitoreo.py` | Registro, referencia y reporte de deriva (PSI) |
| `paso10_rendimiento.py` | Medición de tiempos y memoria |
| `paso10_deriva.py` | Crea la referencia y valida el monitor con 5 escenarios |
| `test_api.py` | 18 pruebas: funcionalidad, seguridad y monitoreo |
| `Dockerfile`, `.dockerignore` | Imagen de despliegue |
| `requirements_app.txt`, `requirements_dev.txt` | Dependencias de producción y de pruebas |
