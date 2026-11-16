// Página de polaridad de comentarios: habla con la API (mismo origen).
"use strict";

const CLASES = ["negativo", "neutral", "positivo"];
const ORDEN = ["positivo", "neutral", "negativo", "ambiguo", "texto_vacio"];
const NOMBRES = { positivo: "Positivo", neutral: "Neutral", negativo: "Negativo",
                  ambiguo: "Ambiguo", texto_vacio: "Vacío" };
const COLOR = { positivo: "var(--positivo)", neutral: "var(--neutral)", negativo: "var(--negativo)",
                ambiguo: "var(--ambiguo)", texto_vacio: "var(--vacio)" };
const MAX_FILAS_TABLA = 300;

const $ = (sel) => document.querySelector(sel);

// Todo texto que viene del usuario se escapa antes de insertarlo en el HTML
function esc(t) {
  return String(t ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}
const pct = (v) => (v == null ? "—" : `${Math.round(v * 100)}%`);

async function api(ruta, opciones = {}) {
  let r;
  try {
    r = await fetch(ruta, opciones);
  } catch {
    throw new Error("No se pudo conectar con el servicio. ¿Está corriendo la API?");
  }
  const datos = await r.json().catch(() => ({}));
  if (!r.ok) {
    let d = datos.detail ?? r.statusText;
    if (Array.isArray(d)) d = d.map((e) => e.msg).join("; ");
    throw new Error(`Error ${r.status}: ${d}`);
  }
  return datos;
}

function caraSVG(clase) {
  const boca = { positivo: "M-9 4 Q0 13 9 4", neutral: "M-8 7 H8", negativo: "M-9 10 Q0 1 9 10" }[clase];
  return `<svg class="cara" viewBox="-26 -26 52 52" aria-hidden="true">
    <circle r="24" fill="${COLOR[clase]}"/>
    <circle cx="-8" cy="-6" r="2.6" fill="#fff"/><circle cx="8" cy="-6" r="2.6" fill="#fff"/>
    <path d="${boca}" fill="none" stroke="#fff" stroke-width="3" stroke-linecap="round"/></svg>`;
}

// ---------------------------------------------------------------------------
// Estado del servicio y métricas
// ---------------------------------------------------------------------------

async function cargarEstado() {
  const pill = $("#estado-api");
  try {
    const [salud, meta] = await Promise.all([api("/salud"), api("/modelo")]);
    pill.className = "estado-api ok";
    pill.lastChild.textContent = `Modelo v${salud.version_modelo} listo`;
    const m = meta.metricas;
    $("#metricas").innerHTML = `
      <li><b>${m.macro_f1_test.toFixed(2)}</b>Macro-F1 en test</li>
      <li><b>${Math.round(m.accuracy_test * 100)}%</b>Aciertos en test</li>
      <li><b>3</b>Clases de polaridad</li>`;
    // La nota inferior sale de los metadatos del modelo cargado, no de un texto fijo
    $("#nota-aciertos").textContent = `${Math.round(m.accuracy_test * 100)}%`;
    $("#nota-f1").textContent = m.macro_f1_test.toFixed(2);
  } catch (e) {
    pill.className = "estado-api error";
    pill.lastChild.textContent = "Servicio no disponible";
  }
}

// ---------------------------------------------------------------------------
// Pestañas
// ---------------------------------------------------------------------------

const tabs = [...document.querySelectorAll('[role="tab"]')];
function activar(tab) {
  tabs.forEach((t) => {
    const activa = t === tab;
    t.setAttribute("aria-selected", activa);
    t.tabIndex = activa ? 0 : -1;
    document.getElementById(t.getAttribute("aria-controls")).hidden = !activa;
  });
  tab.focus();
}
tabs.forEach((t, i) => {
  t.addEventListener("click", () => activar(t));
  t.addEventListener("keydown", (e) => {
    if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
      activar(tabs[(i + (e.key === "ArrowRight" ? 1 : tabs.length - 1)) % tabs.length]);
    }
  });
});

// ---------------------------------------------------------------------------
// Un comentario
// ---------------------------------------------------------------------------

const texto = $("#texto");
const actualizarContador = () => { $("#contador").textContent = `${texto.value.length} / 5000`; };
texto.addEventListener("input", actualizarContador);
document.querySelectorAll("[data-ejemplo]").forEach((b) =>
  b.addEventListener("click", () => { texto.value = b.dataset.ejemplo; actualizarContador(); texto.focus(); }));

$("#form-uno").addEventListener("submit", async (e) => {
  e.preventDefault();
  const boton = $("#btn-uno"), caja = $("#resultado-uno");
  boton.disabled = true;
  boton.innerHTML = '<span class="cargando"></span>Analizando…';
  try {
    const d = await api("/predecir", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ texto: texto.value }),
    });
    caja.innerHTML = d.estado === "texto_vacio" ? `
      <div class="vacio-estado"><p>El comentario está vacío o no tiene palabras con contenido.<br>
      Escribe un comentario para analizarlo.</p></div>` : `
      <div class="clase-predicha">
        ${caraSVG(d.prediccion)}
        <div><small>Polaridad</small><strong style="color:${COLOR[d.prediccion]}">${esc(d.prediccion)}</strong></div>
        <div class="confianza"><small>Confianza</small><br><b>${pct(d.confianza)}</b></div>
      </div>
      <div class="barras">
        ${CLASES.map((c) => `
          <div class="barra-prob">
            <span>${NOMBRES[c]}</span>
            <div class="pista"><div class="relleno" style="background:${COLOR[c]}" data-w="${d.probabilidades[c] * 100}"></div></div>
            <span>${pct(d.probabilidades[c])}</span>
          </div>`).join("")}
      </div>
      ${d.estado === "ambiguo" ? `<div class="aviso">Predicción <b>ambigua</b>: ninguna clase llega al 50%.
        Conviene revisar este comentario manualmente.</div>` : ""}
      ${bloqueAspectos(d.aspectos)}`;
    requestAnimationFrame(() => caja.querySelectorAll(".relleno").forEach((r) => { r.style.width = `${r.dataset.w}%`; }));
  } catch (err) {
    caja.innerHTML = `<div class="aviso error">${esc(err.message)}</div>`;
  } finally {
    boton.disabled = false;
    boton.textContent = "Analizar comentario";
  }
});

// Punto 5: los 9 aspectos llegan siempre; se muestran solo los mencionados,
// con su sentimiento en texto (no solo en color) y la probabilidad de mención.
function bloqueAspectos(aspectos) {
  if (!aspectos) return "";
  const mencionados = aspectos.filter((a) => a.mencionado);
  const contenido = mencionados.length
    ? `<ul class="lista-aspectos">${mencionados.map((a) => `
        <li class="aspecto" style="--c:${COLOR[a.sentimiento]}"
            title="Probabilidad de que hable de este aspecto: ${pct(a.probabilidad)}">
          <span class="punto" aria-hidden="true"></span>${esc(a.nombre)}
          <b>${esc(NOMBRES[a.sentimiento])}</b></li>`).join("")}</ul>`
    : '<p class="sin-aspectos">No se detectó ningún aspecto concreto del curso.</p>';
  return `<div class="aspectos"><small>Aspectos del curso</small>${contenido}</div>`;
}

// ---------------------------------------------------------------------------
// Archivo CSV
// ---------------------------------------------------------------------------

let archivoActual = null;
let resultadosCSV = [];
let filtroActivo = new Set(ORDEN);

// Divide una línea CSV respetando comillas
function dividirLinea(linea, sep) {
  const campos = []; let actual = "", comillas = false;
  for (let i = 0; i < linea.length; i++) {
    const c = linea[i];
    if (c === '"') { if (comillas && linea[i + 1] === '"') { actual += '"'; i++; } else comillas = !comillas; }
    else if (c === sep && !comillas) { campos.push(actual); actual = ""; }
    else actual += c;
  }
  campos.push(actual);
  return campos.map((s) => s.trim());
}

async function elegirArchivo(archivo) {
  if (!archivo) return;
  $("#resultado-csv").innerHTML = "";
  if (!/\.csv$/i.test(archivo.name)) {
    $("#resultado-csv").innerHTML = '<div class="aviso error">El archivo debe ser .csv</div>';
    return;
  }
  archivoActual = archivo;
  const tamano = archivo.size < 1024 ? `${archivo.size} B` : `${Math.round(archivo.size / 1024)} KB`;
  $("#nombre-archivo").textContent = `${archivo.name} · ${tamano}`;
  const inicio = await archivo.slice(0, 64 * 1024).text();
  const encabezado = inicio.replace(/^﻿/, "").split(/\r?\n/)[0] ?? "";
  const sep = encabezado.includes(";") && encabezado.split(";").length >= encabezado.split(",").length ? ";" : ",";
  const columnas = dividirLinea(encabezado, sep).filter(Boolean);
  const select = $("#columna");
  select.innerHTML = columnas.map((c) => `<option value="${esc(c)}">${esc(c)}</option>`).join("");
  const sugerida = columnas.find((c) => /feedback|comentario|texto|opini/i.test(c));
  if (sugerida) select.value = sugerida;
  $("#opciones-csv").hidden = columnas.length === 0;
}

const zona = $("#zona-archivo");
$("#archivo").addEventListener("change", (e) => elegirArchivo(e.target.files[0]));
["dragenter", "dragover"].forEach((ev) => zona.addEventListener(ev, (e) => { e.preventDefault(); zona.classList.add("arrastrando"); }));
["dragleave", "drop"].forEach((ev) => zona.addEventListener(ev, (e) => { e.preventDefault(); zona.classList.remove("arrastrando"); }));
zona.addEventListener("drop", (e) => elegirArchivo(e.dataTransfer.files[0]));

const categoria = (r) => (r.estado === "ok" ? r.prediccion : r.estado);

$("#btn-csv").addEventListener("click", async () => {
  if (!archivoActual) return;
  const boton = $("#btn-csv"), caja = $("#resultado-csv");
  boton.disabled = true;
  boton.innerHTML = '<span class="cargando"></span>Clasificando…';
  caja.innerHTML = "";
  try {
    const form = new FormData();
    form.append("archivo", archivoActual);
    const columna = encodeURIComponent($("#columna").value);
    const d = await api(`/predecir/csv?columna=${columna}`, { method: "POST", body: form });
    resultadosCSV = d.resultados;
    filtroActivo = new Set(ORDEN);
    pintarCSV(d);
  } catch (err) {
    caja.innerHTML = `<div class="aviso error">${esc(err.message)}</div>`;
  } finally {
    boton.disabled = false;
    boton.textContent = "Clasificar archivo";
  }
});

function pintarCSV(d) {
  const conteo = Object.fromEntries(ORDEN.map((k) => [k, 0]));
  resultadosCSV.forEach((r) => { conteo[categoria(r)]++; });
  const total = resultadosCSV.length || 1;
  $("#resultado-csv").innerHTML = `
    <div class="resumen">
      ${ORDEN.map((k) => `<div style="--c:${COLOR[k]}"><small>${NOMBRES[k]}</small><b>${conteo[k]}</b>
        <small>${Math.round((conteo[k] / total) * 100)}%</small></div>`).join("")}
    </div>
    <div class="distribucion" role="img" aria-label="Distribución de polaridad">
      ${ORDEN.filter((k) => conteo[k]).map((k) => `<span style="--c:${COLOR[k]};flex:${conteo[k]}" title="${NOMBRES[k]}: ${conteo[k]}"></span>`).join("")}
    </div>
    <div class="barra-acciones">
      <div class="filtros" aria-label="Filtrar por categoría">
        ${ORDEN.map((k) => `<button type="button" class="chip" data-filtro="${k}" aria-pressed="true">${NOMBRES[k]}</button>`).join("")}
      </div>
      <button type="button" class="boton secundario" id="btn-descargar">Descargar resultados (CSV)</button>
    </div>
    <div id="tabla"></div>
    <p class="pie-tabla">${d.n} comentarios clasificados en ${(d.milisegundos / 1000).toFixed(1)} s.</p>`;
  document.querySelectorAll("[data-filtro]").forEach((b) => b.addEventListener("click", () => {
    const k = b.dataset.filtro;
    filtroActivo.has(k) ? filtroActivo.delete(k) : filtroActivo.add(k);
    b.setAttribute("aria-pressed", filtroActivo.has(k));
    pintarTabla();
  }));
  $("#btn-descargar").addEventListener("click", descargar);
  pintarTabla();
}

function pintarTabla() {
  const filas = resultadosCSV.map((r, i) => ({ ...r, fila: i + 1 })).filter((r) => filtroActivo.has(categoria(r)));
  const visibles = filas.slice(0, MAX_FILAS_TABLA);
  $("#tabla").innerHTML = `
    <div class="tabla-envoltura"><table>
      <thead><tr><th>#</th><th>Comentario</th><th>Resultado</th><th>Confianza</th>
        <th>Neg.</th><th>Neu.</th><th>Pos.</th></tr></thead>
      <tbody>${visibles.map((r) => `
        <tr><td class="num">${r.fila}</td>
          <td class="comentario">${esc(r.texto) || "<i>(vacío)</i>"}</td>
          <td><span class="etiqueta-clase" style="--c:${COLOR[categoria(r)]}">${NOMBRES[categoria(r)]}</span>
            ${r.estado === "ambiguo" ? `<br><small>${esc(r.prediccion)}?</small>` : ""}</td>
          <td class="num">${pct(r.confianza)}</td>
          ${CLASES.map((c) => `<td class="num">${pct(r.probabilidades[c])}</td>`).join("")}</tr>`).join("")}
      </tbody></table></div>
    ${filas.length > MAX_FILAS_TABLA ? `<p class="pie-tabla">Se muestran ${MAX_FILAS_TABLA} de ${filas.length}; descarga el CSV para verlos todos.</p>` : ""}`;
}

// Un comentario que empieza por = + - @ (o tab / retorno) se volveria formula
// al abrir el CSV en Excel: =HYPERLINK("http://...") ejecutaria un enlace.
// Se antepone ' para que la hoja de calculo lo muestre como texto.
function sinFormula(v) {
  const s = String(v ?? "");
  return /^[=+\-@\t\r]/.test(s) ? `'${s}` : s;
}

function descargar() {
  const celda = (v) => `"${sinFormula(v).replace(/"/g, '""')}"`;
  // Una columna por aspecto con su sentimiento (vacía si no se menciona)
  const claves = resultadosCSV.find((r) => r.aspectos)?.aspectos.map((a) => a.aspecto) ?? [];
  const lineas = [["fila", "comentario", "estado", "prediccion", "confianza", ...CLASES.map((c) => `prob_${c}`),
    ...claves.map((k) => `aspecto_${k}`)].join(";")];
  resultadosCSV.forEach((r, i) => lineas.push([i + 1, celda(r.texto), r.estado, r.prediccion ?? "",
    r.confianza ?? "", ...CLASES.map((c) => r.probabilidades[c] ?? ""),
    ...claves.map((_, j) => (r.aspectos?.[j]?.mencionado ? r.aspectos[j].sentimiento : ""))].join(";")));
  const blob = new Blob(["﻿" + lineas.join("\n")], { type: "text/csv;charset=utf-8" });
  const a = Object.assign(document.createElement("a"), {
    href: URL.createObjectURL(blob),
    download: `polaridad_${(archivoActual?.name ?? "resultados").replace(/\.csv$/i, "")}.csv`,
  });
  a.click();
  URL.revokeObjectURL(a.href);
}

cargarEstado();
