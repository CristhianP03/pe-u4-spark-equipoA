# Dataset del experimento PE-U4 (dominio ACC)

## Descripción

Dataset **sintético** de dos tablas relacionadas sobre tickets de soporte
técnico ISP (dominio del PFC ACC: *Sistema de Gestión de Soporte Técnico ISP*):

| Tabla | Registros | Columnas | Tamaño en disco |
|---|---|---|---|
| `raw/tickets.csv` | 600 000 | 15 | 75 819 472 bytes (~75,8 MB) |
| `raw/agentes.csv` | 300 | 7 | 19 160 bytes (~19 KB) |

El diseño de **dos tablas relacionadas** (FK `tickets.agente_id` →
`agentes.agente_id`) hace que el join de la transformación T3 sea **genuino**
(no un self-join sobre el propio dataset).

## Fuente y URL

**No existe un dataset real público de tickets ISP con ≥ 500 000 registros**
(verificado). La guía permite explícitamente usar **Faker con semilla fija,
declarado en el informe** (Paso 1). Por tanto:

- **Fuente:** datos sintéticos generados por el propio equipo.
- **URL (origen de la generación):** script `src/generar_dataset_acc.py` del
  repositorio `Soporte-Tecnico-ACC`.
- El esquema (categorías, estados, prioridades con SLA, canales, zonas) está
  **anclado al dominio real del PFC ACC** (`docs/db/schema.sql` y
  `seed-generator` de `PFC-Soporte-ISP`).

## Licencia

Los datos son generados por el propio equipo; **no se redistribuyen datos de
terceros**. El código del proyecto se distribuye bajo la licencia declarada en
`LICENSE` (MIT).

## Fecha de generación

`fecha_generacion_utc`: `2026-08-07T04:50:09+00:00` (última generación). Se
regenera con el propio script y la fecha se actualiza automáticamente en la
salida impresa de la verificación.

## Reproducibilidad y determinismo

Semilla fija (numpy + Faker, locale `es_MX`) y rango de fechas fijo
(`2024-08-01` a `2026-07-31`). Con los mismos argumentos, los ficheros
generados son **bit a bit idénticos** entre ejecuciones.

```bash
# Regenerar (sobrescribe) y verificar registros/esquema
python src/generar_dataset_acc.py --sobrescribir
```

La verificación integrada comprueba (criterio 1.1 Nivel 4): número de filas y
columnas, unicidad de `ticket_id`, integridad referencial de `agente_id` y
conteo de nulos.

## Semántica de nulos

Los valores ausentes se serializan como **cadena vacía `""`** en el CSV:

- `agente_id = ""` → ticket **CREADO** sin agente asignado (~10 %).
- `fecha_resolucion = ""` → ticket **abierto** sin resolución (~40 %).

pandas convierte `""` → `NaN` y PySpark `""` → `NULL` (`leer_crudo_pandas` /
`leer_crudo_spark`). Ese criterio único garantiza la **equivalencia
pandas ↔ PySpark** de las transformaciones T1–T5 (criterio 1.4).

## Esquemas

**`tickets.csv` (15 columnas):** `ticket_id, cliente_id, agente_id, asunto,
categoria, estado, prioridad, canal, zona, sla_horas, fecha_creacion,
fecha_resolucion, hora_creacion, dia_semana_creacion, mes_creacion`.

**`agentes.csv` (7 columnas):** `agente_id, nombre, equipo, zona_operativa,
nivel, activo, fecha_ingreso`.

Distribuciones de los catálogos: categorías `[CONEXION 0.30, VELOCIDAD 0.20,
EQUIPO 0.15, FACTURACION 0.15, INSTALACION 0.10, OTRO 0.10]`; estados `[CREADO
0.10, ASIGNADO 0.15, EN_PROGRESO 0.15, RESUELTO 0.35, CERRADO 0.25]`;
prioridades `[BAJA 0.25, MEDIA 0.40, ALTA 0.25, CRITICA 0.10]` con SLA
`{BAJA: 72, MEDIA: 48, ALTA: 24, CRITICA: 4}` horas; canales `[WEB 0.30, APP
0.20, TELEFONO 0.35, EMAIL 0.15]`.

## Tabla booktabs del informe (criterio 1.1)

Este documento concentra los metadatos que alimentan la **tabla booktabs del
criterio 1.1** del informe LaTeX: fuente y URL de generación, licencia, fecha
de generación, registros, columnas y tamaño en disco de cada fichero.
