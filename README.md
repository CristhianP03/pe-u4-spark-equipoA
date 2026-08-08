# PE-U4: pandas vs PySpark — análisis de escalabilidad y Ley de Amdahl (ACC)

Proyecto de práctica para la unidad de **Big Data / Procesamiento distribuido**:
implementación y comparación de **5 transformaciones (T1–T5)** sobre un dataset
sintético de **600 000 tickets de soporte técnico ISP** (dominio ACC) en
**pandas** y **PySpark 3.5.3**, con protocolo de medición riguroso (1
calentamiento + 5 repeticiones, mediana, `time.perf_counter`), escalado de T3
con **N = 1, 2 y 4 executors**, **análisis de Amdahl**, figuras a 300 DPI
generadas por código y **notebook ejecutado en Google Colab**.

## PFC de referencia
- **Código:** ACC
- **Título del sistema:** Sistema de Gestión de Soporte Técnico ISP

## Integrantes

| Nombre                            | PFC de origen | Rol técnico principal                                          | Aporte al documento LaTeX                                             |
|-----------------------------------|---------------|----------------------------------------------------------------|-----------------------------------------------------------------------|
| Cristhian Daniel Pacheco Cárdenas | ACC           | Implementación PySpark, protocolo de medición y escalado de T3 | Redacción de procedimiento, resultados y protocolo de medición        |
| Robinson Rodrigo Cando Moreno     | ACC           | Análisis de Amdahl, equivalencia y umbral de rentabilidad      | Redacción de fundamento teórico y análisis cuantitativo               |
| Ernesto Gregory Luna Mora         | BCEL          | Repositorio, reproducibilidad, figuras y verificación final    | Redacción de estructura, bibliografía, compilación y revisión general |

> El equipo se conformó de manera independiente a los equipos de PFC del
> período, conforme lo permite la guía de la asignatura. Aunque Ernesto
> proviene del PFC BCEL, el equipo eligió en consenso ACC — Soporte Técnico
> ISP como dominio de referencia único para esta práctica.


## Dataset

Dataset **sintético** de dos tablas relacionadas, generado de forma
determinista (semilla fija `42`) por `src/generar_dataset_acc.py`:

| Tabla | Filas | Columnas | Tamaño |
|---|---|---|---|
| `data/raw/tickets.csv` | 600 000 | 15 | 75,8 MB |
| `data/raw/agentes.csv` | 300 | 7 | 19 KB |

- **Fuente:** datos sintéticos declarados explícitamente (la guía lo permite):
  no existe un dataset real público de tickets ISP con ≥ 500 000 registros.
- El esquema está **anclado al PFC ACC** (`docs/db/schema.sql` y
  `seed-generator` de `PFC-Soporte-ISP`): categorías, estados, prioridades con
  SLA, canales y zonas.
- **Semántica de nulos:** los huecos se serializan como cadena vacía `""`
  (ticket sin agente asignado; ticket abierto sin resolución) y ambos motores
  la convierten en nulo (`NaN` / `NULL`) — criterio único que garantiza la
  equivalencia (1.4).
- **Determinismo:** con los mismos argumentos el dataset es bit a bit
  idéntico (numpy + Faker con semilla fija y rango de fechas fijo).
- Regenerable: `python src/generar_dataset_acc.py --sobrescribir`.
- Metadatos y tipos: `data/README_dataset.md`.

> `data/raw/` **no se versiona** (regenerable, 75,8 MB). En cambio, las **10
> salidas del pipeline** `data/pandas/*.csv` y `data/spark/*.csv` **sí se
> versionan** como evidencia de los criterios 1.2/1.3/1.4 (generadas por el
> notebook ejecutado en Colab).

### Límite de tamaño y decisión de versionado (criterio 1.1)

`tickets.csv` pesa **75,8 MB**, por debajo del **límite práctico de 100 MB**
que fija la guía para repositorios. Aun así, **`data/raw/` no se versiona**
en el repositorio por decisión explícita: al ser un dataset **sintético y
determinista** (semilla fija `42`), se regenera bit a bit con
`python src/generar_dataset_acc.py --sobrescribir`; el repositorio versiona en
su lugar los **metadatos** (`data/README_dataset.md`).
La tabla booktabs del informe (criterio 1.1) documenta fuente, URL de
generación, licencia, fecha, registros, columnas y tamaño a partir de
`data/README_dataset.md`.

### Persistencia del dataset (criterio 1.1 / ejecución en Colab)

El dataset no se almacena en el repositorio; **viaja dentro de la carpeta del
proyecto que se sube a Google Drive** (`data/raw/`). Al abrir el notebook desde
Drive, la celda de detección (sección 1) localiza el proyecto y `data/raw/`
queda disponible en la sesión sin copiar nada a mano.

Alternativas si solo se quiere regenerar o probar en local:
1. **Regeneración in situ:** `python src/generar_dataset_acc.py` dentro de
   Colab (requiere reinstalar dependencias).
2. **Carga manual:** subir los dos CSV arrastrándolos al entorno de Colab.

## Entorno (versiones fijadas)

| Componente  | Versión                |
|-------------|------------------------|
| Python      | 3.12 (requerido por PySpark 3.5.3) |
| PySpark     | 3.5.3                  |
| Motor Spark | 3.5.3 (Java 17.0.19)    |
| pandas      | 2.2.3                  |
| matplotlib  | 3.9.2                  |
| numpy       | 2.5.1                  |
| faker       | 40.36.0                |

`python src/medicion.py --modo versiones` imprime las versiones reales de la
ejecución (la tabla se transcribe al informe, criterio 1.4).

**Plataforma: Google Colab** (requisito del checklist). Los scripts de PySpark
se ejecutan en Colab/Linux; los modos exclusivamente pandas (`--solo pandas`,
`versiones`, `graficas` → fig0) funcionan también en Windows.

## Estructura

```
Soporte-Tecnico-ACC/
├── README.md · LICENSE · .gitignore · requirements.txt
├── notebooks/
│   ├── PE_U4_pipeline_spark.ipynb   # ejecutado en Colab (celdas con salida)
│   └── PE_U4_pipeline_spark.html    # generado desde el notebook ejecutado
├── src/
│   ├── generar_dataset_acc.py         # dataset sintético determinista
│   ├── transformaciones_pandas.py     # carga común + T1-T5 pandas
│   ├── transformaciones_spark.py      # sesión Spark + T1-T5 PySpark
│   ├── medicion.py                    # medición + equivalencia + amdahl + umbral + versiones
│   └── graficas.py                    # 4 figuras PNG a 300 DPI
├── data/
│   ├── README_dataset.md              # metadatos del dataset (criterio 1.1)
│   ├── raw/                           # tickets.csv + agentes.csv (NO versionado)
│   ├── pandas/                        # salidas T1-T5 pandas (versionado)
│   └── spark/                         # salidas T1-T5 PySpark, CSV únicos (versionado)
├── resultados/
│   ├── tiempos_crudos.csv · tiempos_resumen.csv
│   └── figuras/                       # fig0…fig3_*.png (300 DPI)
├── evidencia/
│   └── spark_ui_t3.png                # captura de la Spark UI (Colab)
└── docs/
    ├── PE_U4_Informe.tex · PE_U4_Informe.pdf
    └── references_U4.bib              # informe LaTeX
```

## Reproducción (Google Colab)

1. **Sube el proyecto.** Arrastra la carpeta `Soporte-Tecnico-ACC` **dentro de
   la carpeta `GA-SUM-05`** de tu Google Drive (`drive.google.com`) o, si
   prefieres, al panel de archivos de Colab (icono de carpeta). El notebook
   localiza el proyecto solo (sección 1); no hay que descomprimir nada.
2. Abre `notebooks/PE_U4_pipeline_spark.ipynb` en Colab (`Archivo → Abrir
   notebook → Google Drive`) y, si se pide, concede acceso a Drive.
3. `Entorno de ejecución → Ejecutar todo` (CPU estándar): instalación
   (PySpark 3.5.3 + Java 17 + `requirements.txt`) → sesión Spark → T1–T5 en
   ambos motores → equivalencia → medición (~30 min) → Amdahl → umbral →
   versiones → Parquet → empaquetado. Total ≈ 35–40 min.
4. **Captura de la Spark UI** (criterio 1.5): durante la medición (sección 12
   del notebook) abre `Herramientas → Puertos locales → 4040` en Colab y
   guarda la pestaña SQL/Stages como `evidencia/spark_ui_t3.png`.
5. **Descarga los resultados:** la celda de empaquetado (sección 15) descarga
   `resultados_pe_u4.zip` (resultados + evidencia + tablas) y
   `tablas_items_4_12_13_20_23.txt`; la celda final (sección 15b) consolida y
   **verifica los 10 CSV** (T1–T5 en pandas y Spark) y los descarga en
   `salidas_pandas.zip` / `salidas_spark.zip`. Además descarga el notebook
   ejecutado con `Archivo → Descargar → Descargar .ipynb`.
6. **Trae los archivos al repositorio:** descomprime el zip en la raíz, coloca
   el `.ipynb` ejecutado en `notebooks/PE_U4_pipeline_spark.ipynb`, la captura
   en `evidencia/` y los CSV de `salidas_pandas.zip` / `salidas_spark.zip` en
   `data/pandas/` y `data/spark/`. Con eso se regeneran las figuras
   (`python src/graficas.py`), se genera el `.html` desde el notebook ejecutado
   y se redacta el informe LaTeX en local.

> **Importante (medición):** los tiempos definitivos de
> `tiempos_crudos.csv`/`tiempos_resumen.csv` deben generarse **íntegros en
> Colab** (mismo entorno). Si se parte de un proyecto ya medido localmente,
> borra esos dos CSV antes de medir en Colab para no mezclar entornos.

### Pasos individuales (Colab)

```bash
python src/medicion.py --modo medicion     # protocolo (pandas + Spark + T3 N=1,2,4)
python src/medicion.py --modo equivalencia # equivalencia numérica pandas <-> Spark
python src/medicion.py --modo amdahl       # Amdahl + Gustafson (T3)
python src/medicion.py --modo umbral       # umbral de rentabilidad 10k-600k
python src/medicion.py --modo versiones    # versiones reales del entorno
python src/graficas.py                     # figuras 300 DPI
```

## Transformaciones

| Id | Operación | Descripción |
|---|---|---|
| T1 | Filtrado | Tickets `RESUELTO`/`CERRADO` por `WEB`/`APP` con agente, resolución y zona no nulos |
| T2 | Agrupación | 6 agregaciones (cnt, sum/avg SLA, max hora, avg día, min mes) por `(categoria, zona)` |
| T3 | Join | Inner `tickets` ⨝ `agentes` por `agente_id` — sort-merge (broadcast desactivado), shuffle real |
| T4 | Derivadas | `anio/mes/dia`, tiempo de respuesta (h), SLA cumplido, prioridad crítica, asunto en mayúsculas |
| T5 | Top-N | Los 10 asuntos con más tickets |

## Protocolo de medición (criterio 2.1)

- 1 ejecución de **calentamiento** descartada por (transformación, motor).
- **5 repeticiones** cronometradas con `time.perf_counter()`.
- Estadístico reportado: **mediana**.
- Se mide el ciclo completo **lectura + limpieza + transformación +
  persistencia** (`to_csv` / `write.csv`); en Spark la escritura fuerza la
  **materialización del DAG**.
- El arranque de la SparkSession **no** se cronometra (overhead único).
- pandas es la línea base; T3 en Spark se mide con **N = 1, 2 y 4** executors.

## Resultados

- `tiempos_crudos.csv` — las 5 repeticiones sin promediar por (transformación,
  motor) (criterio 2.1, Listado 1).
- `tiempos_resumen.csv` — medianas y speedups por (transformación, motor).
- `figuras/fig0…fig3_*.png` — 4 figuras a 300 DPI generadas por código
  (criterio 2.3).
- Las **tablas derivadas** (equivalencia pandas ↔ Spark, Amdahl/Gustafson,
  umbral de rentabilidad y versiones del entorno) se imprimen en el notebook y
  se transcriben al **informe LaTeX** (criterios 1.4, 2.2, 2.4); no se persisten
  como ficheros intermedios. La última celda del notebook las vuelca además en
  `tablas_items_4_12_13_20_23.txt` (con metadatos reales del ítem 4) y empaqueta
  `resultados_pe_u4.zip` para descargar.
- `evidencia/spark_ui_t3.png` — captura de la Spark UI (criterio 1.5).

## Notas

- **Informe LaTeX diferido:** `docs/PE_U4_Informe.tex` se completará
  posteriormente (secciones D3–D6 y portada con PFC de ≥ 100 palabras). El
  checklist de verificación final se entregará junto con el informe.
- **Compilación del informe:** la ruta de compilación (`pdflatex + bibtex` o
  `latexmk -pdf`) y la instrucción de compilación en **≤ 15 minutos** se
  documentarán junto con `docs/PE_U4_Informe.tex` (criterio 3.4).
- La Spark UI se captura con **Local port forwarding** de Colab (menú
  *Herramientas → Puertos locales*, puerto 4040; ver notebook, sección 12).
  `ngrok` queda como alternativa (requiere cuenta gratuita).

## Declaración de uso de IA generativa

Se documentará en la sección final del informe LaTeX (herramienta, propósito y
secciones asistidas).
