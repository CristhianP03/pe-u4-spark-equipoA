from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data"
DATA_RAW = DATA / "raw"
PANDAS_OUT = DATA / "pandas"
SPARK_OUT = DATA / "spark"
RESULTADOS = ROOT / "resultados"
FIGURAS = RESULTADOS / "figuras"
EVIDENCIA = ROOT / "evidencia"

RAW_TICKETS = DATA_RAW / "tickets.csv"
RAW_AGENTES = DATA_RAW / "agentes.csv"

PANDAS_PATHS = {
    "T1": PANDAS_OUT / "t1_filtrado.csv",
    "T2": PANDAS_OUT / "t2_agrupacion.csv",
    "T3": PANDAS_OUT / "t3_join.csv",
    "T4": PANDAS_OUT / "t4_derivada.csv",
    "T5": PANDAS_OUT / "t5_topn.csv",
}
SPARK_PATHS = {
    "T1": SPARK_OUT / "t1_filtrado",
    "T2": SPARK_OUT / "t2_agrupacion",
    "T3": SPARK_OUT / "t3_join",
    "T4": SPARK_OUT / "t4_derivada",
    "T5": SPARK_OUT / "t5_topn",
}

# Esquema de tickets (15 columnas) — idéntico a generar_dataset_acc.py.
COLUMNAS = [
    "ticket_id",
    "cliente_id",
    "agente_id",
    "asunto",
    "categoria",
    "estado",
    "prioridad",
    "canal",
    "zona",
    "sla_horas",
    "fecha_creacion",
    "fecha_resolucion",
    "hora_creacion",
    "dia_semana_creacion",
    "mes_creacion",
]
COLUMNAS_AGENTES = [
    "agente_id",
    "nombre",
    "equipo",
    "zona_operativa",
    "nivel",
    "activo",
    "fecha_ingreso",
]
COLUMNAS_NUMERICAS = ["cliente_id", "sla_horas"]
COLUMNAS_ENTERAS = ["hora_creacion", "dia_semana_creacion", "mes_creacion"]
COLUMNAS_FECHA = ["fecha_creacion", "fecha_resolucion"]
FORMATO_FECHA = "%Y-%m-%d %H:%M:%S"

ESTADOS_CERRADOS = ["RESUELTO", "CERRADO"]
CANALES_CERRADOS = ["WEB", "APP"]

LIMITE_FILAS = 600_000
SEMILLA = 42
NUCLEOS = [1, 2, 4]


def leer_crudo_pandas(ruta: Path | None = None) -> pd.DataFrame:
    """Lee tickets.csv en pandas con nulos normalizados ("" -> NaN)."""
    df = pd.read_csv(ruta or RAW_TICKETS, dtype=str, keep_default_na=False)
    df = df.replace("", np.nan)
    for c in COLUMNAS_NUMERICAS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in COLUMNAS_ENTERAS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in COLUMNAS_FECHA:
        df[c] = pd.to_datetime(df[c], errors="coerce", format=FORMATO_FECHA)
    return df


def leer_agentes_pandas(ruta: Path | None = None) -> pd.DataFrame:
    """Lee agentes.csv en pandas con nulos normalizados ("" -> NaN)."""
    df = pd.read_csv(ruta or RAW_AGENTES, dtype=str, keep_default_na=False)
    return df.replace("", np.nan)


def base_filtrada(df: pd.DataFrame) -> pd.DataFrame:
    """Base de T1/T3: tickets cerrados (RESUELTO/CERRADO) por WEB/APP con
    agente, fecha de resolución y zona no nulos."""
    mascara = (
        df["estado"].isin(ESTADOS_CERRADOS)
        & df["canal"].isin(CANALES_CERRADOS)
        & df["agente_id"].notna()
        & df["fecha_resolucion"].notna()
        & df["zona"].notna()
    )
    return df[mascara].copy()


def t1_filtrado(df: pd.DataFrame) -> pd.DataFrame:
    return base_filtrada(df)[
        ["ticket_id", "cliente_id", "agente_id", "categoria", "prioridad",
         "canal", "zona", "fecha_creacion", "fecha_resolucion"]
    ]


def t2_agrupacion(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df["categoria"].notna() & df["zona"].notna()].copy()
    g = (
        d.groupby(["categoria", "zona"], dropna=False)
        .agg(
            cnt=("ticket_id", "count"),
            sum_sla=("sla_horas", "sum"),
            avg_sla=("sla_horas", "mean"),
            max_hora=("hora_creacion", "max"),
            avg_dia=("dia_semana_creacion", "mean"),
            min_mes=("mes_creacion", "min"),
        )
        .reset_index()
    )
    return g


def t3_join(df: pd.DataFrame, agentes: pd.DataFrame) -> pd.DataFrame:
    base = base_filtrada(df)
    unido = base.merge(agentes, on="agente_id", how="inner")
    unido = unido.rename(columns={"nombre": "nombre_agente"})
    unido = unido.sort_values("ticket_id").reset_index(drop=True)
    columnas = [
        "ticket_id", "cliente_id", "agente_id", "categoria", "prioridad",
        "canal", "zona", "fecha_creacion", "nombre_agente", "equipo", "nivel",
    ]
    return unido[columnas]


def t4_derivada(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df["fecha_creacion"].notna()].copy()
    d["anio"] = d["fecha_creacion"].dt.year
    d["mes"] = d["fecha_creacion"].dt.month
    d["dia"] = d["dia_semana_creacion"]
    d["tiempo_respuesta_h"] = (
        (d["fecha_resolucion"] - d["fecha_creacion"]).dt.total_seconds() / 3600.0
    )
    dentro_sla = d["tiempo_respuesta_h"] <= d["sla_horas"]
    d["sla_cumplido"] = np.where(
        d["fecha_resolucion"].isna(), np.nan, np.where(dentro_sla, 1, 0)
    )
    d["prioridad_critica"] = (d["prioridad"] == "CRITICA").astype(int)
    d["asunto_mayus"] = d["asunto"].str.upper()
    columnas = [
        "ticket_id", "anio", "mes", "dia", "tiempo_respuesta_h",
        "sla_cumplido", "prioridad_critica", "asunto_mayus", "zona",
    ]
    return d[columnas]


def t5_topn(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df["asunto"].notna()].copy()
    conteo = d.groupby("asunto").size().rename("cnt").reset_index()
    top = conteo.sort_values(["cnt", "asunto"], ascending=[False, True]).head(10)
    return top.reset_index(drop=True)


def t3_join_pandas(df: pd.DataFrame) -> pd.DataFrame:
    return t3_join(df, leer_agentes_pandas())


TRANSFORMACIONES = {
    "T1": t1_filtrado,
    "T2": t2_agrupacion,
    "T3": t3_join_pandas,
    "T4": t4_derivada,
    "T5": t5_topn,
}


def calcular_amdahl(resumen: pd.DataFrame) -> pd.DataFrame:
    """Análisis de Amdahl de T3 a partir de `tiempos_resumen.csv` (criterio 2.2).

    Cálculo puro (sin E/S): dado el resumen de mediciones, extrae la mediana de
    T3 en pandas (línea base) y en Spark con N = 1, 2, 4 y computa

    Devuelve una fila por N con esos indicadores. `medicion.py --modo amdahl` lo
    imprime y `graficas.py` lo usa para las figuras fig2/fig3.
    """
    def mediana(t: str, motor: str, n_cores: int | None = None) -> float:
        mask = (resumen["transform"] == t) & (resumen["engine"] == motor)
        if n_cores is not None:
            mask = mask & (resumen["n_cores"] == n_cores)
        return float(resumen.loc[mask, "median_s"].iloc[0])

    t_pandas = mediana("T3", "pandas")
    t_spark = {n: mediana("T3", "spark", n) for n in NUCLEOS}
    t_spark_1 = t_spark[1]

    filas = []
    estimaciones_f: list[float] = []
    for n in NUCLEOS:
        s_vs_pandas = t_pandas / t_spark[n]
        s_interno = t_spark_1 / t_spark[n]
        f_est = None
        if n > 1:
            f_est = (1.0 / s_interno - 1.0 / n) / (1.0 - 1.0 / n)
            estimaciones_f.append(f_est)
        filas.append({
            "N": n,
            "t_pandas_s": t_pandas,
            "t_spark_s": t_spark[n],
            "speedup_vs_pandas": s_vs_pandas,
            "speedup_interno": s_interno,
            "f_serial_estimada": f_est,
            "eficiencia_E": s_interno / n,
        })

    f_media = sum(estimaciones_f) / len(estimaciones_f) if estimaciones_f else float("nan")
    p = 1.0 - f_media
    s_max = 1.0 / f_media if f_media and f_media != 0 else float("inf")

    df = pd.DataFrame(filas)
    df["f_serial_media"] = f_media
    df["p_paralela"] = p
    df["S_max_amdahl"] = s_max
    df["S_gustafson"] = df["N"].apply(lambda n: f_media + (1.0 - f_media) * n)
    return df


def main() -> int:
    df = leer_crudo_pandas()
    print(f"raw cargado: {len(df):,} filas x {df.shape[1]} columnas")
    PANDAS_OUT.mkdir(parents=True, exist_ok=True)
    for t, funcion in TRANSFORMACIONES.items():
        resultado = funcion(df)
        ruta = PANDAS_PATHS[t]
        resultado.to_csv(ruta, index=False, encoding="utf-8")
        print(f"{t}: {len(resultado):,} filas x {resultado.shape[1]} col -> {ruta.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
