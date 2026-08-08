from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:  # importación perezosa: los modos pandas funcionan sin PySpark
    from pyspark.sql import functions as F
    from pyspark.sql.types import StringType, StructField, StructType
except ImportError:  # pragma: no cover
    F = None
    StringType = StructField = StructType = None

from transformaciones_pandas import (
    CANALES_CERRADOS,
    COLUMNAS,
    COLUMNAS_AGENTES,
    COLUMNAS_ENTERAS,
    COLUMNAS_FECHA,
    COLUMNAS_NUMERICAS,
    ESTADOS_CERRADOS,
    FORMATO_FECHA,
    RAW_AGENTES,
    RAW_TICKETS,
    SPARK_OUT,
    SPARK_PATHS,
)

FORMATO_FECHA_SPARK = FORMATO_FECHA.replace("%Y", "yyyy").replace("%m", "MM") \
    .replace("%d", "dd").replace("%H", "HH").replace("%M", "mm").replace("%S", "ss")


def base_filtrada(df):
    """Espejo de transformaciones_pandas.base_filtrada para PySpark: tickets
    RESUELTO/CERRADO por WEB/APP con agente, fecha de resolución y zona no
    nulos ("" -> NULL)."""
    return df.filter(
        df["estado"].isin(ESTADOS_CERRADOS)
        & df["canal"].isin(CANALES_CERRADOS)
        & df["agente_id"].isNotNull()
        & df["fecha_resolucion"].isNotNull()
        & df["zona"].isNotNull()
    )


def crear_spark(n_cores: int, nombre: str = "pe-u4-spark-acc", log_nivel: str = "WARN"):
    """Crea una SparkSession local[N] con la configuración fija del experimento."""
    from pyspark.sql import SparkSession

    n = max(1, int(n_cores))
    builder = (
        SparkSession.builder.master(f"local[{n}]")
        .appName(f"{nombre} (N={n})")
        .config("spark.executor.instances", "4")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.sql.autoBroadcastJoinThreshold", "-1")
        .config("spark.sql.shuffle.partitions", str(max(4 * n, 8)))
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.enabled", "true")
        .config("spark.ui.showConsoleProgress", "false")
    )
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel(log_nivel)
    return spark


def leer_crudo_spark(spark, ruta: Path | None = None, n_filas: int | None = None):
    """Lee tickets.csv en PySpark con nulos normalizados ("" -> NULL) y
    esquema fijado explícitamente (ninguna columna se infiere distinta)."""
    ruta = ruta or RAW_TICKETS
    esquema = StructType([StructField(c, StringType(), True) for c in COLUMNAS])

    df = (
        spark.read.format("csv")
        .option("header", True)
        .option("mode", "PERMISSIVE")
        .option("encoding", "UTF-8")
        .schema(esquema)
        .load(str(ruta))
    )

    nulos = [F.when(F.col(c) != "", F.col(c)).alias(c) for c in COLUMNAS]
    df = df.select(*nulos)

    for c in COLUMNAS_NUMERICAS:
        df = df.withColumn(c, F.col(c).cast("double"))
    for c in COLUMNAS_ENTERAS:
        df = df.withColumn(c, F.col(c).cast("int"))
    for c in COLUMNAS_FECHA:
        df = df.withColumn(c, F.to_timestamp(c, FORMATO_FECHA_SPARK))

    if n_filas is not None:
        df = df.limit(n_filas)
    return df


def leer_agentes_spark(spark, ruta: Path | None = None):
    """Lee agentes.csv en PySpark con nulos normalizados ("" -> NULL)."""
    ruta = ruta or RAW_AGENTES
    esquema = StructType([StructField(c, StringType(), True) for c in COLUMNAS_AGENTES])
    df = (
        spark.read.format("csv")
        .option("header", True)
        .option("mode", "PERMISSIVE")
        .option("encoding", "UTF-8")
        .schema(esquema)
        .load(str(ruta))
    )
    nulos = [F.when(F.col(c) != "", F.col(c)).alias(c) for c in COLUMNAS_AGENTES]
    return df.select(*nulos)


def t1_filtrado(df):
    return base_filtrada(df).select(
        "ticket_id", "cliente_id", "agente_id", "categoria", "prioridad",
        "canal", "zona", "fecha_creacion", "fecha_resolucion",
    )


def t2_agrupacion(df):
    d = df.filter(F.col("categoria").isNotNull() & F.col("zona").isNotNull())
    return (
        d.groupBy("categoria", "zona")
        .agg(
            F.count("ticket_id").alias("cnt"),
            F.sum("sla_horas").alias("sum_sla"),
            F.avg("sla_horas").alias("avg_sla"),
            F.max("hora_creacion").alias("max_hora"),
            F.avg("dia_semana_creacion").alias("avg_dia"),
            F.min("mes_creacion").alias("min_mes"),
        )
        .orderBy("categoria", "zona")
    )


def t3_join(df, agentes):
    base = base_filtrada(df)
    unido = base.join(agentes, on="agente_id", how="inner")
    unido = unido.withColumnRenamed("nombre", "nombre_agente")
    return unido.orderBy("ticket_id").select(
        "ticket_id", "cliente_id", "agente_id", "categoria", "prioridad",
        "canal", "zona", "fecha_creacion", "nombre_agente", "equipo", "nivel",
    )


def t4_derivada(df):
    d = df.filter(F.col("fecha_creacion").isNotNull())
    d = (
        d
        .withColumn("anio", F.year("fecha_creacion"))
        .withColumn("mes", F.month("fecha_creacion"))
        .withColumn("dia", F.col("dia_semana_creacion"))
        .withColumn(
            "tiempo_respuesta_h",
            (F.unix_timestamp("fecha_resolucion", FORMATO_FECHA_SPARK)
             - F.unix_timestamp("fecha_creacion", FORMATO_FECHA_SPARK)) / 3600.0,
        )
        .withColumn(
            "sla_cumplido",
            F.when(F.col("fecha_resolucion").isNull(), F.lit(None).cast("double"))
            .otherwise(
                F.when(F.col("tiempo_respuesta_h") <= F.col("sla_horas"), F.lit(1.0))
                .otherwise(F.lit(0.0))
            ),
        )
        .withColumn(
            "prioridad_critica",
            F.when(F.col("prioridad") == "CRITICA", F.lit(1.0)).otherwise(F.lit(0.0)),
        )
        .withColumn("asunto_mayus", F.upper("asunto"))
    )
    return d.select(
        "ticket_id", "anio", "mes", "dia", "tiempo_respuesta_h",
        "sla_cumplido", "prioridad_critica", "asunto_mayus", "zona",
    )


def t5_topn(df):
    d = df.filter(F.col("asunto").isNotNull())
    return (
        d.groupBy("asunto")
        .count()
        .withColumnRenamed("count", "cnt")
        .orderBy(F.col("cnt").desc(), F.col("asunto").asc())
        .limit(10)
    )


def t3_join_spark(df):
    return t3_join(df, leer_agentes_spark(df.sparkSession))


TRANSFORMACIONES = {
    "T1": t1_filtrado,
    "T2": t2_agrupacion,
    "T3": t3_join_spark,
    "T4": t4_derivada,
    "T5": t5_topn,
}
DEFAULT_N = 4


def persistir(df, t: str, n: int) -> None:
    ruta = SPARK_PATHS[t]
    df.write.mode("overwrite").csv(str(ruta), header=True, encoding="utf-8")
    print(f"{t} (N={n}): {df.count():,} filas materializadas -> {ruta}")


def ejecutar_transformacion(spark, t: str, n: int) -> None:
    crudo = leer_crudo_spark(spark)
    resultado = TRANSFORMACIONES[t](crudo)
    persistir(resultado, t, n)


def main() -> int:
    parser = argparse.ArgumentParser(description="Transformaciones T1-T5 en PySpark.")
    parser.add_argument("--solo", choices=list(TRANSFORMACIONES),
                        help="Ejecutar una única transformación.")
    parser.add_argument("--n", type=int, default=DEFAULT_N,
                        help="Núcleos/executors (local[N]).")
    parser.add_argument("--consecutivos", nargs="+", type=int, default=None,
                        help="Núcleos a probar en T3 (por defecto 1 2 4).")
    args = parser.parse_args()

    SPARK_OUT.mkdir(parents=True, exist_ok=True)

    if args.solo:
        spark = crear_spark(args.n, nombre=f"transformacion-{args.solo}")
        ejecutar_transformacion(spark, args.solo, args.n)
        spark.stop()
        return 0

    consecutivos = args.consecutivos or [1, 2, 4]

    spark4 = crear_spark(DEFAULT_N, nombre="transformaciones-base")
    for t in ["T1", "T2", "T4", "T5"]:
        ejecutar_transformacion(spark4, t, DEFAULT_N)
    spark4.stop()

    for n in consecutivos:
        spark_n = crear_spark(n, nombre="t3-escalado")
        ejecutar_transformacion(spark_n, "T3", n)
        spark_n.stop()

    print("Transformaciones Spark completadas (T3 escalado:", consecutivos, ").")
    return 0


if __name__ == "__main__":
    sys.exit(main())
