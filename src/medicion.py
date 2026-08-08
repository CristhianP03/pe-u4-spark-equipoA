from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from transformaciones_pandas import (
    NUCLEOS,
    PANDAS_PATHS,
    SPARK_PATHS,
    RESULTADOS,
    TRANSFORMACIONES as TR_PANDAS,
    calcular_amdahl,
    leer_crudo_pandas,
    t2_agrupacion as t2_pandas,
)
from transformaciones_spark import (
    TRANSFORMACIONES as TR_SPARK,
    crear_spark,
    leer_crudo_spark,
    t2_agrupacion as t2_spark,
)

REPS = 5
CALENTAMIENTO = 1
N_BASE = 4
N_T3 = [1, 2, 4]
TAMANOS_UMBRAL = [10_000, 50_000, 100_000, 250_000, 500_000, 600_000]
REPS_UMBRAL = 3
TOL_EQUIVALENCIA = 1e-6


# --------------------------------------------------------------------------
# Modo: medicion
# --------------------------------------------------------------------------
def paso_pandas(t: str) -> None:
    df = leer_crudo_pandas()
    resultado = TR_PANDAS[t](df)
    resultado.to_csv(PANDAS_PATHS[t], index=False, encoding="utf-8")


def paso_spark(spark, t: str) -> None:
    df = leer_crudo_spark(spark)
    resultado = TR_SPARK[t](df)
    resultado.write.mode("overwrite").csv(str(SPARK_PATHS[t]), header=True, encoding="utf-8")


def medir(func, *args) -> list[float]:
    for _ in range(CALENTAMIENTO):
        func(*args)
    tiempos = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        func(*args)
        tiempos.append(time.perf_counter() - t0)
    return tiempos


def resumen(t: str, motor: str, n_cores, tiempos: list[float], base_pandas: dict[str, float]) -> dict:
    mediana = statistics.median(tiempos)
    media = statistics.fmean(tiempos)
    desv = statistics.stdev(tiempos) if len(tiempos) > 1 else 0.0
    speedup = None
    if motor == "spark" and t in base_pandas:
        speedup = base_pandas[t] / mediana if mediana > 0 else math.inf
    return {
        "transform": t,
        "engine": motor,
        "n_cores": n_cores,
        "median_s": mediana,
        "mean_s": media,
        "std_s": desv,
        "speedup": speedup,
    }


def modo_medicion(args) -> int:
    RESULTADOS.mkdir(parents=True, exist_ok=True)
    crudos: list[dict] = []
    resumenes: list[dict] = []

    ruta_crudos = RESULTADOS / "tiempos_crudos.csv"
    ruta_resumen = RESULTADOS / "tiempos_resumen.csv"

    def cargar_existentes():
        if ruta_crudos.exists():
            crudos.extend(pd.read_csv(ruta_crudos).to_dict("records"))
        if ruta_resumen.exists():
            resumenes.extend(pd.read_csv(ruta_resumen).to_dict("records"))

    def registrar(t, motor, n_cores, tiempos):
        for i, s in enumerate(tiempos, start=1):
            crudos.append({"transform": t, "engine": motor, "n_cores": n_cores,
                           "rep": i, "elapsed_s": s})
        resumenes.append(resumen(t, motor, n_cores, tiempos, base_pandas))

    def guardar():
        dfc = pd.DataFrame(crudos).drop_duplicates(
            ["transform", "engine", "n_cores", "rep"], keep="last")
        dfr = pd.DataFrame(resumenes).drop_duplicates(
            ["transform", "engine", "n_cores"], keep="last")
        dfc.to_csv(ruta_crudos, index=False)
        dfr.to_csv(ruta_resumen, index=False)
        print(f"-> provisional: {ruta_crudos} y {ruta_resumen}", flush=True)

    base_pandas: dict[str, float] = {}
    cargar_existentes()

    if args.solo == "spark":
        if ruta_resumen.exists():
            previo = pd.read_csv(ruta_resumen)
            pp = previo[previo["engine"] == "pandas"]
            for _, r in pp.iterrows():
                base_pandas[str(r["transform"])] = float(r["median_s"])
            print("pandas base cargado desde tiempos_resumen.csv:", base_pandas, flush=True)

    if args.solo != "spark":
        print("--- pandas ---", flush=True)
        for t in ["T1", "T2", "T3", "T4", "T5"]:
            tiempos = medir(paso_pandas, t)
            mediana = statistics.median(tiempos)
            base_pandas[t] = mediana
            registrar(t, "pandas", 0, tiempos)
            print(f"{t}: mediana={mediana:.3f}s (n={len(tiempos)})", flush=True)
        guardar()

    if args.solo != "pandas":
        print("--- spark ---", flush=True)
        spark4 = crear_spark(N_BASE, nombre="medicion-base")
        for t in ["T1", "T2", "T4", "T5"]:
            tiempos = medir(paso_spark, spark4, t)
            registrar(t, "spark", N_BASE, tiempos)
            print(f"{t} (N={N_BASE}): mediana={statistics.median(tiempos):.3f}s", flush=True)
        spark4.stop()
        guardar()

        for n in N_T3:
            spark_n = crear_spark(n, nombre="medicion-t3")
            tiempos = medir(paso_spark, spark_n, "T3")
            registrar("T3", "spark", n, tiempos)
            print(f"T3 (N={n}): mediana={statistics.median(tiempos):.3f}s", flush=True)
            spark_n.stop()
        guardar()

    print(f"-> {ruta_crudos}")
    print(f"-> {ruta_resumen}")
    return 0


# --------------------------------------------------------------------------
# Modo: equivalencia
# --------------------------------------------------------------------------
def leer_spark(directorio: Path) -> pd.DataFrame:
    """Concatena las partes part-*.csv de una salida Spark eliminando la fila
    de cabecera repetida de cada parte excepto la primera."""
    partes = sorted(p for p in directorio.glob("*.csv") if p.name != "_SUCCESS")
    if not partes:
        raise FileNotFoundError(f"No hay partes CSV en {directorio}")
    columnas = list(pd.read_csv(partes[0], nrows=0).columns)
    dfs = []
    for i, p in enumerate(partes):
        if i == 0:
            dfs.append(pd.read_csv(p))
        else:
            dfs.append(pd.read_csv(p, header=None, names=columnas, skiprows=1))
    return pd.concat(dfs, ignore_index=True)


def columnas_numericas(df_p: pd.DataFrame, df_s: pd.DataFrame) -> list[str]:
    comunes = set(df_p.columns) & set(df_s.columns)
    return [c for c in comunes if pd.api.types.is_numeric_dtype(df_p[c])]


def numeros_ok(a: pd.Series, b: pd.Series) -> bool:
    a, b = a.reset_index(drop=True), b.reset_index(drop=True)
    if len(a) != len(b):
        return False
    if a.isna().all() and b.isna().all():
        return True
    aa, bb = a.fillna(0.0).astype(float), b.fillna(0.0).astype(float)
    return bool(np.allclose(aa, bb, rtol=TOL_EQUIVALENCIA, atol=1e-9))


def comparar_transformacion(t: str) -> dict:
    df_p = pd.read_csv(PANDAS_PATHS[t])
    df_s = leer_spark(SPARK_PATHS[t])

    filas_p, filas_s = len(df_p), len(df_s)
    columnas_ok = sorted(df_p.columns) == sorted(df_s.columns)

    numericas = columnas_numericas(df_p, df_s)
    sumas_ok = True
    medias_ok = True
    for c in numericas:
        sumas_ok = sumas_ok and numeros_ok(
            pd.Series([df_p[c].sum()]), pd.Series([df_s[c].sum()])
        )
        medias_ok = medias_ok and numeros_ok(
            pd.Series([df_p[c].mean()]), pd.Series([df_s[c].mean()])
        )

    claves_ok = True
    if t == "T2":
        claves_p = set(map(tuple, df_p[["categoria", "zona"]].astype(str).itertuples(index=False)))
        claves_s = set(map(tuple, df_s[["categoria", "zona"]].astype(str).itertuples(index=False)))
        claves_ok = claves_p == claves_s
    elif t == "T5":
        claves_ok = list(df_p["asunto"]) == list(df_s["asunto"])

    cardinalidad_ok = filas_p == filas_s
    equivalente = cardinalidad_ok and columnas_ok and sumas_ok and medias_ok and claves_ok

    return {
        "transform": t,
        "filas_pandas": filas_p,
        "filas_spark": filas_s,
        "columnas_ok": columnas_ok,
        "cardinalidad_ok": cardinalidad_ok,
        "sumas_ok": sumas_ok,
        "medias_ok": medias_ok,
        "claves_ok": claves_ok,
        "equivalente": equivalente,
    }


def modo_equivalencia() -> pd.DataFrame:
    filas = []
    for t in ["T1", "T2", "T3", "T4", "T5"]:
        try:
            res = comparar_transformacion(t)
        except Exception as e:  # noqa: BLE001
            res = {"transform": t, "equivalente": False, "error": str(e)}
            print(f"{t}: ERROR {e}")
        filas.append(res)
        print(
            f"{t}: filas={res.get('filas_pandas', '?')}/{res.get('filas_spark', '?')} "
            f"equivalentes={res['equivalente']}"
        )
    df = pd.DataFrame(filas)
    print("Tabla comparativa (criterio 1.4):")
    print(df.to_string(index=False))
    return df


# --------------------------------------------------------------------------
# Modo: amdahl
# --------------------------------------------------------------------------
def modo_amdahl() -> pd.DataFrame:
    resumen = pd.read_csv(RESULTADOS / "tiempos_resumen.csv")
    df = calcular_amdahl(resumen)

    t_pandas = float(df["t_pandas_s"].iloc[0])
    t_spark_1 = float(df.loc[df["N"] == 1, "t_spark_s"].iloc[0])
    p = float(df["p_paralela"].iloc[0])
    f = float(df["f_serial_media"].iloc[0])
    s_max = float(df["S_max_amdahl"].iloc[0])

    print(f"T3 pandas (base): {t_pandas:.3f} s | T3 spark N=1: {t_spark_1:.3f} s")
    for _, r in df.iterrows():
        f_str = f"{r['f_serial_estimada']:.3f}" if pd.notna(r["f_serial_estimada"]) else "---"
        print(
            f"N={int(r['N'])}: S_vs_pandas={r['speedup_vs_pandas']:.3f}  "
            f"S_interno={r['speedup_interno']:.3f}  E={r['eficiencia_E']:.3f}  "
            f"f={f_str}"
        )
    print(f"p (fracción paralela, S_int) = {p:.4f} | S_max = {s_max:.4f}")
    print("Tabla de Amdahl (criterio 2.2):")
    print(df.to_string(index=False))
    return df


# --------------------------------------------------------------------------
# Modo: umbral
# --------------------------------------------------------------------------
def paso_umbral_pandas(tam: int) -> None:
    df = leer_crudo_pandas().iloc[:tam]
    t2_pandas(df)


def paso_umbral_spark(spark, tam: int) -> None:
    df = leer_crudo_spark(spark, n_filas=tam)
    t2_spark(df).collect()


def modo_umbral() -> pd.DataFrame:
    filas: list[dict] = []
    spark = crear_spark(4, nombre="umbral-volumen")

    for tam in TAMANOS_UMBRAL:
        t_p = medir_reps(paso_umbral_pandas, REPS_UMBRAL, tam)
        t_s = medir_reps(paso_umbral_spark, REPS_UMBRAL, spark, tam)
        mp, ms = statistics.median(t_p), statistics.median(t_s)
        filas.append({"tamano": tam, "engine": "pandas", "median_s": mp,
                      "mean_s": statistics.fmean(t_p),
                      "std_s": statistics.stdev(t_p) if len(t_p) > 1 else 0.0})
        filas.append({"tamano": tam, "engine": "spark", "median_s": ms,
                      "mean_s": statistics.fmean(t_s),
                      "std_s": statistics.stdev(t_s) if len(t_s) > 1 else 0.0})
        print(f"tam={tam:>7,}: pandas={mp:.3f}s  spark={ms:.3f}s  "
              f"{'SPARK<' if ms < mp else 'pandas<'}pandas", flush=True)

    spark.stop()

    df = pd.DataFrame(filas)
    df_p = df[df["engine"] == "pandas"].set_index("tamano")["median_s"]
    df_s = df[df["engine"] == "spark"].set_index("tamano")["median_s"]
    cruce = None
    for tam in TAMANOS_UMBRAL:
        if df_s[tam] < df_p[tam]:
            cruce = tam
            break
    df["crossover_size"] = cruce if cruce is not None else "None"
    if cruce is None:
        print("Sin punto de cruce en el rango medido (10k-600k): pandas < Spark en todos los tamaños.")
    else:
        print(f"Punto de cruce: Spark supera a pandas a partir de {cruce:,} filas.")
    print("Tabla del umbral de rentabilidad (criterio 2.4):")
    print(df.to_string(index=False))
    return df


def medir_reps(func, reps: int, *args) -> list[float]:
    for _ in range(CALENTAMIENTO):
        func(*args)
    tiempos = []
    for _ in range(reps):
        t0 = time.perf_counter()
        func(*args)
        tiempos.append(time.perf_counter() - t0)
    return tiempos


# --------------------------------------------------------------------------
# Modo: versiones
# --------------------------------------------------------------------------
def version_java() -> str:
    try:
        out = subprocess.run(["java", "-version"], capture_output=True, text=True, timeout=30)
        texto = (out.stderr or out.stdout).splitlines()
        return texto[0].strip() if texto else "no disponible"
    except Exception as e:  # noqa: BLE001
        return f"no disponible ({e})"


def modo_versiones() -> dict:
    from importlib import metadata

    def _ver(nombre: str) -> str:
        try:
            return metadata.version(nombre)
        except Exception as e:  # noqa: BLE001
            return f"no instalado ({e.__class__.__name__})"

    motor_spark = None
    try:
        from pyspark.sql import SparkSession

        spark = SparkSession.builder.master("local[1]").appName("versiones").getOrCreate()
        motor_spark = spark.version
        spark.stop()
    except Exception as e:  # noqa: BLE001
        motor_spark = f"no disponible ({e.__class__.__name__})"

    datos = {
        "python": sys.version.split()[0],
        "pyspark_paquete": _ver("pyspark"),
        "motor_spark": motor_spark,
        "pandas": _ver("pandas"),
        "numpy": _ver("numpy"),
        "matplotlib": _ver("matplotlib"),
        "faker": _ver("Faker"),
        "plataforma": platform.platform(),
        "sistema": platform.system(),
        "java": version_java(),
        "fecha_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    print(json.dumps(datos, ensure_ascii=False, indent=2))
    print("Versiones del entorno (criterio 1.4): se transcriben al informe.")
    return datos


# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Protocolo de medición y análisis.")
    parser.add_argument("--modo", choices=["medicion", "equivalencia", "amdahl",
                                           "umbral", "versiones"],
                        default="medicion", help="Modo a ejecutar (default: medicion).")
    parser.add_argument("--reps", type=int, default=REPS,
                        help="Repeticiones cronometradas (solo --modo medicion).")
    parser.add_argument("--solo", choices=["pandas", "spark"],
                        help="Medir solo un motor (solo --modo medicion).")
    args = parser.parse_args()

    if args.modo == "medicion":
        return modo_medicion(args)
    if args.modo == "equivalencia":
        df = modo_equivalencia()
        return 0 if bool(df["equivalente"].all()) else 1
    if args.modo == "amdahl":
        modo_amdahl()
        return 0
    if args.modo == "umbral":
        modo_umbral()
        return 0
    if args.modo == "versiones":
        modo_versiones()
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
