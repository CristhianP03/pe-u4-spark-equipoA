from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from transformaciones_pandas import FIGURAS, RESULTADOS, calcular_amdahl

P_VALS_TEORICAS = [0.5, 0.75, 0.9, 0.95]
N_RANGO = np.arange(1, 33)


def amdahl(n: np.ndarray, p: float) -> np.ndarray:
    return 1.0 / ((1.0 - p) + p / n)


def fig0_teorico() -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for p in P_VALS_TEORICAS:
        ax.plot(N_RANGO, amdahl(N_RANGO, p), marker="o", markersize=2.5,
                label=f"p = {p}")
    ax.plot(N_RANGO, N_RANGO, "--", color="grey", lw=1, label="Ideal: S = N")
    ax.set_xlabel("N (número de núcleos / executors)")
    ax.set_ylabel("Speedup  S(N)")
    ax.set_title("Ley de Amdahl: curvas teóricas")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, None)
    fig.tight_layout()
    fig.savefig(FIGURAS / "fig0_amdahl_teorico.png", dpi=300)
    plt.close(fig)


def fig1_barras() -> bool:
    ruta = RESULTADOS / "tiempos_resumen.csv"
    if not ruta.exists():
        print("fig1 omitida: falta tiempos_resumen.csv")
        return False
    resumen = pd.read_csv(ruta)
    t_p = resumen.loc[resumen["engine"] == "pandas"].set_index("transform")["median_s"]
    t_s = resumen.loc[(resumen["engine"] == "spark") & (resumen["n_cores"] == 4)].set_index("transform")["median_s"]
    transformaciones = ["T1", "T2", "T3", "T4", "T5"]
    if len(t_s) == 0:
        print("fig1 omitida: no hay mediciones Spark N=4 todavía")
        return False

    x = np.arange(len(transformaciones))
    ancho = 0.35
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - ancho / 2, [t_p[t] for t in transformaciones], ancho, label="pandas")
    ax.bar(x + ancho / 2, [t_s[t] for t in transformaciones], ancho, label="Spark (N=4)")

    valores = list(t_p[t] for t in transformaciones) + list(t_s[t] for t in transformaciones)
    if max(valores) / max(min(valores), 1e-9) > 20:
        ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(transformaciones)
    ax.set_xlabel("Transformación")
    ax.set_ylabel("Tiempo mediano (s)")
    ax.set_title("Tiempo mediano por transformación (5 repeticiones, mediana)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURAS / "fig1_barras.png", dpi=300)
    plt.close(fig)
    return True


def fig2_speedup() -> bool:
    ruta = RESULTADOS / "tiempos_resumen.csv"
    if not ruta.exists():
        print("fig2 omitida: falta tiempos_resumen.csv")
        return False
    resumen = pd.read_csv(ruta)
    t3 = resumen[resumen["transform"] == "T3"]
    if not ((t3["engine"] == "spark") & (t3["n_cores"].isin([1, 2, 4]))).any():
        print("fig2 omitida: faltan mediciones Spark de T3 con N=1,2,4")
        return False
    amd = calcular_amdahl(resumen)
    n = amd["N"].to_numpy()
    s = amd["speedup_interno"].to_numpy()
    p = float(amd["p_paralela"].iloc[0])
    f = float(amd["f_serial_media"].iloc[0])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(n, s, "o-", color="tab:red", label="Experimental S(N) = t(N=1)/t(N)")
    ax.plot([1, max(n)], [1, 1], ":", color="grey", label="S = 1 (sin mejora)")

    if 0.0 < p < 1.0:
        n_teo = np.linspace(1, max(n), 200)
        ax.plot(n_teo, amdahl(n_teo, p), "-", color="tab:blue",
                label=f"Amdahl teórica con p = {p:.3f}")
    else:
        ax.text(
            0.5, 0.9, f"p = {p:.3f} (f = {f:.3f}): régimen no capturado por Amdahl\n"
            "(overhead o superlinealidad)",
            transform=ax.transAxes, ha="center", fontsize=9,
            bbox=dict(boxstyle="round", fc="lightyellow"),
        )
    ax.set_xticks([1, 2, 4])
    ax.set_xlabel("N (executors simulados)")
    ax.set_ylabel("Speedup interno  S(N) = t_spark(N=1) / t_spark(N)")
    ax.set_title("Speedup de T3 vs. Ley de Amdahl (escalado interno del motor)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURAS / "fig2_speedup.png", dpi=300)
    plt.close(fig)
    return True


def fig3_eficiencia() -> bool:
    ruta = RESULTADOS / "tiempos_resumen.csv"
    if not ruta.exists():
        print("fig3 omitida: falta tiempos_resumen.csv")
        return False
    resumen = pd.read_csv(ruta)
    t3 = resumen[resumen["transform"] == "T3"]
    if not ((t3["engine"] == "spark") & (t3["n_cores"].isin([1, 2, 4]))).any():
        print("fig3 omitida: faltan mediciones Spark de T3 con N=1,2,4")
        return False
    amd = calcular_amdahl(resumen)
    n = amd["N"].to_numpy()
    e = amd["eficiencia_E"].to_numpy()

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(n.astype(str), e, color="tab:green", label="E(N) = S_int(N)/N (T3)")
    ax.plot([0, len(n) - 1], [1, 1], "--", color="grey", label="Ideal: E = 1")
    ax.set_xlabel("N (executors simulados)")
    ax.set_ylabel("Eficiencia  E(N)")
    ax.set_title("Eficiencia del escalado de T3")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, max(1.0, e.max()) * 1.15)
    fig.tight_layout()
    fig.savefig(FIGURAS / "fig3_eficiencia.png", dpi=300)
    plt.close(fig)
    return True

def main() -> int:
    FIGURAS.mkdir(parents=True, exist_ok=True)
    fig0_teorico()
    fig1_barras()
    fig2_speedup()
    fig3_eficiencia()
    for f in sorted(FIGURAS.glob("fig*.png")):
        print(f"{f.name}: {f.stat().st_size} bytes")
    print("Figuras generadas en resultados/figuras/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
