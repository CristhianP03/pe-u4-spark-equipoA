from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DATA_RAW = DATA / "raw"

TICKETS_CSV = DATA_RAW / "tickets.csv"
AGENTES_CSV = DATA_RAW / "agentes.csv"

# --------------------------------------------------------------------------
# Parámetros del dataset (declarados en la portada del informe, criterio 1.1)
# --------------------------------------------------------------------------
SEMILLA = 42
FILAS = 600_000
AGENTES = 300
INI = "2024-08-01"
FIN = "2026-07-31"

TICKETS_COLUMNAS = [
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
AGENTES_COLUMNAS = [
    "agente_id",
    "nombre",
    "equipo",
    "zona_operativa",
    "nivel",
    "activo",
    "fecha_ingreso",
]

# Catálogos anclados al esquema real del PFC ACC (schema.sql / seeds.sql).
CATEGORIAS = ["CONEXION", "VELOCIDAD", "EQUIPO", "FACTURACION", "INSTALACION", "OTRO"]
PESO_CATEGORIA = [0.30, 0.20, 0.15, 0.15, 0.10, 0.10]

ESTADOS = ["CREADO", "ASIGNADO", "EN_PROGRESO", "RESUELTO", "CERRADO"]
PESO_ESTADO = [0.10, 0.15, 0.15, 0.35, 0.25]

PRIORIDADES = ["BAJA", "MEDIA", "ALTA", "CRITICA"]
PESO_PRIORIDAD = [0.25, 0.40, 0.25, 0.10]
SLA_PRIORIDAD = {"BAJA": 72, "MEDIA": 48, "ALTA": 24, "CRITICA": 4}

CANALES = ["WEB", "APP", "TELEFONO", "EMAIL"]
PESO_CANAL = [0.30, 0.20, 0.35, 0.15]

ZONAS = ["NORTE", "SUR", "CENTRO", "ORIENTE", "OCCIDENTE"]

ASUNTOS = [
    "No hay conexion a Internet",
    "Velocidad muy lenta",
    "Conexion intermitente",
    "Corte de servicio",
    "No funciona el WiFi",
    "Router no enciende",
    "Solicitud de instalacion",
    "Factura incorrecta",
    "Reclamacion por cobro",
    "Cambio de plan",
    "Configuracion de email",
    "Soporte tecnico remoto",
]
PESO_ASUNTO = [0.16, 0.14, 0.12, 0.10, 0.09, 0.08, 0.08, 0.08, 0.06, 0.05, 0.02, 0.02]

# Log-media (minutos) del tiempo de resolución por prioridad: cuanto más
# crítica, más rápida. sigma = 1.0 y recorte a [5 min, 30 días].
LOG_MEDIA_RESOLUCION = {"BAJA": 8.0, "MEDIA": 6.5, "ALTA": 5.5, "CRITICA": 4.5}
SIGMA_RESOLUCION = 1.0
MIN_RESOLUCION_MIN = 5.0
MAX_RESOLUCION_MIN = 43_200.0

EQUIPOS = ["SOPORTE_L1", "SOPORTE_L2", "FIBRA", "DSL", "FACTURACION", "INSTALACION"]
PESO_EQUIPO = [0.30, 0.20, 0.15, 0.15, 0.10, 0.10]
NIVELES = ["L1", "L2", "L3"]
PESO_NIVEL = [0.50, 0.35, 0.15]
MAX_CLIENTES = 200_000
PROB_AGENTE_ACTIVO = 0.90


def generar_agentes(semilla: int, n: int) -> pd.DataFrame:
    """Genera la tabla de agentes (catálogo pequeño, 200-500 filas).

    El orden de consumo del generador es fijo para garantizar el determinismo:
    equipo -> zona_operativa -> nivel -> activo -> fecha_ingreso.
    """
    rng = np.random.default_rng(semilla)
    Faker.seed(semilla)
    fake = Faker("es_MX")

    ancho = max(3, len(str(n)))
    agentes_id = [f"AG-{i:0{ancho}d}" for i in range(1, n + 1)]
    nombres = [fake.name() for _ in range(n)]
    equipo = rng.choice(EQUIPOS, size=n, p=PESO_EQUIPO)
    zona_operativa = rng.choice(ZONAS, size=n)
    nivel = rng.choice(NIVELES, size=n, p=PESO_NIVEL)
    activo = (rng.random(n) < PROB_AGENTE_ACTIVO).astype(int)

    inicio_ingreso = pd.Timestamp("2019-01-01")
    dias = rng.integers(
        0, int((pd.Timestamp(FIN) - inicio_ingreso).days) + 1, size=n
    )
    fecha_ingreso = (inicio_ingreso + pd.to_timedelta(dias, unit="D")).strftime("%Y-%m-%d")

    return pd.DataFrame(
        {
            "agente_id": agentes_id,
            "nombre": nombres,
            "equipo": equipo,
            "zona_operativa": zona_operativa,
            "nivel": nivel,
            "activo": activo,
            "fecha_ingreso": fecha_ingreso,
        },
        columns=AGENTES_COLUMNAS,
    )


def generar_tickets(
    semilla: int, n_filas: int, n_agentes: int, ini: str, fin: str
) -> pd.DataFrame:
    """Genera la tabla de tickets (~600 000 filas) con distribuciones realistas.

    El orden de consumo del generador es fijo (determinismo):
    estado -> prioridad -> categoria -> canal -> asunto -> zona -> cliente_id
    -> offset_segundos -> minutos_resolucion -> indice_agente.
    """
    rng = np.random.default_rng(semilla)

    estado = rng.choice(ESTADOS, size=n_filas, p=PESO_ESTADO)
    prioridad = rng.choice(PRIORIDADES, size=n_filas, p=PESO_PRIORIDAD)
    categoria = rng.choice(CATEGORIAS, size=n_filas, p=PESO_CATEGORIA)
    canal = rng.choice(CANALES, size=n_filas, p=PESO_CANAL)
    asunto = rng.choice(ASUNTOS, size=n_filas, p=PESO_ASUNTO)
    zona = rng.choice(ZONAS, size=n_filas)
    cliente_id = rng.integers(1, MAX_CLIENTES + 1, size=n_filas)

    inicio = pd.Timestamp(ini)
    fin_ts = pd.Timestamp(fin + " 23:59:59")
    offset_s = rng.integers(0, int((fin_ts - inicio).total_seconds()) + 1, size=n_filas)
    creada = inicio + offset_s.astype("timedelta64[s]")

    mu = np.array([LOG_MEDIA_RESOLUCION[p] for p in prioridad])
    minutos = np.clip(rng.lognormal(mu, SIGMA_RESOLUCION), MIN_RESOLUCION_MIN, MAX_RESOLUCION_MIN)
    resuelta = creada + minutos.astype("timedelta64[m]")

    ancho = max(3, len(str(n_agentes)))
    pool_agentes = np.array([f"AG-{i:0{ancho}d}" for i in range(1, n_agentes + 1)])
    idx_agente = rng.integers(0, n_agentes, size=n_filas)
    agente_id = pool_agentes[idx_agente]

    sla_horas = np.array([SLA_PRIORIDAD[p] for p in prioridad], dtype=int)
    creada_s = pd.Series(creada)
    hora_creacion = creada_s.dt.hour
    dia_semana_creacion = creada_s.dt.day_of_week + 1
    mes_creacion = creada_s.dt.month
    fecha_creacion = creada_s.dt.strftime("%Y-%m-%d %H:%M:%S")

    mascara_resuelto = np.isin(estado, ["RESUELTO", "CERRADO"])
    fecha_resolucion = (
        pd.Series(resuelta).dt.strftime("%Y-%m-%d %H:%M:%S").where(mascara_resuelto, "")
    )
    agente_id = pd.Series(agente_id).where(estado != "CREADO", "")

    return pd.DataFrame(
        {
            "ticket_id": [f"TK-{i:08d}" for i in range(1, n_filas + 1)],
            "cliente_id": cliente_id,
            "agente_id": agente_id,
            "asunto": asunto,
            "categoria": categoria,
            "estado": estado,
            "prioridad": prioridad,
            "canal": canal,
            "zona": zona,
            "sla_horas": sla_horas,
            "fecha_creacion": fecha_creacion,
            "fecha_resolucion": fecha_resolucion,
            "hora_creacion": hora_creacion,
            "dia_semana_creacion": dia_semana_creacion,
            "mes_creacion": mes_creacion,
        },
        columns=TICKETS_COLUMNAS,
    )


def verificar(mostrar_head: bool = True) -> dict:
    """Verifica registros y esquema en pandas (criterio 1.1 Nivel 4).

    Lee ambos CSV con dtype=str y keep_default_na=False (la misma lectura que
    `carga.py`) y comprueba: número de filas, columnas, unicidad de ticket_id,
    integridad referencial de agente_id y conteo de nulos.
    """
    tickets = pd.read_csv(TICKETS_CSV, dtype=str, keep_default_na=False)
    agentes = pd.read_csv(AGENTES_CSV, dtype=str, keep_default_na=False)

    nulos_agente = int((tickets["agente_id"] == "").sum())
    nulos_resolucion = int((tickets["fecha_resolucion"] == "").sum())
    duplicados = int(tickets["ticket_id"].duplicated().sum())

    ids_agentes = set(agentes["agente_id"])
    asignados = set(tickets.loc[tickets["agente_id"] != "", "agente_id"])
    fk_faltantes = sorted(asignados - ids_agentes)

    resumen = {
        "tickets_filas": len(tickets),
        "tickets_columnas": tickets.shape[1],
        "agentes_filas": len(agentes),
        "agentes_columnas": agentes.shape[1],
        "nulos_agente_id": nulos_agente,
        "nulos_fecha_resolucion": nulos_resolucion,
        "ticket_id_duplicados": duplicados,
        "fk_agente_id_invalidos": len(fk_faltantes),
        "tamano_tickets_bytes": TICKETS_CSV.stat().st_size,
        "tamano_agentes_bytes": AGENTES_CSV.stat().st_size,
        "fecha_generacion_utc": datetime.fromtimestamp(
            TICKETS_CSV.stat().st_mtime, tz=timezone.utc
        ).isoformat(timespec="seconds"),
    }

    print("=== VERIFICACION (criterio 1.1, Nivel 4) ===")
    print(f"tickets.csv : {len(tickets):,} filas x {tickets.shape[1]} columnas "
          f"({resumen['tamano_tickets_bytes'] / 1e6:.1f} MB)")
    print(f"agentes.csv : {len(agentes):,} filas x {agentes.shape[1]} columnas "
          f"({resumen['tamano_agentes_bytes'] / 1e6:.1f} MB)")
    print(f"fecha de generacion (UTC): {resumen['fecha_generacion_utc']}")
    print(f"nulos agente_id (ticket sin asignar): {nulos_agente:,}")
    print(f"nulos fecha_resolucion (ticket abierto): {nulos_resolucion:,}")
    print(f"ticket_id duplicados: {duplicados}")
    print(f"FK agente_id no existentes en agentes.csv: {len(fk_faltantes)}")
    if fk_faltantes:
        print(f"  detalle: {fk_faltantes[:10]}")
    print(f"estado (conteos):")
    print(tickets["estado"].value_counts().to_string())
    if mostrar_head:
        print("head(tickets):")
        print(tickets.head(3).to_string(index=False))
    print("=== FIN VERIFICACION ===")
    return resumen


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generador determinista del dataset sintetico de tickets ISP (ACC)."
    )
    parser.add_argument("--filas", type=int, default=FILAS, help="Numero de tickets.")
    parser.add_argument("--agentes", type=int, default=AGENTES, help="Numero de agentes.")
    parser.add_argument("--semilla", type=int, default=SEMILLA, help="Semilla fija.")
    parser.add_argument("--ini", default=INI, help="Inicio del rango de fechas (YYYY-MM-DD).")
    parser.add_argument("--fin", default=FIN, help="Fin del rango de fechas (YYYY-MM-DD).")
    parser.add_argument("--sobrescribir", action="store_true",
                        help="Regenerar aunque los ficheros ya existan.")
    args = parser.parse_args()

    if args.filas < 1 or args.agentes < 1:
        print("ERROR: --filas y --agentes deben ser >= 1.", file=sys.stderr)
        return 2

    if TICKETS_CSV.exists() and AGENTES_CSV.exists() and not args.sobrescribir:
        print("data/raw/tickets.csv y data/raw/agentes.csv ya existen.")
        print("Usa --sobrescribir para regenerarlos.")
        verificar(mostrar_head=False)
        if args.filas < 500_000:
            print("ADVERTENCIA: el dataset tiene menos de 500 000 registros (criterio 1.1).")
            return 2
        return 0

    DATA_RAW.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    print(f"Generando {args.agentes:,} agentes (semilla {args.semilla})...", flush=True)
    agentes = generar_agentes(args.semilla, args.agentes)
    agentes.to_csv(AGENTES_CSV, index=False, encoding="utf-8")

    print(f"Generando {args.filas:,} tickets (semilla {args.semilla})...", flush=True)
    tickets = generar_tickets(args.semilla, args.filas, args.agentes, args.ini, args.fin)
    tickets.to_csv(TICKETS_CSV, index=False, encoding="utf-8")

    print(f"Generacion completada en {time.perf_counter() - t0:.1f}s -> {DATA_RAW}", flush=True)

    verificar(mostrar_head=True)

    if args.filas < 500_000:
        print("ADVERTENCIA CRITICA: menos de 500 000 registros (criterio 1.1 = 0).")
        return 2
    print("Dataset OK: >= 500 000 registros, semilla fija declarada, esquema verificado.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
