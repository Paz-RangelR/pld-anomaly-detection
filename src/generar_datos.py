"""
Generador de datos sintéticos de pagos de arrendamiento.

Todo lo que sale de aquí es inventado. No hay ningún registro real, ni
anonimizado, detrás de este archivo: los clientes, los montos, las fechas y
las contrapartes se construyen con un generador determinista.

La idea central del proyecto está en la forma de estos datos: **cada cliente
tiene su propia escala y su propio ritmo**. Un arrendatario que paga $18,000
puntual el día 3 y otro que paga $420,000 en dos partes a mitad de mes son
los dos normales. Un detector con un umbral global se equivoca con los dos.
Por eso la normalidad se aprende por cliente.

Sobre la serie normal se inyectan cinco tipos de anomalía con etiqueta oculta
(`_es_anomalia`, `_tipo_anomalia`). Esas columnas existen únicamente para
poder evaluar: el modelo nunca las ve, y en producción no existirían.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Parámetros del mundo sintético
# --------------------------------------------------------------------------

SEMILLA = 20260923

N_CLIENTES = 800
N_MESES = 24
MES_INICIAL = "2024-01"

# Umbral de aviso expresado en UMA, como lo fija la LFPIORPI para el
# arrendamiento. El valor de la UMA es ilustrativo: un despliegue real lee la
# UMA vigente publicada por el INEGI en lugar de fijarla en el código.
UMA_ILUSTRATIVA = 113.14
UMBRAL_UMA = 1_605
UMBRAL_AVISO = UMA_ILUSTRATIVA * UMBRAL_UMA  # ~ $181,589 MXN

# Empresas del grupo (ficticias). El modelo no las usa; sirven para que la
# cola de revisión se parezca a lo que vería cumplimiento.
EMPRESAS = [
    "AAI", "ADP", "APE", "AYP", "IBS", "ILP",
    "EDC", "EDP", "BSM", "DIA", "OCI", "PPO",
]

# Giros de arrendatario (ficticios). También son solo de presentación.
GIROS = [
    "Café Bonanza", "Farmacia Lucero", "Banco del Istmo", "Gimnasio Vértice",
    "Telecom Sierra", "Burger Nogal", "Óptica Clara", "Tiendas Almendro",
    "Sushi Kaiso", "Papelería Nova", "Academia Lincoln", "Viajes Orión",
]

# Estacionalidad del arrendamiento comercial: diciembre y julio pesan más,
# enero y febrero menos. Índice por mes del año (1..12).
ESTACIONALIDAD = {
    1: 0.94, 2: 0.96, 3: 1.00, 4: 1.00, 5: 1.02, 6: 1.03,
    7: 1.06, 8: 1.02, 9: 1.00, 10: 1.02, 11: 1.05, 12: 1.12,
}

# Cuántos clientes reciben cada tipo de anomalía.
INYECCIONES = {
    "fraccionamiento": 45,
    "salto_de_escala": 40,
    "reactivacion": 35,
    "rafaga_montos_redondos": 40,
    "contraparte_inusual": 50,
}


@dataclass
class PerfilCliente:
    """La 'personalidad' de pago de un cliente: su escala y su ritmo."""

    cliente_id: int
    renta_base: float
    dia_habitual: int
    dispersion_dias: float
    variacion_monto: float
    parte_en_dos: bool
    prob_mes_sin_pago: float
    contraparte: str
    empresa: str
    arrendatario: str


def _perfiles(rng: np.random.Generator, n_clientes: int) -> list[PerfilCliente]:
    """Construye los perfiles de cliente.

    La escala es lognormal a propósito: en un portafolio de arrendamiento
    conviven locales de plaza y edificios completos, y la distribución de
    rentas es de cola larga, no normal.
    """
    perfiles = []
    for i in range(n_clientes):
        renta_base = float(np.exp(rng.normal(10.6, 0.85)))
        renta_base = float(np.clip(renta_base, 6_000, 900_000))
        perfiles.append(
            PerfilCliente(
                cliente_id=4000 + i,
                renta_base=renta_base,
                dia_habitual=int(rng.integers(1, 26)),
                dispersion_dias=float(rng.gamma(2.0, 1.4)),
                variacion_monto=float(rng.uniform(0.02, 0.10)),
                parte_en_dos=bool(rng.random() < 0.18),
                prob_mes_sin_pago=float(rng.uniform(0.0, 0.06)),
                contraparte=f"BI-{rng.integers(100, 999)}",
                empresa=str(rng.choice(EMPRESAS)),
                arrendatario=f"{rng.choice(GIROS)} {rng.integers(1, 400):03d}",
            )
        )
    return perfiles


def _serie_normal(perfiles, meses, rng) -> list[dict]:
    """La parte aburrida y necesaria: 24 meses de pagos que se ven normales."""
    filas = []
    for p in perfiles:
        for k, mes in enumerate(meses):
            if rng.random() < p.prob_mes_sin_pago:
                continue  # mes sin pago: vacancia, prórroga, lo que sea

            # La renta crece con el tiempo (indexación anual aproximada) y se
            # mueve con la estacionalidad del mes.
            factor = (1.0 + 0.045) ** (k / 12.0) * ESTACIONALIDAD[mes.month]
            facturado = p.renta_base * factor

            # El pago casi siempre cubre la factura; a veces se queda corto.
            cobertura = 1.0 if rng.random() > 0.12 else float(rng.uniform(0.45, 0.95))
            n_partes = 2 if p.parte_en_dos else 1

            for parte in range(n_partes):
                dia = int(np.clip(
                    round(p.dia_habitual + rng.normal(0, p.dispersion_dias) + parte * 11),
                    1, 28,
                ))
                monto = facturado * cobertura / n_partes
                monto *= 1.0 + rng.normal(0, p.variacion_monto)
                filas.append({
                    "fecha": mes + pd.Timedelta(days=dia - 1),
                    "cliente_id": p.cliente_id,
                    "empresa": p.empresa,
                    "arrendatario": p.arrendatario,
                    "monto": round(float(max(monto, 500.0)), 2),
                    "monto_facturado": round(float(facturado / n_partes), 2),
                    "contraparte": p.contraparte,
                    "_es_anomalia": 0,
                    "_tipo_anomalia": "",
                })
    return filas


def _marcar(fila: dict, tipo: str) -> dict:
    fila["_es_anomalia"] = 1
    fila["_tipo_anomalia"] = tipo
    return fila


def _inyectar(df: pd.DataFrame, perfiles, meses, rng, escala: float) -> pd.DataFrame:
    """Inyecta las cinco anomalías.

    Se hace *después* de la serie normal y sobre clientes distintos para cada
    tipo, de forma que ningún cliente acumule dos patrones y las métricas por
    tipo se puedan leer sin ambigüedad.
    """
    por_id = {p.cliente_id: p for p in perfiles}
    disponibles = list(por_id)
    rng.shuffle(disponibles)
    cursor = 0

    def tomar(n: int) -> list[int]:
        """Clientes distintos para cada tipo. `escala` permite generar un
        mundo más chico en los tests sin agotar el padrón."""
        nonlocal cursor
        n = max(1, int(round(n * escala)))
        elegidos = disponibles[cursor:cursor + n]
        cursor += n
        return elegidos

    # Solo se inyecta a partir del mes 7: los primeros seis meses son la
    # ventana de calentamiento con la que se construye la línea base.
    meses_validos = meses[6:]
    nuevas: list[dict] = []
    a_borrar: list[int] = []

    # (1) Fraccionamiento: un monto que cruzaría el umbral de aviso, partido
    #     en pagos chicos que individualmente no lo cruzan.
    for cid in tomar(INYECCIONES["fraccionamiento"]):
        mes = meses_validos[int(rng.integers(0, len(meses_validos)))]
        mask = ((df["cliente_id"] == cid)
                & (df["fecha"].dt.to_period("M") == mes.to_period("M")))
        a_borrar.extend(df.index[mask].tolist())
        p = por_id[cid]
        n_partes = int(rng.integers(4, 7))
        total = UMBRAL_AVISO * float(rng.uniform(1.05, 1.4))
        for j in range(n_partes):
            nuevas.append(_marcar({
                "fecha": mes + pd.Timedelta(days=int(3 + j * 4 + rng.integers(0, 2))),
                "cliente_id": cid,
                "empresa": p.empresa,
                "arrendatario": p.arrendatario,
                "monto": round(total / n_partes, 2),
                "monto_facturado": round(total / n_partes, 2),
                "contraparte": p.contraparte,
            }, "fraccionamiento"))

    # (2) Salto de escala: el cliente paga de golpe varias veces su histórico.
    for cid in tomar(INYECCIONES["salto_de_escala"]):
        idx = df.index[(df["cliente_id"] == cid) & (df["fecha"] >= meses_validos[0])]
        if len(idx) == 0:
            continue
        i = int(rng.choice(idx))
        df.loc[i, "monto"] = round(float(df.loc[i, "monto"]) * float(rng.uniform(6, 11)), 2)
        df.loc[i, "_es_anomalia"] = 1
        df.loc[i, "_tipo_anomalia"] = "salto_de_escala"

    # (3) Reactivación: cuenta dormida diez meses que revive con monto alto.
    #     En un mundo corto (los tests) el letargo se acorta a lo que quepa.
    dormido = min(10, max(2, len(meses) - 7))
    if len(meses) - dormido - 1 > 4:
        clientes_reactivacion = tomar(INYECCIONES["reactivacion"])
    else:
        clientes_reactivacion = []
    for cid in clientes_reactivacion:
        inicio = meses[int(rng.integers(4, len(meses) - dormido - 1))]
        fin = inicio + pd.DateOffset(months=dormido)
        mask = (df["cliente_id"] == cid) & (df["fecha"] >= inicio) & (df["fecha"] < fin)
        a_borrar.extend(df.index[mask].tolist())
        p = por_id[cid]
        nuevas.append(_marcar({
            "fecha": fin + pd.Timedelta(days=int(rng.integers(1, 20))),
            "cliente_id": cid,
            "empresa": p.empresa,
            "arrendatario": p.arrendatario,
            "monto": round(p.renta_base * float(rng.uniform(3.5, 7.0)), 2),
            "monto_facturado": round(p.renta_base, 2),
            "contraparte": p.contraparte,
        }, "reactivacion"))

    # (4) Ráfaga de montos redondos: tres pagos exactos seguidos. Un monto
    #     redondo no dice nada; tres seguidos en un cliente que nunca los
    #     hace, sí.
    for cid in tomar(INYECCIONES["rafaga_montos_redondos"]):
        mes = meses_validos[int(rng.integers(0, len(meses_validos) - 2))]
        p = por_id[cid]
        redondo = float(rng.choice([50_000, 100_000, 150_000, 200_000]))
        for j in range(3):
            nuevas.append(_marcar({
                "fecha": mes + pd.Timedelta(days=int(5 + j * 9)),
                "cliente_id": cid,
                "empresa": p.empresa,
                "arrendatario": p.arrendatario,
                "monto": redondo,
                "monto_facturado": round(p.renta_base, 2),
                "contraparte": p.contraparte,
            }, "rafaga_montos_redondos"))

    # (5) Contraparte inusual: el pago llega de un origen que ese cliente
    #     nunca ha usado. El monto es normal; lo raro es de dónde viene.
    for cid in tomar(INYECCIONES["contraparte_inusual"]):
        idx = df.index[(df["cliente_id"] == cid) & (df["fecha"] >= meses_validos[0])]
        if len(idx) == 0:
            continue
        i = int(rng.choice(idx))
        df.loc[i, "contraparte"] = f"XX-{rng.integers(100, 999)}"
        df.loc[i, "_es_anomalia"] = 1
        df.loc[i, "_tipo_anomalia"] = "contraparte_inusual"

    if a_borrar:
        df = df.drop(index=sorted(set(a_borrar)))
    if nuevas:
        df = pd.concat([df, pd.DataFrame(nuevas)], ignore_index=True)
    return df


def generar(semilla: int = SEMILLA, n_clientes: int = N_CLIENTES,
            n_meses: int = N_MESES) -> pd.DataFrame:
    """Devuelve el padrón completo de operaciones, ordenado por fecha.

    Determinista: la misma semilla da exactamente el mismo dataset, de modo
    que las cifras del README y las de los tests coinciden. `n_clientes` y
    `n_meses` solo se cambian en los tests, para correr el pipeline completo
    en segundos sobre un mundo más chico.
    """
    rng = np.random.default_rng(semilla)
    meses = pd.date_range(MES_INICIAL, periods=n_meses, freq="MS")

    perfiles = _perfiles(rng, n_clientes)
    df = pd.DataFrame(_serie_normal(perfiles, meses, rng))
    df = _inyectar(df, perfiles, meses, rng, escala=n_clientes / N_CLIENTES)

    df = df.sort_values(["fecha", "cliente_id"]).reset_index(drop=True)
    df.insert(0, "operacion_id", range(1, len(df) + 1))
    df["mes"] = df["fecha"].dt.to_period("M").astype(str)
    return df


if __name__ == "__main__":
    datos = generar()
    destino = "data/synthetic/operaciones.csv"
    datos.to_csv(destino, index=False)
    print(f"{len(datos):,} operaciones - {datos['cliente_id'].nunique()} clientes "
          f"- {datos['mes'].nunique()} meses -> {destino}")
    print(datos["_tipo_anomalia"].value_counts().to_string())
