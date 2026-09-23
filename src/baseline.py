"""
Línea base estadística por cliente: mediana + MAD sobre ventana móvil.

Antes del modelo va esto, y va por tres razones:

- **Es explicable sin saber nada de ML.** "Este pago es 9 veces más grande que
  lo que este cliente suele pagar" se defiende frente a un auditor. "El
  Isolation Forest le dio -0.41" no.
- **Es la referencia contra la cual medir si el modelo aporta algo.** Si el
  Isolation Forest no detecta más que esto, no vale la complejidad que cuesta.
- **Es lo que cumplimiento puede seguir usando si el modelo se cae.**

Se usa MAD y no desviación estándar porque la ventana es de seis puntos: con
seis observaciones, un solo pago atípico infla la desviación estándar lo
suficiente como para que el propio pago atípico deje de verse atípico. La MAD
no se mueve con un outlier — ese es todo el punto de usarla.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

VENTANA = 6
MIN_HISTORIA = 3

# 3.5 es el corte convencional para un z robusto. Con MAD, un |z| de 3.5
# equivale aproximadamente a lo que sería 3.5 sigmas si los datos fueran
# normales, pero sin que un outlier contamine la escala.
CORTE_Z = 3.5

# Factor que pone la MAD en la misma escala que una desviación estándar
# bajo normalidad (1 / 0.67449).
CONSISTENCIA = 1.4826


def _mad(serie: pd.Series) -> float:
    valores = serie.to_numpy(dtype=float)
    return float(np.median(np.abs(valores - np.median(valores))))


def calcular(d: pd.DataFrame, columna: str = "monto") -> pd.DataFrame:
    """Agrega `mediana_propia`, `mad_propia` y `z_robusto` al DataFrame.

    La ventana excluye la operación que se está puntuando: `shift(1)`. Sin
    ese shift el pago se compararía contra una mediana que lo incluye, y la
    señal se diluiría justo en los casos que importan.
    """
    d = d.sort_values(["cliente_id", "fecha", "operacion_id"]).copy()
    g = d.groupby("cliente_id", sort=False)[columna]

    d["mediana_propia"] = g.transform(
        lambda s: s.shift(1).rolling(VENTANA, min_periods=MIN_HISTORIA).median())
    d["mad_propia"] = g.transform(
        lambda s: s.shift(1).rolling(VENTANA, min_periods=MIN_HISTORIA).apply(_mad, raw=False))

    # Un piso para la MAD: hay clientes que pagan exactamente lo mismo cada
    # mes y su MAD es 0. Sin piso, cualquier variación de un peso daría un z
    # infinito. El piso es el 2% de la mediana del propio cliente, que es
    # decir "variar menos que esto no cuenta como variar".
    piso = 0.02 * d["mediana_propia"]
    escala = np.maximum(d["mad_propia"] * CONSISTENCIA, piso)

    d["z_robusto"] = (d[columna] - d["mediana_propia"]) / escala.replace(0, np.nan)
    d["z_robusto"] = d["z_robusto"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    d["alerta_baseline"] = (d["z_robusto"].abs() > CORTE_Z).astype(int)
    return d


def explicar(fila: pd.Series) -> str:
    """Una línea en español para la cola de revisión."""
    mediana = fila.get("mediana_propia")
    if pd.isna(mediana) or mediana == 0:
        return "Sin historia suficiente para comparar."
    veces = fila["monto"] / mediana
    direccion = "por encima de" if veces >= 1 else "por debajo de"
    return (f"${fila['monto']:,.0f} es {veces:,.1f}x lo habitual de este cliente "
            f"(${mediana:,.0f}) — {abs(fila['z_robusto']):.1f} desviaciones robustas "
            f"{direccion} su mediana.")
