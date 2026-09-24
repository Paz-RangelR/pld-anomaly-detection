"""
Atribución por feature: por qué se disparó cada alerta.

Una alerta sin explicación no se usa. Cumplimiento no puede abrir un
expediente porque "el modelo la marcó", y con razón: si el analista no puede
decir qué le llamó la atención, tampoco puede descartarla ni sostenerla ante
un auditor. La confianza en una alerta no viene de la precisión del modelo,
viene de poder explicarla.

El método aquí es la **neutralización**: para cada feature se vuelve a
puntuar la operación reemplazando esa feature por su valor mediano en el
entrenamiento, es decir, poniéndola en "normal". Si el score cae mucho, esa
feature era la que sostenía la alerta.

    contribución(f) = score_original - score_con_f_neutralizada

Es el mismo razonamiento que una explicación por ablación, y tiene una
ventaja sobre las importancias globales del modelo: es *de esta alerta*, no
del modelo en promedio. Dos operaciones con el mismo score pueden estar
señaladas por motivos distintos, y eso es justo lo que el analista necesita
saber.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import COLUMNAS_FEATURES

# Cómo se lee cada feature en español, para la cola de revisión.
FRASES = {
    "monto_vs_mediana_propia": "monto fuera de la escala habitual del cliente",
    "dias_desde_pago_anterior": "hueco inusual desde el pago anterior",
    "desv_vs_ritmo_propio": "ritmo de pago distinto al suyo",
    "n_pagos_mes": "varios pagos en el mismo mes",
    "razon_pago_vs_factura": "el pago no corresponde a lo facturado",
    "es_monto_redondo": "monto exacto y redondo",
    "n_contrapartes_distintas": "acumula orígenes de fondos distintos",
    "contraparte_nueva": "origen de fondos que no había usado nunca",
    "meses_inactivo_previos": "cuenta inactiva que se reactiva",
    "pct_del_total_anual": "el pago pesa demasiado en su año",
}


# Version corta de lo mismo, para etiquetas de grafica. Las frases de arriba se
# leen bien dentro de una tabla, pero en el eje de una barra horizontal no
# caben y el navegador las corta a media palabra.
FRASES_CORTAS = {
    "monto_vs_mediana_propia": "monto fuera de escala",
    "dias_desde_pago_anterior": "hueco entre pagos",
    "desv_vs_ritmo_propio": "ritmo distinto",
    "n_pagos_mes": "varios pagos al mes",
    "razon_pago_vs_factura": "pago distinto a factura",
    "es_monto_redondo": "monto redondo",
    "n_contrapartes_distintas": "varios orígenes",
    "contraparte_nueva": "origen nuevo",
    "meses_inactivo_previos": "cuenta reactivada",
    "pct_del_total_anual": "peso en su año",
}


def atribuir(detector, X: pd.DataFrame) -> pd.DataFrame:
    """Matriz de contribuciones: una fila por operación, una columna por feature.

    Positivo = esa feature empujó la operación hacia "anómala".
    """
    if detector.medianas_entrenamiento is None:
        raise RuntimeError("El detector no ha sido entrenado.")

    X = X[COLUMNAS_FEATURES]
    base = detector.puntuar(X)
    contribuciones = {}
    for f in COLUMNAS_FEATURES:
        neutral = X.copy()
        neutral[f] = detector.medianas_entrenamiento[f]
        contribuciones[f] = base - detector.puntuar(neutral)
    return pd.DataFrame(contribuciones, index=X.index)


def principales(contribuciones: pd.DataFrame, n: int = 3) -> pd.Series:
    """Para cada operación, las `n` features que más pesaron, ya en texto."""
    orden = np.argsort(-contribuciones.to_numpy(), axis=1)[:, :n]
    columnas = np.array(contribuciones.columns)
    return pd.Series(
        [" · ".join(FRASES[c] for c in columnas[fila]) for fila in orden],
        index=contribuciones.index,
        name="motivo",
    )


def detalle(fila_features: pd.Series, fila_contrib: pd.Series, n: int = 3) -> pd.DataFrame:
    """Desglose de una sola alerta, para el panel de detalle.

    Muestra la feature, cuánto valió y cuánto aportó al score, ordenado por
    aportación. Es lo que el analista mira antes de abrir o descartar el caso.
    """
    orden = fila_contrib.sort_values(ascending=False).head(n).index
    return pd.DataFrame({
        "señal": [FRASES[f] for f in orden],
        "feature": list(orden),
        "valor": [float(fila_features[f]) for f in orden],
        "aporte_al_score": [float(fila_contrib[f]) for f in orden],
    })
