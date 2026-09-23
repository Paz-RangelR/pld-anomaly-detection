"""
Isolation Forest sobre la matriz de features.

La línea base univariada (`baseline.py`) atrapa lo obvio: el monto que se
dispara. Lo que no puede ver es la *combinación* — monto normal, pero en un
ritmo raro, con una contraparte nueva y varios pagos en el mes. Ninguna de
esas señales por separado cruza un umbral; juntas describen exactamente el
patrón que a cumplimiento le interesa.

Para eso está el Isolation Forest: aísla puntos raros en el espacio conjunto
de las features, sin necesidad de etiquetas y sin suponer que los datos
son normales (no lo son: la distribución de rentas tiene cola larga).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from .features import COLUMNAS_FEATURES, verificar_sin_identidad

SEMILLA = 20260923

# El tamaño de submuestra del paper original (Liu, Ting & Zhou). Mantenerlo
# chico es deliberado: con submuestras grandes los árboles se saturan de
# puntos normales y las anomalías dejan de aislarse en pocos cortes.
MAX_SAMPLES = 256
N_ARBOLES = 300


def contaminacion_por_presupuesto(alertas_por_mes: int, operaciones_por_mes: float) -> float:
    """Traduce una restricción operativa en el hiperparámetro.

    `contamination` se fija casi siempre a ojo (0.01, 0.05, "lo que se vea
    bien"). Aquí no: cumplimiento son tres personas y pueden revisar del orden
    de 60 operaciones al mes con cuidado. Ese número es el que manda.

    Si el modelo marca 400, el equipo revisa las primeras 60 y las otras 340
    son ruido que además genera desconfianza en el sistema. Si marca 10, se
    están dejando casos sobre la mesa. La pregunta correcta no es "¿cuántas
    anomalías hay?" — nadie lo sabe — sino "¿cuántas alcanzo a revisar?".
    """
    if operaciones_por_mes <= 0:
        raise ValueError("operaciones_por_mes debe ser positivo")
    return float(np.clip(alertas_por_mes / operaciones_por_mes, 0.001, 0.25))


@dataclass
class DetectorAnomalias:
    """Envoltura del Isolation Forest con el vocabulario del problema."""

    contaminacion: float = 0.02
    semilla: int = SEMILLA
    modelo: IsolationForest | None = field(default=None, repr=False)
    medianas_entrenamiento: pd.Series | None = field(default=None, repr=False)

    def entrenar(self, X: pd.DataFrame) -> "DetectorAnomalias":
        verificar_sin_identidad(X)

        # Sin escalador, y no por descuido: el Isolation Forest parte cada
        # feature en un valor uniforme dentro de su rango, así que cualquier
        # transformación afín (estandarizar, min-max, RobustScaler) deja los
        # árboles exactamente igual. Meter un escalador aquí sería ritual.
        self.modelo = IsolationForest(
            n_estimators=N_ARBOLES,
            max_samples=MAX_SAMPLES,
            contamination=self.contaminacion,
            random_state=self.semilla,
            n_jobs=-1,
        ).fit(X[COLUMNAS_FEATURES])

        # Se guardan para la atribución por feature (ver explicacion.py): el
        # valor "neutro" contra el que se compara cada variable.
        self.medianas_entrenamiento = X[COLUMNAS_FEATURES].median()
        return self

    def puntuar(self, X: pd.DataFrame) -> np.ndarray:
        """Score de anomalía, **mayor = más raro**.

        `score_samples` de scikit-learn devuelve lo contrario (más bajo = más
        anómalo), lo cual se presta a equivocaciones cuando después se ordena
        la cola de revisión. Se invierte aquí, una sola vez, en el único lugar
        donde el signo debe pensarse.
        """
        if self.modelo is None:
            raise RuntimeError("El detector no ha sido entrenado.")
        return -self.modelo.score_samples(X[COLUMNAS_FEATURES])

    def umbral_por_presupuesto(self, scores: np.ndarray, n_alertas: int) -> float:
        """Corte que deja pasar exactamente las `n_alertas` más raras."""
        if n_alertas <= 0:
            return float("inf")
        if n_alertas >= len(scores):
            return float(np.min(scores))
        return float(np.partition(scores, -n_alertas)[-n_alertas])
