"""
Validación con ventana expansiva. Nunca split aleatorio.

Este archivo es corto y es el más importante del repo.

Con datos temporales, un `train_test_split` aleatorio entrena con operaciones
de diciembre y evalúa con operaciones de marzo. El modelo llega al mes de
prueba habiendo visto ya lo que pasó después, y las métricas salen infladas —
a veces espectacularmente. Después el sistema se despliega, ve el futuro de
verdad por primera vez, y no se parece en nada al reporte.

Lo que se hace aquí es lo único que reproduce las condiciones reales:

    entrena con los meses 1..T  ->  puntúa el mes T+1  ->  avanza T

El modelo del mes 13 nunca vio el mes 14. Igual que en producción, donde el
modelo que corre hoy se entrenó con lo que había ayer.
"""

from __future__ import annotations

import pandas as pd

from .features import COLUMNAS_FEATURES
from .modelo import DetectorAnomalias, contaminacion_por_presupuesto

# Seis meses de calentamiento: antes de eso los clientes no tienen historia
# suficiente para que "su normalidad" signifique algo.
MES_INICIAL_SCORING = 7

# Lo que un equipo de tres personas puede revisar al mes sin que la revisión
# se vuelva un trámite. Es la restricción que calibra el modelo.
PRESUPUESTO_MENSUAL = 60


def validar(
    d: pd.DataFrame,
    presupuesto_mensual: int = PRESUPUESTO_MENSUAL,
    mes_inicial: int = MES_INICIAL_SCORING,
) -> tuple[pd.DataFrame, dict[str, DetectorAnomalias]]:
    """Corre la ventana expansiva sobre todos los meses puntuables.

    Devuelve las operaciones puntuadas —con su score, su lugar en la cola de
    ese mes y si entró al presupuesto de revisión— y el detector entrenado
    para cada mes, que hace falta para explicar sus alertas.
    """
    d = d[d["tiene_historia"]].copy()
    meses = sorted(d["mes"].unique())

    resultados: list[pd.DataFrame] = []
    detectores: dict[str, DetectorAnomalias] = {}

    for i, mes in enumerate(meses):
        if i + 1 < mes_inicial:
            continue

        entrenamiento = d[d["mes"] < mes]
        prueba = d[d["mes"] == mes]
        if len(prueba) == 0 or len(entrenamiento) < 500:
            continue

        # La contaminación se recalcula cada mes porque el volumen de
        # operaciones cambia: el presupuesto de revisión es fijo, la base no.
        ops_por_mes = len(entrenamiento) / entrenamiento["mes"].nunique()
        detector = DetectorAnomalias(
            contaminacion=contaminacion_por_presupuesto(presupuesto_mensual, ops_por_mes)
        ).entrenar(entrenamiento[COLUMNAS_FEATURES])

        scores = detector.puntuar(prueba[COLUMNAS_FEATURES])
        bloque = prueba.copy()
        bloque["score"] = scores
        bloque["rank_mes"] = bloque["score"].rank(ascending=False, method="first").astype(int)
        bloque["alerta"] = (bloque["rank_mes"] <= presupuesto_mensual).astype(int)
        bloque["mes_entrenamiento_hasta"] = meses[i - 1]

        resultados.append(bloque)
        detectores[mes] = detector

    if not resultados:
        raise RuntimeError("No hubo ningún mes puntuable; revisa el rango de fechas.")

    return pd.concat(resultados, ignore_index=True), detectores
