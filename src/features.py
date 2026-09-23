"""
Features de comportamiento, calculadas por cliente y hacia atrás en el tiempo.

Dos reglas gobiernan este archivo, y las dos son decisiones de diseño, no
detalles de implementación:

1. **Ninguna feature mira el futuro.** Todo lo que describe la "normalidad" de
   un cliente se calcula con sus pagos *anteriores*, nunca con el pago que se
   está puntuando ni con los que vienen después. Si una feature usara la
   mediana de toda la serie, el propio evento anómalo estaría dentro de la
   referencia contra la que se compara, y la anomalía se escondería sola.

2. **Ninguna feature contiene identidad.** No entra nombre, RFC, domicilio,
   correo ni razón social. El modelo ve comportamiento; para el algoritmo, el
   cliente 4471 es 4471. En un contexto de PLD eso es minimización de datos, y
   además evita que el modelo aprenda a sospechar de un giro o de una zona en
   lugar de un patrón.

La regla 2 está puesta en código: `verificar_sin_identidad()` revienta si
alguien agrega una columna identificable a la matriz de entrenamiento.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Lo que entra al modelo. Diez features, todas de comportamiento.
COLUMNAS_FEATURES = [
    "monto_vs_mediana_propia",
    "dias_desde_pago_anterior",
    "desv_vs_ritmo_propio",
    "n_pagos_mes",
    "razon_pago_vs_factura",
    "es_monto_redondo",
    "n_contrapartes_distintas",
    "contraparte_nueva",
    "meses_inactivo_previos",
    "pct_del_total_anual",
]

# Lo que nunca entra, aunque esté en el DataFrame.
COLUMNAS_PROHIBIDAS = [
    "arrendatario", "empresa", "contraparte", "nombre", "rfc",
    "domicilio", "correo", "_es_anomalia", "_tipo_anomalia",
]

# Ventana para la línea base de comportamiento: seis pagos previos. Corta a
# propósito — la normalidad de un arrendatario cambia (renegocia, crece,
# reduce superficie), y una ventana larga la promedia con un pasado que ya no
# aplica.
VENTANA = 6
MIN_HISTORIA = 3
EPS = 1e-9


def _mad(serie: pd.Series) -> float:
    """Desviación absoluta mediana. Robusta: un solo outlier no la mueve."""
    valores = serie.to_numpy(dtype=float)
    return float(np.median(np.abs(valores - np.median(valores))))


def _es_redondo(monto: pd.Series) -> pd.Series:
    """Monto 'redondo': múltiplo exacto de mil y de al menos diez mil.

    Un pago de $12,000 es común y no dice nada. La señal no está en un monto
    redondo aislado sino en la repetición, que captura `n_pagos_mes` junto con
    esta bandera.
    """
    return ((monto >= 10_000) & (np.mod(monto, 1_000) == 0)).astype(int)


def construir(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve el DataFrame original más las diez features.

    Espera las columnas `operacion_id`, `fecha`, `cliente_id`, `monto`,
    `monto_facturado` y `contraparte`.
    """
    d = df.copy()
    d["fecha"] = pd.to_datetime(d["fecha"])
    d = d.sort_values(["cliente_id", "fecha", "operacion_id"]).reset_index(drop=True)

    g = d.groupby("cliente_id", sort=False)

    # --- escala: cuánto se sale este monto de lo que ese cliente suele pagar
    mediana_previa = (g["monto"]
                      .transform(lambda s: s.shift(1)
                                 .rolling(VENTANA, min_periods=MIN_HISTORIA).median()))
    d["monto_vs_mediana_propia"] = d["monto"] / (mediana_previa + EPS)

    # --- ritmo: cada cuánto paga, y qué tan raro es este hueco para él
    dias = g["fecha"].diff().dt.days
    d["dias_desde_pago_anterior"] = dias.fillna(0.0)

    gap_mediano = (dias.groupby(d["cliente_id"], sort=False)
                   .transform(lambda s: s.shift(1)
                              .rolling(VENTANA, min_periods=MIN_HISTORIA).median()))
    gap_mad = (dias.groupby(d["cliente_id"], sort=False)
               .transform(lambda s: s.shift(1)
                          .rolling(VENTANA, min_periods=MIN_HISTORIA).apply(_mad, raw=False)))
    # El 0.6745 convierte la MAD en algo comparable con una desviación
    # estándar bajo normalidad, para que el número se lea como un z-score.
    d["desv_vs_ritmo_propio"] = (0.6745 * (dias - gap_mediano) / (gap_mad + 1.0)).fillna(0.0)

    # --- frecuencia: cuántos pagos lleva ese cliente en el mes calendario,
    #     contando este. El fraccionamiento vive aquí.
    d["n_pagos_mes"] = (d.groupby([d["cliente_id"], d["fecha"].dt.to_period("M")], sort=False)
                        .cumcount() + 1)

    # --- cobertura: qué proporción de lo facturado cubre el pago. Por encima
    #     de 1 hay pago en exceso, que es de las cosas que cumplimiento mira.
    d["razon_pago_vs_factura"] = d["monto"] / (d["monto_facturado"] + EPS)

    d["es_monto_redondo"] = _es_redondo(d["monto"])

    # --- contrapartes distintas acumuladas. Casi todos los clientes se quedan
    #     en 1 de por vida; pasar a 2 es informativo justamente por eso.
    d["n_contrapartes_distintas"] = (g["contraparte"]
                                     .transform(lambda s: (~s.duplicated()).cumsum()))

    # Y la bandera puntual: este origen de fondos no lo había usado nunca.
    # Se agregó después de la primera evaluación. El conteo acumulado de
    # arriba se queda elevado para siempre una vez que el cliente estrena
    # contraparte, así que termina marcando al cliente en lugar de marcar la
    # operación: el recall del tipo "contraparte inusual" se quedaba en 43%.
    # Esta bandera vale 1 solo en el pago que estrena el origen. El
    # diagnóstico salió de `recall_por_tipo()`, que es justo para eso.
    d["contraparte_nueva"] = g["contraparte"].transform(
        lambda s: (~s.duplicated()).astype(int))

    d["meses_inactivo_previos"] = np.floor(d["dias_desde_pago_anterior"] / 30.0)

    # --- peso relativo: qué fracción representa este pago del total que ese
    #     cliente movió en los doce meses anteriores.
    total_12m = (d.set_index("fecha")
                 .groupby("cliente_id", sort=False)["monto"]
                 .rolling("365D").sum()
                 .reset_index(level=0, drop=True)
                 .to_numpy())
    previo_12m = total_12m - d["monto"].to_numpy()
    d["pct_del_total_anual"] = d["monto"].to_numpy() / (previo_12m + EPS)

    # Sin historia suficiente no hay normalidad que comparar. Esas filas se
    # marcan y el pipeline las excluye del entrenamiento y del scoring.
    d["tiene_historia"] = mediana_previa.notna()

    for col in COLUMNAS_FEATURES:
        d[col] = (d[col].replace([np.inf, -np.inf], np.nan)
                  .fillna(0.0)
                  .astype(float))
    return d


def matriz(d: pd.DataFrame) -> pd.DataFrame:
    """La matriz que entra al modelo: solo esas columnas, en orden fijo."""
    X = d[COLUMNAS_FEATURES].copy()
    verificar_sin_identidad(X)
    return X


def verificar_sin_identidad(X: pd.DataFrame) -> None:
    """Falla si la matriz trae algo identificable.

    Existe para que la decisión de privacidad no dependa de que alguien se
    acuerde: si en un futuro se agrega una columna con nombre o RFC, el
    pipeline se detiene aquí en lugar de entrenar con ella.
    """
    intrusas = [c for c in X.columns if c.lower() in COLUMNAS_PROHIBIDAS]
    if intrusas:
        raise ValueError(
            f"La matriz de features contiene columnas identificables: {intrusas}. "
            "El modelo no debe ver identidad."
        )
    no_numericas = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    if no_numericas:
        raise ValueError(f"Features no numéricas: {no_numericas}")
