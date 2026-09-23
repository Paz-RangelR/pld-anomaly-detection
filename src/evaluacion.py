"""
Evaluación en dos capas.

Este dataset es sintético, así que tiene etiquetas: se sabe exactamente qué
operaciones se inyectaron y de qué tipo. Eso permite medir precisión y
recall de verdad.

**En producción eso no existe.** Nadie en la historia de la empresa marcó
"esta operación fue lavado". Si el único argumento para desplegar un modelo
es un precision@60 calculado sobre etiquetas que en la realidad no hay, no
hay argumento.

Por eso se calculan las dos capas y se reportan juntas:

- `con_etiquetas()`  — lo que se puede medir aquí y no allá. Sirve para saber
  si el método funciona en absoluto y para comparar variantes entre sí.
- `sin_etiquetas()`  — lo que sí se puede medir en producción desde el primer
  mes: estabilidad, concentración, y la tasa de confirmación de una muestra
  revisada a mano.

La segunda es la que decide si el sistema se queda o se apaga.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TIPOS = [
    "fraccionamiento",
    "salto_de_escala",
    "reactivacion",
    "rafaga_montos_redondos",
    "contraparte_inusual",
]


# --------------------------------------------------------------------------
# Capa 1 — con etiquetas (posible aquí, imposible en producción)
# --------------------------------------------------------------------------

def con_etiquetas(r: pd.DataFrame) -> dict:
    """Precisión, recall y comparación contra la línea base y contra el azar."""
    alertas = r[r["alerta"] == 1]
    n_alertas = len(alertas)
    n_anomalas = int(r["_es_anomalia"].sum())
    tasa_base = r["_es_anomalia"].mean()

    precision = alertas["_es_anomalia"].mean() if n_alertas else 0.0
    recall = alertas["_es_anomalia"].sum() / n_anomalas if n_anomalas else 0.0

    # La línea base univariada sola, con el mismo presupuesto: ordenar por
    # |z| y tomar los mismos N por mes. Es la comparación honesta.
    base = (r.assign(abs_z=r["z_robusto"].abs())
            .sort_values(["mes", "abs_z"], ascending=[True, False])
            .groupby("mes").head(int(r.groupby("mes")["alerta"].sum().mean())))
    precision_base = base["_es_anomalia"].mean() if len(base) else 0.0

    return {
        "operaciones_puntuadas": len(r),
        "anomalias_reales": n_anomalas,
        "tasa_base": float(tasa_base),
        "alertas_emitidas": n_alertas,
        "precision_en_presupuesto": float(precision),
        "recall_en_presupuesto": float(recall),
        "lift_vs_azar": float(precision / tasa_base) if tasa_base else 0.0,
        "precision_solo_linea_base": float(precision_base),
        "lift_vs_linea_base": float(precision / precision_base) if precision_base else 0.0,
    }


def recall_por_tipo(r: pd.DataFrame) -> pd.DataFrame:
    """Qué tipos de anomalía atrapa el modelo y cuáles se le van.

    Esta tabla es más útil que un número global: dice *dónde* falla. Un tipo
    con recall bajo no es necesariamente un defecto del modelo — puede ser una
    señal que las features no representan, y entonces lo que falta es una
    feature, no más árboles. Así apareció `contraparte_nueva`.

    Se reportan dos recalls, y la distinción importa:

    - **por operación** — de cada pago inyectado, ¿entró a la cola?
    - **por caso** — de cada episodio (un cliente en un mes), ¿entró al menos
      uno de sus pagos?

    El que cuenta operacionalmente es el segundo. Un fraccionamiento son cinco
    pagos; con que el analista vea uno, abre el expediente del cliente y
    encuentra los otros cuatro. Medir por operación castiga al modelo por algo
    que en la mesa de revisión no cuesta nada.
    """
    filas = []
    for tipo in TIPOS:
        sub = r[r["_tipo_anomalia"] == tipo]
        if len(sub) == 0:
            continue
        casos = sub.groupby(["cliente_id", "mes"])["alerta"].max()
        filas.append({
            "tipo": tipo,
            "ops": len(sub),
            "recall_op": float(sub["alerta"].mean()),
            "casos": int(len(casos)),
            "recall_caso": float(casos.mean()),
            "rank_mediano": float(sub["rank_mes"].median()),
        })
    return pd.DataFrame(filas).sort_values("recall_caso", ascending=False)


def curva_presupuesto(r: pd.DataFrame, presupuestos=range(10, 201, 10)) -> pd.DataFrame:
    """Detección en función de cuántas alertas al mes se puedan revisar.

    Es la curva que se le enseña al área: "si me das una persona más y puedo
    revisar 90 al mes en lugar de 60, atrapo esto otro". Convierte una
    decisión de plantilla en un número.
    """
    filas = []
    n_anomalas = int(r["_es_anomalia"].sum())
    for k in presupuestos:
        tomadas = (r.sort_values(["mes", "score"], ascending=[True, False])
                   .groupby("mes").head(k))
        filas.append({
            "alertas_por_mes": k,
            "revisiones_totales": len(tomadas),
            "precision": float(tomadas["_es_anomalia"].mean()),
            "recall": float(tomadas["_es_anomalia"].sum() / n_anomalas) if n_anomalas else 0.0,
        })
    return pd.DataFrame(filas)


# --------------------------------------------------------------------------
# Capa 2 — sin etiquetas (lo que de verdad se usaría)
# --------------------------------------------------------------------------

def sin_etiquetas(r: pd.DataFrame, muestra_revision: int = 20,
                  semilla: int = 20260923) -> dict:
    """Las métricas que sí existen en producción.

    Ninguna de las tres necesita saber qué operación fue realmente inusual.
    """
    alertas = r[r["alerta"] == 1]

    # (a) Estabilidad del ranking: qué proporción de los clientes alertados se
    #     repite del mes anterior. Muy alta significa que el modelo se quedó
    #     pegado a los mismos y el equipo revisa lo mismo cada mes. Muy baja
    #     significa que el ranking es ruido. Se busca algo intermedio.
    meses = sorted(alertas["mes"].unique())
    jaccards = []
    for a, b in zip(meses, meses[1:]):
        ca = set(alertas.loc[alertas["mes"] == a, "cliente_id"])
        cb = set(alertas.loc[alertas["mes"] == b, "cliente_id"])
        if ca or cb:
            jaccards.append(len(ca & cb) / len(ca | cb))
    estabilidad = float(np.mean(jaccards)) if jaccards else 0.0

    # (b) Concentración: cuánta de la carga de revisión recae en unos pocos
    #     clientes. Si el 5% de los clientes genera la mitad de las alertas,
    #     el sistema está señalando clientes, no operaciones.
    por_cliente = alertas.groupby("cliente_id").size().sort_values(ascending=False)
    top5pct = max(1, int(len(por_cliente) * 0.05))
    concentracion = float(por_cliente.head(top5pct).sum() / len(alertas)) if len(alertas) else 0.0

    # (c) Tasa de confirmación: se revisa a mano una muestra de cada mes y se
    #     cuenta cuántas el analista confirma como inusuales. Aquí el
    #     "analista" se simula con la etiqueta oculta, porque es una demo; en
    #     producción es una persona llenando un campo, y es la única fuente
    #     de verdad que el sistema llega a tener. Esa retroalimentación es
    #     además lo que con el tiempo permitiría pasar a un modelo supervisado.
    rng = np.random.default_rng(semilla)
    confirmadas, revisadas = 0, 0
    for mes, bloque in alertas.groupby("mes"):
        n = min(muestra_revision, len(bloque))
        idx = rng.choice(bloque.index.to_numpy(), size=n, replace=False)
        confirmadas += int(r.loc[idx, "_es_anomalia"].sum())
        revisadas += n

    return {
        "estabilidad_ranking_jaccard": estabilidad,
        "concentracion_top5pct_clientes": concentracion,
        "clientes_distintos_alertados": int(alertas["cliente_id"].nunique()),
        "muestra_revisada": revisadas,
        "tasa_confirmacion_muestra": float(confirmadas / revisadas) if revisadas else 0.0,
        "score_mediano_alertas": float(alertas["score"].median()) if len(alertas) else 0.0,
        "deriva_score_mensual": float(
            alertas.groupby("mes")["score"].median().std()) if len(meses) > 1 else 0.0,
    }
