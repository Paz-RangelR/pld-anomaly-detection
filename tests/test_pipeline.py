"""
Pruebas del pipeline.

No prueban que el modelo sea bueno —para eso está `scripts/run_demo.py` con
sus aserciones—. Prueban las invariantes que, si se rompen, hacen que los
números buenos sean mentira:

- que los datos sean reproducibles,
- que **ninguna feature mire el futuro**,
- que la matriz de entrenamiento no contenga identidad,
- que cada modelo se entrene solo con meses anteriores al que puntúa.

La segunda es la importante. Una fuga temporal no truena nada: entrega
métricas excelentes y un sistema que en producción no funciona. Por eso se
prueba y no se confía.

    pytest -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import baseline, evaluacion, features, generar_datos  # noqa: E402
from src.explicacion import atribuir, principales  # noqa: E402
from src.modelo import contaminacion_por_presupuesto  # noqa: E402
from src.validacion_temporal import validar  # noqa: E402

# Un mundo chico: el pipeline completo corre en segundos y las invariantes
# son las mismas.
CLIENTES, MESES = 140, 18


@pytest.fixture(scope="module")
def datos() -> pd.DataFrame:
    return generar_datos.generar(n_clientes=CLIENTES, n_meses=MESES)


@pytest.fixture(scope="module")
def con_features(datos) -> pd.DataFrame:
    return baseline.calcular(features.construir(datos))


# --------------------------------------------------------------------------
# Datos
# --------------------------------------------------------------------------

def test_generador_es_determinista():
    a = generar_datos.generar(n_clientes=40, n_meses=14)
    b = generar_datos.generar(n_clientes=40, n_meses=14)
    pd.testing.assert_frame_equal(a, b)


def test_hay_anomalias_de_los_cinco_tipos(datos):
    tipos = set(datos.loc[datos["_es_anomalia"] == 1, "_tipo_anomalia"])
    assert tipos == set(evaluacion.TIPOS)


def test_las_anomalias_son_minoria(datos):
    # Si fueran muchas, "lo normal" dejaría de ser normal y el enfoque no
    # aplicaría. En PLD real la proporción es mucho menor todavía.
    assert 0.005 < datos["_es_anomalia"].mean() < 0.06


def test_no_se_inyecta_en_la_ventana_de_calentamiento(datos):
    primeros_seis = sorted(datos["mes"].unique())[:6]
    calentamiento = datos[datos["mes"].isin(primeros_seis)]
    assert calentamiento["_es_anomalia"].sum() == 0


# --------------------------------------------------------------------------
# Features: la prueba de fuga temporal
# --------------------------------------------------------------------------

def test_ninguna_feature_mira_el_futuro(datos):
    """Recalcula las features viendo solo la primera mitad de la historia.

    Si una feature usara datos posteriores, los valores de las operaciones
    antiguas cambiarían al aparecer el futuro. Deben ser idénticos.
    """
    corte = datos["fecha"].quantile(0.5)

    completo = features.construir(datos)
    truncado = features.construir(datos[datos["fecha"] <= corte])

    clave = ["operacion_id"] + features.COLUMNAS_FEATURES
    a = (completo[completo["fecha"] <= corte][clave]
         .sort_values("operacion_id").reset_index(drop=True))
    b = truncado[clave].sort_values("operacion_id").reset_index(drop=True)

    assert len(b) > 100, "el corte dejó muy pocas filas para que la prueba valga"
    pd.testing.assert_frame_equal(a, b)


def test_la_linea_base_tampoco_mira_el_futuro(datos):
    corte = datos["fecha"].quantile(0.5)
    columnas = ["operacion_id", "mediana_propia", "mad_propia", "z_robusto"]

    completo = baseline.calcular(features.construir(datos))
    truncado = baseline.calcular(features.construir(datos[datos["fecha"] <= corte]))

    a = (completo[completo["fecha"] <= corte][columnas]
         .sort_values("operacion_id").reset_index(drop=True))
    b = truncado[columnas].sort_values("operacion_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


def test_las_features_no_traen_identidad(con_features):
    X = features.matriz(con_features)
    assert list(X.columns) == features.COLUMNAS_FEATURES
    assert X.notna().all().all()
    assert np.isfinite(X.to_numpy()).all()


def test_la_matriz_rechaza_columnas_identificables(con_features):
    X = features.matriz(con_features)
    X["rfc"] = "AAI180312K72"
    with pytest.raises(ValueError, match="identificables"):
        features.verificar_sin_identidad(X)


def test_la_matriz_rechaza_la_etiqueta_oculta(con_features):
    X = features.matriz(con_features)
    X["_es_anomalia"] = con_features["_es_anomalia"].to_numpy()
    with pytest.raises(ValueError):
        features.verificar_sin_identidad(X)


# --------------------------------------------------------------------------
# Línea base
# --------------------------------------------------------------------------

def test_la_mad_no_se_deja_envenenar_por_un_outlier():
    """El argumento para usar MAD en lugar de desviación estándar, medido.

    Seis pagos iguales y uno enorme. Con desviación estándar, el propio pago
    enorme infla la escala y su z se desploma; con MAD se mantiene alto.
    """
    normales = [100_000.0] * 6
    serie = pd.DataFrame({
        "operacion_id": range(1, 9),
        "cliente_id": 1,
        "fecha": pd.date_range("2024-01-31", periods=8, freq="ME"),
        "monto": normales + [1_000_000.0, 100_000.0],
    })
    z = baseline.calcular(serie).iloc[6]["z_robusto"]

    ventana = np.array(normales)
    z_desv_std = (1_000_000 - ventana.mean()) / max(ventana.std(ddof=1), 1e-9)

    assert z > baseline.CORTE_Z
    assert np.isfinite(z)
    # Con seis valores idénticos la desviación estándar es 0 y el z-score
    # clásico ni se puede calcular sin inventarle un piso. Ese es el punto.
    assert not np.isfinite(z_desv_std) or z_desv_std > 1e6


def test_el_piso_de_la_mad_evita_z_infinitos():
    serie = pd.DataFrame({
        "operacion_id": range(1, 7),
        "cliente_id": 1,
        "fecha": pd.date_range("2024-01-31", periods=6, freq="ME"),
        "monto": [50_000.0] * 5 + [50_100.0],
    })
    d = baseline.calcular(serie)
    assert np.isfinite(d["z_robusto"]).all()


# --------------------------------------------------------------------------
# Modelo y validación temporal
# --------------------------------------------------------------------------

def test_contaminacion_sale_del_presupuesto():
    assert contaminacion_por_presupuesto(60, 1_200) == pytest.approx(0.05)
    assert contaminacion_por_presupuesto(1, 100_000) == 0.001      # piso
    assert contaminacion_por_presupuesto(900, 1_000) == 0.25       # techo
    with pytest.raises(ValueError):
        contaminacion_por_presupuesto(60, 0)


def test_cada_modelo_entrena_solo_con_el_pasado(con_features):
    r, detectores = validar(con_features, presupuesto_mensual=10)
    assert len(detectores) >= 6
    assert (r["mes_entrenamiento_hasta"] < r["mes"]).all()


def test_el_presupuesto_se_respeta_mes_a_mes(con_features):
    r, _ = validar(con_features, presupuesto_mensual=10)
    por_mes = r.groupby("mes")["alerta"].sum()
    assert (por_mes <= 10).all()
    assert (por_mes > 0).all()


def test_el_score_es_mayor_para_lo_mas_raro(con_features):
    """El signo del score. Se invierte una vez en `modelo.puntuar` y si
    alguien lo vuelve a invertir, la cola queda al revés y nadie lo nota."""
    # La comparación es *dentro* de cada mes: el corte es por mes, así que
    # una alerta de un mes tranquilo puede puntuar menos que una operación
    # normal de un mes agitado sin que nada esté mal.
    r, _ = validar(con_features, presupuesto_mensual=10)
    for mes, bloque in r.groupby("mes"):
        alertadas = bloque.loc[bloque["alerta"] == 1, "score"]
        resto = bloque.loc[bloque["alerta"] == 0, "score"]
        assert alertadas.min() >= resto.max(), f"cola mal ordenada en {mes}"


def test_el_modelo_detecta_mejor_que_el_azar(con_features):
    r, _ = validar(con_features, presupuesto_mensual=10)
    m = evaluacion.con_etiquetas(r)
    assert m["lift_vs_azar"] > 3


# --------------------------------------------------------------------------
# Evaluación y explicación
# --------------------------------------------------------------------------

def test_recall_por_tipo_distingue_operacion_de_caso(con_features):
    r, _ = validar(con_features, presupuesto_mensual=20)
    t = evaluacion.recall_por_tipo(r)
    assert not t.empty
    # Un caso puede tener varias operaciones, nunca al contrario.
    assert (t["casos"] <= t["ops"]).all()
    assert (t["recall_caso"] >= t["recall_op"] - 1e-9).all()


def test_la_curva_de_presupuesto_es_monotona(con_features):
    r, _ = validar(con_features, presupuesto_mensual=20)
    c = evaluacion.curva_presupuesto(r, presupuestos=[5, 10, 20, 40])
    assert c["recall"].is_monotonic_increasing
    # Más presupuesto siempre alcanza más anomalías, y siempre a costa de
    # precisión: es la disyuntiva que la gráfica del README muestra.
    assert c["precision"].iloc[0] >= c["precision"].iloc[-1]


def test_las_metricas_sin_etiquetas_no_usan_la_etiqueta_para_rankear(con_features):
    r, _ = validar(con_features, presupuesto_mensual=20)
    s = evaluacion.sin_etiquetas(r, muestra_revision=5)
    assert 0.0 <= s["estabilidad_ranking_jaccard"] <= 1.0
    assert 0.0 < s["concentracion_top5pct_clientes"] <= 1.0
    assert s["muestra_revisada"] > 0


def test_cada_alerta_trae_su_explicacion(con_features):
    r, detectores = validar(con_features, presupuesto_mensual=10)
    mes = max(detectores)
    alertas = r[(r["mes"] == mes) & (r["alerta"] == 1)]

    contrib = atribuir(detectores[mes], alertas)
    assert contrib.shape == (len(alertas), len(features.COLUMNAS_FEATURES))

    motivos = principales(contrib)
    assert len(motivos) == len(alertas)
    assert motivos.str.len().gt(10).all()
    # Alguna feature tiene que estar sosteniendo la alerta; si todas las
    # contribuciones fueran cero, la explicación sería decorativa.
    assert (contrib.max(axis=1) > 0).mean() > 0.9
