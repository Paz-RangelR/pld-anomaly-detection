"""
La app de revisión tiene que levantar sin credenciales y sin explotar.

`AppTest` corre el script de Streamlit de verdad, sin navegador. No prueba que
se vea bonita; prueba que renderiza, que la tabla trae casos y que el toggle de
la etiqueta oculta no rompe nada — que es lo que se descubre tarde cuando la
demo se enseña en vivo.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app" / "cola_revision.py"

# Correr el pipeline completo dentro del test tarda; los fixtures hacen que
# esto sea cuestión de segundos.
FIXTURES = (APP.parents[1] / "data" / "synthetic" / "cola_revision.csv")
pytestmark = pytest.mark.skipif(
    not FIXTURES.exists(),
    reason="faltan los fixtures; corre primero `python scripts/run_demo.py`",
)


@pytest.fixture(scope="module")
def app() -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    return at


def test_la_app_levanta_sin_excepciones(app):
    assert not app.exception


def test_declara_que_los_datos_son_sinteticos(app):
    texto = " ".join(e.value for e in app.caption) + " ".join(e.value for e in app.title)
    assert "sintéticos" in texto
    assert "no desplegado" in texto


def test_muestra_la_cola_con_casos(app):
    assert len(app.dataframe) >= 1
    tabla = app.dataframe[0].value
    assert len(tabla) > 0
    assert "Qué llamó la atención" in tabla.columns
    # La identidad se muestra al analista pero nunca entró al modelo.
    assert "Arrendatario" in tabla.columns


def test_los_indicadores_estan_presentes(app):
    etiquetas = [m.label for m in app.metric]
    assert "Casos en cola" in etiquetas
    assert "Monto involucrado" in etiquetas


def test_el_toggle_de_la_etiqueta_oculta_funciona():
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    at.toggle[0].set_value(True).run()
    assert not at.exception
    assert "Etiqueta oculta" in at.dataframe[0].value.columns


def test_bajar_el_presupuesto_acorta_la_cola():
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    largo = len(at.dataframe[0].value)
    at.slider[0].set_value(5).run()
    assert not at.exception
    assert len(at.dataframe[0].value) <= min(5, largo)
