"""
La cola de revisión, como la vería cumplimiento.

    streamlit run app/cola_revision.py

El modelo no decide nada. Ordena. Esta pantalla existe porque una lista de
scores en un CSV no la usa nadie: el analista necesita ver por qué subió cada
caso, qué hacía ese cliente antes, y poder resolverlo sin salir de aquí.

Lee la cola que produce `scripts/run_demo.py`. Si no la encuentra, corre el
pipeline completo en el momento (tarda cerca de medio minuto) para que la app
funcione en un clon recién hecho.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src.explicacion import FRASES_CORTAS  # noqa: E402

DATOS = RAIZ / "data" / "synthetic"
COLA = DATOS / "cola_revision.csv"
HISTORIAL = DATOS / "historial_alertados.csv"

st.set_page_config(page_title="Cola de revisión PLD", page_icon="🔎", layout="wide")


@st.cache_data(show_spinner="Corriendo el pipeline sobre datos sintéticos…")
def cargar() -> tuple[pd.DataFrame, pd.DataFrame]:
    if COLA.exists() and HISTORIAL.exists():
        cola = pd.read_csv(COLA, parse_dates=["fecha"])
        historial = pd.read_csv(HISTORIAL, parse_dates=["fecha"])
        return cola, historial

    # Clon recién hecho y sin fixtures: se genera todo desde cero.
    from src import baseline, explicacion, features, generar_datos
    from src.validacion_temporal import validar

    d = baseline.calcular(features.construir(generar_datos.generar()))
    r, detectores = validar(d)
    alertas = r[r["alerta"] == 1].copy()

    motivos, contribs = [], []
    for mes, bloque in alertas.groupby("mes"):
        c = explicacion.atribuir(detectores[mes], bloque)
        motivos.append(explicacion.principales(c))
        contribs.append(c.add_prefix("aporte_"))
    alertas["motivo"] = pd.concat(motivos)
    alertas = alertas.join(pd.concat(contribs)).sort_values(["mes", "rank_mes"])

    historial = (d.loc[d["cliente_id"].isin(alertas["cliente_id"].unique()),
                       ["cliente_id", "fecha", "monto", "mediana_propia"]]
                 .sort_values(["cliente_id", "fecha"]))
    return alertas, historial


cola, historial = cargar()

# --------------------------------------------------------------------------
# Encabezado
# --------------------------------------------------------------------------

st.title("Cola de revisión de operaciones inusuales")
st.caption(
    "Arrendamiento · LFPIORPI — **implementación de referencia sobre datos "
    "sintéticos.** Ningún dato real, ni anonimizado. El sistema de producción "
    "que la inspira está en diseño, no desplegado."
)

# --------------------------------------------------------------------------
# Controles
# --------------------------------------------------------------------------

lado = st.sidebar
lado.header("Revisión")

meses = sorted(cola["mes"].unique())
mes = lado.selectbox("Mes", meses, index=len(meses) - 1)

presupuesto = lado.slider(
    "Casos que puedo revisar este mes", 5, int(cola["rank_mes"].max()), 25, step=5,
    help="El modelo se calibra contra esta restricción, no al revés: "
         "se revisan los N más raros, no 'los que pasen un umbral'.",
)

revelar = lado.toggle(
    "Mostrar la etiqueta oculta", value=False,
    help="Solo posible porque los datos son sintéticos. En producción esta "
         "columna no existe: nadie sabe de antemano qué operación fue inusual.",
)

lado.divider()
lado.caption(
    "El modelo nunca ve el nombre, el RFC ni el domicilio del arrendatario. "
    "Puntúa comportamiento: escala, ritmo, frecuencia, origen de fondos. "
    "Los nombres de esta pantalla son para el analista, no para el algoritmo."
)

del_mes = cola[(cola["mes"] == mes) & (cola["rank_mes"] <= presupuesto)].copy()

# --------------------------------------------------------------------------
# Indicadores
# --------------------------------------------------------------------------

c1, c2, c3, c4 = st.columns(4)
c1.metric("Casos en cola", len(del_mes))
c2.metric("Clientes distintos", del_mes["cliente_id"].nunique())
c3.metric("Monto involucrado", f"${del_mes['monto'].sum():,.0f}")
if revelar:
    c4.metric("Confirmados", f"{del_mes['_es_anomalia'].mean():.0%}",
              help="Proporción de la cola que corresponde a una anomalía inyectada.")
else:
    c4.metric("Confirmados", "—", help="Actívalo en la barra lateral.")

# --------------------------------------------------------------------------
# La cola
# --------------------------------------------------------------------------

columnas = {
    "rank_mes": "#", "fecha": "Fecha", "cliente_id": "Cliente",
    "arrendatario": "Arrendatario", "empresa": "Empresa",
    "monto": "Monto", "score": "Score", "motivo": "Qué llamó la atención",
}
if revelar:
    columnas["_tipo_anomalia"] = "Etiqueta oculta"

tabla = del_mes[list(columnas)].rename(columns=columnas)

st.dataframe(
    tabla,
    hide_index=True,
    width="stretch",
    column_config={
        "Fecha": st.column_config.DateColumn(format="DD/MM/YYYY"),
        "Monto": st.column_config.NumberColumn(format="dollar"),
        "Score": st.column_config.ProgressColumn(
            format="%.2f", min_value=float(cola["score"].min()),
            max_value=float(cola["score"].max())),
    },
)

# --------------------------------------------------------------------------
# Detalle de un caso
# --------------------------------------------------------------------------

st.subheader("Detalle del caso")

etiquetas = {
    f"#{int(f['rank_mes'])} · {f['arrendatario']} · ${f['monto']:,.0f}": f["operacion_id"]
    for _, f in del_mes.iterrows()
}
if not etiquetas:
    st.info("No hay casos con este presupuesto.")
    st.stop()

elegido = st.selectbox("Caso", list(etiquetas), label_visibility="collapsed")
caso = del_mes[del_mes["operacion_id"] == etiquetas[elegido]].iloc[0]

izq, der = st.columns([3, 2])

with izq:
    st.markdown(f"**Cliente {int(caso['cliente_id'])}** — {caso['arrendatario']} "
                f"· empresa {caso['empresa']}")
    if pd.notna(caso.get("mediana_propia")) and caso.get("mediana_propia"):
        veces = caso["monto"] / caso["mediana_propia"]
        # Los signos de peso van escapados: Streamlit interpreta $...$ como
        # LaTeX, y sin la diagonal todo lo que queda entre los dos montos se
        # renderiza como una formula matematica.
        st.markdown(
            rf"Pagó **\${caso['monto']:,.0f}** el {caso['fecha']:%d/%m/%Y}. "
            rf"Su pago habitual es de **\${caso['mediana_propia']:,.0f}**, "
            f"así que este es **{veces:,.1f}×** lo suyo "
            f"({abs(caso['z_robusto']):.1f} desviaciones robustas)."
        )

    serie = historial[historial["cliente_id"] == caso["cliente_id"]].set_index("fecha")
    if len(serie):
        st.caption("Historial de pagos del cliente, contra su propia mediana móvil")
        st.line_chart(serie[["monto", "mediana_propia"]], height=240)

with der:
    st.caption("Qué sostiene la alerta")
    aportes = (caso.filter(like="aporte_")
               .rename(lambda s: s.replace("aporte_", ""))
               .astype(float)
               .sort_values(ascending=False)
               .head(5))
    st.bar_chart(
        pd.DataFrame({"aporte al score": aportes.to_numpy()},
                     index=[FRASES_CORTAS.get(f, f) for f in aportes.index]),
        horizontal=True, height=260,
    )
    st.caption(
        "Cada barra es cuánto cae el score si esa señal se pone en su valor "
        "normal. Es la explicación de **esta** alerta, no la importancia "
        "promedio del modelo."
    )

    if revelar:
        if caso["_es_anomalia"] == 1:
            st.success(f"Anomalía inyectada: **{caso['_tipo_anomalia'].replace('_', ' ')}**")
        else:
            st.warning("Falso positivo: operación normal que subió en el ranking.")
