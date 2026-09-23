"""
Pipeline completo de punta a punta, sin credenciales y sin red.

    python scripts/run_demo.py

Genera los datos, construye las features, corre la línea base, valida con
ventana expansiva, evalúa en las dos capas, explica cada alerta y escribe la
cola de revisión que consume la app de Streamlit.

Termina con un bloque de aserciones: si el pipeline se degrada, el script
falla en lugar de imprimir números bonitos.
"""

from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

# La consola de Windows abre en cp1252 y los acentos la revientan.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover - solo aplica en algunas terminales
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src import baseline, evaluacion, explicacion, features, generar_datos  # noqa: E402
from src.validacion_temporal import PRESUPUESTO_MENSUAL, validar  # noqa: E402

DATOS = RAIZ / "data" / "synthetic"
DOCS = RAIZ / "docs"


def titulo(texto: str) -> None:
    print(f"\n{texto}\n" + "-" * len(texto))


def main() -> int:
    DATOS.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)

    titulo("1. Datos sintéticos")
    d = generar_datos.generar()
    print(f"{len(d):,} operaciones · {d['cliente_id'].nunique()} clientes · "
          f"{d['mes'].nunique()} meses")
    print(f"anomalías inyectadas: {int(d['_es_anomalia'].sum())} "
          f"({d['_es_anomalia'].mean():.2%} de las operaciones)")
    d.to_csv(DATOS / "operaciones.csv", index=False)

    titulo("2. Features de comportamiento")
    d = features.construir(d)
    print(f"{len(features.COLUMNAS_FEATURES)} features · "
          f"{int(d['tiene_historia'].sum()):,} operaciones con historia suficiente")
    print("el modelo no ve: " + ", ".join(features.COLUMNAS_PROHIBIDAS[:6]) + ", …")

    titulo("3. Línea base por cliente (mediana + MAD)")
    d = baseline.calcular(d)
    marcadas = d.loc[d["tiene_historia"], "alerta_baseline"].sum()
    print(f"|z| > {baseline.CORTE_Z}: {int(marcadas):,} operaciones "
          f"({marcadas / d['tiene_historia'].sum():.2%})")

    titulo("4. Validación temporal (ventana expansiva)")
    r, detectores = validar(d, presupuesto_mensual=PRESUPUESTO_MENSUAL)
    print(f"{len(detectores)} meses puntuados, del {min(detectores)} al {max(detectores)}")
    print(f"cada modelo entrena solo con los meses anteriores · "
          f"{PRESUPUESTO_MENSUAL} alertas/mes de presupuesto")

    titulo("5a. Evaluación CON etiquetas (imposible en producción)")
    m = evaluacion.con_etiquetas(r)
    print(f"operaciones puntuadas      {m['operaciones_puntuadas']:,}")
    print(f"tasa base de anomalía      {m['tasa_base']:.2%}")
    print(f"alertas emitidas           {m['alertas_emitidas']:,}")
    print(f"precisión en presupuesto   {m['precision_en_presupuesto']:.1%}")
    print(f"recall en presupuesto      {m['recall_en_presupuesto']:.1%}")
    print(f"lift vs azar               {m['lift_vs_azar']:.1f}x")
    print(f"precisión de la línea base {m['precision_solo_linea_base']:.1%} "
          f"(el modelo la mejora {m['lift_vs_linea_base']:.1f}x)")

    tipos = evaluacion.recall_por_tipo(r)
    print()
    print(tipos.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    curva = evaluacion.curva_presupuesto(r)
    curva.to_csv(DOCS / "curva_presupuesto.csv", index=False)

    titulo("5b. Evaluación SIN etiquetas (lo que sí se puede medir allá)")
    s = evaluacion.sin_etiquetas(r)
    print(f"estabilidad del ranking (Jaccard mes a mes)  {s['estabilidad_ranking_jaccard']:.2f}")
    print(f"concentración en el 5% de clientes top       {s['concentracion_top5pct_clientes']:.1%}")
    print(f"clientes distintos alertados                 {s['clientes_distintos_alertados']}")
    print(f"tasa de confirmación en muestra revisada     {s['tasa_confirmacion_muestra']:.1%} "
          f"({s['muestra_revisada']} casos)")
    print(f"deriva del score mediano entre meses         {s['deriva_score_mensual']:.4f}")

    titulo("6. Explicación por alerta")
    alertas = r[r["alerta"] == 1].copy()
    motivos, contribs = [], []
    for mes, bloque in alertas.groupby("mes"):
        c = explicacion.atribuir(detectores[mes], bloque)
        motivos.append(explicacion.principales(c))
        contribs.append(c.add_prefix("aporte_"))
    alertas["motivo"] = pd.concat(motivos)
    alertas = alertas.join(pd.concat(contribs))
    print(alertas["motivo"].value_counts().head(5).to_string())

    cola = alertas.sort_values(["mes", "rank_mes"])
    cola.to_csv(DATOS / "cola_revision.csv", index=False)
    print(f"\ncola de revisión -> {DATOS / 'cola_revision.csv'} ({len(cola):,} alertas)")

    # El historial de los clientes alertados, para que la app pueda dibujar la
    # serie de pagos junto a la alerta. Solo esos clientes: el padrón completo
    # son 1.8 MB que el repo no necesita cargar, y se regenera con este script.
    historial = (d.loc[d["cliente_id"].isin(cola["cliente_id"].unique()),
                       ["cliente_id", "fecha", "monto", "mediana_propia"]]
                 .sort_values(["cliente_id", "fecha"]))
    historial.to_csv(DATOS / "historial_alertados.csv", index=False)
    print(f"historial de alertados -> {DATOS / 'historial_alertados.csv'} "
          f"({len(historial):,} pagos de {historial['cliente_id'].nunique()} clientes)")

    titulo("7. Gráficas")
    _grafica_tipos(tipos, DOCS / "recall_por_tipo.png")
    _grafica_curva(curva, DOCS / "curva_presupuesto.png")
    print(f"-> {DOCS / 'recall_por_tipo.png'}")
    print(f"-> {DOCS / 'curva_presupuesto.png'}")

    titulo("8. Aserciones")
    return _verificar(m, tipos, s)


def _grafica_tipos(tipos: pd.DataFrame, destino: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 3.6))
    t = tipos.sort_values("recall_caso")
    ax.barh(t["tipo"].str.replace("_", " "), t["recall_caso"], color="#2b6cb0")
    for y, v in enumerate(t["recall_caso"]):
        ax.text(v + 0.015, y, f"{v:.0%}", va="center", fontsize=9)
    ax.set_xlim(0, 1.1)
    ax.set_xlabel(f"recall por caso, con {PRESUPUESTO_MENSUAL} alertas al mes")
    ax.set_title("Qué tipo de anomalía atrapa el modelo")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(destino, dpi=140)
    plt.close(fig)


def _grafica_curva(curva: pd.DataFrame, destino: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(curva["alertas_por_mes"], curva["recall"], marker="o", ms=3,
            color="#2b6cb0", label="recall")
    ax.plot(curva["alertas_por_mes"], curva["precision"], marker="o", ms=3,
            color="#c05621", label="precisión")
    ax.axvline(PRESUPUESTO_MENSUAL, color="#718096", ls="--", lw=1)
    ax.annotate(f"presupuesto actual\n({PRESUPUESTO_MENSUAL}/mes)",
                xy=(PRESUPUESTO_MENSUAL, 0.92), xytext=(PRESUPUESTO_MENSUAL + 8, 0.92),
                fontsize=9, color="#4a5568", va="top")
    ax.set_xlabel("alertas que el equipo puede revisar al mes")
    ax.set_ylabel("proporción")
    ax.set_ylim(0, 1.0)
    ax.set_title("Detección en función del presupuesto de revisión")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(destino, dpi=140)
    plt.close(fig)


def _verificar(m: dict, tipos: pd.DataFrame, s: dict) -> int:
    recalls = tipos.set_index("tipo")["recall_caso"].to_dict()
    pruebas = [
        ("la precisión supera al azar por 10x o más",
         m["lift_vs_azar"] >= 10),
        ("el modelo supera a la línea base univariada",
         m["precision_en_presupuesto"] > m["precision_solo_linea_base"]),
        # El umbral mas bajo de los cinco, y a proposito: fraccionar es
        # precisamente la tecnica disenada para que ningun pago individual
        # llame la atencion. Ver el README.
        ("atrapa el fraccionamiento (recall por caso >= 60%)",
         recalls.get("fraccionamiento", 0) >= 0.60),
        ("atrapa el salto de escala (recall por caso >= 90%)",
         recalls.get("salto_de_escala", 0) >= 0.90),
        ("atrapa la ráfaga de montos redondos (recall por caso >= 90%)",
         recalls.get("rafaga_montos_redondos", 0) >= 0.90),
        ("atrapa la contraparte inusual (recall por caso >= 70%)",
         recalls.get("contraparte_inusual", 0) >= 0.70),
        ("las alertas no se concentran en pocos clientes (<= 25%)",
         s["concentracion_top5pct_clientes"] <= 0.25),
        ("el ranking no se queda pegado mes a mes (Jaccard <= 0.35)",
         s["estabilidad_ranking_jaccard"] <= 0.35),
    ]
    fallos = 0
    for descripcion, ok in pruebas:
        print(f"  [{'OK ' if ok else 'FALLA'}] {descripcion}")
        fallos += 0 if ok else 1
    print(f"\n{len(pruebas) - fallos}/{len(pruebas)} aserciones pasan")
    return 0 if fallos == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
