import pandas as pd
import streamlit as st
import altair as alt
import json
import os
from pathlib import Path
from services.financial_data import (
    obtener_datos_empresa,
    obtener_historico,
    calcular_variacion_precio,
    obtener_historico_fundamental,
    obtener_dividendos_historicos,
    buscar_empresas,
    buscar_activos,
    METRICAS_INFO,
)
from services.analysis_engine import analizar_empresa

# ============================================================
# FINSCOPE · NAVEGACION DESPLEGABLE V2
# Bloques principales organizados mediante st.expander.
# Todos cerrados por defecto.
# ============================================================

from services.investment_analysis import calcular_rendimientos_historicos, evolucion_inversion, calcular_benchmark, calcular_metricas_riesgo, calcular_beta_correlacion, calcular_rentabilidad_ajustada_riesgo, calcular_per_historico, calcular_dcf, calcular_estadistica_rendimientos, calcular_riesgo_cuantitativo, calcular_analisis_temporal, calcular_monte_carlo, calcular_escenarios_sensibilidad, obtener_serie_rendimiento, calcular_comparacion_quant_pro
from services.investment_analysis import calcular_cartera_quant


# =====================================================================
# FINSCOPE · CAPA DE CACHE DE DATOS EXTERNOS
# =====================================================================
#
# Mantiene los servicios independientes de Streamlit.
# Solo se cachean consultas externas/repetibles.
# Cartera, watchlist y archivos locales NO pasan por esta capa.
#
# TTL:
#   5 min  -> cotización / históricos de mercado
#   15 min -> series analíticas
#   1 h    -> fundamentales, dividendos y PER
# =====================================================================

@st.cache_data(ttl=300, show_spinner=False)
def cache_obtener_datos_empresa(ticker):
    return obtener_datos_empresa(ticker)


@st.cache_data(ttl=3600, show_spinner=False)
def cache_obtener_historico_fundamental(ticker):
    return obtener_historico_fundamental(ticker)


@st.cache_data(ttl=300, show_spinner=False)
def cache_obtener_historico(ticker, periodo="1mo"):
    return obtener_historico(ticker, periodo)


@st.cache_data(ttl=300, show_spinner=False)
def cache_calcular_variacion_precio(ticker, periodo):
    return calcular_variacion_precio(ticker, periodo)


@st.cache_data(ttl=3600, show_spinner=False)
def cache_obtener_dividendos_historicos(ticker, anios=10):
    return obtener_dividendos_historicos(
        ticker,
        anios=anios,
    )


@st.cache_data(ttl=900, show_spinner=False)
def cache_obtener_serie_rendimiento(
    ticker,
    periodo_descarga="max",
):
    return obtener_serie_rendimiento(
        ticker,
        periodo_descarga,
    )


@st.cache_data(ttl=3600, show_spinner=False)
def cache_calcular_per_historico(
    ticker,
    periodo="5y",
):
    return calcular_per_historico(
        ticker,
        periodo,
    )


# =====================================================================
# FIN CAPA CACHE
# =====================================================================


WATCHLIST_FILE = Path("data/watchlist.json")
PORTFOLIO_FILE = Path("data/portfolio.json")


PORTFOLIO_TRANSACTIONS_FILE = Path(
    "data/portfolio_transactions.json"
)


# =====================================================================
# FINSCOPE · CAPA DE ALMACENAMIENTO
# local   -> persistencia JSON en este equipo
# session -> almacenamiento privado por sesión para despliegue web
# =====================================================================

FINSCOPE_STORAGE_MODE = os.environ.get(
    "FINSCOPE_STORAGE_MODE",
    "local",
).strip().lower()

if FINSCOPE_STORAGE_MODE not in {"local", "session"}:
    FINSCOPE_STORAGE_MODE = "local"


def _storage_session():
    return FINSCOPE_STORAGE_MODE == "session"


def _session_get(key):
    valor = st.session_state.get(key, [])
    return list(valor) if isinstance(valor, list) else []


def _session_set(key, valor):
    st.session_state[key] = list(valor)




def cargar_operaciones_cartera():
    if _storage_session():
        return _session_get("finscope_portfolio_transactions")

    try:
        if not PORTFOLIO_TRANSACTIONS_FILE.exists():
            return []

        datos = json.loads(
            PORTFOLIO_TRANSACTIONS_FILE.read_text(
                encoding="utf-8"
            )
        )

        if not isinstance(datos, list):
            return []

        salida = []

        for op in datos:
            if not isinstance(op, dict):
                continue

            ticker = str(
                op.get("ticker") or ""
            ).strip().upper()

            tipo_operacion = str(
                op.get("operacion") or ""
            ).strip().upper()

            try:
                cantidad = float(op.get("cantidad"))
                precio = float(op.get("precio_eur"))
            except (TypeError, ValueError):
                continue

            if (
                not ticker
                or tipo_operacion not in {"COMPRA", "VENTA"}
                or cantidad <= 0
                or precio <= 0
            ):
                continue

            salida.append({
                "id": str(
                    op.get("id")
                    or (
                        ticker
                        + "_"
                        + str(len(salida))
                    )
                ),
                "fecha": str(
                    op.get("fecha") or ""
                ),
                "ticker": ticker,
                "nombre": str(
                    op.get("nombre") or ticker
                ),
                "tipo": str(
                    op.get("tipo") or "Activo"
                ),
                "operacion": tipo_operacion,
                "cantidad": cantidad,
                "precio_eur": precio,
            })

        return salida

    except Exception:
        return []


def guardar_operaciones_cartera(operaciones):
    if _storage_session():
        _session_set(
            "finscope_portfolio_transactions",
            operaciones,
        )
        return

    PORTFOLIO_TRANSACTIONS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporal = PORTFOLIO_TRANSACTIONS_FILE.with_suffix(
        ".json.tmp"
    )

    temporal.write_text(
        json.dumps(
            operaciones,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    temporal.replace(
        PORTFOLIO_TRANSACTIONS_FILE
    )


def registrar_operacion_cartera(
    ticker,
    nombre,
    tipo,
    operacion,
    cantidad,
    precio_eur,
):
    operaciones = cargar_operaciones_cartera()

    ticker = str(ticker).strip().upper()
    operacion = str(operacion).strip().upper()
    cantidad = float(cantidad)
    precio_eur = float(precio_eur)

    if operacion not in {"COMPRA", "VENTA"}:
        raise ValueError("Operación no válida.")

    if cantidad <= 0 or precio_eur <= 0:
        raise ValueError(
            "Cantidad y precio deben ser positivos."
        )

    if operacion == "VENTA":
        estado = reconstruir_cartera_desde_operaciones(
            operaciones
        )

        disponible = float(
            estado.get(
                ticker,
                {},
            ).get(
                "cantidad",
                0.0,
            )
        )

        if cantidad > disponible + 1e-12:
            raise ValueError(
                "No puedes vender más unidades "
                "de las que tienes."
            )

    fecha = pd.Timestamp.now().isoformat(
        timespec="seconds"
    )

    identificador = (
        ticker
        + "_"
        + str(pd.Timestamp.now().value)
    )

    operaciones.append({
        "id": identificador,
        "fecha": fecha,
        "ticker": ticker,
        "nombre": str(nombre or ticker),
        "tipo": str(tipo or "Activo"),
        "operacion": operacion,
        "cantidad": cantidad,
        "precio_eur": precio_eur,
    })

    guardar_operaciones_cartera(
        operaciones
    )

    return operaciones[-1]


def reconstruir_cartera_desde_operaciones(
    operaciones=None,
):
    if operaciones is None:
        operaciones = cargar_operaciones_cartera()

    estado = {}

    for op in operaciones:
        ticker = op["ticker"]
        cantidad = float(op["cantidad"])
        precio = float(op["precio_eur"])
        operacion = op["operacion"]

        if ticker not in estado:
            estado[ticker] = {
                "ticker": ticker,
                "nombre": op.get(
                    "nombre"
                ) or ticker,
                "tipo": op.get(
                    "tipo"
                ) or "Activo",
                "cantidad": 0.0,
                "precio_medio_eur": 0.0,
                "coste_actual_eur": 0.0,
                "pl_realizado_eur": 0.0,
            }

        posicion = estado[ticker]

        if operacion == "COMPRA":
            coste_compra = (
                cantidad * precio
            )

            posicion[
                "coste_actual_eur"
            ] += coste_compra

            posicion[
                "cantidad"
            ] += cantidad

            if posicion["cantidad"] > 0:
                posicion[
                    "precio_medio_eur"
                ] = (
                    posicion[
                        "coste_actual_eur"
                    ]
                    / posicion[
                        "cantidad"
                    ]
                )

        elif operacion == "VENTA":
            disponible = float(
                posicion["cantidad"]
            )

            if cantidad > disponible + 1e-12:
                raise ValueError(
                    "Historial inválido: venta "
                    "superior a la posición."
                )

            precio_medio = float(
                posicion[
                    "precio_medio_eur"
                ]
            )

            posicion[
                "pl_realizado_eur"
            ] += (
                precio
                - precio_medio
            ) * cantidad

            posicion[
                "cantidad"
            ] -= cantidad

            posicion[
                "coste_actual_eur"
            ] -= (
                precio_medio
                * cantidad
            )

            if posicion["cantidad"] <= 1e-12:
                posicion["cantidad"] = 0.0
                posicion["coste_actual_eur"] = 0.0
                posicion["precio_medio_eur"] = 0.0

    return estado


def sincronizar_cartera_desde_operaciones():
    estado = reconstruir_cartera_desde_operaciones()

    cartera = []

    for posicion in estado.values():
        if posicion["cantidad"] <= 1e-12:
            continue

        cartera.append({
            "ticker":
                posicion["ticker"],

            "nombre":
                posicion["nombre"],

            "tipo":
                posicion["tipo"],

            "cantidad":
                posicion["cantidad"],

            "precio_medio_eur":
                posicion["precio_medio_eur"],
        })

    guardar_cartera_personal(
        cartera
    )

    return cartera


def migrar_cartera_v2_a_operaciones():
    if _storage_session():
        if cargar_operaciones_cartera():
            return False
    elif PORTFOLIO_TRANSACTIONS_FILE.exists():
        return False

    cartera = cargar_cartera_personal()

    operaciones = []

    for posicion in cartera:
        precio = posicion.get(
            "precio_medio_eur"
        )

        if precio is None:
            continue

        try:
            precio = float(precio)
            cantidad = float(
                posicion["cantidad"]
            )
        except (TypeError, ValueError):
            continue

        if precio <= 0 or cantidad <= 0:
            continue

        operaciones.append({
            "id":
                "MIGRACION_"
                + posicion["ticker"],

            "fecha":
                "Migración V2",

            "ticker":
                posicion["ticker"],

            "nombre":
                posicion.get("nombre")
                or posicion["ticker"],

            "tipo":
                posicion.get("tipo")
                or "Activo",

            "operacion":
                "COMPRA",

            "cantidad":
                cantidad,

            "precio_eur":
                precio,
        })

    guardar_operaciones_cartera(
        operaciones
    )

    return True


def calcular_pl_realizado_total():
    estado = reconstruir_cartera_desde_operaciones()

    return sum(
        float(
            posicion.get(
                "pl_realizado_eur",
                0.0,
            )
        )
        for posicion in estado.values()
    )




def cargar_cartera_personal():
    """
    Carga posiciones reales.

    V2 añade precio_medio_eur. Las posiciones antiguas de V1
    siguen siendo compatibles y tendrán precio medio pendiente.
    """
    if _storage_session():
        return _session_get("finscope_portfolio")

    try:
        if PORTFOLIO_FILE.exists():
            with open(
                PORTFOLIO_FILE,
                "r",
                encoding="utf-8",
            ) as f:
                datos = json.load(f)

            if not isinstance(datos, list):
                return []

            salida = []

            for p in datos:
                if not isinstance(p, dict):
                    continue

                ticker = str(
                    p.get("ticker") or ""
                ).strip().upper()

                try:
                    cantidad = float(
                        p.get("cantidad")
                    )
                except (TypeError, ValueError):
                    continue

                if not ticker or cantidad <= 0:
                    continue

                precio_medio = p.get(
                    "precio_medio_eur"
                )

                try:
                    if precio_medio is not None:
                        precio_medio = float(
                            precio_medio
                        )

                        if precio_medio <= 0:
                            precio_medio = None

                except (TypeError, ValueError):
                    precio_medio = None

                salida.append({
                    "ticker": ticker,
                    "nombre": str(
                        p.get("nombre") or ticker
                    ),
                    "tipo": str(
                        p.get("tipo") or "Activo"
                    ),
                    "cantidad": cantidad,
                    "precio_medio_eur": precio_medio,
                })

            return salida

    except Exception:
        pass

    return []


def guardar_cartera_personal(cartera):
    if _storage_session():
        _session_set("finscope_portfolio", cartera)
        return

    PORTFOLIO_FILE.parent.mkdir(parents=True, exist_ok=True)

    temporal = PORTFOLIO_FILE.with_suffix(".json.tmp")

    with open(temporal, "w", encoding="utf-8") as f:
        json.dump(
            cartera,
            f,
            ensure_ascii=False,
            indent=2,
        )

    temporal.replace(PORTFOLIO_FILE)


def resolver_activo_cartera(consulta):
    consulta = str(consulta or "").strip()

    if not consulta:
        return None

    resultados = buscar_activos(
        consulta,
        max_resultados=8,
    )

    if not resultados:
        return None

    consulta_upper = consulta.upper()

    exacto = next(
        (
            activo
            for activo in resultados
            if str(activo.get("ticker") or "").upper()
            == consulta_upper
        ),
        None,
    )

    return exacto or resultados[0]


def obtener_cotizacion_cartera_eur(ticker):
    try:
        serie = cache_obtener_serie_rendimiento(
            ticker,
            "1mo",
        )

        if serie is None or serie.empty:
            return None

        if "Close" in serie.columns:
            precios = serie["Close"]
        elif "Adj Close" in serie.columns:
            precios = serie["Adj Close"]
        else:
            return None

        precios = pd.to_numeric(
            precios,
            errors="coerce",
        ).dropna()

        if precios.empty:
            return None

        precio = float(precios.iloc[-1])

        if not pd.notna(precio) or precio <= 0:
            return None

        return {
            "precio_eur": precio,
            "fecha": precios.index[-1],
            "moneda": "EUR",
            "moneda_original": serie.attrs.get(
                "moneda_original"
            ),
        }

    except Exception:
        return None


def construir_resumen_cartera_personal(cartera):
    """
    Valora las posiciones en EUR y calcula coste y P/L cuando
    existe precio medio de compra.
    """
    filas = []
    errores = []

    for posicion in cartera:
        ticker = posicion["ticker"]
        cantidad = float(
            posicion["cantidad"]
        )

        cotizacion = (
            obtener_cotizacion_cartera_eur(
                ticker
            )
        )

        if not cotizacion:
            errores.append(ticker)
            continue

        precio_actual = float(
            cotizacion["precio_eur"]
        )

        valor_actual = (
            cantidad * precio_actual
        )

        precio_medio = posicion.get(
            "precio_medio_eur"
        )

        capital_invertido = None
        beneficio_eur = None
        rentabilidad_pct = None

        try:
            if precio_medio is not None:
                precio_medio = float(
                    precio_medio
                )

                if precio_medio > 0:
                    capital_invertido = (
                        cantidad * precio_medio
                    )

                    beneficio_eur = (
                        valor_actual
                        - capital_invertido
                    )

                    if capital_invertido > 0:
                        rentabilidad_pct = (
                            beneficio_eur
                            / capital_invertido
                            * 100.0
                        )

        except (TypeError, ValueError):
            precio_medio = None

        filas.append({
            "Activo":
                posicion.get("nombre")
                or ticker,

            "Ticker":
                ticker,

            "Tipo":
                posicion.get("tipo")
                or "Activo",

            "Cantidad":
                cantidad,

            "Precio medio EUR":
                precio_medio,

            "Precio EUR":
                precio_actual,

            "Invertido EUR":
                capital_invertido,

            "Valor EUR":
                valor_actual,

            "P/L EUR":
                beneficio_eur,

            "Rentabilidad %":
                rentabilidad_pct,

            "Fecha":
                cotizacion.get("fecha"),
        })

    if not filas:
        return pd.DataFrame(), errores

    df = pd.DataFrame(filas)

    total_actual = float(
        df["Valor EUR"].sum()
    )

    if total_actual > 0:
        df["Peso %"] = (
            df["Valor EUR"]
            / total_actual
            * 100.0
        )
    else:
        df["Peso %"] = 0.0

    return df, errores



def cargar_watchlist():
    if _storage_session():
        return _session_get("finscope_watchlist")

    try:
        if WATCHLIST_FILE.exists():
            with open(
                WATCHLIST_FILE,
                "r",
                encoding="utf-8"
            ) as f:
                datos = json.load(f)

            if isinstance(datos, list):
                return datos

    except Exception:
        pass

    return []


def guardar_watchlist(watchlist):
    if _storage_session():
        _session_set("finscope_watchlist", watchlist)
        return

    WATCHLIST_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        WATCHLIST_FILE,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            watchlist,
            f,
            ensure_ascii=False,
            indent=2
        )





def resolver_busqueda_empresa(
    consulta,
    key_resultados,
    contenedor=None,
):
    """
    Convierte una búsqueda por nombre o ticker en un ticker
    utilizable por FinScope.

    Si existe una coincidencia exacta de ticker, la utiliza
    directamente. Si hay varias empresas posibles, muestra
    un selector para que el usuario elija.
    """

    consulta = str(consulta or "").strip()

    if not consulta:
        return ""

    resultados = buscar_empresas(
        consulta,
        max_resultados=6,
    )

    if not resultados:
        return ""

    consulta_normalizada = consulta.upper()

    coincidencia_ticker = next(
        (
            empresa
            for empresa in resultados
            if empresa["ticker"].upper()
            == consulta_normalizada
        ),
        None,
    )

    if coincidencia_ticker:
        return coincidencia_ticker["ticker"]

    if len(resultados) == 1:
        return resultados[0]["ticker"]

    opciones = {
        (
            f'{empresa["nombre"]} · '
            f'{empresa["ticker"]} · '
            f'{empresa["exchange"]}'
        ): empresa["ticker"]
        for empresa in resultados
    }

    destino = contenedor if contenedor is not None else st

    seleccion = destino.selectbox(
        "Coincidencias",
        options=list(opciones.keys()),
        index=0,
        key=key_resultados,
        label_visibility="collapsed",
    )

    return opciones[seleccion]


def interpretar_empresa(datos):
    resultado = []

    # ─────────────────────────────
    # PER
    # ─────────────────────────────

    per = datos.get("per")

    if per is not None:
        resultado.append((
            "Valoración · PER",
            "neutral",
            (
                f"El PER registrado es {per:.2f}x. "
                "Esto indica cuántas veces el beneficio por acción "
                "está pagando actualmente el mercado. "
                "Por sí solo no permite determinar si la acción está "
                "barata o cara: debe compararse con empresas similares, "
                "su historial, crecimiento y situación financiera."
            )
        ))

    # ─────────────────────────────
    # MARGEN OPERATIVO
    # ─────────────────────────────

    margen = datos.get("margen_operativo")

    if margen is not None:
        valor = margen * 100

        resultado.append((
            "Margen operativo",
            "neutral",
            (
                f"El margen operativo registrado es {valor:.2f}%. "
                "Representa aproximadamente qué porcentaje de los "
                "ingresos permanece como resultado operativo antes "
                "de intereses e impuestos. "
                "Su interpretación depende especialmente del sector "
                "y del modelo de negocio."
            )
        ))

    # ─────────────────────────────
    # ROE
    # ─────────────────────────────

    roe = datos.get("roe")

    if roe is not None:
        valor = roe * 100

        resultado.append((
            "ROE",
            "neutral",
            (
                f"El ROE registrado es {valor:.2f}%. "
                "Relaciona el beneficio con el patrimonio de los "
                "accionistas. Un ROE elevado no implica por sí mismo "
                "una mayor calidad empresarial, ya que puede verse "
                "afectado por el nivel de deuda, recompras de acciones "
                "o un patrimonio contable reducido."
            )
        ))

    # ─────────────────────────────
    # CRECIMIENTO DE INGRESOS
    # ─────────────────────────────

    crecimiento = datos.get("crecimiento_ingresos")

    if crecimiento is not None:
        valor = crecimiento * 100

        if valor >= 0:
            direccion = "crecimiento"
        else:
            direccion = "descenso"

        resultado.append((
            "Ingresos TTM",
            "neutral",
            (
                f"La variación de ingresos registrada es {valor:+.2f}%, "
                f"lo que representa un {direccion} respecto al periodo "
                "de comparación utilizado por la fuente de datos. "
                "Esta cifra no demuestra por sí sola una tendencia "
                "estructural; conviene observar varios ejercicios."
            )
        ))

    # ─────────────────────────────
    # FREE CASH FLOW
    # ─────────────────────────────

    fcf = datos.get("free_cash_flow_ttm")

    if fcf is not None:

        if fcf >= 1_000_000_000:
            fcf_texto = f"{fcf / 1_000_000_000:,.2f} B"
        elif fcf >= 1_000_000:
            fcf_texto = f"{fcf / 1_000_000:,.2f} M"
        elif fcf <= -1_000_000_000:
            fcf_texto = f"{fcf / 1_000_000_000:,.2f} B"
        elif fcf <= -1_000_000:
            fcf_texto = f"{fcf / 1_000_000:,.2f} M"
        else:
            fcf_texto = f"{fcf:,.0f}"

        resultado.append((
            "Free Cash Flow TTM",
            "neutral",
            (
                f"El Free Cash Flow registrado es {fcf_texto}. "
                "Representa la caja disponible después de determinadas "
                "necesidades de inversión del negocio. "
                "Que sea positivo o negativo en un único periodo no "
                "permite concluir por sí solo que la situación "
                "financiera de la empresa sea buena o mala."
            )
        ))

    return resultado


def formato_numero(valor, tipo="numero"):
    if valor is None:
        return "—"

    if tipo == "porcentaje":
        return f"{valor * 100:.2f}%"

    if tipo == "multiplo":
        return f"{valor:.2f}x"

    if tipo == "billones":
        return f"{valor / 1_000_000_000:,.2f} B"

    return f"{valor:,.2f}"



st.set_page_config(
    page_title="FinScope",
    page_icon="📈",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: #0b0f17;
}

.block-container {
    max-width: 1280px;
    padding-top: 3rem;
    padding-bottom: 5rem;
}

h1, h2, h3 {
    letter-spacing: -0.03em;
}

h1 {
    font-size: 3rem !important;
}

div[data-testid="stTextInput"] input {
    background: #111827;
    border: 1px solid #263244;
    border-radius: 12px;
    min-height: 48px;
}

div[data-testid="stMetric"] {
    background: #111827;
    border: 1px solid #202b3c;
    border-radius: 14px;
    padding: 18px 20px;
    min-height: 112px;
}

div[data-testid="stMetricLabel"] {
    opacity: 0.72;
}

div[data-testid="stMetricValue"] {
    font-size: 1.8rem;
    letter-spacing: -0.03em;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# FINSCOPE · PORTADA PROFESIONAL V2
# ============================================================

st.markdown(
"""<div class="fs-home-v2">
<div class="fs-eyebrow"><span></span>FINSCOPE · FINANCIAL INTELLIGENCE</div>

<div class="fs-hero-top">

<div class="fs-hero-copy">
<h1>Analiza mejor.<br><em>Decide con más información.</em></h1>
<p>Datos de mercado, fundamentales, valoración, riesgo y análisis cuantitativo en una sola plataforma.</p>
</div>

</div>

<div class="fs-hero-divider"></div>

<div class="fs-hero-grid">

<div class="fs-hero-item">
<small>01</small>
<div>
<strong>Analiza</strong>
<span>Mercado, fundamentales, valoración y riesgo.</span>
</div>
</div>

<div class="fs-hero-item">
<small>02</small>
<div>
<strong>Compara</strong>
<span>Contrasta activos bajo un mismo marco.</span>
</div>
</div>

<div class="fs-hero-item">
<small>03</small>
<div>
<strong>Profundiza</strong>
<span>Quant, Monte Carlo, cartera y Machine Learning.</span>
</div>
</div>

</div>
</div>""",
unsafe_allow_html=True,
)

st.markdown("""
<style>
.company-hero {
    margin-top: 1.25rem;
    margin-bottom: 1rem;
    padding: 22px 24px;
    background: #111827;
    border: 1px solid #202b3c;
    border-radius: 16px;
}

.company-hero h2 {
    margin: 0 0 6px 0 !important;
    font-size: 2rem !important;
}

.company-hero p {
    margin: 0;
    opacity: 0.70;
}

div[data-testid="stMetric"] {
    min-height: 96px !important;
    padding: 14px 18px !important;
}

div[data-testid="stHorizontalBlock"] {
    gap: 1rem;
}

div[data-testid="stVegaLiteChart"],
div[data-testid="stArrowVegaLiteChart"] {
    background: #0e1522;
    border: 1px solid #202b3c;
    border-radius: 14px;
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)


st.markdown("""
<style>

.finscope-home {
    position: relative;
    overflow: hidden;

    margin: 1rem 0 2rem 0;
    padding: 64px 60px 52px;

    border: 1px solid #253249;
    border-radius: 24px;

    background:
        radial-gradient(
            circle at 84% 20%,
            rgba(59, 130, 246, 0.20),
            transparent 30%
        ),
        radial-gradient(
            circle at 70% 90%,
            rgba(20, 184, 166, 0.08),
            transparent 30%
        ),
        linear-gradient(
            135deg,
            #121b2b 0%,
            #0e1623 55%,
            #0a1019 100%
        );

    box-shadow:
        0 25px 70px rgba(0, 0, 0, 0.30);
}


.home-badge {
    display: flex;
    align-items: center;
    gap: 10px;

    margin-bottom: 24px;

    color: #9db6d8;

    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.15em;
}


.home-badge span {
    width: 8px;
    height: 8px;

    border-radius: 50%;

    background: #60a5fa;

    box-shadow:
        0 0 16px rgba(96, 165, 250, 0.85);
}


.finscope-home h1 {
    margin: 0 !important;

    max-width: 850px;

    color: #f8fafc;

    font-size: clamp(
        3.5rem,
        6vw,
        5.8rem
    ) !important;

    line-height: 0.96 !important;

    letter-spacing:
        -0.065em !important;

    font-weight: 750 !important;
}


.finscope-home h1 span {
    background:
        linear-gradient(
            90deg,
            #78b7ff,
            #8bd8d1
        );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;

    background-clip: text;
}


.home-lead {
    max-width: 720px;

    margin:
        28px 0 0 0 !important;

    color: #a9b6c9 !important;

    font-size: 1.10rem !important;

    line-height: 1.7 !important;
}


.home-features {
    display: grid;

    grid-template-columns:
        repeat(3, minmax(0, 1fr));

    gap: 14px;

    max-width: 900px;

    margin-top: 40px;
}


.home-features > div {
    padding: 17px 18px;

    border:
        1px solid
        rgba(148, 163, 184, 0.15);

    border-radius: 14px;

    background:
        rgba(255, 255, 255, 0.025);
}


.home-features strong {
    display: block;

    margin-bottom: 5px;

    color: #edf3fb;

    font-size: 0.92rem;
}


.home-features small {
    display: block;

    color: #7f8da2;

    font-size: 0.77rem;
}


/* BUSCADOR */

div[data-testid="stTextInput"] {
    max-width: 1000px;

    margin:
        0 auto 1rem auto;
}


div[data-testid="stTextInput"] label p {
    color: #a9b6c9;

    font-size: 0.80rem;

    font-weight: 600;
}


div[data-testid="stTextInput"] input {
    min-height: 58px;

    padding-left: 18px;

    background: #101827;

    border: 1px solid #29364a;

    border-radius: 15px;

    font-size: 1rem;

    box-shadow:
        0 10px 30px
        rgba(0, 0, 0, 0.18);
}


div[data-testid="stTextInput"] input:focus {
    border-color: #5d9ee8;

    box-shadow:
        0 0 0 1px #5d9ee8,
        0 12px 32px
        rgba(0, 0, 0, 0.22);
}


@media (max-width: 800px) {

    .finscope-home {
        padding:
            40px 28px 36px;
    }

    .home-features {
        grid-template-columns: 1fr;
    }
}

</style>
""", unsafe_allow_html=True)


st.markdown("""
<style>

.section-kicker {
    margin-top: 1.8rem;
    margin-bottom: -0.35rem;
    color: #6f83a1;
    font-size: 0.70rem;
    font-weight: 700;
    letter-spacing: 0.15em;
}

.quote-summary {
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    gap: 24px;

    margin: 0.4rem 0 1.2rem 0;
    padding: 24px 26px;

    border: 1px solid #202b3c;
    border-radius: 16px;

    background:
        linear-gradient(
            135deg,
            rgba(17,24,39,.97),
            rgba(13,21,34,.97)
        );
}

.quote-label {
    display: block;
    margin-bottom: 8px;

    color: #8392a8;

    font-size: 0.78rem;
    font-weight: 600;
}

.quote-price {
    color: #f6f8fb;

    font-size: 2.7rem;
    font-weight: 700;

    line-height: 1;
    letter-spacing: -0.045em;
}

.quote-price small {
    color: #8795aa;

    font-size: 0.9rem;
    font-weight: 600;

    letter-spacing: 0;
}

.quote-change {
    margin-top: 11px;

    font-size: 0.94rem;
    font-weight: 650;
}

.quote-change.positive {
    color: #55c59a;
}

.quote-change.negative {
    color: #ef7777;
}

.quote-change span {
    margin-left: 5px;

    color: #718096;

    font-size: 0.76rem;
    font-weight: 500;
}

.quote-period-card {
    min-width: 190px;

    padding: 14px 17px;

    border:
        1px solid
        rgba(148,163,184,.13);

    border-radius: 12px;

    background:
        rgba(255,255,255,.025);
}

.quote-period-card span,
.quote-period-card small {
    display: block;

    color: #718096;

    font-size: 0.70rem;
}

.quote-period-card strong {
    display: block;

    margin: 4px 0;

    color: #eaf0f8;

    font-size: 1.2rem;
}

div[data-testid="stSegmentedControl"] {
    margin-bottom: 1rem;
}

@media (max-width: 760px) {

    .quote-summary {
        flex-direction: column;
        align-items: stretch;
    }

    .quote-period-card {
        min-width: 0;
    }
}

</style>
""", unsafe_allow_html=True)


st.markdown("""
<style>

.subsection-space {
    margin-top: 2.8rem;
}

</style>
""", unsafe_allow_html=True)


st.markdown("""
<style>

.analysis-section {
    margin-top: 2.8rem;
}

.analysis-caption {
    color: #718096 !important;
    font-size: 0.82rem !important;
    margin-top: -0.25rem !important;
    margin-bottom: 1.1rem !important;
}

.mini-title {
    margin-top: 2rem;
    margin-bottom: 0.8rem;

    color: #aeb9c9;

    font-size: 0.82rem;
    font-weight: 650;

    letter-spacing: 0.02em;
}

</style>
""", unsafe_allow_html=True)


st.markdown("""
<style>

/* SIDEBAR */

section[data-testid="stSidebar"] {
    background: #0d1420;
    border-right: 1px solid #1d2939;
}

section[data-testid="stSidebar"] > div {
    padding-top: 1.4rem;
}

.sidebar-brand {
    padding: 8px 4px 26px 4px;
}

.sidebar-brand-name {
    color: #f3f6fb;
    font-size: 1.35rem;
    font-weight: 750;
    letter-spacing: -0.04em;
}

.sidebar-brand-name span {
    color: #78b7ff;
}

.sidebar-brand-sub {
    margin-top: 4px;
    color: #64748b;
    font-size: 0.72rem;
}

.sidebar-label {
    margin: 10px 0 8px 0;
    color: #617087;
    font-size: 0.67rem;
    font-weight: 700;
    letter-spacing: 0.14em;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# FINSCOPE · DESIGN SYSTEM V1
# =========================================================

st.markdown("""
<style>

/* FINSCOPE DESIGN SYSTEM V1 */

:root {
    --fs-bg: #080d14;
    --fs-bg-soft: #0b111b;
    --fs-sidebar: #090f18;

    --fs-surface: #0f1724;
    --fs-surface-2: #121c2b;
    --fs-surface-3: #162234;

    --fs-border: rgba(148, 163, 184, 0.13);
    --fs-border-strong: rgba(148, 163, 184, 0.20);

    --fs-text: #f4f7fb;
    --fs-text-2: #b6c1d1;
    --fs-muted: #718096;
    --fs-muted-2: #526176;

    --fs-blue: #69aaf7;
    --fs-blue-soft: rgba(105, 170, 247, 0.12);

    --fs-green: #4fc49a;
    --fs-red: #ef7777;

    --fs-radius-sm: 10px;
    --fs-radius: 14px;
    --fs-radius-lg: 20px;
}


/* ======================================================
   APP
   ====================================================== */

html,
body,
[class*="css"] {
    font-family:
        Inter,
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        sans-serif;
}

.stApp {
    background:
        radial-gradient(
            circle at 70% -10%,
            rgba(59, 130, 246, 0.055),
            transparent 25rem
        ),
        var(--fs-bg);
    color: var(--fs-text);
}

.block-container {
    max-width: 1320px;
    padding-top: 2.4rem;
    padding-bottom: 6rem;
    padding-left: 2.6rem;
    padding-right: 2.6rem;
}


/* Ocultar elementos Streamlit que restan aspecto de producto */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header[data-testid="stHeader"] {
    background: transparent;
}


/* ======================================================
   TIPOGRAFÍA
   ====================================================== */

h1,
h2,
h3,
h4 {
    color: var(--fs-text);
    letter-spacing: -0.035em;
}

h1 {
    font-size: clamp(
        2.3rem,
        4vw,
        3.35rem
    ) !important;

    line-height: 1.06 !important;
    font-weight: 720 !important;
}

h2 {
    font-size: 1.72rem !important;
    font-weight: 680 !important;
}

h3 {
    font-size: 1.18rem !important;
    font-weight: 650 !important;
}

p {
    color: var(--fs-text-2);
}


/* ======================================================
   PORTADA
   ====================================================== */

.finscope-home {
    position: relative;
    overflow: hidden;

    margin:
        0.35rem 0
        2rem 0;

    padding:
        72px 64px
        58px;

    border:
        1px solid
        var(--fs-border);

    border-radius: 26px;

    background:
        radial-gradient(
            circle at 86% 12%,
            rgba(72, 135, 235, 0.20),
            transparent 28%
        ),
        radial-gradient(
            circle at 68% 105%,
            rgba(56, 189, 178, 0.075),
            transparent 28%
        ),
        linear-gradient(
            135deg,
            #111b2a 0%,
            #0d1521 48%,
            #090f17 100%
        );

    box-shadow:
        0 24px 80px
        rgba(0, 0, 0, 0.26);
}

.finscope-home::after {
    content: "";

    position: absolute;
    width: 390px;
    height: 390px;
    right: -140px;
    top: -170px;

    border:
        1px solid
        rgba(105, 170, 247, 0.10);

    border-radius: 50%;

    box-shadow:
        0 0 0 55px
            rgba(105,170,247,.018),
        0 0 0 110px
            rgba(105,170,247,.012);

    pointer-events: none;
}

.home-badge {
    display: inline-flex;
    align-items: center;
    gap: 9px;

    margin-bottom: 25px;

    color: #91a9c8;

    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.17em;
}

.home-badge span {
    width: 7px;
    height: 7px;

    border-radius: 50%;
    background: var(--fs-blue);

    box-shadow:
        0 0 14px
        rgba(105,170,247,.75);
}

.finscope-home h1 {
    max-width: 850px;

    margin: 0 !important;

    font-size:
        clamp(
            3.6rem,
            6vw,
            5.9rem
        ) !important;

    line-height: 0.94 !important;

    font-weight: 740 !important;

    letter-spacing:
        -0.067em !important;
}

.finscope-home h1 span {
    background:
        linear-gradient(
            90deg,
            #6daeff,
            #8ed9d3
        );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.home-lead {
    max-width: 680px;

    margin:
        30px 0 0 0 !important;

    color:
        #a9b6c8 !important;

    font-size:
        1.06rem !important;

    line-height:
        1.72 !important;
}

.home-features {
    display: grid;

    grid-template-columns:
        repeat(
            3,
            minmax(0, 1fr)
        );

    gap: 12px;

    max-width: 900px;

    margin-top: 42px;
}

.home-features > div {
    padding: 17px 18px;

    border:
        1px solid
        rgba(148,163,184,.12);

    border-radius: 13px;

    background:
        rgba(255,255,255,.022);

    backdrop-filter:
        blur(10px);
}

.home-features strong {
    display: block;

    margin-bottom: 4px;

    color: #eaf0f8;

    font-size: .88rem;
    font-weight: 650;
}

.home-features small {
    color: #718096;
    font-size: .75rem;
}


/* ======================================================
   SIDEBAR
   ====================================================== */

section[data-testid="stSidebar"] {
    background:
        linear-gradient(
            180deg,
            #0a111b,
            #080e16
        );

    border-right:
        1px solid
        rgba(148,163,184,.10);
}

section[data-testid="stSidebar"] > div {
    padding-top: 1.25rem;
}

.sidebar-brand {
    padding:
        7px 6px
        30px;
}

.sidebar-brand-name {
    color: #f5f8fc;

    font-size: 1.42rem;
    font-weight: 760;

    letter-spacing: -0.05em;
}

.sidebar-brand-name span {
    color: var(--fs-blue);
}

.sidebar-brand-sub {
    margin-top: 4px;

    color: #526176;

    font-size: .68rem;

    letter-spacing: .03em;
}

.sidebar-label {
    margin:
        8px 6px
        10px;

    color: #536277;

    font-size: .62rem;
    font-weight: 750;

    letter-spacing: .17em;
}


/* navegación */

section[data-testid="stSidebar"]
div[role="radiogroup"] {
    gap: 5px;
}

section[data-testid="stSidebar"]
div[role="radiogroup"] label {
    padding:
        10px 11px;

    border-radius: 10px;

    transition:
        background .16s ease,
        color .16s ease;
}

section[data-testid="stSidebar"]
div[role="radiogroup"] label:hover {
    background:
        rgba(255,255,255,.035);
}

section[data-testid="stSidebar"]
div[role="radiogroup"]
label:has(input:checked) {
    background:
        rgba(105,170,247,.105);
}

section[data-testid="stSidebar"]
div[role="radiogroup"]
label:has(input:checked) p {
    color: #dcecff;
    font-weight: 620;
}

section[data-testid="stSidebar"]
hr {
    border-color:
        rgba(148,163,184,.09);
}

section[data-testid="stSidebar"]
[data-testid="stCaptionContainer"] p {
    color: #536277;
    font-size: .70rem;
}


/* ======================================================
   INPUTS
   ====================================================== */

div[data-testid="stTextInput"] {
    max-width: none;
}

div[data-testid="stTextInput"] label p {
    color: #8998ac;

    font-size: .75rem;
    font-weight: 620;
}

div[data-testid="stTextInput"] input {
    min-height: 52px;

    padding:
        0 16px;

    color: #eef4fb;

    background:
        #0d1622;

    border:
        1px solid
        #263448;

    border-radius: 12px;

    font-size: .94rem;

    box-shadow:
        none;

    transition:
        border-color .15s ease,
        box-shadow .15s ease,
        background .15s ease;
}

div[data-testid="stTextInput"]
input:hover {
    border-color:
        #34465e;
}

div[data-testid="stTextInput"]
input:focus {
    background:
        #0f1927;

    border-color:
        #5b9ce9;

    box-shadow:
        0 0 0 3px
        rgba(91,156,233,.09);
}


/* ======================================================
   BOTONES
   ====================================================== */

.stButton > button {
    min-height: 42px;

    border:
        1px solid
        #29384c;

    border-radius: 10px;

    color: #dbe5f1;

    background:
        #111b29;

    font-weight: 620;

    transition:
        all .15s ease;
}

.stButton > button:hover {
    color: #ffffff;

    border-color:
        #4779b5;

    background:
        #16243a;
}

.stButton > button:active {
    transform:
        translateY(1px);
}


/* ======================================================
   MÉTRICAS
   ====================================================== */

div[data-testid="stMetric"] {
    min-height: 100px !important;

    padding:
        16px 17px !important;

    border:
        1px solid
        var(--fs-border);

    border-radius:
        var(--fs-radius);

    background:
        linear-gradient(
            145deg,
            rgba(17,27,42,.96),
            rgba(13,21,33,.96)
        );

    box-shadow:
        inset 0 1px 0
        rgba(255,255,255,.018);
}

div[data-testid="stMetricLabel"] {
    color: #8291a6;
    opacity: 1;
}

div[data-testid="stMetricLabel"] p {
    font-size: .74rem;
    font-weight: 590;
}

div[data-testid="stMetricValue"] {
    color: #f2f6fb;

    font-size:
        1.52rem !important;

    font-weight: 680;

    letter-spacing:
        -0.035em;
}

div[data-testid="stMetricDelta"] {
    font-size: .74rem;
}


/* ======================================================
   EMPRESA
   ====================================================== */

.company-hero {
    position: relative;

    margin:
        1.1rem 0
        1.2rem;

    padding:
        25px 27px;

    border:
        1px solid
        var(--fs-border);

    border-radius:
        var(--fs-radius-lg);

    background:
        linear-gradient(
            135deg,
            #111b2a,
            #0d1520
        );

    box-shadow:
        inset 0 1px 0
        rgba(255,255,255,.02);
}

.company-hero::before {
    content: "";

    position: absolute;

    top: 20px;
    bottom: 20px;
    left: 0;

    width: 2px;

    border-radius:
        0 3px 3px 0;

    background:
        linear-gradient(
            #6daeff,
            #6ccdc3
        );
}

.company-hero h2 {
    margin:
        0 0 7px
        0 !important;

    color: #f5f8fc;

    font-size:
        2.05rem !important;

    font-weight: 690 !important;
}

.company-hero p {
    margin: 0;

    color: #7f8da1;

    font-size: .85rem;
}


/* ======================================================
   SECCIONES
   ====================================================== */

.section-kicker {
    margin-top: 3rem;
    margin-bottom: -.32rem;

    color: #60728c;

    font-size: .64rem;
    font-weight: 750;

    letter-spacing: .17em;
}

.subsection-space,
.analysis-section {
    margin-top: 3.2rem;
}

.analysis-caption {
    max-width: 850px;

    margin-top:
        -.2rem !important;

    margin-bottom:
        1.2rem !important;

    color:
        #6f7e92 !important;

    font-size:
        .79rem !important;

    line-height:
        1.55 !important;
}


/* ======================================================
   COTIZACIÓN
   ====================================================== */

.quote-summary {
    display: flex;

    align-items:
        flex-end;

    justify-content:
        space-between;

    gap: 28px;

    margin:
        .5rem 0
        1.25rem;

    padding:
        25px 27px;

    border:
        1px solid
        var(--fs-border);

    border-radius:
        var(--fs-radius-lg);

    background:
        linear-gradient(
            135deg,
            #111b2a,
            #0c141f
        );
}

.quote-label {
    display: block;

    margin-bottom: 8px;

    color: #75869d;

    font-size: .72rem;
    font-weight: 620;
}

.quote-price {
    color: #f6f9fc;

    font-size: 2.75rem;
    font-weight: 700;

    line-height: 1;

    letter-spacing:
        -.05em;
}

.quote-price small {
    color: #718096;

    font-size: .82rem;
    font-weight: 600;

    letter-spacing: 0;
}

.quote-change {
    margin-top: 11px;

    font-size: .88rem;
    font-weight: 650;
}

.quote-change.positive {
    color: var(--fs-green);
}

.quote-change.negative {
    color: var(--fs-red);
}

.quote-change span {
    margin-left: 6px;

    color: #637289;

    font-size: .70rem;
    font-weight: 500;
}

.quote-period-card {
    min-width: 195px;

    padding: 15px 17px;

    border:
        1px solid
        rgba(148,163,184,.11);

    border-radius: 12px;

    background:
        rgba(255,255,255,.018);
}

.quote-period-card span,
.quote-period-card small {
    display: block;

    color: #64748b;

    font-size: .67rem;
}

.quote-period-card strong {
    display: block;

    margin: 5px 0;

    color: #e9eff7;

    font-size: 1.15rem;
}


/* ======================================================
   SEGMENTED CONTROL
   ====================================================== */

div[data-testid="stSegmentedControl"] {
    margin-bottom: 1rem;
}

div[data-testid="stSegmentedControl"]
button {
    border-radius: 8px;
}


/* ======================================================
   GRÁFICOS
   ====================================================== */

div[data-testid="stVegaLiteChart"],
div[data-testid="stArrowVegaLiteChart"] {
    padding: 14px;

    border:
        1px solid
        var(--fs-border);

    border-radius:
        var(--fs-radius);

    background:
        #0b131e;

    box-shadow:
        inset 0 1px 0
        rgba(255,255,255,.015);
}


/* ======================================================
   DATAFRAMES
   ====================================================== */

div[data-testid="stDataFrame"] {
    overflow: hidden;

    border:
        1px solid
        var(--fs-border);

    border-radius:
        var(--fs-radius);
}


/* ======================================================
   EXPANDERS
   ====================================================== */

div[data-testid="stExpander"] {
    overflow: hidden;

    border:
        1px solid
        var(--fs-border) !important;

    border-radius:
        var(--fs-radius) !important;

    background:
        rgba(15,23,36,.65);
}


/* ======================================================
   ALERTAS
   ====================================================== */

div[data-testid="stAlert"] {
    border-radius:
        var(--fs-radius);

    border:
        1px solid
        var(--fs-border);
}


/* ======================================================
   COLUMNAS
   ====================================================== */

div[data-testid="stHorizontalBlock"] {
    gap: 1rem;
}


/* ======================================================
   SCROLLBAR
   ====================================================== */

::-webkit-scrollbar {
    width: 9px;
    height: 9px;
}

::-webkit-scrollbar-track {
    background:
        #080d14;
}

::-webkit-scrollbar-thumb {
    background:
        #263448;

    border-radius:
        20px;

    border:
        2px solid
        #080d14;
}

::-webkit-scrollbar-thumb:hover {
    background:
        #354861;
}


/* ======================================================
   RESPONSIVE
   ====================================================== */

@media (max-width: 900px) {

    .block-container {
        padding-left: 1.3rem;
        padding-right: 1.3rem;
    }

    .finscope-home {
        padding:
            48px 34px
            42px;
    }

    .finscope-home h1 {
        font-size:
            clamp(
                3rem,
                10vw,
                4.5rem
            ) !important;
    }

    .home-features {
        grid-template-columns:
            1fr;
    }
}


@media (max-width: 700px) {

    .block-container {
        padding-top:
            1.5rem;
    }

    .finscope-home {
        padding:
            38px 24px
            32px;

        border-radius:
            18px;
    }

    .quote-summary {
        flex-direction:
            column;

        align-items:
            stretch;
    }

    .quote-period-card {
        min-width: 0;
    }

    .company-hero {
        padding:
            21px 22px;
    }
}

</style>
""", unsafe_allow_html=True)



st.markdown("""
<style>

/* FINSCOPE DESIGN V1.1 */


/* ================================
   PORTADA
   ================================ */

.finscope-home {
    margin:
        0.15rem 0
        1.15rem 0;

    padding:
        38px 44px
        25px;

    border-radius:
        22px;

    background:
        radial-gradient(
            circle at 82% 12%,
            rgba(70, 137, 230, .20),
            transparent 28%
        ),
        radial-gradient(
            circle at 75% 105%,
            rgba(66, 190, 180, .07),
            transparent 30%
        ),
        linear-gradient(
            135deg,
            #111b2a 0%,
            #0c1521 55%,
            #090f17 100%
        );
}


.home-layout {
    display: grid;

    grid-template-columns:
        minmax(0, 1.15fr)
        minmax(320px, .85fr);

    align-items: center;

    gap: 48px;
}


.home-copy {
    position: relative;
    z-index: 2;
}


.finscope-home h1 {
    max-width: 720px;

    font-size:
        clamp(
            3rem,
            5vw,
            4.65rem
        ) !important;

    line-height:
        .96 !important;
}


.home-badge {
    margin-bottom:
        18px;
}


.home-lead {
    max-width: 620px;

    margin-top:
        20px !important;

    font-size:
        .96rem !important;

    line-height:
        1.62 !important;
}


.home-pills {
    display: flex;
    flex-wrap: wrap;

    gap: 8px;

    margin-top:
        22px;
}


.home-pills span {
    padding:
        7px 11px;

    border:
        1px solid
        rgba(148,163,184,.13);

    border-radius:
        999px;

    background:
        rgba(255,255,255,.025);

    color:
        #8494aa;

    font-size:
        .67rem;

    font-weight:
        620;
}


/* PANEL DERECHO */

.home-visual {
    padding:
        18px;

    border:
        1px solid
        rgba(148,163,184,.11);

    border-radius:
        17px;

    background:
        rgba(7,14,23,.48);

    box-shadow:
        inset 0 1px 0
        rgba(255,255,255,.02);
}


.visual-label {
    color:
        #60738e;

    font-size:
        .57rem;

    font-weight:
        750;

    letter-spacing:
        .16em;
}


.visual-chart {
    position: relative;

    height: 155px;

    margin-top:
        11px;

    overflow: hidden;

    border-radius:
        10px;
}


.chart-grid {
    position: absolute;
    inset: 0;

    background:
        linear-gradient(
            rgba(148,163,184,.055)
            1px,
            transparent 1px
        ),
        linear-gradient(
            90deg,
            rgba(148,163,184,.045)
            1px,
            transparent 1px
        );

    background-size:
        100% 38px,
        70px 100%;
}


.hero-chart-svg {
    position: absolute;
    inset: 0;

    width: 100%;
    height: 100%;

    filter:
        drop-shadow(
            0 0 8px
            rgba(94,169,244,.18)
        );
}


.visual-stats {
    display: grid;

    grid-template-columns:
        repeat(
            3,
            minmax(0,1fr)
        );

    gap: 7px;

    margin-top:
        10px;
}


.visual-stats > div {
    padding:
        8px 9px;

    border:
        1px solid
        rgba(148,163,184,.09);

    border-radius:
        9px;

    background:
        rgba(255,255,255,.018);
}


.visual-stats small {
    display: block;

    color:
        #596b83;

    font-size:
        .52rem;
}


.visual-stats strong {
    display: block;

    margin-top:
        2px;

    color:
        #cdd9e7;

    font-size:
        .65rem;
}


.home-start {
    display: flex;

    align-items: center;

    gap: 8px;

    margin-top:
        18px;

    padding-top:
        14px;

    border-top:
        1px solid
        rgba(148,163,184,.08);

    color:
        #64768e;

    font-size:
        .67rem;
}


.home-start span {
    color:
        #70b1fc;

    font-size:
        .9rem;
}


/* ================================
   BUSCADOR MÁS ACCESIBLE
   ================================ */

.finscope-home + div[data-testid="stTextInput"] {
    margin-top:
        0 !important;
}


div[data-testid="stTextInput"] input {
    min-height:
        54px;
}


/* ================================
   JERARQUÍA DE SECCIONES
   ================================ */

.section-kicker {
    margin-top:
        3.9rem;

    margin-bottom:
        .22rem;

    color:
        #7592b8;

    font-size:
        .71rem;

    font-weight:
        780;

    letter-spacing:
        .18em;
}


.subsection-space,
.analysis-section {
    margin-top:
        4.2rem;
}


h2 {
    margin-top:
        .25rem !important;

    margin-bottom:
        .65rem !important;

    font-size:
        2rem !important;

    line-height:
        1.12 !important;

    font-weight:
        710 !important;

    letter-spacing:
        -.045em !important;
}


.analysis-caption {
    margin-top:
        .05rem !important;

    margin-bottom:
        1.45rem !important;

    color:
        #78889e !important;

    font-size:
        .86rem !important;

    line-height:
        1.55 !important;
}


.mini-title {
    margin-top:
        2.6rem;

    margin-bottom:
        .95rem;

    color:
        #b4c1d2;

    font-size:
        .9rem;

    font-weight:
        680;
}


/* Separadores más discretos pero visibles */

hr {
    margin:
        3.2rem 0 !important;

    border-color:
        rgba(148,163,184,.14) !important;
}


/* ================================
   GRÁFICO
   ================================ */

div[data-testid="stVegaLiteChart"],
div[data-testid="stArrowVegaLiteChart"] {
    padding:
        16px 18px
        22px;

    border-radius:
        15px;
}


/* ================================
   RESPONSIVE
   ================================ */

@media (
    max-width: 1050px
) {

    .home-layout {
        grid-template-columns:
            1fr;
    }

    .home-visual {
        display:
            none;
    }

}


@media (
    max-width: 700px
) {

    .finscope-home {
        padding:
            30px 22px
            23px;
    }

    .finscope-home h1 {
        font-size:
            clamp(
                2.6rem,
                12vw,
                3.5rem
            ) !important;
    }

    .section-kicker,
    .analysis-section,
    .subsection-space {
        margin-top:
            3rem;
    }

}

</style>
""", unsafe_allow_html=True)



st.markdown("""
<style>

/* FINSCOPE DESIGN V1.2 */


/* ======================================================
   PORTADA · MÁS COMPACTA
   ====================================================== */

.finscope-home {
    margin-bottom: .45rem !important;
    padding-bottom: 22px !important;
}


/* ======================================================
   GRÁFICO DECORATIVO DE PORTADA
   ====================================================== */

.visual-chart {
    position: relative;

    height: 155px;

    margin-top: 10px;

    overflow: hidden;

    border-radius: 10px;

    background:
        linear-gradient(
            180deg,
            rgba(17,29,45,.25),
            rgba(7,14,23,.12)
        );
}


.chart-grid {
    position: absolute;
    inset: 0;

    background:
        linear-gradient(
            rgba(148,163,184,.055)
            1px,
            transparent 1px
        ),
        linear-gradient(
            90deg,
            rgba(148,163,184,.045)
            1px,
            transparent 1px
        );

    background-size:
        100% 38px,
        70px 100%;
}


/* Línea construida con segmentos CSS */

.market-line {
    position: absolute;
    inset: 0;
}


.market-line i {
    position: absolute;

    height: 2px;

    border-radius: 999px;

    transform-origin: left center;

    background:
        linear-gradient(
            90deg,
            #5ea8f5,
            #76d0c7
        );

    box-shadow:
        0 0 8px
        rgba(94,168,245,.22);
}


.s1 {
    left: 7%;
    top: 72%;

    width: 18%;

    transform:
        rotate(-17deg);
}


.s2 {
    left: 24%;
    top: 61%;

    width: 17%;

    transform:
        rotate(10deg);
}


.s3 {
    left: 40%;
    top: 65%;

    width: 18%;

    transform:
        rotate(-27deg);
}


.s4 {
    left: 56%;
    top: 48%;

    width: 19%;

    transform:
        rotate(9deg);
}


.s5 {
    left: 74%;
    top: 53%;

    width: 18%;

    transform:
        rotate(-30deg);
}


.market-line .point {
    position: absolute;

    width: 6px;
    height: 6px;

    border-radius: 50%;

    background: #78c9d4;

    box-shadow:
        0 0 10px
        rgba(120,201,212,.45);
}


.p1 { left: 7%; top: 70%; }
.p2 { left: 24%; top: 59%; }
.p3 { left: 40%; top: 63%; }
.p4 { left: 56%; top: 46%; }
.p5 { left: 74%; top: 51%; }
.p6 { left: 91%; top: 31%; }


.chart-caption {
    position: absolute;

    left: 14px;
    bottom: 12px;

    padding:
        7px 9px;

    border:
        1px solid
        rgba(148,163,184,.09);

    border-radius:
        8px;

    background:
        rgba(7,14,23,.78);
}


.chart-caption span,
.chart-caption strong {
    display: block;
}


.chart-caption span {
    color:
        #60738e;

    font-size:
        .49rem;

    letter-spacing:
        .12em;
}


.chart-caption strong {
    margin-top:
        2px;

    color:
        #d3deeb;

    font-size:
        .62rem;
}


/* ======================================================
   BUSCADOR PRINCIPAL
   ====================================================== */

/*
El espacio grande venía principalmente de la separación entre
el hero y el widget. Lo reducimos y hacemos que el buscador
forme visualmente parte de la portada.
*/

.finscope-home {
    margin-bottom:
        .35rem !important;
}


.finscope-home + div {
    margin-top:
        0 !important;
}


div[data-testid="stTextInput"] {
    margin-top:
        0 !important;

    margin-bottom:
        1.6rem !important;
}


div[data-testid="stTextInput"] label p {
    color:
        #aebdd0 !important;

    font-size:
        .84rem !important;

    font-weight:
        680 !important;
}


div[data-testid="stTextInput"] input {
    min-height:
        64px !important;

    padding-left:
        20px !important;

    border:
        1px solid
        #31445d !important;

    border-radius:
        14px !important;

    background:
        linear-gradient(
            135deg,
            #101a29,
            #0d1623
        ) !important;

    color:
        #f2f6fb !important;

    font-size:
        1.05rem !important;

    box-shadow:
        0 10px 30px
        rgba(0,0,0,.16) !important;
}


div[data-testid="stTextInput"] input:focus {
    border-color:
        #68aaf7 !important;

    box-shadow:
        0 0 0 3px
        rgba(104,170,247,.10),
        0 12px 32px
        rgba(0,0,0,.18) !important;
}


/* ======================================================
   TÍTULOS DE GRANDES APARTADOS
   ====================================================== */

/*
Mercado
Finanzas
Calidad del negocio
Evolución del negocio
Múltiplos
Evolución
Lectura de la empresa
*/

.section-kicker {
    margin-top:
        4.2rem !important;

    margin-bottom:
        .42rem !important;

    color:
        #7195c2 !important;

    font-size:
        .72rem !important;

    font-weight:
        800 !important;

    letter-spacing:
        .19em !important;
}


.subsection-space,
.analysis-section {
    margin-top:
        4.4rem !important;
}


/* Streamlit subheaders */

div[data-testid="stHeadingWithActionElements"]
h2 {
    margin:
        0 0 .75rem 0 !important;

    color:
        #f4f7fb !important;

    font-size:
        2.35rem !important;

    line-height:
        1.08 !important;

    font-weight:
        730 !important;

    letter-spacing:
        -.048em !important;
}


/* Compatibilidad con versiones distintas de Streamlit */

h2 {
    color:
        #f4f7fb !important;

    font-size:
        2.35rem !important;

    line-height:
        1.08 !important;

    font-weight:
        730 !important;

    letter-spacing:
        -.048em !important;
}


.analysis-caption {
    max-width:
        900px;

    margin-top:
        .05rem !important;

    margin-bottom:
        1.65rem !important;

    color:
        #8493a8 !important;

    font-size:
        .9rem !important;

    line-height:
        1.6 !important;
}


/* ======================================================
   SUBAPARTADOS INTERNOS
   ====================================================== */

.mini-title {
    margin-top:
        2.8rem !important;

    margin-bottom:
        1rem !important;

    color:
        #c4cfdd !important;

    font-size:
        1rem !important;

    font-weight:
        700 !important;
}


/* ======================================================
   SEPARADORES ENTRE GRANDES BLOQUES
   ====================================================== */

hr {
    margin:
        3.6rem 0 !important;

    border:
        none !important;

    border-top:
        1px solid
        rgba(148,163,184,.15) !important;
}


/* ======================================================
   RESPONSIVE
   ====================================================== */

@media (
    max-width: 800px
) {

    div[data-testid="stHeadingWithActionElements"]
    h2,
    h2 {
        font-size:
            1.9rem !important;
    }

    div[data-testid="stTextInput"]
    input {
        min-height:
            58px !important;
    }

}

</style>
""", unsafe_allow_html=True)



st.markdown("""
<style>

/* ==========================================================
   FINSCOPE · SEARCH + TITLES V3
   ========================================================== */


/* ----------------------------------------------------------
   HERO MÁS PEGADO AL BUSCADOR
   ---------------------------------------------------------- */

.finscope-home {
    margin-bottom: 0 !important;
}

.home-start {
    margin-top: 8px !important;
    padding-top: 8px !important;
}


/* ----------------------------------------------------------
   CABECERA DEL BUSCADOR
   ---------------------------------------------------------- */

.main-search-label {
    margin:
        8px 0
        9px 2px !important;
}

.main-search-label strong {
    display: block;

    color: #f3f7fc;

    font-size: 1.12rem;

    font-weight: 740;

    letter-spacing: -.02em;
}

.main-search-label span {
    display: block;

    margin-top: 2px;

    color: #74869e;

    font-size: .76rem;
}


/* ----------------------------------------------------------
   BUSCADOR PRINCIPAL GRANDE
   ---------------------------------------------------------- */

div[data-testid="stTextInput"]:has(
    input[placeholder^="Buscar activo ·"]
) {
    width: 100% !important;

    max-width: none !important;

    margin:
        0 0
        1.4rem 0 !important;
}

div[data-testid="stTextInput"]:has(
    input[placeholder^="Buscar activo ·"]
) input {

    width: 100% !important;

    min-height: 82px !important;

    padding:
        0 26px !important;

    border:
        1px solid
        #3b5777 !important;

    border-radius:
        17px !important;

    background:
        linear-gradient(
            135deg,
            #121f30,
            #0d1825
        ) !important;

    color:
        #f5f8fc !important;

    font-size:
        1.22rem !important;

    font-weight:
        580 !important;

    letter-spacing:
        -.02em !important;

    box-shadow:
        0 14px 36px
        rgba(0,0,0,.22),
        inset 0 1px 0
        rgba(255,255,255,.03)
        !important;
}

div[data-testid="stTextInput"]:has(
    input[placeholder^="Buscar activo ·"]
) input::placeholder {
    color:
        #73869f !important;

    opacity: 1 !important;
}

div[data-testid="stTextInput"]:has(
    input[placeholder^="Buscar activo ·"]
) input:focus {

    border-color:
        #6cb1ff !important;

    box-shadow:
        0 0 0 4px
        rgba(108,177,255,.10),
        0 16px 40px
        rgba(0,0,0,.24)
        !important;
}


/* ----------------------------------------------------------
   TÍTULOS PRINCIPALES
   ---------------------------------------------------------- */

.fs-big-title {

    margin:
        .25rem 0
        1.15rem 0 !important;

    padding: 0 !important;

    color:
        #f5f8fc !important;

    font-size:
        2.85rem !important;

    line-height:
        1.04 !important;

    font-weight:
        780 !important;

    letter-spacing:
        -.06em !important;
}


/* Etiquetas superiores */

.section-kicker {

    margin-bottom:
        .62rem !important;

    color:
        #79a8df !important;

    font-size:
        .78rem !important;

    font-weight:
        820 !important;

    letter-spacing:
        .20em !important;
}


/* Más separación entre bloques */

.analysis-section,
.subsection-space {

    margin-top:
        4.8rem !important;
}


/* Subapartados internos */

.mini-title {

    margin-top:
        2.8rem !important;

    margin-bottom:
        1rem !important;

    color:
        #c9d5e4 !important;

    font-size:
        1.10rem !important;

    font-weight:
        720 !important;

    letter-spacing:
        -.015em !important;
}


/* Texto bajo títulos */

.analysis-caption {

    margin-top:
        -.25rem !important;

    margin-bottom:
        1.55rem !important;

    font-size:
        .90rem !important;

    line-height:
        1.6 !important;
}


/* ----------------------------------------------------------
   MÓVIL
   ---------------------------------------------------------- */

@media (max-width: 800px) {

    .fs-big-title {
        font-size:
            2.15rem !important;
    }

    div[data-testid="stTextInput"]:has(
        input[placeholder^="Buscar activo ·"]
    ) input {

        min-height:
            68px !important;

        font-size:
            1.05rem !important;
    }

}

</style>
""", unsafe_allow_html=True)


st.sidebar.markdown(
    """<div class="sidebar-brand">
<div class="sidebar-brand-name">Fin<span>Scope</span></div>
<div class="sidebar-brand-sub">Financial Intelligence</div>
</div>""",
    unsafe_allow_html=True
)

st.sidebar.markdown(
    '<div class="sidebar-label">NAVEGACIÓN</div>',
    unsafe_allow_html=True
)

pagina = st.sidebar.radio(
    "Navegación",
    [
        "Analizar empresa",
        "Comparador",
        "Watchlist",
        "Mi cartera"
    ],
    label_visibility="collapsed"
)

# FinScope · modo claro permanente
tema = "Claro"


st.sidebar.markdown("---")

# ============================================================
# FINSCOPE · VALOR ACTUAL DE MI CARTERA EN SIDEBAR
# ============================================================

try:
    _cartera_sidebar = cargar_cartera_personal()

    (
        _resumen_sidebar,
        _errores_sidebar,
    ) = construir_resumen_cartera_personal(
        _cartera_sidebar
    )

    if (
        isinstance(_resumen_sidebar, pd.DataFrame)
        and not _resumen_sidebar.empty
        and "Valor EUR" in _resumen_sidebar.columns
    ):
        _valor_sidebar = float(
            _resumen_sidebar["Valor EUR"]
            .fillna(0)
            .sum()
        )
    else:
        _valor_sidebar = 0.0

    _valor_sidebar_txt = (
        f"{_valor_sidebar:,.2f}"
        .replace(",", "X")
        .replace(".", ",")
        .replace("X", ".")
    )

except Exception:
    _valor_sidebar_txt = "—"

st.sidebar.markdown(
    "##### MI CARTERA"
)

st.sidebar.markdown(
    f'<div style="font-size:22px;font-weight:700;color:#172033;'
    f'line-height:1.15;margin-top:-8px;margin-bottom:8px;">'
    f'{_valor_sidebar_txt} €</div>',
    unsafe_allow_html=True,
)

# ============================================================
# FINSCOPE · SISTEMA CLARO / OSCURO
# ============================================================

if tema == "Claro":

    st.markdown(
        """
        <style>

        /* ================================================
           FINSCOPE LIGHT MODE
           ================================================ */

        :root {
            --fs-bg: #f5f7fa !important;
            --fs-bg-soft: #eef2f7 !important;
            --fs-sidebar: #ffffff !important;

            --fs-surface: #ffffff !important;
            --fs-surface-2: #f8fafc !important;
            --fs-surface-3: #eef3f8 !important;

            --fs-border: rgba(44, 67, 95, .13) !important;
            --fs-border-strong: rgba(44, 67, 95, .21) !important;

            --fs-text: #142033 !important;
            --fs-text-2: #44536a !important;
            --fs-muted: #68788e !important;
            --fs-muted-2: #8794a6 !important;

            --fs-blue: #347fda !important;
            --fs-blue-soft: rgba(52,127,218,.10) !important;
        }


        /* PÁGINA */

        html,
        body,
        [data-testid="stAppViewContainer"],
        .stApp {
            background:
                #f5f7fa !important;

            color:
                #142033 !important;
        }

        [data-testid="stAppViewBlockContainer"] {
            background:
                #f5f7fa !important;
        }


        /* SIDEBAR */

        section[data-testid="stSidebar"] {
            background:
                linear-gradient(
                    180deg,
                    #ffffff,
                    #f7f9fc
                ) !important;

            border-right:
                1px solid
                #e1e7ef !important;
        }

        section[data-testid="stSidebar"] * {
            color:
                #46566c;
        }

        .sidebar-brand-name {
            color:
                #172337 !important;
        }

        .sidebar-brand-name span {
            color:
                #347fda !important;
        }

        .sidebar-brand-sub,
        .sidebar-label {
            color:
                #8290a3 !important;
        }

        section[data-testid="stSidebar"]
        div[role="radiogroup"]
        label {
            color:
                #536276 !important;
        }

        section[data-testid="stSidebar"]
        div[role="radiogroup"]
        label:has(input:checked) {
            background:
                #eaf2fc !important;

            border-color:
                #d4e4f8 !important;
        }

        section[data-testid="stSidebar"]
        div[role="radiogroup"]
        label:has(input:checked) p {
            color:
                #173a66 !important;
        }

        section[data-testid="stSidebar"]
        hr {
            border-color:
                #e3e9f0 !important;
        }


        /* TEXTO GENERAL */

        h1,
        h2,
        h3,
        h4,
        h5,
        h6,
        p,
        label {
            color:
                #172337;
        }

        .fs-big-title {
            color:
                #172337 !important;
        }

        .analysis-caption {
            color:
                #68788e !important;
        }

        .section-kicker {
            color:
                #397fcf !important;
        }

        .mini-title {
            color:
                #314157 !important;
        }


        /* HERO */

        .finscope-home {
            background:
                radial-gradient(
                    circle at 88% 0%,
                    rgba(66,135,219,.13),
                    transparent 32%
                ),
                linear-gradient(
                    135deg,
                    #ffffff,
                    #f2f6fb
                ) !important;

            border-color:
                #dce5ef !important;

            box-shadow:
                0 18px 55px
                rgba(41,62,89,.08) !important;
        }

        .finscope-home h1 {
            color:
                #172337 !important;
        }

        .home-lead {
            color:
                #5e6e83 !important;
        }

        .home-pills span {
            background:
                rgba(255,255,255,.72) !important;

            border-color:
                #d8e2ed !important;

            color:
                #52647a !important;
        }

        .home-start {
            color:
                #718299 !important;
        }


        /* PANEL VISUAL HERO */

        .home-visual {
            background:
                rgba(255,255,255,.64) !important;

            border-color:
                #dce5ef !important;
        }

        .visual-label {
            color:
                #71849b !important;
        }

        .visual-chart {
            background:
                linear-gradient(
                    180deg,
                    rgba(235,242,250,.8),
                    rgba(255,255,255,.45)
                ) !important;
        }

        .chart-grid {
            opacity:
                .65;
        }

        .chart-caption {
            background:
                rgba(255,255,255,.92) !important;

            border-color:
                #dce5ef !important;
        }

        .chart-caption strong {
            color:
                #34455b !important;
        }

        .visual-stats > div {
            background:
                rgba(255,255,255,.70) !important;

            border-color:
                #dce5ef !important;
        }

        .visual-stats strong {
            color:
                #28394f !important;
        }


        /* TARJETAS */

        [data-testid="stMetric"],
        .metric-card,
        .financial-card,
        .analysis-card,
        .insight-card,
        .compare-card,
        .watchlist-card,
        .quote-summary,
        .company-hero {
            background:
                #ffffff !important;

            border-color:
                #dce4ed !important;

            box-shadow:
                0 8px 25px
                rgba(37,57,82,.055) !important;
        }

        [data-testid="stMetric"] label,
        [data-testid="stMetricLabel"] {
            color:
                #607086 !important;
        }

        [data-testid="stMetricValue"] {
            color:
                #172337 !important;
        }


        /* INPUTS */

        div[data-testid="stTextInput"] input {
            background:
                #ffffff !important;

            color:
                #172337 !important;

            border-color:
                #cad6e3 !important;

            box-shadow:
                0 6px 18px
                rgba(35,55,80,.05) !important;
        }

        div[data-testid="stTextInput"] input::placeholder {
            color:
                #8996a7 !important;
        }

        div[data-testid="stTextInput"] input:focus {
            border-color:
                #4e94e5 !important;

            box-shadow:
                0 0 0 3px
                rgba(78,148,229,.11) !important;
        }

        .main-search-label strong {
            color:
                #172337 !important;
        }

        .main-search-label span {
            color:
                #738398 !important;
        }


        /* BOTONES */

        .stButton button {
            background:
                #ffffff !important;

            color:
                #26384f !important;

            border-color:
                #ccd7e4 !important;
        }

        .stButton button:hover {
            background:
                #f2f6fb !important;

            border-color:
                #7baee8 !important;
        }


        /* EXPANDERS */

        [data-testid="stExpander"] {
            background:
                #ffffff !important;

            border-color:
                #d7e0ea !important;
        }

        [data-testid="stExpander"] summary {
            color:
                #34465d !important;
        }


        /* GRÁFICOS */

        div[data-testid="stVegaLiteChart"] {
            background:
                #ffffff !important;

            border:
                1px solid
                #dde5ee !important;

            box-shadow:
                0 8px 24px
                rgba(35,55,80,.045) !important;
        }


        /* TABLAS */

        [data-testid="stDataFrame"] {
            background:
                #ffffff !important;

            border-radius:
                14px !important;

            overflow:
                hidden !important;
        }


        /* SEPARADORES */

        hr {
            border-top-color:
                #dde5ed !important;
        }


        /* ALERTAS */

        [data-testid="stAlert"] {
            border-color:
                #d5e0ec !important;
        }


        /* SEGMENTED CONTROL */

        [data-testid="stSegmentedControl"] {
            color:
                #35475e !important;
        }


        /* TOOLBAR SUPERIOR */

        [data-testid="stHeader"] {
            background:
                rgba(245,247,250,.92) !important;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )

if pagina == "Comparador":

    st.markdown(
        '<div class="section-kicker">COMPARADOR</div>',
        unsafe_allow_html=True
    )
    st.title("Comparar empresas")
    st.markdown(
        '<p class="analysis-caption">'
        'Compara dos compañías con una lectura adaptada '
        'al tipo de negocio.'
        '</p>',
        unsafe_allow_html=True
    )

    comp1, comp2 = st.columns(2)

    consulta_1 = comp1.text_input(
        "Primera empresa",
        placeholder="Apple, Inditex, AAPL...",
        key="comparador_1"
    ).strip()

    consulta_2 = comp2.text_input(
        "Segunda empresa",
        placeholder="Microsoft, Santander, MSFT...",
        key="comparador_2"
    ).strip()

    ticker_1 = ""
    ticker_2 = ""

    if consulta_1:
        with st.spinner("Buscando primera empresa..."):
            ticker_1 = resolver_busqueda_empresa(
                consulta_1,
                "comparador_resultado_1",
                comp1,
            )

        if not ticker_1:
            comp1.warning(
                "No se ha encontrado una empresa "
                "con esa búsqueda."
            )

    if consulta_2:
        with st.spinner("Buscando segunda empresa..."):
            ticker_2 = resolver_busqueda_empresa(
                consulta_2,
                "comparador_resultado_2",
                comp2,
            )

        if not ticker_2:
            comp2.warning(
                "No se ha encontrado una empresa "
                "con esa búsqueda."
            )

    if ticker_1 and ticker_2:

        try:
            with st.spinner("Comparando empresas..."):
                empresa_1 = cache_obtener_datos_empresa(ticker_1)
                empresa_2 = cache_obtener_datos_empresa(ticker_2)

            financiera_1 = bool(empresa_1.get("es_financiera"))
            financiera_2 = bool(empresa_2.get("es_financiera"))

            st.markdown(
                f"""<div class="compare-header">
<div>
<small>{empresa_1["ticker"]}</small>
<strong>{empresa_1["nombre"]}</strong>
<span>{empresa_1.get("sector") or "Sector no disponible"}</span>
</div>
<div class="compare-vs">VS</div>
<div>
<small>{empresa_2["ticker"]}</small>
<strong>{empresa_2["nombre"]}</strong>
<span>{empresa_2.get("sector") or "Sector no disponible"}</span>
</div>
</div>""",
                unsafe_allow_html=True
            )

            if financiera_1 != financiera_2:
                st.warning(
                    "Las empresas pertenecen a modelos financieros "
                    "distintos. Algunas métricas convencionales no son "
                    "directamente comparables."
                )

            elif empresa_1.get("sector") != empresa_2.get("sector"):
                st.info(
                    "Las compañías pertenecen a sectores distintos. "
                    "Los ratios deben interpretarse dentro de las "
                    "características de cada sector."
                )

            def comparar_metrica(titulo, clave, tipo="numero"):
                st.markdown(f"#### {titulo}")
                c1, c2 = st.columns(2)

                c1.metric(
                    empresa_1["ticker"],
                    formato_numero(empresa_1.get(clave), tipo)
                )

                c2.metric(
                    empresa_2["ticker"],
                    formato_numero(empresa_2.get(clave), tipo)
                )

            st.markdown("### Mercado")
            comparar_metrica("Precio", "precio")
            comparar_metrica(
                "Capitalización",
                "capitalizacion",
                "billones"
            )

            st.markdown("### Crecimiento")
            comparar_metrica(
                "Crecimiento de ingresos",
                "crecimiento_ingresos",
                "porcentaje"
            )
            comparar_metrica(
                "Crecimiento de beneficios",
                "crecimiento_beneficios",
                "porcentaje"
            )

            st.markdown("### Rentabilidad")
            comparar_metrica("ROE", "roe", "porcentaje")
            comparar_metrica("ROA", "roa", "porcentaje")

            st.markdown("### Valoración")
            comparar_metrica("PER", "per", "multiplo")
            comparar_metrica(
                "Price / Book",
                "price_to_book",
                "multiplo"
            )

            if not financiera_1 and not financiera_2:

                comparar_metrica(
                    "EV / EBITDA",
                    "ev_ebitda",
                    "multiplo"
                )

                st.markdown("### Calidad del negocio")

                comparar_metrica(
                    "Margen bruto",
                    "margen_bruto",
                    "porcentaje"
                )
                comparar_metrica(
                    "Margen operativo",
                    "margen_operativo",
                    "porcentaje"
                )
                comparar_metrica(
                    "Margen neto",
                    "margen_neto",
                    "porcentaje"
                )

            if financiera_1 and financiera_2:

                st.markdown("### Actividad bancaria")

                comparar_metrica(
                    "Ingresos netos por intereses",
                    "net_interest_income",
                    "billones"
                )
                comparar_metrica(
                    "Préstamos netos",
                    "net_loans",
                    "billones"
                )
                comparar_metrica(
                    "Patrimonio común",
                    "common_stock_equity",
                    "billones"
                )
                comparar_metrica(
                    "Valor contable tangible",
                    "tangible_book_value",
                    "billones"
                )
                comparar_metrica(
                    "Activos totales",
                    "total_assets",
                    "billones"
                )

                st.caption(
                    "Métricas específicas del negocio bancario "
                    "correspondientes al último periodo disponible."
                )

            elif financiera_1 or financiera_2:

                st.markdown("### Métricas específicas")

                st.info(
                    "FinScope no enfrenta directamente métricas "
                    "bancarias con métricas operativas convencionales "
                    "de una empresa no financiera."
                )

            st.caption(
                "El comparador aporta contexto financiero. "
                "Las diferencias entre ratios no constituyen por sí "
                "solas una conclusión sobre qué inversión es preferible."
            )

        except Exception:
            st.error(
                "No se ha podido completar la comparación. "
                "Comprueba los tickers e inténtalo de nuevo."
            )


elif pagina == "Watchlist":

    st.markdown(
        '<div class="section-kicker">SEGUIMIENTO</div>',
        unsafe_allow_html=True
    )

    st.title("Watchlist")

    st.markdown(
        '<p class="analysis-caption">'
        'Sigue tus empresas y consulta rápidamente '
        'sus principales métricas.'
        '</p>',
        unsafe_allow_html=True
    )

    if "watchlist" not in st.session_state:
        st.session_state.watchlist = cargar_watchlist()

    consulta_watchlist = st.text_input(
        "Añadir empresa",
        placeholder="Apple, Inditex, AAPL...",
        key="watchlist_input"
    ).strip()

    nuevo_ticker = ""

    if consulta_watchlist:
        with st.spinner("Buscando empresa..."):
            nuevo_ticker = resolver_busqueda_empresa(
                consulta_watchlist,
                "watchlist_resultado_busqueda",
            )

        if not nuevo_ticker:
            st.warning(
                "No se ha encontrado una empresa "
                "con esa búsqueda."
            )

    if st.button(
        "Añadir a Watchlist",
        width="stretch"
    ):

        if not consulta_watchlist:
            st.warning(
                "Introduce el nombre de una empresa "
                "o un ticker."
            )

        elif not nuevo_ticker:
            st.error(
                "No se ha podido encontrar esa empresa."
            )

        elif nuevo_ticker in st.session_state.watchlist:
            st.info("La empresa ya está en tu Watchlist.")

        else:
            try:
                empresa_nueva = cache_obtener_datos_empresa(
                    nuevo_ticker
                )

                if empresa_nueva:
                    st.session_state.watchlist.append(
                        nuevo_ticker
                    )
                    guardar_watchlist(
                        st.session_state.watchlist
                    )
                    st.rerun()
                else:
                    st.error(
                        "No se ha podido encontrar esa empresa."
                    )

            except Exception:
                st.error(
                    "No se ha podido encontrar esa empresa."
                )

    if not st.session_state.watchlist:

        st.info(
            "Tu Watchlist está vacía. "
            "Añade una empresa para empezar."
        )

    else:

        st.caption(
            f"{len(st.session_state.watchlist)} "
            "empresas en seguimiento"
        )

        for simbolo in list(st.session_state.watchlist):

            try:
                empresa = cache_obtener_datos_empresa(simbolo)

                st.divider()

                st.markdown(
                    f"### {empresa['nombre']} · "
                    f"{empresa['ticker']}"
                )

                st.caption(
                    f"{empresa.get('sector') or 'Sector no disponible'}"
                    " · "
                    f"{empresa.get('industria') or 'Industria no disponible'}"
                )

                c1, c2, c3, c4 = st.columns(4)

                c1.metric(
                    "Precio",
                    formato_numero(empresa.get("precio"))
                )

                c2.metric(
                    "Capitalización",
                    formato_numero(
                        empresa.get("capitalizacion"),
                        "billones"
                    )
                )

                c3.metric(
                    "PER",
                    formato_numero(
                        empresa.get("per"),
                        "multiplo"
                    )
                )

                c4.metric(
                    "Crec. ingresos",
                    formato_numero(
                        empresa.get("crecimiento_ingresos"),
                        "porcentaje"
                    )
                )

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "ROE",
                    formato_numero(
                        empresa.get("roe"),
                        "porcentaje"
                    )
                )

                c2.metric(
                    "Price / Book",
                    formato_numero(
                        empresa.get("price_to_book"),
                        "multiplo"
                    )
                )

                if empresa.get("es_financiera"):

                    c3.metric(
                        "Activos",
                        formato_numero(
                            empresa.get("total_assets"),
                            "billones"
                        )
                    )

                    st.caption(
                        "Entidad financiera · FinScope utiliza "
                        "métricas adaptadas al negocio bancario."
                    )

                else:

                    c3.metric(
                        "Margen operativo",
                        formato_numero(
                            empresa.get("margen_operativo"),
                            "porcentaje"
                        )
                    )

                if st.button(
                    f"Eliminar {empresa['ticker']}",
                    key=f"eliminar_{simbolo}"
                ):
                    st.session_state.watchlist.remove(simbolo)
                    guardar_watchlist(st.session_state.watchlist)
                    st.rerun()

            except Exception:
                st.warning(
                    f"No se han podido actualizar los datos "
                    f"de {simbolo}."
                )



if pagina == "Mi cartera":

    st.markdown(
        '<div class="section-kicker">PATRIMONIO</div>',
        unsafe_allow_html=True,
    )

    st.title("Mi cartera")

    st.markdown(
        '<p class="analysis-caption">'
        'Registra tus posiciones reales y FinScope captura '
        'automáticamente su precio de mercado en euros.'
        '</p>',
        unsafe_allow_html=True,
    )

    st.caption(
        "Los precios de mercado corresponden al último dato "
        "disponible y pueden no ser cotizaciones en tiempo real."
    )

    migrar_cartera_v2_a_operaciones()

    if "cartera_personal" not in st.session_state:
        st.session_state.cartera_personal = (
            sincronizar_cartera_desde_operaciones()
        )

    # ------------------------------------------------------------
    # AÑADIR COMPRA / POSICIÓN
    # ------------------------------------------------------------

    st.markdown("### Añadir compra")

    consulta_cartera = st.text_input(
        "Buscar activo",
        placeholder=(
            "Apple, Bitcoin, Inditex, "
            "MSCI World..."
        ),
        key="cartera_busqueda",
    ).strip()

    activo_cartera = None

    if consulta_cartera:

        with st.spinner(
            "Buscando activo..."
        ):
            activo_cartera = (
                resolver_activo_cartera(
                    consulta_cartera
                )
            )

        if activo_cartera:

            ticker_cartera = str(
                activo_cartera.get(
                    "ticker"
                ) or ""
            ).upper()

            nombre_cartera = (
                activo_cartera.get(
                    "nombre"
                )
                or ticker_cartera
            )

            tipo_cartera = (
                activo_cartera.get(
                    "tipo_nombre"
                )
                or activo_cartera.get(
                    "tipo"
                )
                or "Activo"
            )

            st.info(
                "Seleccionado: "
                + str(nombre_cartera)
                + " · "
                + ticker_cartera
                + " · "
                + str(tipo_cartera)
            )

        else:

            st.warning(
                "No se ha encontrado un activo "
                "con esa búsqueda."
            )

    cantidad_cartera = st.number_input(
        "Cantidad comprada",
        min_value=0.0,
        value=0.0,
        step=0.00000001,
        format="%.8f",
        key="cartera_cantidad",
        help=(
            "Número de unidades adquiridas. "
            "Se admiten cantidades fraccionarias."
        ),
    )

    cotizacion_registro = None
    precio_compra_cartera = None

    if activo_cartera:

        ticker_registro = str(
            activo_cartera.get("ticker")
            or ""
        ).upper()

        with st.spinner(
            "Obteniendo precio de registro..."
        ):
            cotizacion_registro = (
                obtener_cotizacion_cartera_eur(
                    ticker_registro
                )
            )

        if cotizacion_registro:

            precio_compra_cartera = float(
                cotizacion_registro[
                    "precio_eur"
                ]
            )

            fecha_registro_mercado = (
                cotizacion_registro.get(
                    "fecha"
                )
            )

            c_precio, c_capital = st.columns(2)

            c_precio.metric(
                "Precio de registro",
                f"{precio_compra_cartera:,.2f} €",
            )

            if cantidad_cartera > 0:

                c_capital.metric(
                    "Capital registrado",
                    f"{cantidad_cartera * precio_compra_cartera:,.2f} €",
                )

            else:

                c_capital.metric(
                    "Capital registrado",
                    "—",
                )

            st.caption(
                "Precio de registro = último dato de mercado "
                "disponible en EUR. Puede diferir del precio "
                "real de ejecución de tu bróker."
            )

            if fecha_registro_mercado is not None:

                st.caption(
                    "Fecha del dato de mercado: "
                    + str(fecha_registro_mercado)
                )

        else:

            st.warning(
                "No se ha podido obtener un precio de mercado "
                "válido para registrar esta compra."
            )

    if st.button(
        "Añadir compra",
        width="stretch",
        key="cartera_anadir",
    ):

        if not consulta_cartera:

            st.warning(
                "Busca primero un activo."
            )

        elif not activo_cartera:

            st.error(
                "No se ha podido resolver "
                "el activo."
            )

        elif cantidad_cartera <= 0:

            st.warning(
                "La cantidad debe ser "
                "superior a cero."
            )

        elif (
            precio_compra_cartera is None
            or precio_compra_cartera <= 0
        ):

            st.warning(
                "No hay una cotización válida disponible "
                "para registrar la compra."
            )

        else:

            ticker_nuevo = str(
                activo_cartera.get(
                    "ticker"
                ) or ""
            ).upper()

            nombre_nuevo = (
                activo_cartera.get(
                    "nombre"
                )
                or ticker_nuevo
            )

            tipo_nuevo = (
                activo_cartera.get(
                    "tipo_nombre"
                )
                or activo_cartera.get(
                    "tipo"
                )
                or "Activo"
            )

            cantidad_nueva = float(
                cantidad_cartera
            )

            precio_nuevo = float(
                precio_compra_cartera
            )

            # V3: cada alta se guarda como una operación
            # individual. El estado de la cartera se reconstruye
            # después desde el libro de operaciones.

            registrar_operacion_cartera(
                ticker=ticker_nuevo,
                nombre=nombre_nuevo,
                tipo=tipo_nuevo,
                operacion="COMPRA",
                cantidad=cantidad_nueva,
                precio_eur=precio_nuevo,
            )

            st.session_state.cartera_personal = (
                sincronizar_cartera_desde_operaciones()
            )

            st.rerun()

    st.divider()

    # ------------------------------------------------------------
    # CARTERA VACÍA
    # ------------------------------------------------------------

    if not st.session_state.cartera_personal:

        st.info(
            "Tu cartera está vacía. "
            "Añade una compra para empezar."
        )

    else:

        # --------------------------------------------------------
        # VALORACIÓN
        # --------------------------------------------------------

        with st.spinner(
            "Actualizando cartera..."
        ):
            (
                resumen_cartera,
                errores_cartera,
            ) = construir_resumen_cartera_personal(
                st.session_state.cartera_personal
            )

        if errores_cartera:

            st.warning(
                "Sin cotización válida para: "
                + ", ".join(
                    errores_cartera
                )
            )

        if not resumen_cartera.empty:

            valor_total = float(
                resumen_cartera[
                    "Valor EUR"
                ].sum()
            )

            filas_coste = (
                resumen_cartera[
                    resumen_cartera[
                        "Invertido EUR"
                    ].notna()
                ]
            )

            coste_completo = (
                len(filas_coste)
                == len(resumen_cartera)
            )

            capital_invertido = (
                float(
                    filas_coste[
                        "Invertido EUR"
                    ].sum()
                )
                if not filas_coste.empty
                else None
            )

            beneficio_total = None
            rentabilidad_total = None

            if (
                coste_completo
                and capital_invertido is not None
                and capital_invertido > 0
            ):

                beneficio_total = (
                    valor_total
                    - capital_invertido
                )

                rentabilidad_total = (
                    beneficio_total
                    / capital_invertido
                    * 100.0
                )

            mayor = (
                resumen_cartera
                .sort_values(
                    "Peso %",
                    ascending=False,
                )
                .iloc[0]
            )

            # ----------------------------------------------------
            # MÉTRICAS PRINCIPALES
            # ----------------------------------------------------

            m1, m2, m3, m4 = st.columns(4)

            m1.metric(
                "Valor actual",
                f"{valor_total:,.2f} €",
            )

            if coste_completo:

                m2.metric(
                    "Capital invertido",
                    f"{capital_invertido:,.2f} €",
                )

            else:

                m2.metric(
                    "Capital invertido",
                    "Pendiente",
                )

            if beneficio_total is not None:

                m3.metric(
                    "Beneficio / pérdida",
                    f"{beneficio_total:,.2f} €",
                    delta=(
                        f"{rentabilidad_total:+.2f}%"
                    ),
                )

            else:

                m3.metric(
                    "Beneficio / pérdida",
                    "Pendiente",
                )

            m4.metric(
                "Mayor posición",
                (
                    f"{mayor['Ticker']} · "
                    f"{mayor['Peso %']:.1f}%"
                ),
            )

            if not coste_completo:

                st.info(
                    "Hay posiciones creadas antes de V2 "
                    "sin precio medio de compra. "
                    "Edítalas para completar el cálculo "
                    "de rentabilidad total."
                )

            # ----------------------------------------------------
            # DISTRIBUCIÓN
            # ----------------------------------------------------

            st.markdown(
                "### Distribución actual"
            )

            grafico = (
                alt.Chart(
                    resumen_cartera
                )
                .mark_arc(
                    innerRadius=65
                )
                .encode(
                    theta=alt.Theta(
                        "Valor EUR:Q",
                        title="Valor",
                    ),
                    color=alt.Color(
                        "Ticker:N",
                        title="Activo",
                    ),
                    tooltip=[
                        alt.Tooltip(
                            "Activo:N"
                        ),
                        alt.Tooltip(
                            "Ticker:N"
                        ),
                        alt.Tooltip(
                            "Cantidad:Q",
                            format=".8f",
                        ),
                        alt.Tooltip(
                            "Precio EUR:Q",
                            title="Precio actual",
                            format=",.2f",
                        ),
                        alt.Tooltip(
                            "Valor EUR:Q",
                            title="Valor",
                            format=",.2f",
                        ),
                        alt.Tooltip(
                            "Peso %:Q",
                            title="Peso",
                            format=".2f",
                        ),
                    ],
                )
                .properties(
                    height=360
                )
            )

            st.altair_chart(
                grafico,
                width="stretch",
            )

            # ----------------------------------------------------
            # TABLA
            # ----------------------------------------------------

            st.markdown(
                "### Posiciones"
            )

            tabla = resumen_cartera[
                [
                    "Activo",
                    "Ticker",
                    "Cantidad",
                    "Precio medio EUR",
                    "Precio EUR",
                    "Invertido EUR",
                    "Valor EUR",
                    "P/L EUR",
                    "Rentabilidad %",
                    "Peso %",
                ]
            ].copy()

            tabla = tabla.sort_values(
                "Valor EUR",
                ascending=False,
            )

            st.dataframe(
                tabla,
                width="stretch",
                hide_index=True,
                column_config={
                    "Cantidad":
                        st.column_config.NumberColumn(
                            "Cantidad",
                            format="%.8f",
                        ),

                    "Precio medio EUR":
                        st.column_config.NumberColumn(
                            "Precio medio",
                            format="%.2f €",
                        ),

                    "Precio EUR":
                        st.column_config.NumberColumn(
                            "Precio actual",
                            format="%.2f €",
                        ),

                    "Invertido EUR":
                        st.column_config.NumberColumn(
                            "Invertido",
                            format="%.2f €",
                        ),

                    "Valor EUR":
                        st.column_config.NumberColumn(
                            "Valor actual",
                            format="%.2f €",
                        ),

                    "P/L EUR":
                        st.column_config.NumberColumn(
                            "P/L",
                            format="%.2f €",
                        ),

                    "Rentabilidad %":
                        st.column_config.NumberColumn(
                            "Rentabilidad",
                            format="%.2f %%",
                        ),

                    "Peso %":
                        st.column_config.NumberColumn(
                            "Peso",
                            format="%.2f %%",
                        ),
                },
            )

        # --------------------------------------------------------
        # EDITAR POSICIONES
        # --------------------------------------------------------


        # --------------------------------------------------------
        # OPERACIONES
        # --------------------------------------------------------

        st.markdown(
            "### Registrar venta"
        )

        estado_operaciones = (
            reconstruir_cartera_desde_operaciones()
        )

        tickers_disponibles = [
            ticker
            for ticker, posicion
            in estado_operaciones.items()
            if posicion["cantidad"] > 1e-12
        ]

        if tickers_disponibles:

            ticker_venta = st.selectbox(
                "Activo a vender",
                tickers_disponibles,
                key="cartera_ticker_venta",
            )

            posicion_venta = (
                estado_operaciones[
                    ticker_venta
                ]
            )

            disponible_venta = float(
                posicion_venta[
                    "cantidad"
                ]
            )

            st.caption(
                "Disponible: "
                + f"{disponible_venta:.8f}"
                + " unidades"
            )

            cantidad_venta = st.number_input(
                "Cantidad a vender",
                min_value=0.0,
                max_value=disponible_venta,
                value=0.0,
                step=0.00000001,
                format="%.8f",
                key="cartera_cantidad_venta",
            )

            cotizacion_venta = (
                obtener_cotizacion_cartera_eur(
                    ticker_venta
                )
            )

            if cotizacion_venta:

                precio_venta = float(
                    cotizacion_venta[
                        "precio_eur"
                    ]
                )

                precio_medio_venta = float(
                    posicion_venta[
                        "precio_medio_eur"
                    ]
                )

                v1, v2, v3 = st.columns(3)

                v1.metric(
                    "Precio de venta registrado",
                    f"{precio_venta:,.2f} €",
                )

                if cantidad_venta > 0:

                    importe_venta = (
                        cantidad_venta
                        * precio_venta
                    )

                    pl_estimado = (
                        precio_venta
                        - precio_medio_venta
                    ) * cantidad_venta

                    v2.metric(
                        "Importe de venta",
                        f"{importe_venta:,.2f} €",
                    )

                    v3.metric(
                        "P/L realizado estimado",
                        f"{pl_estimado:,.2f} €",
                    )

                else:

                    v2.metric(
                        "Importe de venta",
                        "—",
                    )

                    v3.metric(
                        "P/L realizado estimado",
                        "—",
                    )

                if st.button(
                    "Registrar venta",
                    width="stretch",
                    key="cartera_registrar_venta",
                ):

                    if cantidad_venta <= 0:

                        st.warning(
                            "Introduce una cantidad "
                            "superior a cero."
                        )

                    else:

                        try:

                            registrar_operacion_cartera(
                                ticker=ticker_venta,
                                nombre=posicion_venta[
                                    "nombre"
                                ],
                                tipo=posicion_venta[
                                    "tipo"
                                ],
                                operacion="VENTA",
                                cantidad=cantidad_venta,
                                precio_eur=precio_venta,
                            )

                            st.session_state.cartera_personal = (
                                sincronizar_cartera_desde_operaciones()
                            )

                            st.rerun()

                        except ValueError as exc:

                            st.error(
                                str(exc)
                            )

            else:

                st.warning(
                    "No se ha podido obtener "
                    "una cotización para la venta."
                )

        st.markdown(
            "### Historial de operaciones"
        )

        operaciones_historial = (
            cargar_operaciones_cartera()
        )

        if operaciones_historial:

            filas_historial = []

            for op in reversed(
                operaciones_historial
            ):

                filas_historial.append({
                    "Fecha":
                        op["fecha"],

                    "Operación":
                        op["operacion"],

                    "Activo":
                        op["nombre"],

                    "Ticker":
                        op["ticker"],

                    "Cantidad":
                        op["cantidad"],

                    "Precio EUR":
                        op["precio_eur"],

                    "Importe EUR":
                        (
                            op["cantidad"]
                            * op["precio_eur"]
                        ),
                })

            df_historial = pd.DataFrame(
                filas_historial
            )

            st.dataframe(
                df_historial,
                width="stretch",
                hide_index=True,
                column_config={
                    "Cantidad":
                        st.column_config.NumberColumn(
                            "Cantidad",
                            format="%.8f",
                        ),

                    "Precio EUR":
                        st.column_config.NumberColumn(
                            "Precio",
                            format="%.2f €",
                        ),

                    "Importe EUR":
                        st.column_config.NumberColumn(
                            "Importe",
                            format="%.2f €",
                        ),
                },
            )

            pl_realizado_total = (
                calcular_pl_realizado_total()
            )

            st.metric(
                "P/L realizado acumulado",
                f"{pl_realizado_total:,.2f} €",
            )

        else:

            st.info(
                "Todavía no hay operaciones "
                "registradas."
            )


        st.markdown(
            "### Gestionar posiciones"
        )

        st.caption(
            "Las cantidades se modifican registrando compras "
            "o ventas para conservar un historial coherente. "
            "El libro de operaciones es ahora la fuente principal "
            "de la cartera."
        )

        for posicion in list(
            st.session_state.cartera_personal
        ):

            ticker_posicion = (
                posicion["ticker"]
            )

            nombre_posicion = (
                posicion.get("nombre")
                or ticker_posicion
            )

            with st.expander(
                str(nombre_posicion)
                + " · "
                + ticker_posicion,
                expanded=False,
            ):

                cantidad_editada = (
                    st.number_input(
                        "Cantidad total",
                        min_value=0.00000001,
                        value=float(
                            posicion[
                                "cantidad"
                            ]
                        ),
                        step=0.00000001,
                        format="%.8f",
                        key=(
                            "editar_cantidad_"
                            + ticker_posicion
                        ),
                    )
                )

                precio_actual_guardado = (
                    posicion.get(
                        "precio_medio_eur"
                    )
                )

                if precio_actual_guardado is None:
                    precio_actual_guardado = 0.0

                precio_editado = (
                    st.number_input(
                        "Precio medio de compra (€)",
                        min_value=0.0,
                        value=float(
                            precio_actual_guardado
                        ),
                        step=0.01,
                        format="%.4f",
                        key=(
                            "editar_precio_"
                            + ticker_posicion
                        ),
                    )
                )

                b1, b2 = st.columns(2)

                if b1.button(
                    "Guardar cambios",
                    key=(
                        "guardar_posicion_"
                        + ticker_posicion
                    ),
                    width="stretch",
                ):

                    if precio_editado <= 0:

                        st.warning(
                            "El precio medio debe "
                            "ser superior a cero."
                        )

                    else:

                        posicion[
                            "cantidad"
                        ] = float(
                            cantidad_editada
                        )

                        posicion[
                            "precio_medio_eur"
                        ] = float(
                            precio_editado
                        )

                        guardar_cartera_personal(
                            st.session_state.cartera_personal
                        )

                        st.rerun()

                if b2.button(
                    "Eliminar posición",
                    key=(
                        "eliminar_posicion_"
                        + ticker_posicion
                    ),
                    width="stretch",
                ):

                    st.session_state.cartera_personal = [
                        p
                        for p
                        in st.session_state.cartera_personal
                        if p["ticker"]
                        != ticker_posicion
                    ]

                    guardar_cartera_personal(
                        st.session_state.cartera_personal
                    )

                    st.rerun()

        st.caption(
            "Todos los valores agregados se expresan en EUR. "
            "La rentabilidad mostrada compara el valor actual "
            "con el coste de adquisición introducido por el usuario."
        )


if pagina == "Analizar empresa":
    st.markdown(
        """<div class="main-search-label">
        <strong>Buscar activo</strong>
        <span>Empresas, índices, ETF, criptomonedas y materias primas</span>
        </div>""",
        unsafe_allow_html=True,
    )

    consulta_empresa = st.text_input(
        "Buscar activo",
        placeholder="Apple, Bitcoin, S&P 500, MSCI World, Gold...",
        label_visibility="collapsed",
        key="main_company_search",
    ).strip()

    ticker = ""
    activo_seleccionado = None

    if consulta_empresa:

        with st.spinner("Buscando activo..."):
            resultados_busqueda = buscar_activos(
                consulta_empresa,
                max_resultados=8,
            )

        if resultados_busqueda:

            consulta_normalizada = consulta_empresa.upper()

            coincidencia_ticker = next(
                (
                    activo
                    for activo in resultados_busqueda
                    if activo["ticker"].upper()
                    == consulta_normalizada
                ),
                None,
            )

            if coincidencia_ticker:
                activo_seleccionado = coincidencia_ticker

            elif len(resultados_busqueda) == 1:
                activo_seleccionado = resultados_busqueda[0]

            else:
                opciones_activos = {
                    (
                        f'{activo["tipo_nombre"]} · '
                        f'{activo["nombre"]} · '
                        f'{activo["ticker"]} · '
                        f'{activo["exchange"]}'
                    ): activo
                    for activo in resultados_busqueda
                }

                seleccion_activo = st.selectbox(
                    "Coincidencias",
                    options=list(opciones_activos.keys()),
                    index=0,
                    key="asset_search_result",
                )

                activo_seleccionado = opciones_activos[
                    seleccion_activo
                ]

            if activo_seleccionado:
                ticker = activo_seleccionado["ticker"]

        else:
            st.warning(
                "No hemos encontrado un activo "
                "con esa búsqueda."
            )

else:
    ticker = ""
    activo_seleccionado = None

if ticker:

    es_empresa = (
        activo_seleccionado is None
        or activo_seleccionado.get("tipo") == "EMPRESA"
    )

    if es_empresa:
        with st.spinner("Analizando empresa..."):
            datos = cache_obtener_datos_empresa(ticker)

    else:
        import yfinance as yf

        nombre_multi = activo_seleccionado.get(
            "nombre",
            ticker,
        )

        tipo_multi = activo_seleccionado.get(
            "tipo",
            "ACTIVO",
        )

        tipo_nombre_multi = activo_seleccionado.get(
            "tipo_nombre",
            "Activo",
        )

        mercado_multi = activo_seleccionado.get(
            "exchange",
            "",
        )

        precio_multi = None
        moneda_multi = ""

        try:
            info_multi = yf.Ticker(ticker).info or {}

            precio_multi = (
                info_multi.get("currentPrice")
                or info_multi.get("regularMarketPrice")
            )

            moneda_multi = (
                info_multi.get("currency")
                or ""
            )

        except Exception:
            info_multi = {}

        if precio_multi is None:
            try:
                hist_multi = cache_obtener_historico(
                    ticker,
                    "5d",
                )

                if (
                    hist_multi is not None
                    and not hist_multi.empty
                    and "Close" in hist_multi.columns
                ):
                    serie_close_multi = (
                        hist_multi["Close"]
                        .dropna()
                    )

                    if not serie_close_multi.empty:
                        precio_multi = float(
                            serie_close_multi.iloc[-1]
                        )

            except Exception:
                precio_multi = None

        datos = {
            "ticker": ticker,
            "nombre": nombre_multi,
            "sector": tipo_nombre_multi,
            "industria": mercado_multi,
            "pais": "",
            "precio": precio_multi,
            "moneda": moneda_multi,
            "capitalizacion": None,
            "per": None,
            "fcf_aplicable": False,
        }

    st.markdown(
        f"""<div class="company-hero">
        <h2>{datos["nombre"]} · {datos["ticker"]}</h2>
        <p>{datos["sector"]} · {datos["industria"]} · {datos["pais"]}</p>
        </div>""",
        unsafe_allow_html=True
    )

    if es_empresa:
        resumen1, resumen2, resumen3 = st.columns(3)

        try:
            _serie_precio_eur = cache_obtener_serie_rendimiento(
                datos["ticker"],
                "1mo",
            )
            _precio_actual_eur = float(
                _serie_precio_eur["Close"].dropna().iloc[-1]
            )
        except Exception:
            _precio_actual_eur = None

        if _precio_actual_eur is not None:
            resumen1.metric(
                "Precio actual",
                f"{_precio_actual_eur:,.2f} €",
            )

        if datos["capitalizacion"] is not None:
            resumen2.metric(
                "Capitalización",
                f'{datos["capitalizacion"] / 1_000_000_000:,.2f} B'
            )

        if datos["per"] is not None:
            resumen3.metric(
                "PER",
                f'{datos["per"]:.2f}x'
            )

    else:
        resumen_multi_1, resumen_multi_2 = st.columns(2)

        try:
            _serie_precio_eur = cache_obtener_serie_rendimiento(
                datos["ticker"],
                "1mo",
            )
            _precio_actual_eur = float(
                _serie_precio_eur["Close"].dropna().iloc[-1]
            )
        except Exception:
            _precio_actual_eur = None

        if _precio_actual_eur is not None:
            resumen_multi_1.metric(
                "Precio actual",
                f"{_precio_actual_eur:,.2f} €",
            )
        else:
            resumen_multi_1.metric(
                "Precio actual",
                "N/D",
            )

        resumen_multi_2.metric(
            "Tipo de activo",
            activo_seleccionado.get(
                "tipo_nombre",
                "Activo",
            ),
        )

        st.info(
            "FinScope aplica a este instrumento los análisis "
            "basados en precios y rendimientos. Los módulos "
            "empresariales que no corresponden a este tipo de "
            "activo se omiten automáticamente."
        )

    st.divider()

    # ============================================================
    # FINSCOPE · PANEL DE ANÁLISIS
    # ============================================================

    st.markdown(
        """
        <style>

        .fs-modules-header {
            margin-top: 1.15rem;
            margin-bottom: 0.9rem;
        }

        .fs-modules-kicker {
            color: #71849c;
            font-size: 0.70rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            margin-bottom: 0.35rem;
        }

        .fs-modules-title {
            font-size: 1.42rem;
            font-weight: 700;
            line-height: 1.2;
            margin-bottom: 0.3rem;
        }

        .fs-modules-subtitle {
            color: #7e8fa4;
            font-size: 0.88rem;
            margin-bottom: 0.35rem;
        }

        .fs-active-module {
            margin-top: 1rem;
            margin-bottom: 1.15rem;
            padding: 12px 16px;
            border:
                1px solid rgba(120,145,175,0.18);
            border-radius: 12px;
            background:
                rgba(30,70,115,0.08);
        }

        .fs-active-module span {
            display: block;
            color: #71849c;
            font-size: 0.66rem;
            font-weight: 700;
            letter-spacing: 0.11em;
            margin-bottom: 0.18rem;
        }

        .fs-active-module strong {
            font-size: 0.98rem;
            font-weight: 650;
        }

        /*
        Botones utilizados como tarjetas.
        */
        div[data-testid="stColumn"]
        div[data-testid="stButton"] > button {
            min-height: 76px;
            border-radius: 13px;
            white-space: normal;
            line-height: 1.22;
            padding-left: 0.75rem;
            padding-right: 0.75rem;
        }

        div[data-testid="stColumn"]
        div[data-testid="stButton"] > button:hover {
            transform: translateY(-1px);
        }

        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="fs-modules-header">
            <div class="fs-modules-kicker">
                PANEL DE ANÁLISIS
            </div>
            <div class="fs-modules-title">
                Elige qué quieres analizar
            </div>
            <div class="fs-modules-subtitle">
                Selecciona un módulo para mostrar su análisis.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if es_empresa:
        modulos_finscope = [
            "Cotización y mercado",
            "Empresa y fundamentales",
            "Rendimiento histórico",
            "Comparación con el mercado",
            "Riesgo histórico",
            "Beta y correlación",
            "Rentabilidad ajustada al riesgo",
            "Valoración histórica",
            "Evolución del negocio",
            "FinScope Insight",
            "Dividendos",
            "Valoración DCF",
            "Estadística cuantitativa",
            "Riesgo cuantitativo avanzado",
            "Análisis temporal",
            "Monte Carlo",
            "Escenarios y sensibilidad",
            "Comparación cuantitativa",
        "Carteras y diversificación",
            "Machine Learning",
        ]
    else:
        modulos_finscope = [
            "Cotización y mercado",
            "Rendimiento histórico",
            "Comparación con el mercado",
            "Riesgo histórico",
            "Beta y correlación",
            "Rentabilidad ajustada al riesgo",
            "Estadística cuantitativa",
            "Riesgo cuantitativo avanzado",
            "Análisis temporal",
            "Monte Carlo",
            "Escenarios y sensibilidad",
            "Comparación cuantitativa",
            "Machine Learning",
        ]

    ticker_estado = (
        str(ticker)
        .replace("^", "IDX_")
        .replace("=", "_")
        .replace(".", "_")
        .replace("-", "_")
        .replace(" ", "_")
    )

    clave_modulo = (
        f"finscope_modulo_{ticker_estado}"
    )

    if (
        clave_modulo not in st.session_state
        or st.session_state[clave_modulo]
        not in modulos_finscope
    ):
        st.session_state[clave_modulo] = (
            "Cotización y mercado"
        )

    modulo_finscope = st.session_state[
        clave_modulo
    ]

    # --------------------------------------------------------
    # CUADRÍCULA · 4 TARJETAS POR FILA
    # --------------------------------------------------------

    for inicio_fila in range(
        0,
        len(modulos_finscope),
        4,
    ):

        fila_modulos = modulos_finscope[
            inicio_fila:inicio_fila + 4
        ]

        columnas_modulos = st.columns(
            4,
            gap="small",
        )

        for indice_columna in range(4):

            with columnas_modulos[indice_columna]:

                if indice_columna < len(fila_modulos):

                    nombre_modulo = fila_modulos[
                        indice_columna
                    ]

                    seleccionado = (
                        nombre_modulo
                        == modulo_finscope
                    )

                    if st.button(
                        nombre_modulo,
                        key=(
                            f"fs_module_"
                            f"{ticker_estado}_"
                            f"{inicio_fila}_"
                            f"{indice_columna}"
                        ),
                        type=(
                            "primary"
                            if seleccionado
                            else "secondary"
                        ),
                        use_container_width=True,
                    ):

                        st.session_state[
                            clave_modulo
                        ] = nombre_modulo

                        st.rerun()

    st.markdown(
        f"""
        <div class="fs-active-module">
            <span>MÓDULO SELECCIONADO</span>
            <strong>{modulo_finscope}</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if modulo_finscope == "Cotización y mercado":

        # COTIZACIÓN

        st.markdown(
            '<div class="section-kicker">MERCADO</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="fs-big-title">Cotización</div>',
            unsafe_allow_html=True,
        )

        periodo = st.radio(
            "Periodo",
            options=[
                "1h", "1d", "5d", "1mo", "3mo",
                "6mo", "1y", "2y", "5y", "max"
            ],
            index=6,
            horizontal=True,
            label_visibility="collapsed",
            key="quote_period"
        )

        historico = cache_obtener_historico(ticker, periodo)

        if historico is not None and not historico.empty:

            variacion = cache_calcular_variacion_precio(
                ticker,
                periodo,
            )

            if variacion is None:
                st.warning(
                    "No hay datos suficientes para calcular "
                    "la variación de este periodo."
                )
                precio_final = float(historico["Close"].iloc[-1])
                precio_inicial = precio_final
                variacion_abs = 0.0
                rentabilidad = 0.0
                fecha_inicio_variacion = "No disponible"
                fecha_fin_variacion = "No disponible"
            else:
                precio_inicial = variacion["precio_inicial"]
                precio_final = variacion["precio_final"]
                variacion_abs = variacion["variacion_abs"]
                rentabilidad = variacion["variacion_pct"]
                fecha_inicio_variacion = variacion["fecha_inicio"]
                fecha_fin_variacion = variacion["fecha_fin"]

            direccion = (
                "positive"
                if rentabilidad >= 0
                else "negative"
            )

            signo = "+" if rentabilidad >= 0 else ""

            # Construimos explícitamente los textos en cada rerun.
            # El id cambia con ticker y periodo para que Streamlit/frontend
            # no reutilice el contenido de la tarjeta anterior.
            variacion_abs_texto = f"{variacion_abs:,.2f}"
            rentabilidad_texto = f"{rentabilidad:.2f}"
            quote_id = f"quote-{ticker}-{periodo}".replace(".", "-")

            tarjeta_cotizacion = f"""
    <div
        id="{quote_id}"
        class="quote-summary"
        data-ticker="{ticker}"
        data-periodo="{periodo}"
    >
        <div class="quote-main">
            <span class="quote-label">
                {datos.get("nombre")} · {datos.get("ticker")}
            </span>
            <div class="quote-price">
                {precio_final:,.2f}
                <small>EUR</small>
            </div>
            <div class="quote-change {direccion}">
                {signo}{variacion_abs_texto} ·
                {signo}{rentabilidad_texto}%
                <span>en {periodo}</span>
            </div>
        </div>
        <div class="quote-period-card">
            <span>Periodo seleccionado</span>
            <strong>{periodo}</strong>
            <small>Variación del precio</small>
        </div>
    </div>
    """

            st.markdown(
                tarjeta_cotizacion,
                unsafe_allow_html=True,
            )

            if variacion is not None:
                st.caption(
                    f"Variación del precio entre "
                    f"{fecha_inicio_variacion} y {fecha_fin_variacion}. "
                    f"Calculada con precios de cierre sin ajustar por "
                    f"dividendos. Fuente: Yahoo Finance vía yfinance."
                )

            df_precio = (
                historico[["Close"]]
                .reset_index()
            )

            columna_fecha = df_precio.columns[0]

            grafico_precio = (
                alt.Chart(df_precio)
                .mark_line(
                    strokeWidth=2
                )
                .encode(
                    x=alt.X(
                        columna_fecha,
                        type="temporal",
                        axis=alt.Axis(
                            format="%b %d",
                            labelAngle=0,
                            labelOverlap="greedy",
                            tickCount=8,
                            title=None,
                        ),
                    ),
                    y=alt.Y(
                        "Close:Q",
                        scale=alt.Scale(
                            zero=False
                        ),
                        axis=alt.Axis(
                            title=None
                        ),
                    ),
                    tooltip=[
                        alt.Tooltip(
                            columna_fecha,
                            type="temporal",
                            title="Fecha",
                        ),
                        alt.Tooltip(
                            "Close:Q",
                            title="Precio",
                            format=",.2f",
                        ),
                    ],
                )
                .properties(
                    height=410
                )
                .interactive()
            )

            st.altair_chart(
                grafico_precio,
                width="stretch"
            )

        else:
            st.warning(
                "No se ha podido obtener el histórico."
            )

        # MERCADO

        st.divider()

        st.markdown(
            '<div class="section-kicker">DATOS CLAVE</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="fs-big-title">Mercado</div>',
            unsafe_allow_html=True,
        )

        c1, c2, c3, c4 = st.columns(4)

        if datos.get("precio") is not None:
            c1.metric(
                "Precio",
                f'{datos.get("precio"):,.2f} {datos.get("moneda")}'
            )

        if datos.get("capitalizacion") is not None:
            c2.metric(
                "Capitalización",
                f'{datos.get("capitalizacion") / 1_000_000_000:,.2f} B'
            )

        if datos.get("minimo_52_semanas") is not None:
            c3.metric(
                "Mínimo 52 semanas",
                f'{datos.get("minimo_52_semanas"):,.2f}'
            )

        if datos.get("maximo_52_semanas") is not None:
            c4.metric(
                "Máximo 52 semanas",
                f'{datos.get("maximo_52_semanas"):,.2f}'
            )



    if es_empresa:
        if modulo_finscope == "Empresa y fundamentales":

            # FINANZAS

            st.markdown(
                '<div class="section-kicker subsection-space">RESULTADOS</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="fs-big-title">Finanzas</div>',
                unsafe_allow_html=True,
            )

            c1, c2, c3, c4 = st.columns(4)

            if datos["ingresos"] is not None:
                c1.metric(
                    "Ingresos TTM",
                    f'{datos["ingresos"] / 1_000_000_000:,.2f} B'
                )

            if datos["beneficio_neto"] is not None:
                c2.metric(
                    "Beneficio neto TTM",
                    f'{datos["beneficio_neto"] / 1_000_000_000:,.2f} B'
                )

            if datos["ebitda"] is not None:
                c3.metric(
                    "EBITDA TTM",
                    f'{datos["ebitda"] / 1_000_000_000:,.2f} B'
                )

            if datos.get("free_cash_flow_ttm") is not None:
                c4.metric(
                    "Free Cash Flow TTM",
                    f'{datos["free_cash_flow_ttm"] / 1_000_000_000:,.2f} B'
                )

            # REFERENCIA TEMPORAL DE LOS DATOS TTM
            periodos_fcf = datos.get("free_cash_flow_ttm_periodos", [])

            if periodos_fcf:
                try:
                    from datetime import datetime

                    fecha_ultimo_trimestre = datetime.strptime(
                        periodos_fcf[0],
                        "%Y-%m-%d"
                    ).strftime("%d/%m/%Y")

                    st.caption(
                        f"Datos financieros TTM · "
                        f"Último trimestre disponible: {fecha_ultimo_trimestre} · "
                        f"Free Cash Flow TTM calculado con los "
                        f"{len(periodos_fcf)} últimos trimestres disponibles."
                    )
                except (ValueError, TypeError):
                    st.caption(
                        "Datos financieros TTM · "
                        "Free Cash Flow calculado con los 4 últimos "
                        "trimestres disponibles."
                    )
            else:
                st.caption(
                    "Datos financieros TTM · "
                    "La disponibilidad y fecha de los fundamentales "
                    "dependen de la información proporcionada por la fuente."
                )

            # ANÁLISIS FUNDAMENTAL

            st.divider()

            st.markdown(
                '<div class="section-kicker analysis-section">ANÁLISIS FUNDAMENTAL</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="fs-big-title">Calidad del negocio</div>',
                unsafe_allow_html=True,
            )

            if datos.get("es_financiera"):

                st.markdown(
                    '<p class="analysis-caption">'
                    'Análisis adaptado a entidades financieras. En banca, FinScope '
                    'prioriza magnitudes específicas del negocio en lugar de márgenes '
                    'operativos propios de empresas no financieras.'
                    '</p>',
                    unsafe_allow_html=True
                )

                def formato_bancario(valor):
                    if valor is None:
                        return "No disponible"

                    valor = float(valor)

                    if abs(valor) >= 1_000_000_000_000:
                        return f"€{valor / 1_000_000_000_000:.2f} T"

                    if abs(valor) >= 1_000_000_000:
                        return f"€{valor / 1_000_000_000:.2f} B"

                    if abs(valor) >= 1_000_000:
                        return f"€{valor / 1_000_000:.2f} M"

                    return f"€{valor:,.0f}"

                fecha_banco = datos.get("fecha_resultados_bancarios")

                if fecha_banco:
                    st.caption(f"Últimos datos bancarios disponibles: {fecha_banco}")

                st.markdown(
                    '<div class="mini-title">Actividad bancaria</div>',
                    unsafe_allow_html=True
                )

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "Ingresos netos por intereses",
                    formato_bancario(datos.get("net_interest_income"))
                )

                c2.metric(
                    "Ingresos por intereses",
                    formato_bancario(datos.get("interest_income"))
                )

                c3.metric(
                    "Gastos por intereses",
                    formato_bancario(datos.get("interest_expense"))
                )

                st.markdown(
                    '<div class="mini-title">Balance bancario</div>',
                    unsafe_allow_html=True
                )

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "Préstamos netos",
                    formato_bancario(datos.get("net_loans"))
                )

                c2.metric(
                    "Patrimonio común",
                    formato_bancario(datos.get("common_stock_equity"))
                )

                c3.metric(
                    "Activos totales",
                    formato_bancario(datos.get("total_assets"))
                )

                if datos.get("tangible_book_value") is not None:
                    st.metric(
                        "Valor contable tangible",
                        formato_bancario(datos.get("tangible_book_value"))
                    )

                st.markdown(
                    '<div class="mini-title">Rentabilidad del capital</div>',
                    unsafe_allow_html=True
                )

                c1, c2 = st.columns(2)

                if datos.get("roe") is not None:
                    c1.metric(
                        "ROE",
                        f'{datos["roe"] * 100:.2f}%'
                    )

                if datos.get("roa") is not None:
                    c2.metric(
                        "ROA",
                        f'{datos["roa"] * 100:.2f}%'
                    )

            else:

                st.markdown(
                    '<p class="analysis-caption">'
                    'Márgenes y eficiencia con la que la empresa convierte '
                    'su actividad en beneficios.'
                    '</p>',
                    unsafe_allow_html=True
                )

                c1, c2, c3 = st.columns(3)

                if datos.get("margen_bruto") is not None:
                    c1.metric(
                        "Margen bruto",
                        f'{datos["margen_bruto"] * 100:.2f}%'
                    )

                if datos.get("margen_operativo") is not None:
                    c2.metric(
                        "Margen operativo",
                        f'{datos["margen_operativo"] * 100:.2f}%'
                    )

                if datos.get("margen_neto") is not None:
                    c3.metric(
                        "Margen neto",
                        f'{datos["margen_neto"] * 100:.2f}%'
                    )

                st.markdown(
                    '<div class="mini-title">Rentabilidad del capital</div>',
                    unsafe_allow_html=True
                )

                c1, c2 = st.columns(2)

                if datos.get("roe") is not None:
                    c1.metric(
                        "ROE",
                        f'{datos["roe"] * 100:.2f}%'
                    )

                if datos.get("roa") is not None:
                    c2.metric(
                        "ROA",
                        f'{datos["roa"] * 100:.2f}%'
                    )


            # =====================================================
            # EVOLUCIÓN FUNDAMENTAL
            # =====================================================

            st.divider()

            st.markdown(
                '<div class="section-kicker analysis-section">'
                'EVOLUCIÓN FUNDAMENTAL'
                '</div>',
                unsafe_allow_html=True,
            )




    if modulo_finscope == "Rendimiento histórico":

        # ============================================================
        # NIVEL 2 · RENDIMIENTO HISTÓRICO
        # ============================================================

        st.markdown(
            '<div class="section-kicker">INVERSIÓN</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="fs-big-title">Rendimiento histórico</div>',
            unsafe_allow_html=True,
        )

        st.caption(
            "Rentabilidad histórica del activo utilizando precios "
            "de mercado. FinScope separa la variación del precio de "
            "la rentabilidad total ajustada por dividendos y eventos "
            "corporativos cuando Yahoo Finance proporciona la serie "
            "ajustada."
        )

        with st.spinner(
            "Calculando rendimiento histórico..."
        ):
            paquete_rendimiento = (
                calcular_rendimientos_historicos(
                    ticker
                )
            )

        rendimientos = paquete_rendimiento.get(
            "resultados",
            {}
        )

        etiquetas_periodos = {
            "1m": "1M",
            "3m": "3M",
            "6m": "6M",
            "1y": "1A",
            "3y": "3A",
            "5y": "5A",
            "10y": "10A",
        }

        if rendimientos:

            # --------------------------------------------------------
            # RENTABILIDADES
            # --------------------------------------------------------

            periodos_rendimiento = [
                "1m",
                "3m",
                "6m",
                "1y",
                "3y",
                "5y",
                "10y",
            ]

            columnas_rendimiento = st.columns(
                len(periodos_rendimiento)
            )

            for columna, codigo in zip(
                columnas_rendimiento,
                periodos_rendimiento,
            ):
                resultado_periodo = rendimientos.get(
                    codigo
                )

                with columna:
                    if not resultado_periodo:
                        st.metric(
                            etiquetas_periodos[codigo],
                            "N/D",
                        )
                        continue

                    retorno_precio = (
                        resultado_periodo.get(
                            "rentabilidad_precio_pct"
                        )
                    )

                    retorno_total = (
                        resultado_periodo.get(
                            "rentabilidad_total_pct"
                        )
                    )

                    if retorno_precio is None:
                        valor_precio = "N/D"
                    else:
                        valor_precio = (
                            f"{retorno_precio:+.2f}%"
                        )

                    if retorno_total is None:
                        delta_total = None
                    else:
                        delta_total = (
                            f"Total {retorno_total:+.2f}%"
                        )

                    st.metric(
                        etiquetas_periodos[codigo],
                        valor_precio,
                        delta=delta_total,
                        delta_color="off",
                    )

                    if not resultado_periodo.get(
                        "periodo_completo",
                        True,
                    ):
                        st.caption(
                            "Historial disponible"
                        )

            st.caption(
                "Cifra principal: rentabilidad del precio "
                "(Close sin ajustar). El dato “Total” utiliza "
                "Adj Close de Yahoo Finance como aproximación "
                "histórica ajustada por dividendos y eventos "
                "corporativos."
            )

            # --------------------------------------------------------
            # CAGR
            # --------------------------------------------------------

            st.markdown(
                "#### Rentabilidad anualizada · CAGR"
            )

            periodos_cagr = [
                "1y",
                "3y",
                "5y",
                "10y",
            ]

            columnas_cagr = st.columns(
                len(periodos_cagr)
            )

            for columna, codigo in zip(
                columnas_cagr,
                periodos_cagr,
            ):
                resultado_periodo = rendimientos.get(
                    codigo
                )

                with columna:
                    if not resultado_periodo:
                        st.metric(
                            f"CAGR · {etiquetas_periodos[codigo]}",
                            "N/D",
                        )
                        continue

                    cagr_precio = (
                        resultado_periodo.get(
                            "cagr_precio_pct"
                        )
                    )

                    cagr_total = (
                        resultado_periodo.get(
                            "cagr_total_pct"
                        )
                    )

                    if cagr_precio is None:
                        valor_cagr = "N/D"
                    else:
                        valor_cagr = (
                            f"{cagr_precio:+.2f}%"
                        )

                    if cagr_total is None:
                        delta_cagr = None
                    else:
                        delta_cagr = (
                            f"Total {cagr_total:+.2f}%"
                        )

                    st.metric(
                        f"CAGR · {etiquetas_periodos[codigo]}",
                        valor_cagr,
                        delta=delta_cagr,
                        delta_color="off",
                    )

            st.caption(
                "El CAGR expresa la tasa anual compuesta que "
                "conectaría el valor inicial con el final durante "
                "el tiempo realmente transcurrido. No representa "
                "la rentabilidad obtenida cada año."
            )

            # --------------------------------------------------------
            # EVOLUCIÓN DE UNA INVERSIÓN
            # --------------------------------------------------------

            st.markdown(
                "#### Evolución de una inversión"
            )

            control_1, control_2 = st.columns(
                [1.35, 1]
            )

            with control_1:
                periodo_inversion = st.selectbox(
                    "Periodo de la inversión",
                    options=[
                        "1y",
                        "3y",
                        "5y",
                        "10y",
                    ],
                    format_func=lambda x: {
                        "1y": "1 año",
                        "3y": "3 años",
                        "5y": "5 años",
                        "10y": "10 años",
                    }[x],
                    index=2,
                    key="nivel2_inversion_periodo",
                )

            with control_2:
                incluir_dividendos = st.toggle(
                    "Usar rentabilidad total ajustada",
                    value=True,
                    key="nivel2_incluir_dividendos",
                    help=(
                        "Utiliza la serie Adj Close cuando "
                        "está disponible."
                    ),
                )

            inversion = evolucion_inversion(
                ticker,
                capital_inicial=1000.0,
                periodo=periodo_inversion,
                incluir_dividendos=incluir_dividendos,
            )

            if inversion:
                inv_1, inv_2, inv_3 = st.columns(3)

                inv_1.metric(
                    "Capital inicial",
                    "1.000",
                )

                inv_2.metric(
                    "Capital final",
                    f"{inversion['capital_final']:,.2f}",
                )

                inv_3.metric(
                    "Rentabilidad",
                    f"{inversion['rentabilidad_pct']:+.2f}%",
                )

                fecha_inicio_inv = (
                    inversion["fecha_inicio"]
                    .strftime("%d/%m/%Y")
                )

                fecha_fin_inv = (
                    inversion["fecha_fin"]
                    .strftime("%d/%m/%Y")
                )

                metodologia_inv = (
                    "rentabilidad total ajustada"
                    if incluir_dividendos
                    else "rentabilidad del precio"
                )

                st.caption(
                    f"Del {fecha_inicio_inv} al "
                    f"{fecha_fin_inv} · "
                    f"{metodologia_inv}. "
                    "Simulación sin comisiones, impuestos ni "
                    "efecto del cambio de divisa."
                )

            else:
                st.info(
                    "No hay historial suficiente para calcular "
                    "esta simulación."
                )

        else:
            st.info(
                "No hay suficiente historial disponible para "
                "calcular el rendimiento de este activo."
            )



    if modulo_finscope == "Comparación con el mercado":

        # ============================================================
        # NIVEL 2 · BENCHMARK
        # ============================================================

        st.markdown(
            '<div class="section-kicker">COMPARACIÓN</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="fs-big-title">Benchmark</div>',
            unsafe_allow_html=True,
        )

        st.caption(
            "Compara la evolución de la inversión con un índice de "
            "referencia durante exactamente el mismo intervalo."
        )

        bench_control_1, bench_control_2 = st.columns(
            [1, 1]
        )

        with bench_control_1:
            periodo_benchmark = st.selectbox(
                "Periodo de comparación",
                options=[
                    "1y",
                    "3y",
                    "5y",
                    "10y",
                ],
                format_func=lambda x: {
                    "1y": "1 año",
                    "3y": "3 años",
                    "5y": "5 años",
                    "10y": "10 años",
                }[x],
                index=2,
                key="nivel2_benchmark_periodo",
            )

        with bench_control_2:
            total_benchmark = st.toggle(
                "Usar rentabilidad total ajustada",
                value=True,
                key="nivel2_benchmark_total",
                help=(
                    "Cuando está disponible, utiliza Adj Close "
                    "para ambos activos."
                ),
            )

        if ticker == "^GSPC":
            benchmark_ticker = "^NDX"
            benchmark_nombre = "Nasdaq 100"
        else:
            benchmark_ticker = "^GSPC"
            benchmark_nombre = "S&P 500"

        with st.spinner(
            "Comparando con el benchmark..."
        ):
            comparacion_benchmark = calcular_benchmark(
                ticker,
                benchmark=benchmark_ticker,
                periodo=periodo_benchmark,
                incluir_dividendos=total_benchmark,
            )

        if comparacion_benchmark:

            retorno_empresa = (
                comparacion_benchmark[
                    "rentabilidad_activo_pct"
                ]
            )

            retorno_indice = (
                comparacion_benchmark[
                    "rentabilidad_benchmark_pct"
                ]
            )

            diferencia_retorno = (
                comparacion_benchmark[
                    "diferencia_rentabilidad_pp"
                ]
            )

            bench_1, bench_2, bench_3 = st.columns(3)

            bench_1.metric(
                f"{ticker} · Rentabilidad",
                f"{retorno_empresa:+.2f}%",
            )

            bench_2.metric(
                f"{benchmark_nombre} · Rentabilidad",
                f"{retorno_indice:+.2f}%",
            )

            bench_3.metric(
                "Diferencia",
                f"{diferencia_retorno:+.2f} pp",
            )

            cagr_empresa = (
                comparacion_benchmark.get(
                    "cagr_activo_pct"
                )
            )

            cagr_indice = (
                comparacion_benchmark.get(
                    "cagr_benchmark_pct"
                )
            )

            diferencia_cagr = (
                comparacion_benchmark.get(
                    "diferencia_cagr_pp"
                )
            )

            if (
                cagr_empresa is not None
                and cagr_indice is not None
            ):
                cagr_b1, cagr_b2, cagr_b3 = st.columns(3)

                cagr_b1.metric(
                    f"{ticker} · CAGR",
                    f"{cagr_empresa:+.2f}%",
                )

                cagr_b2.metric(
                    f"{benchmark_nombre} · CAGR",
                    f"{cagr_indice:+.2f}%",
                )

                cagr_b3.metric(
                    "Diferencia CAGR",
                    (
                        f"{diferencia_cagr:+.2f} pp"
                        if diferencia_cagr is not None
                        else "N/D"
                    ),
                )

            st.markdown(
                "#### Evolución comparada"
            )

            serie_benchmark = (
                comparacion_benchmark[
                    "serie_normalizada"
                ]
                .reset_index()
            )

            fecha_col = (
                serie_benchmark.columns[0]
            )

            serie_benchmark = (
                serie_benchmark.rename(
                    columns={
                        fecha_col: "Fecha",
                        "Activo_normalizado":
                            ticker,
                        "Benchmark_normalizado":
                            benchmark_nombre,
                    }
                )
            )

            serie_larga = serie_benchmark.melt(
                id_vars=["Fecha"],
                value_vars=[
                    ticker,
                    benchmark_nombre,
                ],
                var_name="Serie",
                value_name="Índice",
            )

            grafico_benchmark = (
                alt.Chart(
                    serie_larga
                )
                .mark_line(
                    strokeWidth=2.4
                )
                .encode(
                    x=alt.X(
                        "Fecha:T",
                        title=None,
                    ),
                    y=alt.Y(
                        "Índice:Q",
                        title="Valor normalizado",
                        scale=alt.Scale(
                            zero=False
                        ),
                    ),
                    color=alt.Color(
                        "Serie:N",
                        title=None,
                    ),
                    tooltip=[
                        alt.Tooltip(
                            "Fecha:T",
                            title="Fecha",
                        ),
                        alt.Tooltip(
                            "Serie:N",
                            title="Serie",
                        ),
                        alt.Tooltip(
                            "Índice:Q",
                            title="Valor",
                            format=".2f",
                        ),
                    ],
                )
                .properties(
                    height=380
                )
            )

            st.altair_chart(
                grafico_benchmark,
                width="stretch",
            )

            st.caption(
                "Ambas series comienzan en 100 en la misma fecha. "
                "Esto permite comparar su crecimiento relativo sin "
                "que el precio nominal de cada activo distorsione "
                "la visualización."
            )

            st.markdown(
                "#### ¿Qué habría ocurrido con 1.000?"
            )

            capital_activo = (
                comparacion_benchmark[
                    "capital_final_activo"
                ]
            )

            capital_benchmark = (
                comparacion_benchmark[
                    "capital_final_benchmark"
                ]
            )

            capital_diferencia = (
                capital_activo
                - capital_benchmark
            )

            cap_1, cap_2, cap_3 = st.columns(3)

            cap_1.metric(
                ticker,
                f"{capital_activo:,.2f}",
            )

            cap_2.metric(
                benchmark_nombre,
                f"{capital_benchmark:,.2f}",
            )

            cap_3.metric(
                "Diferencia",
                f"{capital_diferencia:+,.2f}",
            )

            fecha_bench_inicio = (
                comparacion_benchmark[
                    "fecha_inicio"
                ].strftime("%d/%m/%Y")
            )

            fecha_bench_fin = (
                comparacion_benchmark[
                    "fecha_fin"
                ].strftime("%d/%m/%Y")
            )

            metodologia_benchmark = (
                "rentabilidad total ajustada"
                if comparacion_benchmark[
                    "metodologia"
                ] == "total_return_ajustado"
                else "rentabilidad del precio"
            )

            st.caption(
                f"Periodo común: {fecha_bench_inicio} – "
                f"{fecha_bench_fin} · "
                f"{metodologia_benchmark}. "
                "La simulación parte de 1.000 en cada alternativa "
                "y no incluye comisiones, impuestos ni efecto divisa."
            )

        else:
            st.info(
                "No hay historial común suficiente para realizar "
                "la comparación con el benchmark."
            )




    if modulo_finscope == "Riesgo histórico":

        # ============================================================
        # NIVEL 2 · RIESGO
        # ============================================================

        st.markdown(
            '<div class="section-kicker">RIESGO</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="fs-big-title">Riesgo histórico</div>',
            unsafe_allow_html=True,
        )

        st.caption(
            "Analiza cuánto ha fluctuado históricamente el activo "
            "y cuál ha sido su mayor caída desde un máximo previo. "
            "Estas métricas describen el pasado y no predicen el "
            "riesgo futuro."
        )

        riesgo_control_1, riesgo_control_2 = st.columns(
            [1, 1]
        )

        with riesgo_control_1:
            periodo_riesgo = st.selectbox(
                "Periodo de riesgo",
                options=[
                    "1y",
                    "3y",
                    "5y",
                    "10y",
                ],
                format_func=lambda x: {
                    "1y": "1 año",
                    "3y": "3 años",
                    "5y": "5 años",
                    "10y": "10 años",
                }[x],
                index=2,
                key="nivel2_riesgo_periodo",
            )

        with riesgo_control_2:
            riesgo_ajustado = st.toggle(
                "Usar precios ajustados",
                value=True,
                key="nivel2_riesgo_ajustado",
                help=(
                    "Utiliza Adj Close cuando está disponible."
                ),
            )

        with st.spinner(
            "Calculando riesgo histórico..."
        ):
            metricas_riesgo = (
                calcular_metricas_riesgo(
                    ticker,
                    periodo=periodo_riesgo,
                    usar_ajustado=riesgo_ajustado,
                )
            )

        if metricas_riesgo:

            volatilidad = metricas_riesgo[
                "volatilidad_anual_pct"
            ]

            max_drawdown = metricas_riesgo[
                "max_drawdown_pct"
            ]

            fecha_pico = metricas_riesgo[
                "fecha_pico"
            ]

            fecha_valle = metricas_riesgo[
                "fecha_valle"
            ]

            fecha_recuperacion = (
                metricas_riesgo[
                    "fecha_recuperacion"
                ]
            )

            riesgo_1, riesgo_2, riesgo_3 = st.columns(3)

            riesgo_1.metric(
                "Volatilidad anualizada",
                f"{volatilidad:.2f}%",
            )

            riesgo_2.metric(
                "Máximo drawdown",
                f"{max_drawdown:.2f}%",
            )

            if fecha_recuperacion is not None:
                texto_recuperacion = (
                    fecha_recuperacion
                    .strftime("%d/%m/%Y")
                )
            else:
                texto_recuperacion = (
                    "No recuperado"
                )

            riesgo_3.metric(
                "Recuperación del máximo",
                texto_recuperacion,
            )

            st.caption(
                "Volatilidad anualizada a partir de rendimientos "
                "diarios y 252 sesiones bursátiles por año. "
                "Una volatilidad mayor indica que los rendimientos "
                "históricos han presentado una dispersión mayor."
            )

            # --------------------------------------------------------
            # EPISODIO DE MÁXIMO DRAWDOWN
            # --------------------------------------------------------

            st.markdown(
                "#### Mayor caída del periodo"
            )

            dd_1, dd_2, dd_3 = st.columns(3)

            dd_1.metric(
                "Máximo previo",
                fecha_pico.strftime(
                    "%d/%m/%Y"
                ),
            )

            dd_2.metric(
                "Mínimo posterior",
                fecha_valle.strftime(
                    "%d/%m/%Y"
                ),
            )

            dd_3.metric(
                "Duración de la caída",
                f"{metricas_riesgo['dias_caida']} días",
            )

            if fecha_recuperacion is None:
                st.caption(
                    "Dentro del periodo analizado, el precio todavía "
                    "no había recuperado el máximo previo al finalizar "
                    "la serie."
                )
            else:
                st.caption(
                    "Desde el mínimo fueron necesarios "
                    f"{metricas_riesgo['dias_recuperacion']} días "
                    "naturales para volver al nivel del máximo previo."
                )

            # --------------------------------------------------------
            # GRÁFICO DRAWDOWN
            # --------------------------------------------------------

            st.markdown(
                "#### Drawdown a lo largo del tiempo"
            )

            serie_dd = (
                metricas_riesgo[
                    "serie_drawdown"
                ][["Drawdown_pct"]]
                .reset_index()
            )

            fecha_dd_col = (
                serie_dd.columns[0]
            )

            serie_dd = serie_dd.rename(
                columns={
                    fecha_dd_col: "Fecha",
                    "Drawdown_pct": "Drawdown",
                }
            )

            grafico_drawdown = (
                alt.Chart(
                    serie_dd
                )
                .mark_area(
                    opacity=0.55
                )
                .encode(
                    x=alt.X(
                        "Fecha:T",
                        title=None,
                    ),
                    y=alt.Y(
                        "Drawdown:Q",
                        title="Drawdown (%)",
                    ),
                    tooltip=[
                        alt.Tooltip(
                            "Fecha:T",
                            title="Fecha",
                        ),
                        alt.Tooltip(
                            "Drawdown:Q",
                            title="Drawdown",
                            format=".2f",
                        ),
                    ],
                )
                .properties(
                    height=330
                )
            )

            linea_cero = (
                alt.Chart(
                    serie_dd
                )
                .mark_rule(
                    opacity=0.35
                )
                .encode(
                    y=alt.datum(0)
                )
            )

            st.altair_chart(
                grafico_drawdown
                + linea_cero,
                width="stretch",
            )

            fecha_riesgo_inicio = (
                metricas_riesgo[
                    "fecha_inicio"
                ].strftime("%d/%m/%Y")
            )

            fecha_riesgo_fin = (
                metricas_riesgo[
                    "fecha_fin"
                ].strftime("%d/%m/%Y")
            )

            st.caption(
                f"Periodo: {fecha_riesgo_inicio} – "
                f"{fecha_riesgo_fin} · "
                f"{metricas_riesgo['observaciones']} "
                "rendimientos diarios · "
                f"{metricas_riesgo['metodologia']} · "
                "Fuente: Yahoo Finance vía yfinance."
            )

        else:
            st.info(
                "No hay suficiente historial para calcular "
                "las métricas de riesgo."
            )




    if modulo_finscope == "Beta y correlación":

        # ============================================================
        # NIVEL 2 · BETA Y CORRELACIÓN
        # ============================================================

        st.markdown(
            '<div class="section-kicker">MERCADO Y RIESGO</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="fs-big-title">Sensibilidad al mercado</div>',
            unsafe_allow_html=True,
        )

        st.caption(
            "Beta mide cuánto han respondido históricamente los "
            "rendimientos del activo a los movimientos del mercado. "
            "La correlación mide hasta qué punto ambos han tendido a "
            "moverse en la misma dirección."
        )

        bc_control_1, bc_control_2 = st.columns(
            [1, 1]
        )

        with bc_control_1:
            periodo_bc = st.selectbox(
                "Periodo de análisis",
                options=[
                    "1y",
                    "3y",
                    "5y",
                    "10y",
                ],
                format_func=lambda x: {
                    "1y": "1 año",
                    "3y": "3 años",
                    "5y": "5 años",
                    "10y": "10 años",
                }[x],
                index=2,
                key="nivel2_beta_periodo",
            )

        with bc_control_2:
            bc_ajustado = st.toggle(
                "Usar precios ajustados",
                value=True,
                key="nivel2_beta_ajustado",
                help=(
                    "Utiliza Adj Close cuando está disponible."
                ),
            )

        benchmark_bc = "^GSPC"
        benchmark_bc_nombre = "S&P 500"

        with st.spinner(
            "Calculando sensibilidad al mercado..."
        ):
            resultado_bc = calcular_beta_correlacion(
                ticker,
                benchmark=benchmark_bc,
                periodo=periodo_bc,
                usar_ajustado=bc_ajustado,
            )

        if resultado_bc:

            beta_bc = resultado_bc["beta"]
            correlacion_bc = resultado_bc[
                "correlacion"
            ]

            bc_1, bc_2, bc_3 = st.columns(3)

            bc_1.metric(
                "Beta",
                f"{beta_bc:.2f}",
            )

            bc_2.metric(
                f"Correlación · {benchmark_bc_nombre}",
                f"{correlacion_bc:.2f}",
            )

            bc_3.metric(
                "Observaciones",
                f"{resultado_bc['observaciones']:,}",
            )

            # --------------------------------------------------------
            # INTERPRETACIÓN DESCRIPTIVA DE BETA
            # --------------------------------------------------------

            if beta_bc < 0:
                lectura_beta = (
                    "La beta histórica es negativa. Durante este "
                    "periodo, la relación estimada con los movimientos "
                    "del mercado ha sido inversa."
                )

            elif beta_bc < 0.8:
                lectura_beta = (
                    "La acción ha mostrado históricamente una "
                    "sensibilidad menor que la del mercado."
                )

            elif beta_bc <= 1.2:
                lectura_beta = (
                    "La acción ha mostrado históricamente una "
                    "sensibilidad relativamente próxima a la del mercado."
                )

            else:
                lectura_beta = (
                    "La acción ha mostrado históricamente una "
                    "sensibilidad mayor que la del mercado."
                )

            # --------------------------------------------------------
            # INTERPRETACIÓN DESCRIPTIVA DE CORRELACIÓN
            # --------------------------------------------------------

            abs_corr = abs(correlacion_bc)

            if abs_corr < 0.2:
                intensidad_corr = "muy débil"
            elif abs_corr < 0.4:
                intensidad_corr = "débil"
            elif abs_corr < 0.6:
                intensidad_corr = "moderada"
            elif abs_corr < 0.8:
                intensidad_corr = "elevada"
            else:
                intensidad_corr = "muy elevada"

            if correlacion_bc > 0.05:
                direccion_corr = "positiva"
            elif correlacion_bc < -0.05:
                direccion_corr = "negativa"
            else:
                direccion_corr = "prácticamente neutra"

            st.markdown(
                "#### Cómo interpretar estos datos"
            )

            st.info(
                f"{lectura_beta} "
                f"La correlación con {benchmark_bc_nombre} ha sido "
                f"{direccion_corr} y {intensidad_corr} "
                f"({correlacion_bc:.2f})."
            )

            st.caption(
                "Una beta de 1 representa una sensibilidad histórica "
                "similar a la del mercado. Una beta superior a 1 indica "
                "mayor sensibilidad y una inferior a 1, menor. "
                "La correlación se mueve entre −1 y +1 y mide la "
                "intensidad de la relación lineal entre los rendimientos."
            )

            fecha_bc_inicio = (
                resultado_bc["fecha_inicio"]
                .strftime("%d/%m/%Y")
            )

            fecha_bc_fin = (
                resultado_bc["fecha_fin"]
                .strftime("%d/%m/%Y")
            )

            st.caption(
                f"Periodo común: {fecha_bc_inicio} – "
                f"{fecha_bc_fin} · "
                f"{resultado_bc['observaciones']} rendimientos "
                f"diarios · {resultado_bc['metodologia']} · "
                f"Benchmark: {benchmark_bc_nombre} ({benchmark_bc}) · "
                "Fuente: Yahoo Finance vía yfinance. "
                "Beta y correlación son estimaciones históricas y "
                "pueden cambiar significativamente según el periodo."
            )

        else:
            st.info(
                "No existe suficiente historial común con el "
                "S&P 500 para calcular beta y correlación."
            )




    if modulo_finscope == "Rentabilidad ajustada al riesgo":

        # ============================================================
        # NIVEL 2 · SHARPE Y SORTINO
        # ============================================================

        st.markdown(
            '<div class="section-kicker">RENTABILIDAD Y RIESGO</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="fs-big-title">Rentabilidad ajustada al riesgo</div>',
            unsafe_allow_html=True,
        )

        st.caption(
            "Sharpe relaciona el exceso de rentabilidad con toda la "
            "volatilidad histórica. Sortino utiliza únicamente las "
            "desviaciones negativas respecto al objetivo."
        )

        ras_control_1, ras_control_2 = st.columns(
            [1, 1]
        )

        with ras_control_1:
            periodo_ras = st.selectbox(
                "Periodo de análisis",
                options=[
                    "1y",
                    "3y",
                    "5y",
                    "10y",
                ],
                format_func=lambda x: {
                    "1y": "1 año",
                    "3y": "3 años",
                    "5y": "5 años",
                    "10y": "10 años",
                }[x],
                index=2,
                key="nivel2_ras_periodo",
            )

        with ras_control_2:
            ras_ajustado = st.toggle(
                "Usar precios ajustados",
                value=True,
                key="nivel2_ras_ajustado",
                help=(
                    "Utiliza Adj Close cuando está disponible."
                ),
            )

        tasa_rf = st.number_input(
            "Tasa libre de riesgo anual (%)",
            min_value=0.0,
            max_value=20.0,
            value=0.0,
            step=0.25,
            format="%.2f",
            key="nivel2_tasa_rf",
            help=(
                "Se utiliza para calcular el exceso de rentabilidad. "
                "FinScope no presupone una tasa actual: el valor "
                "introducido queda visible y forma parte del cálculo."
            ),
        )

        with st.spinner(
            "Calculando rentabilidad ajustada al riesgo..."
        ):
            resultado_ras = (
                calcular_rentabilidad_ajustada_riesgo(
                    ticker,
                    periodo=periodo_ras,
                    tasa_libre_riesgo_pct=tasa_rf,
                    usar_ajustado=ras_ajustado,
                )
            )

        if resultado_ras:

            sharpe_ras = resultado_ras[
                "sharpe"
            ]

            sortino_ras = resultado_ras[
                "sortino"
            ]

            downside_ras = resultado_ras[
                "downside_deviation_anual_pct"
            ]

            ras_1, ras_2, ras_3 = st.columns(3)

            ras_1.metric(
                "Ratio de Sharpe",
                (
                    f"{sharpe_ras:.2f}"
                    if sharpe_ras is not None
                    else "N/D"
                ),
            )

            ras_2.metric(
                "Ratio de Sortino",
                (
                    f"{sortino_ras:.2f}"
                    if sortino_ras is not None
                    else "N/D"
                ),
            )

            ras_3.metric(
                "Downside deviation anualizada",
                f"{downside_ras:.2f}%",
            )

            st.markdown(
                "#### Cómo leer estas métricas"
            )

            st.info(
                "Sharpe muestra cuánto exceso de rentabilidad histórica "
                "se obtuvo por unidad de volatilidad total. Sortino "
                "cambia el enfoque y penaliza únicamente las "
                "desviaciones por debajo del objetivo. Por eso ambos "
                "ratios pueden ser diferentes incluso utilizando la "
                "misma acción y el mismo periodo."
            )

            st.caption(
                "Un valor mayor implica más exceso de rentabilidad "
                "histórica por unidad del riesgo medido por cada ratio; "
                "no implica que la inversión sea mejor ni predice "
                "rentabilidades futuras. Los ratios también pueden "
                "cambiar de forma importante al modificar el periodo "
                "o la tasa libre de riesgo."
            )

            fecha_ras_inicio = (
                resultado_ras[
                    "fecha_inicio"
                ].strftime("%d/%m/%Y")
            )

            fecha_ras_fin = (
                resultado_ras[
                    "fecha_fin"
                ].strftime("%d/%m/%Y")
            )

            st.caption(
                f"Periodo: {fecha_ras_inicio} – "
                f"{fecha_ras_fin} · "
                f"{resultado_ras['observaciones']} rendimientos "
                f"diarios · {resultado_ras['metodologia']} · "
                f"Tasa libre de riesgo anual utilizada: "
                f"{resultado_ras['tasa_libre_riesgo_pct']:.2f}% · "
                "Anualización: 252 sesiones · "
                "Fuente: Yahoo Finance vía yfinance."
            )

        else:
            st.info(
                "No existe suficiente historial para calcular "
                "Sharpe y Sortino."
            )




    if es_empresa:
        if modulo_finscope == "Valoración histórica":

            # ============================================================
            # NIVEL 2 · PER HISTORICO
            # ============================================================

            st.markdown(
                '<div class="section-kicker">VALORACIÓN</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                '<div class="fs-big-title">Valoración histórica</div>',
                unsafe_allow_html=True,
            )

            st.caption(
                "Compara cuánto ha pagado históricamente el mercado "
                "por cada unidad de beneficio de la empresa."
            )

            periodo_per = st.selectbox(
                "Periodo de valoración",
                ["1y", "3y", "5y", "10y"],
                format_func=lambda x: {
                    "1y": "1 año",
                    "3y": "3 años",
                    "5y": "5 años",
                    "10y": "10 años",
                }[x],
                index=2,
                key="nivel2_per_periodo",
            )

            with st.spinner(
                "Reconstruyendo valoración histórica..."
            ):
                resultado_per = cache_calcular_per_historico(
                    ticker,
                    periodo_per,
                )

            if resultado_per:

                p1, p2, p3 = st.columns(3)

                p1.metric(
                    "PER actual calculado",
                    f"{resultado_per['per_actual']:.1f}x",
                )

                p2.metric(
                    "Mediana histórica",
                    f"{resultado_per['per_mediana']:.1f}x",
                )

                p3.metric(
                    "Percentil actual",
                    f"{resultado_per['percentil_actual']:.0f}",
                )

                serie_per = resultado_per["serie"].copy()

                grafico_per = (
                    alt.Chart(serie_per)
                    .mark_line()
                    .encode(
                        x=alt.X(
                            "fecha:T",
                            title=None,
                        ),
                        y=alt.Y(
                            "per:Q",
                            title="PER",
                            scale=alt.Scale(
                                zero=False
                            ),
                        ),
                        tooltip=[
                            alt.Tooltip(
                                "fecha:T",
                                title="Fecha",
                            ),
                            alt.Tooltip(
                                "precio:Q",
                                title="Precio",
                                format=".2f",
                            ),
                            alt.Tooltip(
                                "eps_ttm:Q",
                                title="EPS TTM",
                                format=".2f",
                            ),
                            alt.Tooltip(
                                "per:Q",
                                title="PER",
                                format=".2f",
                            ),
                        ],
                    )
                )

                mediana_per = (
                    alt.Chart(serie_per)
                    .mark_rule(
                        strokeDash=[6, 6],
                    )
                    .encode(
                        y=alt.datum(
                            resultado_per[
                                "per_mediana"
                            ]
                        )
                    )
                )

                st.altair_chart(
                    grafico_per + mediana_per,
                    width="stretch",
                )

                r1, r2, r3 = st.columns(3)

                r1.metric(
                    "PER mínimo",
                    f"{resultado_per['per_min']:.1f}x",
                )

                r2.metric(
                    "PER medio",
                    f"{resultado_per['per_media']:.1f}x",
                )

                r3.metric(
                    "PER máximo",
                    f"{resultado_per['per_max']:.1f}x",
                )

                st.markdown(
                    "#### Cómo leerlo"
                )

                st.info(
                    "El percentil muestra qué proporción de las "
                    "observaciones del periodo tuvo un PER igual o "
                    "inferior al actual. Un percentil 80 significa "
                    "que aproximadamente el 80% de las observaciones "
                    "presentaron un PER inferior o igual al actual."
                )

                st.caption(
                    "Un PER superior o inferior a su histórico no "
                    "determina por sí solo que una acción esté cara "
                    "o barata. La valoración también depende del "
                    "crecimiento, expectativas, tipos de interés, "
                    "riesgo y evolución de los beneficios."
                )

                fecha_per_inicio = resultado_per[
                    "fecha_inicio"
                ].strftime("%d/%m/%Y")

                fecha_per_fin = resultado_per[
                    "fecha_fin"
                ].strftime("%d/%m/%Y")

                st.caption(
                    f"Periodo disponible: {fecha_per_inicio} – "
                    f"{fecha_per_fin} · "
                    f"{resultado_per['observaciones']} observaciones · "
                    f"{resultado_per['metodologia']} · "
                    f"Fuente: {resultado_per['fuente']}."
                )

                st.warning(
                    "Esta no es una serie histórica oficial de PER "
                    "publicada por Yahoo Finance. FinScope reconstruye "
                    "el múltiplo utilizando precios históricos y los "
                    "datos de EPS trimestral disponibles. La cobertura "
                    "real puede ser menor que el periodo seleccionado."
                )

            else:
                st.info(
                    "No hay suficiente información histórica de "
                    "beneficios para reconstruir el PER."
                )



        if modulo_finscope == "Evolución del negocio":

            st.markdown(
                '<div class="fs-big-title">Evolución del negocio</div>',
                unsafe_allow_html=True,
            )

            st.caption(
                "Datos anuales disponibles. "
                "FinScope no rellena ni estima ejercicios ausentes."
            )

            historico_fundamental = cache_obtener_historico_fundamental(ticker)
            historial_f = historico_fundamental.get("datos", [])

            if historial_f:

                df_f = pd.DataFrame(historial_f)

                # -----------------------------------------------------
                # RESULTADOS
                # -----------------------------------------------------

                columnas_resultados = [
                    c for c in [
                        "ingresos",
                        "beneficio_neto",
                        "ebitda",
                    ]
                    if c in df_f.columns
                ]

                if columnas_resultados:
                    st.markdown(
                        '<div class="mini-title">'
                        'Resultados históricos'
                        '</div>',
                        unsafe_allow_html=True,
                    )

                    nombres = {
                        "ingresos": "Ingresos",
                        "beneficio_neto": "Beneficio neto",
                        "ebitda": "EBITDA",
                    }

                    grafico = (
                        df_f[
                            ["anio"] + columnas_resultados
                        ]
                        .set_index("anio")
                        .rename(columns=nombres)
                    )

                    # Mostrar en miles de millones para mejorar lectura.
                    grafico = grafico / 1_000_000_000

                    st.line_chart(
                        grafico,
                        width="stretch",
                    )

                    st.caption(
                        "Importes del gráfico en miles de millones "
                        f"de {datos.get('moneda') or 'la moneda reportada'}."
                    )

                # -----------------------------------------------------
                # FCF · SOLO NO FINANCIERAS
                # -----------------------------------------------------

                if (
                    not datos.get("es_financiera")
                    and "free_cash_flow" in df_f.columns
                ):
                    st.markdown(
                        '<div class="mini-title">'
                        'Generación de caja'
                        '</div>',
                        unsafe_allow_html=True,
                    )

                    grafico_fcf = (
                        df_f[
                            ["anio", "free_cash_flow"]
                        ]
                        .set_index("anio")
                        .rename(
                            columns={
                                "free_cash_flow": "Free Cash Flow"
                            }
                        )
                        / 1_000_000_000
                    )

                    st.bar_chart(
                        grafico_fcf,
                        width="stretch",
                    )

                    st.caption(
                        "Free Cash Flow anual. "
                        "No se utiliza esta métrica convencional "
                        "para entidades financieras."
                    )

                # -----------------------------------------------------
                # BALANCE
                # -----------------------------------------------------

                columnas_balance = [
                    c for c in [
                        "deuda_total",
                        "efectivo",
                        "patrimonio",
                        "activos_totales",
                    ]
                    if c in df_f.columns
                ]

                if columnas_balance:
                    st.markdown(
                        '<div class="mini-title">'
                        'Evolución del balance'
                        '</div>',
                        unsafe_allow_html=True,
                    )

                    nombres_balance = {
                        "deuda_total": "Deuda",
                        "efectivo": "Caja",
                        "patrimonio": "Patrimonio",
                        "activos_totales": "Activos",
                    }

                    grafico_balance = (
                        df_f[
                            ["anio"] + columnas_balance
                        ]
                        .set_index("anio")
                        .rename(columns=nombres_balance)
                        / 1_000_000_000
                    )

                    st.line_chart(
                        grafico_balance,
                        width="stretch",
                    )

                    if datos.get("es_financiera"):
                        st.caption(
                            "En entidades financieras, deuda, caja y "
                            "activos tienen una estructura distinta a la "
                            "de compañías no financieras y deben "
                            "interpretarse dentro del negocio bancario."
                        )
                    else:
                        st.caption(
                            "Importes en miles de millones de "
                            f"{datos.get('moneda') or 'la moneda reportada'}."
                        )

                # -----------------------------------------------------
                # TABLA ANUAL
                # -----------------------------------------------------

                with st.expander("Ver datos históricos anuales"):
                    tabla = df_f.copy()

                    nombres_tabla = {
                        "fecha": "Fecha",
                        "anio": "Año",
                        "ingresos": "Ingresos",
                        "beneficio_neto": "Beneficio neto",
                        "ebitda": "EBITDA",
                        "free_cash_flow": "FCF",
                        "flujo_caja_operativo": "Flujo operativo",
                        "deuda_total": "Deuda",
                        "efectivo": "Caja",
                        "patrimonio": "Patrimonio",
                        "activos_totales": "Activos",
                    }

                    tabla = tabla.rename(columns=nombres_tabla)

                    columnas_dinero = [
                        c for c in tabla.columns
                        if c not in ["Fecha", "Año"]
                    ]

                    for c in columnas_dinero:
                        tabla[c] = tabla[c].apply(
                            lambda x: (
                                formato_numero(x)
                                if pd.notna(x)
                                else "—"
                            )
                        )

                    st.dataframe(
                        tabla,
                        width="stretch",
                        hide_index=True,
                    )

            else:
                st.info(
                    "No hay suficientes datos históricos anuales "
                    "disponibles para esta empresa."
                )

            # VALORACIÓN

            st.markdown(
                '<div class="section-kicker analysis-section">VALORACIÓN</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="fs-big-title">Múltiplos</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                '<p class="analysis-caption">Indicadores para contextualizar cuánto está pagando el mercado por el negocio.</p>',
                unsafe_allow_html=True
            )

            c1, c2, c3 = st.columns(3)

            if datos["per"] is not None:
                c1.metric(
                    "PER",
                    f'{datos["per"]:.2f}x'
                )

            if datos["price_to_book"] is not None:
                c2.metric(
                    "Price / Book",
                    f'{datos["price_to_book"]:.2f}x'
                )

            if datos["ev_ebitda"] is not None:
                c3.metric(
                    "EV / EBITDA",
                    f'{datos["ev_ebitda"]:.2f}x'
                )


            # CRECIMIENTO

            st.markdown(
                '<div class="section-kicker analysis-section">CRECIMIENTO</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="fs-big-title">Evolución</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                '<p class="analysis-caption">Ritmo reciente de crecimiento de los principales resultados de la empresa.</p>',
                unsafe_allow_html=True
            )

            c1, c2 = st.columns(2)

            if datos["crecimiento_ingresos"] is not None:
                c1.metric(
                    "Crecimiento de ingresos",
                    f'{datos["crecimiento_ingresos"] * 100:.2f}%'
                )

            if datos["crecimiento_beneficios"] is not None:
                c2.metric(
                    "Crecimiento de beneficios",
                    f'{datos["crecimiento_beneficios"] * 100:.2f}%'
                )



        if modulo_finscope == "FinScope Insight":

            # FINSCOPE INSIGHT

            st.divider()

            st.markdown(
                '<div class="section-kicker analysis-section">FINSCOPE INSIGHT</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="fs-big-title">Lectura de la empresa</div>',
                unsafe_allow_html=True,
            )

            st.caption(
                "FinScope organiza los datos disponibles para explicar "
                "crecimiento, rentabilidad, balance, caja y valoración. "
                "La lectura se adapta al tipo de empresa y al contexto "
                "de los datos disponibles."
            )

            analisis = analizar_empresa(datos)

            resumen_1, resumen_2, resumen_3 = st.columns(3)

            resumen_1.metric(
                "Crecimiento ingresos",
                formato_numero(
                    datos.get("crecimiento_ingresos"),
                    "porcentaje"
                )
            )

            resumen_2.metric(
                "ROE",
                formato_numero(
                    datos.get("roe"),
                    "porcentaje"
                )
            )

            resumen_3.metric(
                "PER",
                formato_numero(
                    datos.get("per"),
                    "multiplo"
                )
            )

            tipo_empresa = analisis.get("tipo_empresa")

            if tipo_empresa == "financiera":
                st.info(
                    "Empresa financiera · FinScope adapta el análisis porque "
                    "determinadas métricas convencionales de deuda, liquidez y "
                    "flujo de caja no son directamente comparables con las de "
                    "una empresa no financiera."
                )

            nombres_bloques = {
                "crecimiento": "Crecimiento",
                "rentabilidad": "Rentabilidad",
                "caja": "Caja",
                "balance": "Balance",
                "valoracion": "Valoración",
            }

            for bloque, elementos in analisis.get("bloques", {}).items():

                if not elementos:
                    continue

                st.markdown(
                    f"### {nombres_bloques.get(bloque, bloque.title())}"
                )

                for elemento in elementos:

                    texto = elemento.get("texto")

                    if texto:
                        st.write(f"• {texto}")

                    advertencia = elemento.get("advertencia")

                    if advertencia:
                        st.caption(f"↳ {advertencia}")


        # ============================================================
        # NIVEL 2 · DIVIDENDOS
        # ============================================================

        if modulo_finscope == "Dividendos":

            st.markdown(
                '<div class="section-kicker">RENTA DEL ACCIONISTA</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                '<div class="fs-big-title">Dividendos</div>',
                unsafe_allow_html=True,
            )

            st.caption(
                "Historial de pagos en efectivo por acción. "
                "Los importes proceden de Yahoo Finance vía yfinance "
                "y no incluyen impuestos."
            )

            with st.spinner(
                "Analizando historial de dividendos..."
            ):
                datos_dividendos = (
                    cache_obtener_dividendos_historicos(
                        ticker,
                        anios=10,
                    )
                )

            if not datos_dividendos["tiene_dividendos"]:

                st.info(
                    "No se han encontrado dividendos en efectivo "
                    "positivos para este activo en los datos "
                    "disponibles. Esto puede significar que la "
                    "empresa no reparte dividendos o que la fuente "
                    "no dispone del historial."
                )

            else:

                div_ttm = datos_dividendos[
                    "dividendo_ttm"
                ]

                ultimo_div = datos_dividendos[
                    "ultimo_dividendo"
                ]

                fecha_ultimo = datos_dividendos[
                    "fecha_ultimo_dividendo"
                ]

                frecuencia = datos_dividendos[
                    "frecuencia_aprox"
                ] or "No determinada"

                d1, d2, d3, d4 = st.columns(4)

                d1.metric(
                    "Dividendo últimos 12 meses",
                    (
                        f"{div_ttm:.4f}"
                        if div_ttm is not None
                        else "N/D"
                    ),
                )

                d2.metric(
                    "Último pago por acción",
                    (
                        f"{ultimo_div:.4f}"
                        if ultimo_div is not None
                        else "N/D"
                    ),
                )

                d3.metric(
                    "Pagos últimos 12 meses",
                    datos_dividendos[
                        "pagos_ultimos_12m"
                    ],
                )

                d4.metric(
                    "Frecuencia aproximada",
                    frecuencia,
                )

                if fecha_ultimo is not None:
                    st.caption(
                        "Último pago registrado: "
                        f"{fecha_ultimo.strftime('%d/%m/%Y')}."
                    )

                st.markdown("#### Crecimiento del dividendo")

                g1, g2, g3, g4 = st.columns(4)

                crecimiento_1a = datos_dividendos[
                    "crecimiento_1a_pct"
                ]

                cagr_3 = datos_dividendos[
                    "cagr_3a_pct"
                ]

                cagr_5 = datos_dividendos[
                    "cagr_5a_pct"
                ]

                g1.metric(
                    "Variación anual",
                    (
                        f"{crecimiento_1a:+.2f}%"
                        if crecimiento_1a is not None
                        else "N/D"
                    ),
                )

                g2.metric(
                    "CAGR · 3 años",
                    (
                        f"{cagr_3:+.2f}%"
                        if cagr_3 is not None
                        else "N/D"
                    ),
                )

                g3.metric(
                    "CAGR · 5 años",
                    (
                        f"{cagr_5:+.2f}%"
                        if cagr_5 is not None
                        else "N/D"
                    ),
                )

                g4.metric(
                    "Años consecutivos con pagos",
                    datos_dividendos[
                        "anios_consecutivos"
                    ],
                )

                st.caption(
                    "El crecimiento y los CAGR utilizan únicamente "
                    "años naturales completos. El año actual puede "
                    "aparecer en el gráfico, pero no se utiliza para "
                    "medir crecimiento hasta que haya terminado."
                )

                anual_div = datos_dividendos["anual"]

                if (
                    anual_div is not None
                    and not anual_div.empty
                ):

                    st.markdown(
                        "#### Dividendo pagado por año"
                    )


                    anio_actual_div = pd.Timestamp.now().year

                    anual_div = anual_div.copy()

                    anual_div["Periodo"] = anual_div["Año"].apply(
                        lambda x: (
                            f"{int(x)} · en curso"
                            if int(x) == anio_actual_div
                            else str(int(x))
                        )
                    )

                    grafico_dividendos = (
                        alt.Chart(anual_div)
                        .mark_bar()
                        .encode(
                            x=alt.X(
                                "Periodo:O",
                                title="Año",
                            ),
                            y=alt.Y(
                                "Dividendo anual:Q",
                                title="Dividendo por acción",
                            ),
                            tooltip=[
                                alt.Tooltip(
                                    "Periodo:O",
                                    title="Año",
                                ),
                                alt.Tooltip(
                                    "Dividendo anual:Q",
                                    title="Dividendo",
                                    format=".4f",
                                ),
                            ],
                        )
                        .properties(
                            height=320,
                        )
                    )

                    st.altair_chart(
                        grafico_dividendos,
                        width="stretch",
                    )

                # ----------------------------------------------------
                # SIMULACIÓN
                # ----------------------------------------------------

                st.markdown(
                    "#### ¿Cuánto habría cobrado en dividendos?"
                )

                pagos_div = datos_dividendos["pagos"]

                opciones_anios = [1, 3, 5, 10]

                sim1, sim2 = st.columns(2)

                with sim1:
                    capital_div = st.number_input(
                        "Inversión inicial",
                        min_value=100.0,
                        value=1000.0,
                        step=100.0,
                        key="dividendos_capital",
                    )

                with sim2:
                    periodo_div = st.selectbox(
                        "Periodo",
                        opciones_anios,
                        index=2,
                        format_func=lambda x: (
                            f"{x} año"
                            if x == 1
                            else f"{x} años"
                        ),
                        key="dividendos_periodo",
                    )

                try:
                    import yfinance as yf

                    fecha_final_div = (
                        pagos_div["Fecha"].max()
                    )

                    fecha_inicial_div = (
                        fecha_final_div
                        - pd.DateOffset(
                            years=periodo_div
                        )
                    )

                    pagos_periodo = pagos_div[
                        pagos_div["Fecha"]
                        > fecha_inicial_div
                    ]

                    # Descargamos un margen suficientemente amplio
                    # para encontrar la primera sesión bursátil
                    # disponible a partir de la fecha objetivo.
                    hist_sim = yf.download(
                        ticker,
                        start=fecha_inicial_div.strftime("%Y-%m-%d"),
                        end=(
                            fecha_inicial_div
                            + pd.DateOffset(days=30)
                        ).strftime("%Y-%m-%d"),
                        auto_adjust=False,
                        actions=False,
                        progress=False,
                        threads=False,
                    )

                    if (
                        hist_sim is not None
                        and not hist_sim.empty
                        and "Close" in hist_sim
                    ):
                        close_sim = hist_sim["Close"].dropna()

                        # yf.download puede devolver DataFrame incluso
                        # para un único ticker.
                        if hasattr(close_sim, "columns"):
                            close_sim = close_sim.iloc[:, 0]

                        precio_inicial_div = float(
                            close_sim.iloc[0]
                        )

                        acciones_div = (
                            capital_div
                            / precio_inicial_div
                        )

                        div_por_accion_periodo = float(
                            pagos_periodo[
                                "Dividendo"
                            ].sum()
                        )

                        cobrado_div = (
                            acciones_div
                            * div_por_accion_periodo
                        )

                        rentabilidad_div = (
                            cobrado_div
                            / capital_div
                            * 100
                        )

                        s1, s2, s3 = st.columns(3)

                        s1.metric(
                            "Acciones aproximadas",
                            f"{acciones_div:.2f}",
                        )

                        s2.metric(
                            "Dividendos cobrados",
                            f"{cobrado_div:,.2f}",
                        )

                        s3.metric(
                            "Dividendos / inversión inicial",
                            f"{rentabilidad_div:.2f}%",
                        )

                        st.caption(
                            "Simulación simplificada: compra al "
                            "primer precio Close disponible del "
                            "periodo y mantiene las acciones. "
                            "No reinvierte dividendos y no incluye "
                            "impuestos, comisiones ni efecto divisa."
                        )

                    else:
                        st.info(
                            "No hay precio histórico suficiente "
                            "para realizar esta simulación."
                        )

                except Exception:
                    st.info(
                        "No se ha podido completar la simulación "
                        "con los datos disponibles."
                    )

                with st.expander(
                    "Ver historial de pagos"
                ):
                    tabla_div = pagos_div.copy()

                    tabla_div["Fecha"] = (
                        tabla_div["Fecha"]
                        .dt.strftime("%d/%m/%Y")
                    )

                    st.dataframe(
                        tabla_div.sort_values(
                            "Fecha",
                            ascending=False,
                        ),
                        width="stretch",
                        hide_index=True,
                    )

                st.caption(
                    "Fuente: "
                    f"{datos_dividendos['fuente']}. "
                    "Los dividendos históricos describen pagos "
                    "registrados y no garantizan pagos futuros."
                )


        # ============================================================
        # NIVEL 2 · DCF
        # ============================================================

        if modulo_finscope == "Valoración DCF":

            st.markdown(
                '<div class="section-kicker">'
                'VALORACIÓN INTRÍNSECA'
                '</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                '<div class="fs-big-title">'
                'Flujo de Caja Descontado · DCF'
                '</div>',
                unsafe_allow_html=True,
            )

            st.caption(
                "El DCF estima el valor de una empresa descontando "
                "los flujos de caja libre futuros. El resultado no "
                "es un precio objetivo ni una predicción: depende "
                "directamente de los supuestos utilizados."
            )

            fcf_dcf = datos.get("free_cash_flow_ttm")
            efectivo_dcf = datos.get("efectivo")
            deuda_dcf = datos.get("deuda_total")

            if not datos.get("fcf_aplicable", True):

                st.info(
                    "El DCF convencional basado en Free Cash Flow "
                    "no se aplica aquí porque FinScope ha "
                    "identificado una entidad financiera. La "
                    "estructura de caja y deuda de bancos y otras "
                    "entidades financieras requiere métodos de "
                    "valoración específicos."
                )

            elif fcf_dcf is None or fcf_dcf <= 0:

                st.info(
                    "FinScope no dispone de un Free Cash Flow TTM "
                    "positivo y verificable para construir un DCF "
                    "convencional. No se genera una valoración "
                    "estimada con datos inventados."
                )

            else:

                # ----------------------------------------------------
                # DATOS DE MERCADO PARA EL DCF
                # ----------------------------------------------------

                import yfinance as yf

                acciones_dcf = None
                precio_dcf = None

                try:
                    info_dcf = yf.Ticker(ticker).info

                    acciones_dcf = info_dcf.get(
                        "sharesOutstanding"
                    )

                    precio_dcf = (
                        info_dcf.get("currentPrice")
                        or info_dcf.get("regularMarketPrice")
                    )

                except Exception:
                    pass

                efectivo_dcf = float(
                    efectivo_dcf or 0
                )

                deuda_dcf = float(
                    deuda_dcf or 0
                )

                # ----------------------------------------------------
                # BASE DEL MODELO
                # ----------------------------------------------------

                st.markdown("#### Base del modelo")

                b1, b2, b3, b4 = st.columns(4)

                b1.metric(
                    "FCF TTM",
                    f"{fcf_dcf / 1_000_000_000:,.2f} B",
                )

                b2.metric(
                    "Caja",
                    f"{efectivo_dcf / 1_000_000_000:,.2f} B",
                )

                b3.metric(
                    "Deuda",
                    f"{deuda_dcf / 1_000_000_000:,.2f} B",
                )

                b4.metric(
                    "Acciones",
                    (
                        f"{acciones_dcf / 1_000_000_000:,.2f} B"
                        if acciones_dcf
                        else "N/D"
                    ),
                )

                st.caption(
                    "FCF base: últimos cuatro trimestres "
                    "disponibles. Caja, deuda y acciones proceden "
                    "de los datos disponibles para la empresa."
                )

                # ----------------------------------------------------
                # SUPUESTOS
                # ----------------------------------------------------

                st.markdown("#### Supuestos del modelo")

                escenario_dcf = st.segmented_control(
                    "Escenario",
                    options=[
                        "Conservador",
                        "Base",
                        "Expansivo",
                        "Personalizado",
                    ],
                    default="Base",
                    key="dcf_escenario",
                )

                escenarios_dcf = {
                    "Conservador": {
                        "crecimiento": 3.0,
                        "descuento": 10.0,
                        "terminal": 2.0,
                    },
                    "Base": {
                        "crecimiento": 5.0,
                        "descuento": 9.0,
                        "terminal": 2.5,
                    },
                    "Expansivo": {
                        "crecimiento": 7.0,
                        "descuento": 8.0,
                        "terminal": 3.0,
                    },
                }

                if escenario_dcf in escenarios_dcf:
                    supuestos_dcf = escenarios_dcf[escenario_dcf]

                    crecimiento_dcf = float(
                        supuestos_dcf["crecimiento"]
                    )
                    descuento_dcf = float(
                        supuestos_dcf["descuento"]
                    )
                    terminal_dcf = float(
                        supuestos_dcf["terminal"]
                    )

                    e1, e2, e3 = st.columns(3)

                    e1.metric(
                        "Crecimiento FCF",
                        f"{crecimiento_dcf:.1f}%",
                    )
                    e2.metric(
                        "Tasa de descuento",
                        f"{descuento_dcf:.1f}%",
                    )
                    e3.metric(
                        "Crecimiento terminal",
                        f"{terminal_dcf:.2f}%",
                    )

                    st.caption(
                        "Este escenario es una plantilla de "
                        "sensibilidad de FinScope, no una previsión "
                        "específica sobre esta empresa. Los valores "
                        "no se han estimado a partir de expectativas "
                        "futuras de la compañía."
                    )

                else:
                    s1, s2, s3 = st.columns(3)

                    with s1:
                        crecimiento_dcf = st.number_input(
                            "Crecimiento anual del FCF",
                            min_value=-20.0,
                            max_value=30.0,
                            value=5.0,
                            step=0.5,
                            format="%.1f",
                            key="dcf_crecimiento_personalizado",
                            help=(
                                "Crecimiento anual supuesto del Free "
                                "Cash Flow durante los próximos "
                                "cinco años."
                            ),
                        )

                    with s2:
                        descuento_dcf = st.number_input(
                            "Tasa de descuento",
                            min_value=1.0,
                            max_value=30.0,
                            value=9.0,
                            step=0.5,
                            format="%.1f",
                            key="dcf_descuento_personalizado",
                            help=(
                                "Tasa utilizada para descontar los "
                                "flujos futuros al presente."
                            ),
                        )

                    with s3:
                        terminal_dcf = st.number_input(
                            "Crecimiento terminal",
                            min_value=-2.0,
                            max_value=8.0,
                            value=2.5,
                            step=0.25,
                            format="%.2f",
                            key="dcf_terminal_personalizado",
                            help=(
                                "Crecimiento perpetuo supuesto "
                                "después de los cinco años "
                                "proyectados."
                            ),
                        )

                    st.caption(
                        "Modo personalizado: los tres supuestos son "
                        "introducidos por el usuario. FinScope "
                        "calcula el resultado, pero no valida que "
                        "representen una previsión futura probable."
                    )

                # ----------------------------------------------------
                # CÁLCULO
                # ----------------------------------------------------

                resultado_dcf = calcular_dcf(
                    fcf_base=fcf_dcf,
                    crecimiento_pct=crecimiento_dcf,
                    tasa_descuento_pct=descuento_dcf,
                    crecimiento_terminal_pct=terminal_dcf,
                    efectivo=efectivo_dcf,
                    deuda=deuda_dcf,
                    acciones=acciones_dcf,
                    precio_actual=precio_dcf,
                    anos=5,
                )

                if not resultado_dcf["valido"]:

                    st.warning(
                        "No puede calcularse el DCF con estos "
                        "supuestos: "
                        f"{resultado_dcf.get('motivo', 'N/D')}."
                    )

                else:

                    # ------------------------------------------------
                    # RESULTADO PRINCIPAL
                    # ------------------------------------------------

                    st.markdown("#### Resultado")

                    r1, r2, r3, r4 = st.columns(4)

                    r1.metric(
                        "Valor empresa",
                        (
                            f"{resultado_dcf['enterprise_value'] / 1_000_000_000:,.2f} B"
                        ),
                    )

                    r2.metric(
                        "Valor del equity",
                        (
                            f"{resultado_dcf['equity_value'] / 1_000_000_000:,.2f} B"
                        ),
                    )

                    valor_accion_dcf = resultado_dcf.get(
                        "valor_por_accion"
                    )

                    diferencia_dcf = resultado_dcf.get(
                        "diferencia_pct"
                    )

                    r3.metric(
                        "Valor DCF por acción",
                        (
                            f"{valor_accion_dcf:,.2f}"
                            if valor_accion_dcf is not None
                            else "N/D"
                        ),
                    )

                    r4.metric(
                        "Precio actual",
                        (
                            f"{precio_dcf:,.2f}"
                            if precio_dcf is not None
                            else "N/D"
                        ),
                        delta=(
                            f"{diferencia_dcf:+.2f}% DCF vs precio"
                            if diferencia_dcf is not None
                            else None
                        ),
                    )

                    st.info(
                        f"Con el escenario {escenario_dcf.lower()}, "
                        "el valor mostrado es el resultado matemático "
                        "de los supuestos visibles arriba. No es una "
                        "predicción del precio futuro. Una diferencia "
                        "positiva o negativa frente al mercado no "
                        "constituye por sí sola una señal de compra "
                        "o venta."
                    )

                    # ------------------------------------------------
                    # PROYECCIÓN 5 AÑOS
                    # ------------------------------------------------

                    st.markdown(
                        "#### Proyección de Free Cash Flow"
                    )


                    tabla_dcf = pd.DataFrame(
                        resultado_dcf["flujos"]
                    )

                    tabla_dcf["FCF proyectado"] = (
                        tabla_dcf["FCF proyectado"]
                        / 1_000_000_000
                    )

                    tabla_dcf["Valor presente"] = (
                        tabla_dcf["Valor presente"]
                        / 1_000_000_000
                    )

                    grafico_dcf = (
                        alt.Chart(tabla_dcf)
                        .transform_fold(
                            [
                                "FCF proyectado",
                                "Valor presente",
                            ],
                            as_=["Serie", "Valor"],
                        )
                        .mark_line(point=True)
                        .encode(
                            x=alt.X(
                                "Año:O",
                                title="Año proyectado",
                            ),
                            y=alt.Y(
                                "Valor:Q",
                                title="Miles de millones",
                            ),
                            color=alt.Color(
                                "Serie:N",
                                title=None,
                            ),
                            tooltip=[
                                "Año:O",
                                "Serie:N",
                                alt.Tooltip(
                                    "Valor:Q",
                                    format=".2f",
                                ),
                            ],
                        )
                        .properties(height=320)
                    )

                    st.altair_chart(
                        grafico_dcf,
                        width="stretch",
                    )

                    # ------------------------------------------------
                    # COMPOSICIÓN
                    # ------------------------------------------------

                    terminal_pct = resultado_dcf.get(
                        "peso_terminal_pct"
                    )

                    if terminal_pct is not None:
                        st.metric(
                            "Peso del valor terminal",
                            f"{terminal_pct:.1f}%",
                        )

                        if terminal_pct >= 70:
                            st.caption(
                                "Una parte elevada de la valoración "
                                "procede del valor terminal. Esto "
                                "hace que el resultado sea "
                                "especialmente sensible a la tasa "
                                "de descuento y al crecimiento "
                                "terminal."
                            )

                    # ------------------------------------------------
                    # SENSIBILIDAD
                    # ------------------------------------------------

                    st.markdown(
                        "#### Sensibilidad del valor por acción"
                    )

                    st.caption(
                        "La tabla mantiene constante el crecimiento "
                        "del FCF de los primeros cinco años y muestra "
                        "cómo cambia la valoración al modificar dos "
                        "supuestos especialmente sensibles: la tasa "
                        "de descuento y el crecimiento terminal."
                    )

                    descuentos_sens = [
                        max(
                            terminal_dcf + 0.5,
                            descuento_dcf - 1.0,
                        ),
                        descuento_dcf,
                        descuento_dcf + 1.0,
                    ]

                    terminales_sens = [
                        terminal_dcf - 0.5,
                        terminal_dcf,
                        terminal_dcf + 0.5,
                    ]

                    sensibilidad = []

                    for terminal_s in terminales_sens:

                        fila = {
                            "Crecimiento terminal":
                                f"{terminal_s:.2f}%"
                        }

                        for descuento_s in descuentos_sens:

                            res_s = calcular_dcf(
                                fcf_base=fcf_dcf,
                                crecimiento_pct=crecimiento_dcf,
                                tasa_descuento_pct=descuento_s,
                                crecimiento_terminal_pct=terminal_s,
                                efectivo=efectivo_dcf,
                                deuda=deuda_dcf,
                                acciones=acciones_dcf,
                                precio_actual=precio_dcf,
                                anos=5,
                            )

                            columna = (
                                f"Descuento {descuento_s:.1f}%"
                            )

                            if (
                                res_s["valido"]
                                and res_s.get(
                                    "valor_por_accion"
                                ) is not None
                            ):
                                fila[columna] = round(
                                    res_s["valor_por_accion"],
                                    2,
                                )
                            else:
                                fila[columna] = None

                        sensibilidad.append(fila)

                    st.dataframe(
                        pd.DataFrame(sensibilidad),
                        width="stretch",
                        hide_index=True,
                    )

                    st.caption(
                        "La celda correspondiente a los supuestos "
                        "seleccionados representa el DCF principal. "
                        "La tabla no asigna probabilidades a los "
                        "distintos resultados."
                    )

                    # ------------------------------------------------
                    # METODOLOGÍA
                    # ------------------------------------------------

                    with st.expander(
                        "Cómo calcula FinScope este DCF"
                    ):

                        st.write(
                            "1. Parte del Free Cash Flow TTM "
                            "disponible."
                        )

                        st.write(
                            "2. Proyecta el FCF durante cinco años "
                            "utilizando el crecimiento seleccionado."
                        )

                        st.write(
                            "3. Descuenta cada flujo mediante la "
                            "tasa de descuento seleccionada."
                        )

                        st.write(
                            "4. Calcula el valor terminal mediante "
                            "el modelo de crecimiento perpetuo de "
                            "Gordon."
                        )

                        st.write(
                            "5. Suma caja y resta deuda para pasar "
                            "del valor de empresa al valor del "
                            "equity."
                        )

                        st.write(
                            "6. Divide el equity entre las acciones "
                            "en circulación para obtener el valor "
                            "estimado por acción."
                        )

                        st.warning(
                            "El DCF es especialmente sensible a "
                            "sus supuestos. Pequeños cambios en "
                            "crecimiento, descuento o crecimiento "
                            "terminal pueden producir diferencias "
                            "importantes en la valoración."
                        )

                st.caption(
                    "Fuente de datos base: Yahoo Finance vía "
                    "yfinance. Modelo DCF calculado por FinScope."
                )


        # DICCIONARIO DE MÉTRICAS


    if modulo_finscope == "Estadística cuantitativa":
        st.markdown("### Estadística de rendimientos")

        st.caption(
            "Describe cómo se han distribuido históricamente los "
            "rendimientos del activo. No constituye una predicción "
            "del comportamiento futuro."
        )

        q1, q2, q3 = st.columns(3)

        with q1:
            periodo_quant = st.selectbox(
                "Histórico",
                ["1 año", "3 años", "5 años", "10 años"],
                index=2,
                key="quant_periodo_estadistica",
            )

        with q2:
            frecuencia_quant = st.selectbox(
                "Frecuencia",
                ["Diaria", "Semanal", "Mensual", "Anual"],
                index=0,
                key="quant_frecuencia_estadistica",
            )

        with q3:
            ajustado_quant = st.toggle(
                "Precio ajustado",
                value=True,
                key="quant_ajustado_estadistica",
                help=(
                    "Utiliza Adj Close cuando está disponible."
                ),
            )

        mapa_periodos_quant = {
            "1 año": "1y",
            "3 años": "3y",
            "5 años": "5y",
            "10 años": "10y",
        }

        try:
            import yfinance as yf
            import numpy as np
            import matplotlib.pyplot as plt

            hist_quant = yf.download(
                ticker,
                period=mapa_periodos_quant[periodo_quant],
                interval="1d",
                auto_adjust=False,
                actions=False,
                progress=False,
                threads=False,
            )

            if isinstance(
                hist_quant.columns,
                pd.MultiIndex,
            ):
                try:
                    hist_quant = hist_quant.xs(
                        ticker,
                        axis=1,
                        level=1,
                    )
                except Exception:
                    hist_quant.columns = [
                        c[0] if isinstance(c, tuple) else c
                        for c in hist_quant.columns
                    ]

            estad_quant = calcular_estadistica_rendimientos(
                hist_quant,
                frecuencia=frecuencia_quant,
                usar_ajustado=ajustado_quant,
            )

            if not estad_quant.get("valido"):
                st.info(
                    "No hay suficientes datos para realizar "
                    "el análisis con esta selección."
                )

            else:
                a1, a2, a3, a4 = st.columns(4)

                a1.metric(
                    "Rentabilidad media",
                    f"{estad_quant['media'] * 100:.3f}%",
                )

                a2.metric(
                    "Mediana",
                    f"{estad_quant['mediana'] * 100:.3f}%",
                )

                a3.metric(
                    "Desviación estándar",
                    f"{estad_quant['desviacion'] * 100:.3f}%",
                )

                a4.metric(
                    "Observaciones",
                    f"{estad_quant['observaciones']:,}",
                )

                st.caption(
                    f"Periodo: "
                    f"{estad_quant['inicio'].strftime('%d/%m/%Y')} "
                    f"– "
                    f"{estad_quant['fin'].strftime('%d/%m/%Y')} · "
                    f"{estad_quant['frecuencia']} · "
                    f"{estad_quant['columna_precio']}."
                )

                st.markdown("#### Extremos históricos")

                b1, b2 = st.columns(2)

                b1.metric(
                    "Mejor periodo",
                    f"{estad_quant['maximo'] * 100:+.2f}%",
                )

                b1.caption(
                    estad_quant["fecha_maximo"].strftime(
                        "%d/%m/%Y"
                    )
                )

                b2.metric(
                    "Peor periodo",
                    f"{estad_quant['minimo'] * 100:+.2f}%",
                )

                b2.caption(
                    estad_quant["fecha_minimo"].strftime(
                        "%d/%m/%Y"
                    )
                )

                st.markdown(
                    "#### Periodos positivos y negativos"
                )

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "Positivos",
                    f"{estad_quant['positivos']:,}",
                    f"{estad_quant['pct_positivos']:.1f}%",
                )

                c2.metric(
                    "Negativos",
                    f"{estad_quant['negativos']:,}",
                    f"{estad_quant['pct_negativos']:.1f}%",
                )

                c3.metric(
                    "Sin variación",
                    f"{estad_quant['neutros']:,}",
                )

                st.markdown("#### Forma de la distribución")

                d1, d2 = st.columns(2)

                d1.metric(
                    "Asimetría · Skewness",
                    f"{estad_quant['skewness']:.2f}",
                )

                d2.metric(
                    "Curtosis · Exceso",
                    f"{estad_quant['curtosis_exceso']:.2f}",
                )

                st.caption(
                    "Skewness indica hacia qué lado es más "
                    "asimétrica la distribución. El exceso de "
                    "curtosis mide el peso de las colas respecto "
                    "a una distribución normal, cuyo valor es 0."
                )

                st.markdown("#### Percentiles históricos")

                percentiles_q = estad_quant["percentiles"]

                tabla_q = pd.DataFrame(
                    {
                        "Percentil": [
                            "P1", "P5", "P25", "P50",
                            "P75", "P95", "P99",
                        ],
                        "Rendimiento": [
                            f"{percentiles_q[p] * 100:+.2f}%"
                            for p in [
                                1, 5, 25, 50,
                                75, 95, 99,
                            ]
                        ],
                    }
                )

                st.dataframe(
                    tabla_q,
                    width="stretch",
                    hide_index=True,
                )

                st.markdown(
                    "#### Distribución histórica de rendimientos"
                )

                r = estad_quant["rendimientos"].dropna()

                fig, ax = plt.subplots(figsize=(10, 4.8))

                ax.hist(
                    r * 100,
                    bins=40,
                    density=True,
                    alpha=0.65,
                    label="Rendimientos observados",
                )

                mu = float(r.mean())
                sigma = float(r.std(ddof=1))

                if np.isfinite(sigma) and sigma > 0:
                    x = np.linspace(
                        float(r.min()),
                        float(r.max()),
                        300,
                    )

                    normal = (
                        1.0
                        / (
                            sigma
                            * np.sqrt(2.0 * np.pi)
                        )
                        * np.exp(
                            -0.5
                            * ((x - mu) / sigma) ** 2
                        )
                    )

                    # El histograma está expresado en puntos
                    # porcentuales, por eso ajustamos la densidad.
                    ax.plot(
                        x * 100,
                        normal / 100,
                        linewidth=2,
                        label="Normal equivalente",
                    )

                ax.axvline(
                    mu * 100,
                    linestyle="--",
                    linewidth=1.5,
                    label="Media histórica",
                )

                ax.set_xlabel("Rendimiento (%)")
                ax.set_ylabel("Densidad")
                ax.legend()
                ax.grid(alpha=0.15)

                st.pyplot(fig)
                plt.close(fig)

                st.caption(
                    "La curva normal tiene la misma media y "
                    "desviación estándar que la muestra y se "
                    "incluye únicamente como referencia."
                )

                with st.expander(
                    "Cómo calcula FinScope estas estadísticas",
                    expanded=False,
                ):
                    st.markdown(
                        """
    1. Obtiene los precios históricos disponibles.
    2. Los transforma a la frecuencia seleccionada.
    3. Calcula cada rendimiento como `P(t) / P(t-1) - 1`.
    4. Calcula media, mediana y desviación estándar.
    5. Identifica extremos y periodos positivos/negativos.
    6. Calcula percentiles, skewness y exceso de curtosis.
    7. Compara la distribución observada con una normal equivalente.

    Los resultados describen únicamente la muestra histórica disponible.
                        """
                    )

                st.caption(
                    "Fuente de precios: Yahoo Finance vía yfinance. "
                    "Cálculos estadísticos: FinScope."
                )

        except Exception:
            st.info(
                "No se ha podido completar el análisis "
                "estadístico con los datos disponibles."
            )

    if modulo_finscope == "Riesgo cuantitativo avanzado":
        st.markdown("### Riesgo cuantitativo avanzado")

        st.caption(
            "Analiza la distribución histórica de pérdidas y "
            "la evolución del riesgo. Las métricas describen "
            "el pasado y no garantizan pérdidas máximas futuras."
        )

        rq1, rq2, rq3 = st.columns(3)

        with rq1:
            periodo_rq = st.selectbox(
                "Histórico",
                ["1 año", "3 años", "5 años", "10 años"],
                index=2,
                key="riesgo_quant_periodo",
            )

        with rq2:
            confianza_rq = st.selectbox(
                "Nivel de confianza VaR / CVaR",
                [90, 95, 99],
                index=1,
                format_func=lambda x: f"{x}%",
                key="riesgo_quant_confianza",
            )

        with rq3:
            ventana_rq = st.selectbox(
                "Ventana volatilidad móvil",
                [21, 63, 126],
                index=0,
                format_func=lambda x: (
                    "21 sesiones · ~1 mes"
                    if x == 21
                    else (
                        "63 sesiones · ~3 meses"
                        if x == 63
                        else "126 sesiones · ~6 meses"
                    )
                ),
                key="riesgo_quant_ventana",
            )

        mapa_rq = {
            "1 año": "1y",
            "3 años": "3y",
            "5 años": "5y",
            "10 años": "10y",
        }

        try:
            import yfinance as yf
            import numpy as np

            hist_rq = yf.download(
                ticker,
                period=mapa_rq[periodo_rq],
                interval="1d",
                auto_adjust=False,
                actions=False,
                progress=False,
                threads=False,
            )

            if isinstance(
                hist_rq.columns,
                pd.MultiIndex,
            ):
                try:
                    hist_rq = hist_rq.xs(
                        ticker,
                        axis=1,
                        level=1,
                    )
                except Exception:
                    hist_rq.columns = [
                        c[0] if isinstance(c, tuple) else c
                        for c in hist_rq.columns
                    ]

            riesgo_q = calcular_riesgo_cuantitativo(
                hist_rq,
                nivel_confianza=confianza_rq,
                usar_ajustado=True,
                ventana_volatilidad=ventana_rq,
            )

            if not riesgo_q.get("valido"):
                st.info(
                    "No hay suficientes datos históricos para "
                    "calcular estas métricas de riesgo."
                )

            else:
                st.markdown(
                    "#### Riesgo de cola · VaR y CVaR"
                )

                v1, v2, v3 = st.columns(3)

                v1.metric(
                    f"VaR histórico · {confianza_rq}%",
                    f"{riesgo_q['var_historico'] * 100:.2f}%",
                )

                v2.metric(
                    f"CVaR · {confianza_rq}%",
                    f"{riesgo_q['cvar_historico'] * 100:.2f}%",
                )

                v3.metric(
                    "Observaciones",
                    f"{riesgo_q['observaciones']:,}",
                )

                st.caption(
                    f"Con los datos históricos seleccionados, "
                    f"el VaR al {confianza_rq}% corresponde al "
                    f"umbral empírico de la cola inferior. "
                    "El CVaR muestra la pérdida media de las "
                    "observaciones situadas en esa cola. "
                    "No representan una pérdida máxima posible."
                )

                capital_var = st.number_input(
                    "Capital de referencia",
                    min_value=100.0,
                    value=10000.0,
                    step=500.0,
                    key="riesgo_quant_capital",
                    help=(
                        "Sirve únicamente para traducir los "
                        "porcentajes históricos a una cantidad "
                        "monetaria de referencia."
                    ),
                )

                cv1, cv2 = st.columns(2)

                cv1.metric(
                    "VaR sobre capital",
                    f"{capital_var * riesgo_q['var_historico']:,.2f}",
                )

                cv2.metric(
                    "CVaR sobre capital",
                    f"{capital_var * riesgo_q['cvar_historico']:,.2f}",
                )

                st.caption(
                    "Ejemplo interpretativo: el VaR no significa "
                    "que esa sea la pérdida máxima. Históricamente, "
                    "una fracción aproximada de las sesiones quedó "
                    "más allá de ese umbral."
                )

                st.markdown(
                    "#### Riesgo a la baja"
                )

                d1, d2 = st.columns(2)

                d1.metric(
                    "Downside deviation anualizada",
                    (
                        f"{riesgo_q['downside_dev_anual'] * 100:.2f}%"
                    ),
                )

                d2.metric(
                    "Semidesviación diaria",
                    (
                        f"{riesgo_q['semidesviacion_diaria'] * 100:.2f}%"
                    ),
                )

                st.caption(
                    "Downside deviation considera únicamente la "
                    "parte negativa respecto a un objetivo diario "
                    "del 0%. La semidesviación mide la dispersión "
                    "de las observaciones situadas por debajo de "
                    "la media histórica."
                )

                st.markdown(
                    "#### Frecuencia de grandes caídas"
                )

                tabla_caidas = []

                for umbral in [2, 3, 5]:
                    dato = riesgo_q["umbrales"][umbral]

                    tabla_caidas.append(
                        {
                            "Caída diaria": f"≤ -{umbral}%",
                            "Sesiones": dato["cantidad"],
                            "% de sesiones": (
                                f"{dato['porcentaje']:.2f}%"
                            ),
                            "Caída media en esos eventos": (
                                (
                                    f"{dato['media'] * 100:.2f}%"
                                )
                                if dato["media"] is not None
                                else "—"
                            ),
                        }
                    )

                st.dataframe(
                    pd.DataFrame(tabla_caidas),
                    width="stretch",
                    hide_index=True,
                )

                st.markdown(
                    "#### Volatilidad móvil anualizada"
                )

                rv1, rv2, rv3 = st.columns(3)

                rv1.metric(
                    "Actual",
                    (
                        f"{riesgo_q['vol_actual'] * 100:.2f}%"
                        if riesgo_q["vol_actual"] is not None
                        else "—"
                    ),
                )

                rv2.metric(
                    "Media histórica",
                    (
                        f"{riesgo_q['vol_media'] * 100:.2f}%"
                        if riesgo_q["vol_media"] is not None
                        else "—"
                    ),
                )

                rv3.metric(
                    "Máxima",
                    (
                        f"{riesgo_q['vol_max'] * 100:.2f}%"
                        if riesgo_q["vol_max"] is not None
                        else "—"
                    ),
                )

                rolling_rq = riesgo_q["rolling_vol"]

                if len(rolling_rq):
                    chart_rq = pd.DataFrame(
                        {
                            "Fecha": rolling_rq.index,
                            "Volatilidad": (
                                rolling_rq.values * 100
                            ),
                        }
                    )

                    st.line_chart(
                        chart_rq,
                        x="Fecha",
                        y="Volatilidad",
                        height=350,
                    )

                    if riesgo_q["fecha_vol_max"] is not None:
                        st.caption(
                            "Máximo de volatilidad móvil del "
                            f"periodo: "
                            f"{riesgo_q['fecha_vol_max'].strftime('%d/%m/%Y')}."
                        )

                st.markdown(
                    "#### Peores sesiones del periodo"
                )

                peores_rq = riesgo_q[
                    "peores_sesiones"
                ]

                tabla_peores = pd.DataFrame(
                    {
                        "Fecha": [
                            fecha.strftime("%d/%m/%Y")
                            for fecha in peores_rq.index
                        ],
                        "Rendimiento": [
                            f"{valor * 100:.2f}%"
                            for valor in peores_rq.values
                        ],
                    }
                )

                st.dataframe(
                    tabla_peores,
                    width="stretch",
                    hide_index=True,
                )

                with st.expander(
                    "Cómo calcula FinScope el riesgo cuantitativo",
                    expanded=False,
                ):
                    st.markdown(
                        """
    **VaR histórico:** percentil de la cola inferior de los rendimientos observados. No presupone una distribución normal.

    **CVaR / Expected Shortfall:** pérdida media de las observaciones iguales o peores que el umbral VaR.

    **Downside deviation:** raíz de la media cuadrática de los rendimientos situados por debajo del objetivo diario del 0%. Se anualiza con √252.

    **Semidesviación:** dispersión de los rendimientos situados por debajo de la media de la muestra.

    **Volatilidad móvil:** desviación estándar de una ventana de rendimientos diarios, anualizada con √252.

    Estas métricas se calculan sobre datos históricos y no establecen un límite para las pérdidas futuras.
                        """
                    )

                st.caption(
                    f"Periodo: "
                    f"{riesgo_q['inicio'].strftime('%d/%m/%Y')} "
                    f"– "
                    f"{riesgo_q['fin'].strftime('%d/%m/%Y')} · "
                    f"Precio: {riesgo_q['columna_precio']} · "
                    "Fuente: Yahoo Finance vía yfinance. "
                    "Cálculos: FinScope."
                )

        except Exception:
            st.info(
                "No se ha podido completar el análisis de riesgo "
                "cuantitativo con los datos disponibles."
            )

    if modulo_finscope == "Análisis temporal":
        st.markdown("### Análisis temporal")

        st.caption(
            "Estudia cómo se han distribuido los rendimientos "
            "a lo largo del calendario. Los patrones históricos "
            "no implican que vayan a repetirse."
        )

        t1, t2 = st.columns(2)

        with t1:
            periodo_temp = st.selectbox(
                "Histórico",
                ["3 años", "5 años", "10 años"],
                index=1,
                key="temporal_periodo",
            )

        with t2:
            ajustado_temp = st.toggle(
                "Precio ajustado",
                value=True,
                key="temporal_ajustado",
            )

        mapa_temp = {
            "3 años": "3y",
            "5 años": "5y",
            "10 años": "10y",
        }

        try:
            import yfinance as yf
            import numpy as np
            import matplotlib.pyplot as plt

            hist_temp = yf.download(
                ticker,
                period=mapa_temp[periodo_temp],
                interval="1d",
                auto_adjust=False,
                actions=False,
                progress=False,
                threads=False,
            )

            if isinstance(
                hist_temp.columns,
                pd.MultiIndex,
            ):
                try:
                    hist_temp = hist_temp.xs(
                        ticker,
                        axis=1,
                        level=1,
                    )
                except Exception:
                    hist_temp.columns = [
                        c[0] if isinstance(c, tuple) else c
                        for c in hist_temp.columns
                    ]

            temp = calcular_analisis_temporal(
                hist_temp,
                usar_ajustado=ajustado_temp,
            )

            if not temp.get("valido"):

                st.info(
                    "No hay suficientes datos históricos "
                    "para completar el análisis temporal."
                )

            else:
                nombres_meses = {
                    1: "Ene",
                    2: "Feb",
                    3: "Mar",
                    4: "Abr",
                    5: "May",
                    6: "Jun",
                    7: "Jul",
                    8: "Ago",
                    9: "Sep",
                    10: "Oct",
                    11: "Nov",
                    12: "Dic",
                }

                # ====================================================
                # RESUMEN
                # ====================================================

                st.markdown(
                    "#### Comportamiento mensual"
                )

                m1, m2, m3 = st.columns(3)

                m1.metric(
                    "Meses positivos",
                    (
                        f"{temp['meses_positivos']} · "
                        f"{temp['pct_meses_positivos']:.1f}%"
                    ),
                )

                m2.metric(
                    "Meses negativos",
                    (
                        f"{temp['meses_negativos']} · "
                        f"{temp['pct_meses_negativos']:.1f}%"
                    ),
                )

                m3.metric(
                    "Meses analizados",
                    f"{temp['total_meses']}",
                )

                # ====================================================
                # RENTABILIDADES ANUALES
                # ====================================================

                st.markdown(
                    "#### Rentabilidad por año"
                )

                anual_temp = temp["anual"]

                if len(anual_temp):
                    tabla_anual = pd.DataFrame(
                        {
                            "Año": [
                                (
                                    f"{fecha.year} · en curso"
                                    if fecha.year == temp["fin"].year
                                    else str(fecha.year)
                                )
                                for fecha in anual_temp.index
                            ],
                            "Rentabilidad": [
                                f"{valor * 100:+.2f}%"
                                for valor in anual_temp.values
                            ],
                        }
                    )

                    st.dataframe(
                        tabla_anual,
                        width="stretch",
                        hide_index=True,
                    )

                    chart_anual = pd.DataFrame(
                        {
                            "Año": [
                                (
                                    f"{fecha.year} · en curso"
                                    if fecha.year == temp["fin"].year
                                    else str(fecha.year)
                                )
                                for fecha in anual_temp.index
                            ],
                            "Rentabilidad (%)": (
                                anual_temp.values * 100
                            ),
                        }
                    )

                    st.bar_chart(
                        chart_anual,
                        x="Año",
                        y="Rentabilidad (%)",
                        height=330,
                    )

                    st.caption(
                        "El último año se marca como «en curso»: "
                        "su rentabilidad corresponde al periodo transcurrido "
                        "hasta la última sesión disponible y no a un año completo."
                    )

                # ====================================================
                # HEATMAP
                # ====================================================

                st.markdown(
                    "#### Mapa de rentabilidades mensuales"
                )

                heat = temp["heatmap"].copy()

                heat = heat.reindex(
                    columns=range(1, 13)
                )

                heat.columns = [
                    nombres_meses[m]
                    for m in heat.columns
                ]

                heat_pct = heat * 100

                fig_h, ax_h = plt.subplots(
                    figsize=(12, max(3.5, len(heat) * 0.65))
                )

                matriz = heat_pct.to_numpy(
                    dtype=float
                )

                masked = np.ma.masked_invalid(
                    matriz
                )

                imagen = ax_h.imshow(
                    masked,
                    aspect="auto",
                    cmap="RdYlGn",
                    vmin=-15,
                    vmax=15,
                )

                ax_h.set_xticks(
                    range(len(heat_pct.columns))
                )

                ax_h.set_xticklabels(
                    heat_pct.columns
                )

                ax_h.set_yticks(
                    range(len(heat_pct.index))
                )

                ax_h.set_yticklabels(
                    heat_pct.index.astype(str)
                )

                for i in range(
                    len(heat_pct.index)
                ):
                    for j in range(
                        len(heat_pct.columns)
                    ):
                        valor = matriz[i, j]

                        if np.isfinite(valor):
                            ax_h.text(
                                j,
                                i,
                                f"{valor:+.1f}%",
                                ha="center",
                                va="center",
                                fontsize=8,
                            )

                ax_h.set_xlabel("Mes")
                ax_h.set_ylabel("Año")

                fig_h.colorbar(
                    imagen,
                    ax=ax_h,
                    label="Rentabilidad mensual (%)",
                )

                st.pyplot(fig_h)
                plt.close(fig_h)

                st.caption(
                    "Cada celda representa la rentabilidad "
                    "del mes correspondiente. El color sirve "
                    "como apoyo visual; el porcentaje mostrado "
                    "es el dato relevante. El último mes disponible "
                    "puede estar todavía en curso; en ese caso, su "
                    "rentabilidad corresponde únicamente al periodo "
                    "transcurrido hasta la última sesión disponible."
                )

                # ====================================================
                # COMPORTAMIENTO POR MES DEL CALENDARIO
                # ====================================================

                st.markdown(
                    "#### Comportamiento histórico por mes"
                )

                por_mes = temp["por_mes"].copy()

                filas_mes = []

                for numero_mes, fila in por_mes.iterrows():

                    filas_mes.append(
                        {
                            "Mes": nombres_meses[
                                int(numero_mes)
                            ],
                            "Media": (
                                f"{fila['mean'] * 100:+.2f}%"
                            ),
                            "Mediana": (
                                f"{fila['median'] * 100:+.2f}%"
                            ),
                            "% positivo": (
                                f"{fila['pct_positivo']:.1f}%"
                            ),
                            "Observaciones": int(
                                fila["count"]
                            ),
                        }
                    )

                st.dataframe(
                    pd.DataFrame(filas_mes),
                    width="stretch",
                    hide_index=True,
                )

                st.caption(
                    "Las medias mensuales históricas son "
                    "descriptivas. Un mes que haya tenido una "
                    "media positiva no tiene mayor probabilidad "
                    "garantizada de subir en el futuro."
                )

                # ====================================================
                # MEJORES / PEORES
                # ====================================================

                st.markdown(
                    "#### Mejores y peores meses"
                )

                mp1, mp2 = st.columns(2)

                mejores = temp["mejores_meses"]
                peores = temp["peores_meses"]

                with mp1:
                    st.markdown("**5 mejores**")

                    tabla_mejores = pd.DataFrame(
                        {
                            "Mes": [
                                f"{nombres_meses[d.month]} "
                                f"{d.year}"
                                for d in mejores.index
                            ],
                            "Rentabilidad": [
                                f"{v * 100:+.2f}%"
                                for v in mejores.values
                            ],
                        }
                    )

                    st.dataframe(
                        tabla_mejores,
                        width="stretch",
                        hide_index=True,
                    )

                with mp2:
                    st.markdown("**5 peores**")

                    tabla_peores_temp = pd.DataFrame(
                        {
                            "Mes": [
                                f"{nombres_meses[d.month]} "
                                f"{d.year}"
                                for d in peores.index
                            ],
                            "Rentabilidad": [
                                f"{v * 100:+.2f}%"
                                for v in peores.values
                            ],
                        }
                    )

                    st.dataframe(
                        tabla_peores_temp,
                        width="stretch",
                        hide_index=True,
                    )

                # ====================================================
                # RACHAS
                # ====================================================

                st.markdown(
                    "#### Rachas mensuales"
                )

                r1, r2 = st.columns(2)

                r1.metric(
                    "Mayor racha positiva",
                    (
                        f"{temp['racha_positiva']} "
                        "meses"
                    ),
                )

                if (
                    temp["racha_positiva_inicio"]
                    is not None
                    and temp["racha_positiva_fin"]
                    is not None
                ):
                    r1.caption(
                        f"{temp['racha_positiva_inicio'].strftime('%m/%Y')}"
                        " → "
                        f"{temp['racha_positiva_fin'].strftime('%m/%Y')}"
                    )

                r2.metric(
                    "Mayor racha negativa",
                    (
                        f"{temp['racha_negativa']} "
                        "meses"
                    ),
                )

                if (
                    temp["racha_negativa_inicio"]
                    is not None
                    and temp["racha_negativa_fin"]
                    is not None
                ):
                    r2.caption(
                        f"{temp['racha_negativa_inicio'].strftime('%m/%Y')}"
                        " → "
                        f"{temp['racha_negativa_fin'].strftime('%m/%Y')}"
                    )

                # ====================================================
                # METODOLOGÍA
                # ====================================================

                with st.expander(
                    "Cómo calcula FinScope el análisis temporal",
                    expanded=False,
                ):
                    st.markdown(
                        """
    **Rentabilidad mensual:** variación entre el último precio disponible de un mes y el último precio disponible del mes anterior.

    **Rentabilidad anual:** variación entre el último precio disponible de un año y el del año anterior.

    **Mapa mensual:** organiza los rendimientos mensuales por año y mes.

    **Comportamiento por mes:** calcula media, mediana y proporción de observaciones positivas para cada mes del calendario.

    **Rachas:** cuenta meses consecutivos con rendimiento positivo o negativo.

    Todo el análisis es retrospectivo. La existencia de patrones históricos o estacionales no demuestra que vayan a mantenerse.
                        """
                    )

                st.caption(
                    f"Periodo disponible: "
                    f"{temp['inicio'].strftime('%d/%m/%Y')} "
                    f"– "
                    f"{temp['fin'].strftime('%d/%m/%Y')} · "
                    f"Precio: {temp['columna_precio']} · "
                    "Fuente: Yahoo Finance vía yfinance. "
                    "Cálculos: FinScope."
                )

        except Exception:
            st.info(
                "No se ha podido completar el análisis "
                "temporal con los datos disponibles."
            )

    if modulo_finscope == "Monte Carlo":
        st.markdown("### Simulación Monte Carlo")

        st.caption(
            "Explora miles de trayectorias matemáticamente posibles "
            "bajo un modelo parametrizado con el comportamiento histórico "
            "del activo. No es una predicción del precio futuro."
        )

        mc1, mc2, mc3 = st.columns(3)

        with mc1:
            historico_mc = st.selectbox(
                "Histórico para estimar parámetros",
                ["3 años", "5 años", "10 años"],
                index=1,
                key="mc_historico",
            )

        with mc2:
            horizonte_mc = st.selectbox(
                "Horizonte de simulación",
                [
                    "1 mes · 21 sesiones",
                    "3 meses · 63 sesiones",
                    "6 meses · 126 sesiones",
                    "1 año · 252 sesiones",
                    "2 años · 504 sesiones",
                    "5 años · 1260 sesiones",
                ],
                index=3,
                key="mc_horizonte",
            )

        with mc3:
            simulaciones_mc = st.selectbox(
                "Número de simulaciones",
                [1000, 5000, 10000],
                index=1,
                key="mc_simulaciones",
            )

        mc4, mc5 = st.columns(2)

        with mc4:
            capital_mc = st.number_input(
                "Capital de referencia",
                min_value=0.0,
                value=10000.0,
                step=500.0,
                key="mc_capital",
            )

        with mc5:
            ajustado_mc = st.toggle(
                "Precio ajustado",
                value=True,
                key="mc_ajustado",
            )

        mapa_hist_mc = {
            "3 años": "3y",
            "5 años": "5y",
            "10 años": "10y",
        }

        mapa_horizonte_mc = {
            "1 mes · 21 sesiones": 21,
            "3 meses · 63 sesiones": 63,
            "6 meses · 126 sesiones": 126,
            "1 año · 252 sesiones": 252,
            "2 años · 504 sesiones": 504,
            "5 años · 1260 sesiones": 1260,
        }

        try:
            import yfinance as yf
            import numpy as np
            import matplotlib.pyplot as plt

            hist_mc = yf.download(
                ticker,
                period=mapa_hist_mc[historico_mc],
                interval="1d",
                auto_adjust=False,
                actions=False,
                progress=False,
                threads=False,
            )

            if isinstance(
                hist_mc.columns,
                pd.MultiIndex,
            ):
                try:
                    hist_mc = hist_mc.xs(
                        ticker,
                        axis=1,
                        level=1,
                    )
                except Exception:
                    hist_mc.columns = [
                        c[0] if isinstance(c, tuple) else c
                        for c in hist_mc.columns
                    ]

            mc = calcular_monte_carlo(
                hist_mc,
                dias=mapa_horizonte_mc[horizonte_mc],
                simulaciones=int(simulaciones_mc),
                usar_ajustado=ajustado_mc,
                capital_inicial=float(capital_mc),
                seed=42,
        periodos_anuales=(
            365
            if str(ticker).upper().endswith("-USD")
            else 252
        ),
)

            if not mc.get("valido"):

                st.info(
                    "No hay suficientes datos para completar "
                    "la simulación Monte Carlo."
                )

            else:
                # ================================================================
                # PARÁMETROS
                # ================================================================

                st.markdown(
                    "#### Parámetros estimados"
                )

                p1, p2, p3, p4 = st.columns(4)

                p1.metric(
                    "Precio inicial",
                    f"{mc['precio_inicial']:,.2f}",
                )

                p2.metric(
                    "Volatilidad anualizada",
                    f"{mc['volatilidad_anualizada'] * 100:.2f}%",
                )

                p3.metric(
                    "Retorno log. anualizado",
                    f"{mc['retorno_log_anualizado'] * 100:+.2f}%",
                )

                p4.metric(
                    "Observaciones históricas",
                    f"{mc['observaciones']:,}",
                )

                st.caption(
                    "El retorno y la volatilidad utilizados por el modelo "
                    "se estiman a partir de rendimientos logarítmicos diarios "
                    "del periodo histórico seleccionado."
                )

                # ================================================================
                # TRAYECTORIAS
                # ================================================================

                st.markdown(
                    "#### Trayectorias simuladas"
                )

                tray = mc[
                    "trayectorias_visual"
                ]

                fig_t, ax_t = plt.subplots(
                    figsize=(12, 5.5)
                )

                eje_dias = np.arange(
                    tray.shape[0] + 1
                )

                for j in range(
                    tray.shape[1]
                ):
                    serie = np.concatenate(
                        (
                            [mc["precio_inicial"]],
                            tray[:, j],
                        )
                    )

                    ax_t.plot(
                        eje_dias,
                        serie,
                        alpha=0.18,
                        linewidth=0.8,
                    )

                ax_t.axhline(
                    mc["precio_inicial"],
                    linestyle="--",
                    linewidth=1.2,
                    label="Precio inicial",
                )

                ax_t.set_xlabel(
                    "Sesiones simuladas"
                )

                ax_t.set_ylabel(
                    "Precio simulado"
                )

                ax_t.legend()

                st.pyplot(fig_t)
                plt.close(fig_t)

                st.caption(
                    "Se muestran como máximo 100 trayectorias para mantener "
                    "el gráfico legible. Los cálculos utilizan todas las "
                    f"{mc['simulaciones']:,} simulaciones."
                )

                # ================================================================
                # DISTRIBUCIÓN FINAL
                # ================================================================

                st.markdown(
                    "#### Distribución del precio final"
                )

                r1, r2, r3 = st.columns(3)

                r1.metric(
                    "Mediana simulada",
                    f"{mc['mediana_final']:,.2f}",
                )

                r2.metric(
                    "Percentil 5",
                    f"{mc['percentiles'][5]:,.2f}",
                )

                r3.metric(
                    "Percentil 95",
                    f"{mc['percentiles'][95]:,.2f}",
                )

                st.caption(
                    "El 5.º y el 95.º percentil delimitan el 90% central "
                    "de los resultados simulados. No constituyen límites "
                    "mínimos o máximos posibles."
                )

                fig_d, ax_d = plt.subplots(
                    figsize=(12, 5)
                )

                ax_d.hist(
                    mc["finales"],
                    bins=60,
                    density=True,
                    alpha=0.75,
                )

                ax_d.axvline(
                    mc["precio_inicial"],
                    linestyle="--",
                    linewidth=1.3,
                    label="Precio inicial",
                )

                ax_d.axvline(
                    mc["percentiles"][5],
                    linestyle=":",
                    linewidth=1.3,
                    label="P5",
                )

                ax_d.axvline(
                    mc["percentiles"][50],
                    linestyle="-.",
                    linewidth=1.3,
                    label="Mediana",
                )

                ax_d.axvline(
                    mc["percentiles"][95],
                    linestyle=":",
                    linewidth=1.3,
                    label="P95",
                )

                ax_d.set_xlabel(
                    "Precio final simulado"
                )

                ax_d.set_ylabel(
                    "Densidad"
                )

                ax_d.legend()

                st.pyplot(fig_d)
                plt.close(fig_d)

                # ================================================================
                # PERCENTILES
                # ================================================================

                st.markdown(
                    "#### Escenarios por percentiles"
                )

                tabla_percentiles = pd.DataFrame(
                    {
                        "Percentil": [
                            "P5",
                            "P25",
                            "P50 · mediana",
                            "P75",
                            "P95",
                        ],
                        "Precio final": [
                            f"{mc['percentiles'][5]:,.2f}",
                            f"{mc['percentiles'][25]:,.2f}",
                            f"{mc['percentiles'][50]:,.2f}",
                            f"{mc['percentiles'][75]:,.2f}",
                            f"{mc['percentiles'][95]:,.2f}",
                        ],
                        "Variación vs inicial": [
                            (
                                f"{(mc['percentiles'][p] / mc['precio_inicial'] - 1) * 100:+.2f}%"
                            )
                            for p in [5, 25, 50, 75, 95]
                        ],
                        "Capital equivalente": [
                            f"{mc['capital_percentiles'][p]:,.2f}"
                            for p in [5, 25, 50, 75, 95]
                        ],
                    }
                )

                st.dataframe(
                    tabla_percentiles,
                    width="stretch",
                    hide_index=True,
                )

                # ================================================================
                # FRECUENCIAS DE LA SIMULACIÓN
                # ================================================================

                st.markdown(
                    "#### Frecuencias dentro de la simulación"
                )

                q1, q2 = st.columns(2)

                q1.metric(
                    "Final > precio inicial",
                    f"{mc['prob_sobre_inicial'] * 100:.1f}%",
                )

                q2.metric(
                    "Final < precio inicial",
                    f"{mc['prob_bajo_inicial'] * 100:.1f}%",
                )

                q3, q4 = st.columns(2)

                q3.metric(
                    "Final ≤ -10%",
                    f"{mc['prob_perdida_10'] * 100:.1f}%",
                )

                q4.metric(
                    "Final ≥ +10%",
                    f"{mc['prob_ganancia_10'] * 100:.1f}%",
                )

                st.caption(
                    "Estos porcentajes son frecuencias de los escenarios "
                    "generados por este modelo con estos parámetros. "
                    "No son probabilidades objetivas ni garantías sobre "
                    "el comportamiento futuro del activo."
                )

                # ================================================================
                # CAPITAL
                # ================================================================

                st.markdown(
                    "#### Capital simulado"
                )

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "Capital inicial",
                    f"{mc['capital_inicial']:,.2f}",
                )

                c2.metric(
                    "Capital mediano",
                    f"{mc['capital_percentiles'][50]:,.2f}",
                )

                c3.metric(
                    "Capital P5",
                    f"{mc['capital_percentiles'][5]:,.2f}",
                )

                st.caption(
                    "Conversión simplificada del movimiento del precio al "
                    "capital de referencia. No incluye impuestos, comisiones, "
                    "dividendos separados, aportaciones, retiradas ni efectos "
                    "de divisa."
                )

                # ================================================================
                # RESULTADO FINAL
                # ================================================================

                st.markdown(
                    "#### Resultado final de la simulación"
                )

                capital_inicio_mc = float(mc["capital_inicial"])
                capital_final_mc = float(mc["capital_percentiles"][50])

                resultado_mc = capital_final_mc - capital_inicio_mc
                rentabilidad_mc = (
                    capital_final_mc / capital_inicio_mc - 1
                ) * 100

                rf1, rf2, rf3 = st.columns(3)

                rf1.metric(
                    "Capital inicial",
                    f"{capital_inicio_mc:,.2f}",
                )

                rf2.metric(
                    "Capital final · escenario mediano",
                    f"{capital_final_mc:,.2f}",
                    delta=f"{resultado_mc:+,.2f}",
                )

                rf3.metric(
                    "Rentabilidad del escenario mediano",
                    f"{rentabilidad_mc:+.2f}%",
                )

                st.markdown(
                    f"""
**Lectura sencilla:** partiendo de un capital de referencia de
**{capital_inicio_mc:,.2f}**, el escenario central de esta simulación
(P50 o mediana) termina en **{capital_final_mc:,.2f}**.

Eso representa un resultado de **{resultado_mc:+,.2f}**
(**{rentabilidad_mc:+.2f}%**) durante el horizonte seleccionado.
                    """
                )

                resultado_escenarios_mc = pd.DataFrame(
                    {
                        "Escenario": [
                            "P5 · escenario inferior",
                            "P25",
                            "P50 · mediana",
                            "P75",
                            "P95 · escenario superior",
                        ],
                        "Capital final": [
                            f"{mc['capital_percentiles'][p]:,.2f}"
                            for p in [5, 25, 50, 75, 95]
                        ],
                        "Resultado": [
                            (
                                f"{mc['capital_percentiles'][p] - capital_inicio_mc:+,.2f}"
                            )
                            for p in [5, 25, 50, 75, 95]
                        ],
                        "Rentabilidad": [
                            (
                                f"{(mc['capital_percentiles'][p] / capital_inicio_mc - 1) * 100:+.2f}%"
                            )
                            for p in [5, 25, 50, 75, 95]
                        ],
                    }
                )

                st.dataframe(
                    resultado_escenarios_mc,
                    width="stretch",
                    hide_index=True,
                )

                st.caption(
                    "El capital final mostrado como escenario central corresponde "
                    "a la mediana (P50) de las simulaciones, no a una predicción "
                    "del capital que se obtendrá realmente. P5, P25, P75 y P95 "
                    "permiten observar otros puntos de la distribución simulada."
                )

                # ================================================================
                # METODOLOGÍA
                # ================================================================

                with st.expander(
                    "Cómo calcula FinScope Monte Carlo",
                    expanded=False,
                ):
                    st.markdown(
                        """
    **1. Datos históricos.** Se toman los precios diarios del periodo seleccionado.

    **2. Rendimientos logarítmicos.** FinScope calcula `ln(Pₜ / Pₜ₋₁)`.

    **3. Parámetros.** Se estiman la media y la desviación estándar de esos rendimientos diarios.

    **4. Simulación.** Se generan miles de secuencias aleatorias bajo un movimiento browniano geométrico (GBM) con parámetros constantes.

    **5. Precio final.** Cada trayectoria produce un precio al final del horizonte seleccionado.

    **6. Percentiles.** P5, P25, P50, P75 y P95 resumen la distribución generada por el modelo.

    **Limitaciones importantes:** el GBM supone parámetros constantes e incrementos logarítmicos normales e independientes. Los mercados reales pueden presentar cambios de régimen, volatilidad variable, autocorrelación, saltos y colas más extremas. Por ello, Monte Carlo no debe interpretarse como una predicción ni como una estimación completa de todos los riesgos posibles.
                        """
                    )

                st.warning(
                    "Monte Carlo genera escenarios condicionados al modelo "
                    "y a los datos históricos utilizados. Un resultado fuera "
                    "de los percentiles simulados sigue siendo posible."
                )

                st.caption(
                    f"Histórico: "
                    f"{mc['inicio_historico'].strftime('%d/%m/%Y')} – "
                    f"{mc['fin_historico'].strftime('%d/%m/%Y')} · "
                    f"Precio: {mc['columna_precio']} · "
                    f"Modelo: {mc['modelo']} · "
                    f"Semilla reproducible: {mc['seed']} · "
                    "Fuente: Yahoo Finance vía yfinance. "
                    "Cálculos: FinScope."
                )

        except Exception:
            st.info(
                "No se ha podido completar la simulación Monte Carlo "
                "con los datos disponibles."
            )


    if modulo_finscope == "Escenarios y sensibilidad":
        st.markdown("### Escenarios y análisis de sensibilidad")

        st.caption(
            "Permite estudiar cómo cambiaría matemáticamente el "
            "resultado de una inversión bajo distintos supuestos de "
            "rentabilidad. Los escenarios no son predicciones ni "
            "probabilidades futuras."
        )

        try:
            # ========================================================
            # HISTÓRICO DE REFERENCIA
            # ========================================================

            hist_esc = cache_obtener_serie_rendimiento(
                ticker,
                "5y",
            )

            if hist_esc is None or hist_esc.empty:
                st.info(
                    "No existe histórico suficiente para construir "
                    "los escenarios."
                )
            else:
                import numpy as np

                columna_esc = (
                    "Adj Close"
                    if "Adj Close" in hist_esc.columns
                    and hist_esc["Adj Close"].notna().sum() > 2
                    else "Close"
                )

                precios_esc = (
                    hist_esc[columna_esc]
                    .dropna()
                    .astype(float)
                )

                retornos_log_esc = np.log(
                    precios_esc / precios_esc.shift(1)
                ).dropna()

                precio_inicial_esc = float(
                    precios_esc.iloc[-1]
                )

                retorno_log_anual_esc = float(
                    retornos_log_esc.mean() * 252
                )

                # Convertimos el rendimiento logarítmico anual
                # histórico a rentabilidad compuesta equivalente.
                retorno_hist_esc = (
                    np.exp(retorno_log_anual_esc) - 1
                ) * 100

                volatilidad_hist_esc = float(
                    retornos_log_esc.std(ddof=1)
                    * np.sqrt(252)
                    * 100
                )

                st.markdown(
                    "#### Parámetros de referencia"
                )

                ref1, ref2, ref3 = st.columns(3)

                ref1.metric(
                    "Precio actual de referencia",
                    f"{precio_inicial_esc:,.2f}",
                )

                ref2.metric(
                    "Rentabilidad histórica anualizada",
                    f"{retorno_hist_esc:+.2f}%",
                )

                ref3.metric(
                    "Volatilidad histórica anualizada",
                    f"{volatilidad_hist_esc:.2f}%",
                )

                st.caption(
                    "La rentabilidad y volatilidad mostradas se "
                    "estiman con rendimientos logarítmicos diarios "
                    "del histórico de cinco años disponible. Son "
                    "referencias históricas, no previsiones."
                )

                # ====================================================
                # CONTROLES
                # ====================================================

                st.markdown("#### Supuestos")

                es1, es2, es3 = st.columns(3)

                with es1:
                    capital_esc = st.number_input(
                        "Capital inicial",
                        min_value=100.0,
                        value=10000.0,
                        step=500.0,
                        format="%.2f",
                        key=f"escenarios_capital_{ticker}",
                    )

                with es2:
                    horizonte_esc = st.selectbox(
                        "Horizonte",
                        [
                            0.25,
                            0.5,
                            1.0,
                            2.0,
                            3.0,
                            5.0,
                        ],
                        index=2,
                        format_func=lambda x: {
                            0.25: "3 meses",
                            0.5: "6 meses",
                            1.0: "1 año",
                            2.0: "2 años",
                            3.0: "3 años",
                            5.0: "5 años",
                        }[x],
                        key=f"escenarios_horizonte_{ticker}",
                    )

                with es3:
                    retorno_base_esc = st.number_input(
                        "Rentabilidad anual · escenario base",
                        min_value=-90.0,
                        max_value=200.0,
                        value=float(
                            round(retorno_hist_esc, 2)
                        ),
                        step=1.0,
                        format="%.2f",
                        key=f"escenarios_retorno_base_{ticker}",
                        help=(
                            "Supuesto introducido para el escenario "
                            "central. Por defecto se propone la "
                            "rentabilidad histórica anualizada."
                        ),
                    )

                sh1, sh2 = st.columns(2)

                with sh1:
                    shock_adverso_esc = st.number_input(
                        "Shock adverso",
                        min_value=-200.0,
                        max_value=0.0,
                        value=float(
                            round(
                                -volatilidad_hist_esc,
                                2,
                            )
                        ),
                        step=1.0,
                        format="%.2f",
                        key=f"escenarios_shock_adverso_{ticker}",
                        help=(
                            "Variación en puntos porcentuales que "
                            "se resta al supuesto base."
                        ),
                    )

                with sh2:
                    shock_favorable_esc = st.number_input(
                        "Shock favorable",
                        min_value=0.0,
                        max_value=200.0,
                        value=float(
                            round(
                                volatilidad_hist_esc,
                                2,
                            )
                        ),
                        step=1.0,
                        format="%.2f",
                        key=f"escenarios_shock_favorable_{ticker}",
                        help=(
                            "Variación en puntos porcentuales que "
                            "se suma al supuesto base."
                        ),
                    )

                resultado_esc = (
                    calcular_escenarios_sensibilidad(
                        precio_inicial=precio_inicial_esc,
                        capital_inicial=capital_esc,
                        horizonte_anios=horizonte_esc,
                        rentabilidad_base_pct=retorno_base_esc,
                        volatilidad_anual_pct=
                            volatilidad_hist_esc,
                        shock_adverso_pct=
                            shock_adverso_esc,
                        shock_favorable_pct=
                            shock_favorable_esc,
                    )
                )

                # ====================================================
                # TRES ESCENARIOS
                # ====================================================

                st.markdown(
                    "#### Resultado de los escenarios"
                )

                escenarios_esc = resultado_esc[
                    "escenarios"
                ]

                col_adv, col_base, col_fav = st.columns(3)

                for columna, nombre in [
                    (col_adv, "Adverso"),
                    (col_base, "Base"),
                    (col_fav, "Favorable"),
                ]:
                    escenario = escenarios_esc[nombre]

                    with columna:
                        st.markdown(f"**{nombre}**")

                        st.metric(
                            "Capital final",
                            (
                                f"{escenario['capital_final']:,.2f}"
                            ),
                            delta=(
                                f"{escenario['resultado_capital']:+,.2f}"
                            ),
                        )

                        st.metric(
                            "Precio resultante",
                            (
                                f"{escenario['precio_final']:,.2f}"
                            ),
                        )

                        st.caption(
                            "Rentabilidad anual supuesta: "
                            f"{escenario['rentabilidad_anual_pct']:+.2f}%"
                            " · Rentabilidad acumulada: "
                            f"{escenario['rentabilidad_acumulada_pct']:+.2f}%"
                        )

                # ====================================================
                # TABLA COMPARATIVA
                # ====================================================

                tabla_esc = []

                for nombre in [
                    "Adverso",
                    "Base",
                    "Favorable",
                ]:
                    e = escenarios_esc[nombre]

                    tabla_esc.append(
                        {
                            "Escenario": nombre,
                            "Rentabilidad anual":
                                f"{e['rentabilidad_anual_pct']:+.2f}%",
                            "Precio resultante":
                                f"{e['precio_final']:,.2f}",
                            "Capital final":
                                f"{e['capital_final']:,.2f}",
                            "Resultado":
                                f"{e['resultado_capital']:+,.2f}",
                            "Rentabilidad acumulada":
                                f"{e['rentabilidad_acumulada_pct']:+.2f}%",
                        }
                    )

                st.dataframe(
                    pd.DataFrame(tabla_esc),
                    width="stretch",
                    hide_index=True,
                )

                # ====================================================
                # SENSIBILIDAD
                # ====================================================

                st.markdown(
                    "#### Sensibilidad al supuesto de rentabilidad"
                )

                st.caption(
                    "Muestra cómo cambia el resultado manteniendo "
                    "constante el horizonte y desplazando el supuesto "
                    "de rentabilidad alrededor del escenario base."
                )

                sensibilidad_esc = pd.DataFrame(
                    resultado_esc["sensibilidad"]
                )

                sensibilidad_grafico = (
                    sensibilidad_esc[
                        [
                            "rentabilidad_anual_pct",
                            "capital_final",
                        ]
                    ]
                    .rename(
                        columns={
                            "rentabilidad_anual_pct":
                                "Rentabilidad anual (%)",
                            "capital_final":
                                "Capital final",
                        }
                    )
                )

                grafico_sensibilidad = (
                    alt.Chart(sensibilidad_grafico)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X(
                            "Rentabilidad anual (%):Q",
                            title="Rentabilidad anual supuesta (%)",
                        ),
                        y=alt.Y(
                            "Capital final:Q",
                            title="Capital final",
                            scale=alt.Scale(zero=False),
                        ),
                        tooltip=[
                            alt.Tooltip(
                                "Rentabilidad anual (%):Q",
                                format=".2f",
                            ),
                            alt.Tooltip(
                                "Capital final:Q",
                                format=",.2f",
                            ),
                        ],
                    )
                    .properties(height=360)
                )

                st.altair_chart(
                    grafico_sensibilidad,
                    width="stretch",
                )

                tabla_sens = pd.DataFrame(
                    {
                        "Variación vs base": [
                            f"{x:+.2f} pp"
                            for x in sensibilidad_esc[
                                "variacion_pct"
                            ]
                        ],
                        "Rentabilidad anual": [
                            f"{x:+.2f}%"
                            for x in sensibilidad_esc[
                                "rentabilidad_anual_pct"
                            ]
                        ],
                        "Precio resultante": [
                            f"{x:,.2f}"
                            for x in sensibilidad_esc[
                                "precio_final"
                            ]
                        ],
                        "Capital final": [
                            f"{x:,.2f}"
                            for x in sensibilidad_esc[
                                "capital_final"
                            ]
                        ],
                        "Rentabilidad acumulada": [
                            f"{x:+.2f}%"
                            for x in sensibilidad_esc[
                                "rentabilidad_acumulada_pct"
                            ]
                        ],
                    }
                )

                st.dataframe(
                    tabla_sens,
                    width="stretch",
                    hide_index=True,
                )

                # ====================================================
                # LECTURA DEL RESULTADO
                # ====================================================

                base_esc = escenarios_esc["Base"]
                adverso_esc = escenarios_esc["Adverso"]
                favorable_esc = escenarios_esc["Favorable"]

                st.markdown(
                    "#### Lectura del análisis"
                )

                st.write(
                    f"Con un capital inicial de "
                    f"{capital_esc:,.2f}, el escenario base "
                    f"terminaría matemáticamente en "
                    f"{base_esc['capital_final']:,.2f}. "
                    f"Con los shocks seleccionados, el escenario "
                    f"adverso terminaría en "
                    f"{adverso_esc['capital_final']:,.2f} y el "
                    f"favorable en "
                    f"{favorable_esc['capital_final']:,.2f}."
                )

                st.warning(
                    "Estos resultados responden a supuestos definidos "
                    "por el usuario y a una referencia histórica. "
                    "No indican cuál de los escenarios es más probable "
                    "ni constituyen una previsión, recomendación de "
                    "inversión o límite máximo de pérdida o ganancia."
                )

                with st.expander(
                    "Cómo calcula FinScope los escenarios",
                    expanded=False,
                ):
                    st.markdown(
                        """
1. FinScope obtiene cinco años de precios históricos disponibles.

2. Calcula rendimientos logarítmicos diarios.

3. Estima una rentabilidad histórica anualizada y una volatilidad anualizada como referencias.

4. El usuario define el supuesto anual del escenario base.

5. Los shocks adverso y favorable modifican ese supuesto en puntos porcentuales.

6. Cada escenario se capitaliza durante el horizonte seleccionado:

**Valor final = Valor inicial × (1 + rentabilidad anual)^años**

7. La sensibilidad repite el cálculo desplazando el supuesto base en -1, -0,5, 0, +0,5 y +1 veces la volatilidad histórica anual.

Este análisis es determinista: no asigna probabilidades a los escenarios. Monte Carlo, situado en el módulo anterior, responde a una pregunta distinta al generar una distribución de trayectorias bajo un modelo estocástico.
                        """
                    )

                st.caption(
                    "Histórico de referencia: 5 años · "
                    f"Precio utilizado: {columna_esc} · "
                    "Fuente de precios: Yahoo Finance vía yfinance · "
                    "Cálculos de escenarios: FinScope."
                )

        except Exception as exc:
            st.info(
                "No se ha podido completar el análisis de "
                "escenarios con los datos disponibles."
            )

    # ============================================================
    # FIN · ESCENARIOS Y SENSIBILIDAD
    # ============================================================



    # ============================================================
    # MACHINE LEARNING
    # ============================================================

    if modulo_finscope == "Machine Learning":

        st.markdown("## Machine Learning")

        st.caption(
            "FinScope utiliza patrones históricos para generar "
            "estimaciones cuantitativas sobre distintos horizontes. "
            "Las estimaciones no son certezas ni recomendaciones de "
            "compra o venta. Su utilidad depende de la capacidad "
            "predictiva demostrada fuera de muestra."
        )

        try:
            from services.machine_learning import (
                analizar_machine_learning,
            )

            with st.spinner(
                "Entrenando y validando modelos temporales..."
            ):
                resultado_ml = analizar_machine_learning(
                    ticker,
                    periodo="5y",
                    horizontes=(1, 5, 20, 60),
                )

            if resultado_ml.get("error"):
                st.warning(
                    "No se ha podido completar el análisis "
                    f"Machine Learning: {resultado_ml['error']}"
                )

            else:
                resultados_ml = resultado_ml["resultados"]

                opciones_ml = {
                    "1 sesión": 1,
                    "1 semana": 5,
                    "1 mes": 20,
                    "3 meses": 60,
                }

                horizonte_texto = st.segmented_control(
                    "Horizonte de estimación",
                    options=list(opciones_ml.keys()),
                    default="1 mes",
                    key=f"ml_horizonte_{ticker_estado}",
                )

                if horizonte_texto is None:
                    horizonte_texto = "1 mes"

                horizonte_ml = opciones_ml[
                    horizonte_texto
                ]

                ml = resultados_ml.get(
                    horizonte_ml,
                    {},
                )

                if ml.get("error"):
                    st.warning(
                        "No hay una estimación válida para este "
                        f"horizonte: {ml['error']}"
                    )

                else:
                    precio_actual_ml = float(
                        ml["precio_actual_eur"]
                    )

                    precio_estimado_ml = float(
                        ml["precio_estimado_eur"]
                    )

                    retorno_estimado_ml = float(
                        ml["retorno_estimado"]
                    )

                    probabilidad_ml = float(
                        ml["probabilidad_subida"]
                    )

                    calidad_ml = str(
                        ml["calidad"]
                    ).lower()

                    rango_bajo_ml, rango_alto_ml = (
                        ml["intervalo_precio_80_eur"]
                    )

                    # ------------------------------------------------
                    # LECTURA PRINCIPAL
                    # ------------------------------------------------

                    if retorno_estimado_ml > 0.002:
                        direccion_ml = "Alcista"
                    elif retorno_estimado_ml < -0.002:
                        direccion_ml = "Bajista"
                    else:
                        direccion_ml = "Neutral"

                    st.markdown(
                        "### Estimación del modelo"
                    )

                    m1, m2, m3, m4 = st.columns(4)

                    with m1:
                        st.metric(
                            "Precio actual",
                            f"{precio_actual_ml:,.2f} €",
                        )

                    with m2:
                        st.metric(
                            f"Estimación · {horizonte_texto}",
                            f"{precio_estimado_ml:,.2f} €",
                            delta=(
                                f"{retorno_estimado_ml * 100:+.2f}%"
                            ),
                        )

                    with m3:
                        st.metric(
                            "Salida probabilística",
                            f"{probabilidad_ml * 100:.1f}%",
                            help=(
                                "Probabilidad producida por el "
                                "clasificador de que la rentabilidad "
                                "del horizonte sea positiva. No debe "
                                "interpretarse como una probabilidad "
                                "objetiva del futuro."
                            ),
                        )

                    with m4:
                        st.metric(
                            "Respaldo histórico",
                            calidad_ml.capitalize(),
                        )

                    st.markdown(
                        f"""
                        <div style="
                            border:1px solid rgba(120,150,180,.22);
                            border-radius:16px;
                            padding:18px 20px;
                            margin-top:8px;
                            margin-bottom:14px;
                            background:rgba(20,32,48,.30);
                        ">
                            <div style="
                                font-size:.78rem;
                                opacity:.68;
                                margin-bottom:6px;
                                letter-spacing:.04em;
                            ">
                                LECTURA FINSCOPE
                            </div>
                            <div style="
                                font-size:1.22rem;
                                line-height:1.55;
                            ">
                                El modelo presenta una estimación
                                <strong>{direccion_ml.lower()}</strong>
                                para {horizonte_texto.lower()}:
                                aproximadamente
                                <strong>
                                    {retorno_estimado_ml * 100:+.2f}%
                                </strong>,
                                hasta unos
                                <strong>
                                    {precio_estimado_ml:,.2f} €
                                </strong>.
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # ------------------------------------------------
                    # INCERTIDUMBRE
                    # ------------------------------------------------

                    st.markdown(
                        "### Incertidumbre de la estimación"
                    )

                    r1, r2 = st.columns([1.2, 1.8])

                    with r1:
                        st.metric(
                            "Rango empírico 80%",
                            (
                                f"{rango_bajo_ml:,.2f} € — "
                                f"{rango_alto_ml:,.2f} €"
                            ),
                        )

                    with r2:
                        if calidad_ml == "alta":
                            st.success(
                                "Este horizonte muestra el mayor "
                                "respaldo histórico según las pruebas "
                                "fuera de muestra del modelo. Esto no "
                                "garantiza que la estimación futura "
                                "se cumpla."
                            )

                        elif calidad_ml == "moderada":
                            st.info(
                                "El modelo ha mostrado cierta capacidad "
                                "predictiva histórica, aunque la "
                                "incertidumbre sigue siendo relevante."
                            )

                        elif calidad_ml == "limitada":
                            st.warning(
                                "La capacidad predictiva histórica es "
                                "limitada. La estimación debe tratarse "
                                "con cautela."
                            )

                        else:
                            st.warning(
                                "El modelo no ha demostrado una ventaja "
                                "predictiva suficiente en este horizonte. "
                                "La cifra estimada se muestra como salida "
                                "del modelo, no como una previsión robusta."
                            )

                    # ------------------------------------------------
                    # CALIDAD TÉCNICA
                    # ------------------------------------------------

                    with st.expander(
                        "Calidad del modelo y validación"
                    ):
                        reg = ml[
                            "metricas_regresion"
                        ]

                        cla = ml[
                            "metricas_clasificacion"
                        ]

                        t1, t2, t3, t4 = st.columns(4)

                        with t1:
                            st.metric(
                                "MAE",
                                f"{reg['mae'] * 100:.2f}%",
                            )

                        with t2:
                            st.metric(
                                "MAE baseline",
                                (
                                    f"{reg['baseline_mae'] * 100:.2f}%"
                                ),
                            )

                        with t3:
                            st.metric(
                                "ROC-AUC",
                                f"{cla['roc_auc']:.3f}",
                            )

                        with t4:
                            st.metric(
                                "Balanced accuracy",
                                (
                                    f"{cla['balanced_accuracy'] * 100:.1f}%"
                                ),
                            )

                        t5, t6, t7, t8 = st.columns(4)

                        with t5:
                            st.metric(
                                "Brier Score",
                                f"{cla['brier_score']:.3f}",
                            )

                        with t6:
                            st.metric(
                                "Brier baseline",
                                (
                                    f"{cla['baseline_brier_score']:.3f}"
                                ),
                            )

                        with t7:
                            st.metric(
                                "Predicciones OOS",
                                str(
                                    ml["observaciones_oos"]
                                ),
                            )

                        with t8:
                            st.metric(
                                "Folds temporales",
                                str(
                                    ml["numero_folds"]
                                ),
                            )

                        mejora_mae_ml = float(
                            reg[
                                "mejora_mae_vs_baseline"
                            ]
                        )

                        st.write(
                            "Mejora del MAE frente al baseline: "
                            f"**{mejora_mae_ml * 100:+.2f}%**"
                        )

                        st.caption(
                            "La validación es walk-forward: cada "
                            "predicción de prueba se realiza sobre "
                            "fechas posteriores a las utilizadas para "
                            "entrenar el modelo. Las métricas mostradas "
                            "son fuera de muestra."
                        )

                        principales_ml = ml.get(
                            "features_principales",
                            {},
                        )

                        if principales_ml:
                            st.markdown(
                                "**Variables con mayor peso en la "
                                "regresión final**"
                            )

                            nombres_features_ml = {
                                "ret_1": "Rentabilidad · 1 sesión",
                                "ret_2": "Rentabilidad · 2 sesiones",
                                "ret_5": "Rentabilidad · 5 sesiones",
                                "ret_10": "Rentabilidad · 10 sesiones",
                                "ret_20": "Rentabilidad · 20 sesiones",
                                "ret_60": "Rentabilidad · 60 sesiones",
                                "vol_5": "Volatilidad · 5 sesiones",
                                "vol_10": "Volatilidad · 10 sesiones",
                                "vol_20": "Volatilidad · 20 sesiones",
                                "vol_60": "Volatilidad · 60 sesiones",
                                "mom_5": "Momentum · 5 sesiones",
                                "mom_20": "Momentum · 20 sesiones",
                                "mom_60": "Momentum · 60 sesiones",
                                "dist_ma_5": "Distancia a media móvil · 5 sesiones",
                                "dist_ma_20": "Distancia a media móvil · 20 sesiones",
                                "dist_ma_60": "Distancia a media móvil · 60 sesiones",
                                "drawdown_20": "Drawdown · 20 sesiones",
                                "drawdown_60": "Drawdown · 60 sesiones",
                                "rango_ret_20": "Rango de retornos · 20 sesiones",
                                "rango_ret_60": "Rango de retornos · 60 sesiones",
                            }

                            tabla_features_ml = pd.DataFrame(
                                [
                                    {
                                        "Variable": nombres_features_ml.get(
                                            nombre,
                                            nombre,
                                        ),
                                        "Importancia": importancia * 100,
                                    }
                                    for nombre, importancia
                                    in principales_ml.items()
                                ]
                            )

                            tabla_features_ml["Importancia"] = (
                                tabla_features_ml["Importancia"]
                                .map(lambda x: f"{x:.2f}%")
                            )

                            st.dataframe(
                                tabla_features_ml,
                                hide_index=True,
                            )

                    st.caption(
                        "El rango mostrado se construye a partir de "
                        "los errores históricos fuera de muestra. "
                        "La salida probabilística pertenece al modelo "
                        "de clasificación y no equivale a una "
                        "probabilidad objetiva o garantizada de mercado."
                    )

        except Exception as exc:
            st.error(
                "No se ha podido completar el módulo "
                f"Machine Learning: {exc}"
            )

    # ============================================================
    # FIN · MACHINE LEARNING
    # ============================================================


    if modulo_finscope == "Comparación cuantitativa":

        st.markdown("## Comparación cuantitativa")

        st.caption(
            "FinScope Quant Pro analiza rendimiento, riesgo, distribución, "
            "drawdowns y relación estadística entre dos activos. "
            "Los resultados describen datos históricos; no predicen "
            "el comportamiento futuro."
        )

        q1, q2, q3, q4 = st.columns(
            [1.15, 1.0, 1.0, 1.0]
        )

        with q1:
            ticker_comparado = st.text_input(
                "Activo comparado",
                value="MSFT" if ticker != "MSFT" else "AAPL",
                key=f"quant_ticker_{ticker}",
            ).strip().upper()

        with q2:
            periodo_quant = st.selectbox(
                "Periodo",
                ["1y", "3y", "5y", "10y", "max"],
                index=2,
                key=f"quant_periodo_{ticker}",
            )

        with q3:
            serie_quant_ui = st.selectbox(
                "Serie analizada",
                [
                    "Rentabilidad ajustada",
                    "Precio",
                ],
                index=0,
                key=f"quant_serie_{ticker}",
                help=(
                    "Rentabilidad ajustada utiliza Adj Close cuando Yahoo "
                    "lo proporciona. Precio utiliza Close sin ajustar."
                ),
            )

        with q4:
            capital_quant = st.number_input(
                "Capital de referencia",
                min_value=100.0,
                max_value=100000000.0,
                value=10000.0,
                step=1000.0,
                key=f"quant_capital_{ticker}",
            )

        rf_quant_pct = st.number_input(
            "Tasa libre de riesgo anual (%)",
            min_value=0.0,
            max_value=30.0,
            value=0.0,
            step=0.25,
            key=f"quant_rf_{ticker}",
            help=(
                "Se utiliza para Sharpe, Sortino y alpha. "
                "0% permite estudiar únicamente el comportamiento "
                "histórico de los activos."
            ),
        )

        if not ticker_comparado:

            st.info(
                "Introduce el ticker de un segundo activo."
            )

        elif ticker_comparado == ticker:

            st.info(
                "Selecciona un activo diferente al activo principal."
            )

        else:

            tipo_serie_quant = (
                "ajustada"
                if serie_quant_ui == "Rentabilidad ajustada"
                else "precio"
            )

            with st.spinner(
                "Calculando análisis Quant Pro..."
            ):

                resultado_quant = calcular_comparacion_quant_pro(
                    ticker,
                    ticker_comparado,
                    periodo=periodo_quant,
                    tipo_serie=tipo_serie_quant,
                    rf_anual=rf_quant_pct / 100.0,
                    capital_inicial=capital_quant,
                )

            error_quant = resultado_quant.get(
                "error"
            )

            if error_quant:

                st.warning(
                    error_quant
                )

            else:

                A = resultado_quant[
                    "metricas_a"
                ]

                B = resultado_quant[
                    "metricas_b"
                ]

                R = resultado_quant[
                    "relativas"
                ]

                st.caption(
                    f"Serie efectiva · {ticker}: "
                    f"{resultado_quant['columna_a']} · "
                    f"{ticker_comparado}: "
                    f"{resultado_quant['columna_b']} · "
                    f"Frecuencia: diaria"
                )

                # ============================================================
                # RESUMEN PRINCIPAL
                # ============================================================

                st.markdown("### Resumen")

                r1, r2, r3, r4 = st.columns(4)

                with r1:
                    st.metric(
                        f"CAGR · {ticker}",
                        (
                            f"{A['cagr']*100:.2f}%"
                            if A.get("cagr") is not None
                            else "N/D"
                        ),
                    )

                with r2:
                    st.metric(
                        f"CAGR · {ticker_comparado}",
                        (
                            f"{B['cagr']*100:.2f}%"
                            if B.get("cagr") is not None
                            else "N/D"
                        ),
                    )

                with r3:
                    st.metric(
                        "Correlación",
                        (
                            f"{R['correlacion']:.2f}"
                            if R.get("correlacion") is not None
                            else "N/D"
                        ),
                    )

                with r4:
                    st.metric(
                        f"Beta {ticker}/{ticker_comparado}",
                        (
                            f"{R['beta']:.2f}"
                            if R.get("beta") is not None
                            else "N/D"
                        ),
                    )

                # ============================================================
                # BASE 100
                # ============================================================

                st.markdown("### Evolución comparable · Base 100")

                base100_quant = resultado_quant[
                    "base100"
                ]

                if (
                    base100_quant is not None
                    and not base100_quant.empty
                ):
                    st.line_chart(
                        base100_quant
                    )

                # ============================================================
                # RENDIMIENTO Y RIESGO
                # ============================================================

                st.markdown("### Rendimiento y riesgo")

                tabla_rr = pd.DataFrame(
                    {
                        "Métrica": [
                            "Rentabilidad acumulada",
                            "CAGR",
                            "Volatilidad anualizada",
                            "Downside deviation",
                            "Máximo drawdown",
                            "Sharpe",
                            "Sortino",
                            "Calmar",
                        ],
                        ticker: [
                            A.get("rentabilidad_acumulada"),
                            A.get("cagr"),
                            A.get("volatilidad_anualizada"),
                            A.get("downside_deviation"),
                            A.get("max_drawdown"),
                            A.get("sharpe"),
                            A.get("sortino"),
                            A.get("calmar"),
                        ],
                        ticker_comparado: [
                            B.get("rentabilidad_acumulada"),
                            B.get("cagr"),
                            B.get("volatilidad_anualizada"),
                            B.get("downside_deviation"),
                            B.get("max_drawdown"),
                            B.get("sharpe"),
                            B.get("sortino"),
                            B.get("calmar"),
                        ],
                    }
                )

                tabla_rr_mostrar = tabla_rr.copy()

                metricas_pct = {
                    "Rentabilidad acumulada",
                    "CAGR",
                    "Volatilidad anualizada",
                    "Downside deviation",
                    "Máximo drawdown",
                }

                for col in [
                    ticker,
                    ticker_comparado,
                ]:

                    tabla_rr_mostrar[col] = [
                        (
                            "N/D"
                            if pd.isna(valor)
                            else (
                                f"{valor*100:.2f}%"
                                if metrica in metricas_pct
                                else f"{valor:.2f}"
                            )
                        )
                        for metrica, valor in zip(
                            tabla_rr_mostrar["Métrica"],
                            tabla_rr_mostrar[col],
                        )
                    ]

                st.dataframe(
                    tabla_rr_mostrar,
                    width="stretch",
                    hide_index=True,
                )

                # ============================================================
                # DISTRIBUCIÓN Y COLAS
                # ============================================================

                st.markdown(
                    "### Distribución y riesgo de cola"
                )

                tabla_tail = pd.DataFrame(
                    {
                        "Métrica": [
                            "VaR histórico 95%",
                            "CVaR histórico 95%",
                            "VaR histórico 99%",
                            "CVaR histórico 99%",
                            "Mejor sesión",
                            "Peor sesión",
                            "Sesiones positivas",
                            "Skewness",
                            "Kurtosis exceso",
                            "Omega",
                        ],
                        ticker: [
                            A.get("var_95"),
                            A.get("cvar_95"),
                            A.get("var_99"),
                            A.get("cvar_99"),
                            A.get("mejor_sesion"),
                            A.get("peor_sesion"),
                            A.get("porcentaje_sesiones_positivas"),
                            A.get("skewness"),
                            A.get("kurtosis_exceso"),
                            A.get("omega"),
                        ],
                        ticker_comparado: [
                            B.get("var_95"),
                            B.get("cvar_95"),
                            B.get("var_99"),
                            B.get("cvar_99"),
                            B.get("mejor_sesion"),
                            B.get("peor_sesion"),
                            B.get("porcentaje_sesiones_positivas"),
                            B.get("skewness"),
                            B.get("kurtosis_exceso"),
                            B.get("omega"),
                        ],
                    }
                )

                tail_pct = {
                    "VaR histórico 95%",
                    "CVaR histórico 95%",
                    "VaR histórico 99%",
                    "CVaR histórico 99%",
                    "Mejor sesión",
                    "Peor sesión",
                    "Sesiones positivas",
                }

                tabla_tail_mostrar = (
                    tabla_tail.copy()
                )

                for col in [
                    ticker,
                    ticker_comparado,
                ]:

                    tabla_tail_mostrar[col] = [
                        (
                            "N/D"
                            if pd.isna(valor)
                            else (
                                f"{valor*100:.2f}%"
                                if metrica in tail_pct
                                else f"{valor:.2f}"
                            )
                        )
                        for metrica, valor in zip(
                            tabla_tail_mostrar["Métrica"],
                            tabla_tail_mostrar[col],
                        )
                    ]

                st.dataframe(
                    tabla_tail_mostrar,
                    width="stretch",
                    hide_index=True,
                )

                st.caption(
                    "VaR y CVaR se muestran como magnitud positiva de "
                    "pérdida histórica por sesión. No representan una "
                    "pérdida máxima posible."
                )

                # ============================================================
                # RELACIÓN ENTRE ACTIVOS
                # ============================================================

                st.markdown(
                    "### Relación cuantitativa entre activos"
                )

                tabla_rel = pd.DataFrame(
                    {
                        "Métrica": [
                            "Correlación",
                            "R²",
                            f"Beta {ticker}/{ticker_comparado}",
                            "Alpha anualizado",
                            "Tracking error",
                            "Information ratio",
                            "Upside capture",
                            "Downside capture",
                            "Observaciones comunes",
                        ],
                        "Valor": [
                            R.get("correlacion"),
                            R.get("r2"),
                            R.get("beta"),
                            R.get("alpha_anualizado"),
                            R.get("tracking_error"),
                            R.get("information_ratio"),
                            R.get("upside_capture"),
                            R.get("downside_capture"),
                            R.get("observaciones_comunes"),
                        ],
                    }
                )

                def _fmt_rel(metrica, valor):

                    if valor is None or pd.isna(valor):
                        return "N/D"

                    if metrica in {
                        "Alpha anualizado",
                        "Tracking error",
                    }:
                        return f"{valor*100:.2f}%"

                    if metrica in {
                        "Upside capture",
                        "Downside capture",
                    }:
                        return f"{valor*100:.1f}%"

                    if metrica == "Observaciones comunes":
                        return f"{int(valor)}"

                    return f"{valor:.3f}"

                tabla_rel["Valor"] = [
                    _fmt_rel(m, v)
                    for m, v in zip(
                        tabla_rel["Métrica"],
                        tabla_rel["Valor"],
                    )
                ]

                st.dataframe(
                    tabla_rel,
                    width="stretch",
                    hide_index=True,
                )

                # ============================================================
                # ROLLING
                # ============================================================

                rolling_quant = resultado_quant[
                    "rolling"
                ]

                ventana_quant = rolling_quant.get(
                    "ventana"
                )

                if ventana_quant:

                    st.markdown(
                        "### Riesgo y relación dinámica"
                    )

                    st.caption(
                        f"Ventana rolling utilizada: "
                        f"{ventana_quant} observaciones comunes."
                    )

                    corr_roll = rolling_quant.get(
                        "correlacion_rolling"
                    )

                    if (
                        corr_roll is not None
                        and not corr_roll.empty
                    ):

                        st.markdown(
                            "#### Correlación rolling"
                        )

                        st.line_chart(
                            corr_roll.rename(
                                "Correlación"
                            )
                        )

                    beta_roll = rolling_quant.get(
                        "beta_rolling"
                    )

                    if (
                        beta_roll is not None
                        and not beta_roll.empty
                    ):

                        st.markdown(
                            "#### Beta rolling"
                        )

                        st.line_chart(
                            beta_roll.rename(
                                "Beta"
                            )
                        )

                    vol_a_roll = rolling_quant.get(
                        "volatilidad_a_rolling"
                    )

                    vol_b_roll = rolling_quant.get(
                        "volatilidad_b_rolling"
                    )

                    if (
                        vol_a_roll is not None
                        and vol_b_roll is not None
                        and not vol_a_roll.empty
                        and not vol_b_roll.empty
                    ):

                        vols_roll = pd.concat(
                            [
                                vol_a_roll.rename(ticker),
                                vol_b_roll.rename(
                                    ticker_comparado
                                ),
                            ],
                            axis=1,
                        ).dropna()

                        if not vols_roll.empty:

                            st.markdown(
                                "#### Volatilidad rolling"
                            )

                            st.line_chart(
                                vols_roll
                            )

                # ============================================================
                # DRAWDOWN
                # ============================================================

                st.markdown(
                    "### Drawdown histórico"
                )

                dd_quant = pd.concat(
                    [
                        A["serie_drawdown"].rename(
                            ticker
                        ),
                        B["serie_drawdown"].rename(
                            ticker_comparado
                        ),
                    ],
                    axis=1,
                )

                st.line_chart(
                    dd_quant
                )

                d1, d2 = st.columns(2)

                with d1:

                    st.markdown(
                        f"#### Principales episodios · {ticker}"
                    )

                    episodios_a = A.get(
                        "episodios_drawdown"
                    )

                    if (
                        episodios_a is not None
                        and not episodios_a.empty
                    ):

                        ep_a = episodios_a.copy()

                        ep_a["Drawdown"] = (
                            ep_a["Drawdown"]
                            * 100
                        ).map(
                            lambda x: f"{x:.2f}%"
                        )

                        st.dataframe(
                            ep_a,
                            width="stretch",
                            hide_index=True,
                        )

                with d2:

                    st.markdown(
                        f"#### Principales episodios · {ticker_comparado}"
                    )

                    episodios_b = B.get(
                        "episodios_drawdown"
                    )

                    if (
                        episodios_b is not None
                        and not episodios_b.empty
                    ):

                        ep_b = episodios_b.copy()

                        ep_b["Drawdown"] = (
                            ep_b["Drawdown"]
                            * 100
                        ).map(
                            lambda x: f"{x:.2f}%"
                        )

                        st.dataframe(
                            ep_b,
                            width="stretch",
                            hide_index=True,
                        )

                # ============================================================
                # HORIZONTES
                # ============================================================

                st.markdown(
                    "### Rentabilidad por horizontes"
                )

                ha = resultado_quant[
                    "horizontes_a"
                ].copy()

                hb = resultado_quant[
                    "horizontes_b"
                ].copy()

                if not ha.empty and not hb.empty:

                    ha = ha[
                        [
                            "Horizonte",
                            "Rentabilidad",
                            "CAGR",
                        ]
                    ].rename(
                        columns={
                            "Rentabilidad": ticker,
                            "CAGR": f"CAGR {ticker}",
                        }
                    )

                    hb = hb[
                        [
                            "Horizonte",
                            "Rentabilidad",
                            "CAGR",
                        ]
                    ].rename(
                        columns={
                            "Rentabilidad": ticker_comparado,
                            "CAGR": f"CAGR {ticker_comparado}",
                        }
                    )

                    tabla_h = ha.merge(
                        hb,
                        on="Horizonte",
                        how="outer",
                    )

                    for col in tabla_h.columns:

                        if col != "Horizonte":

                            tabla_h[col] = tabla_h[
                                col
                            ].map(
                                lambda x: (
                                    "N/D"
                                    if pd.isna(x)
                                    else f"{x*100:.2f}%"
                                )
                            )

                    st.dataframe(
                        tabla_h,
                        width="stretch",
                        hide_index=True,
                    )

                # ============================================================
                # INVERSIÓN EQUIVALENTE
                # ============================================================

                st.markdown(
                    "### Misma inversión inicial"
                )

                inversion_quant = resultado_quant[
                    "inversion"
                ]

                if (
                    inversion_quant is not None
                    and not inversion_quant.empty
                ):

                    st.line_chart(
                        inversion_quant
                    )

                    valor_a = float(
                        inversion_quant[
                            ticker
                        ].iloc[-1]
                    )

                    valor_b = float(
                        inversion_quant[
                            ticker_comparado
                        ].iloc[-1]
                    )

                    i1, i2, i3 = st.columns(3)

                    with i1:
                        st.metric(
                            "Capital inicial",
                            f"{capital_quant:,.2f}",
                        )

                    with i2:
                        st.metric(
                            ticker,
                            f"{valor_a:,.2f}",
                            f"{valor_a-capital_quant:+,.2f}",
                        )

                    with i3:
                        st.metric(
                            ticker_comparado,
                            f"{valor_b:,.2f}",
                            f"{valor_b-capital_quant:+,.2f}",
                        )

                # ============================================================
                # LECTURA AUTOMÁTICA
                # ============================================================

                st.markdown(
                    "### Lectura cuantitativa"
                )

                for texto_quant in resultado_quant[
                    "observaciones"
                ]:
                    st.write(
                        f"• {texto_quant}"
                    )

                # ============================================================
                # METODOLOGÍA
                # ============================================================

                with st.expander(
                    "Metodología Quant Pro"
                ):

                    st.markdown(
                        """
FinScope descarga series diarias mediante Yahoo Finance vía
`yfinance`, sin ajuste automático.

**Rentabilidad ajustada** utiliza `Adj Close` cuando está
disponible. Esta serie incorpora los ajustes distribuidos por
Yahoo por eventos corporativos y distribuciones y se utiliza como
aproximación al comportamiento total ajustado histórico.

**Precio** utiliza `Close` sin ajustar.

Las métricas individuales se calculan sobre la serie completa de
cada activo. Los rendimientos de los dos activos solo se alinean
posteriormente para calcular correlación, beta, alpha y otras
métricas conjuntas. De esta forma, la ausencia de una sesión en un
activo no altera artificialmente la volatilidad o el drawdown del
otro.

La volatilidad utiliza la desviación estándar muestral de los
rendimientos y anualización por raíz del número de periodos.

Sharpe utiliza el exceso de rentabilidad respecto a la tasa libre
de riesgo seleccionada.

Sortino utiliza downside deviation respecto al objetivo periódico
derivado de esa tasa.

VaR histórico representa un cuantil de la distribución observada.
CVaR representa la pérdida media de las observaciones que quedan
más allá de ese cuantil.

Beta se calcula como covarianza(A,B) / varianza(B).

R² se muestra como el cuadrado de la correlación en esta regresión
simple de un solo factor.

Alpha es una estimación histórica de Jensen basada en rendimientos
periódicos y se anualiza aritméticamente.

Tracking error es la volatilidad anualizada de A − B.

Upside y downside capture utilizan los meses en los que el segundo
activo tuvo respectivamente rentabilidad positiva o negativa. Para
cada subconjunto se componen geométricamente los rendimientos
mensuales, se anualizan y se calcula la relación entre el rendimiento
del primer activo y el del segundo.

Las métricas rolling usan una ventana de observaciones comunes,
mostrada en pantalla.

Para acciones y activos de mercado tradicional se utiliza una
convención de 252 sesiones por año. Para criptomonedas analizadas
individualmente se utilizan 365 días. Cuando se comparan activos
con calendarios diferentes, las métricas conjuntas se calculan
solo sobre las observaciones coincidentes.

Los datos históricos pueden contener errores, revisiones,
ausencias o diferencias metodológicas del proveedor. Ninguna
métrica constituye una predicción ni una recomendación de
inversión.
                        """
                    )

                st.caption(
                    "Fuente de mercado: Yahoo Finance vía yfinance. "
                    "FinScope no rellena datos históricos ausentes para "
                    "forzar resultados cuantitativos."
                )


    # ============================================================
    # CARTERAS Y DIVERSIFICACIÓN
    # ============================================================

    if modulo_finscope == "Carteras y diversificación":

        st.markdown("## Carteras y diversificación")

        st.caption(
            "Analiza cómo se comportan varios activos cuando se combinan "
            "en una cartera. FinScope estudia rentabilidad, riesgo, "
            "correlaciones y contribución de cada posición utilizando "
            "datos históricos diarios."
        )

        p1, p2, p3, p4 = st.columns(4)

        with p1:
            activos_txt = st.text_input(
                "Activos",
                value=f"{ticker}, MSFT, ^GSPC",
                key=f"portfolio_assets_{ticker}",
                help=(
                    "Introduce entre 2 y 8 activos separados por comas. "
                    "Puedes utilizar tickers o nombres reconocidos, por "
                    "ejemplo: AAPL, Bitcoin, S&P 500, Oro."
                ),
            )

        # ------------------------------------------------------------
        # Resolución universal de activos
        # ------------------------------------------------------------
        #
        # Carteras reutiliza exactamente el mismo buscador que el
        # buscador principal de FinScope. De esta forma nombres como
        # Apple, Microsoft, Inditex, S&P 500, Bitcoin u Oro se
        # convierten al ticker principal reconocido por FinScope.
        #
        # Si el usuario introduce directamente un ticker válido,
        # buscar_activos() también lo prioriza.
        # ------------------------------------------------------------

        activos_originales = [
            x.strip()
            for x in activos_txt.split(",")
            if x.strip()
        ]

        activos_resueltos = []
        detalles_resolucion_cartera = []
        activos_no_resueltos = []

        for activo_original in activos_originales:

            resultados_activo = buscar_activos(
                activo_original,
                max_resultados=8,
            )

            if resultados_activo:

                # El buscador universal ya ordena los resultados
                # aplicando aliases, coincidencia de ticker, nombre,
                # mercado principal y tipo de activo.
                #
                # Carteras debe respetar ese ranking sin volver a
                # reinterpretar la consulta.
                activo_elegido = resultados_activo[0]

                ticker_resuelto = str(
                    activo_elegido.get("ticker") or ""
                ).strip().upper()

                nombre_resuelto = str(
                    activo_elegido.get("nombre")
                    or ticker_resuelto
                ).strip()

                tipo_resuelto = str(
                    activo_elegido.get("tipo_nombre")
                    or activo_elegido.get("tipo")
                    or "Activo"
                ).strip()

                if ticker_resuelto:
                    activos_resueltos.append(
                        ticker_resuelto
                    )

                    detalles_resolucion_cartera.append(
                        {
                            "consulta": activo_original,
                            "ticker": ticker_resuelto,
                            "nombre": nombre_resuelto,
                            "tipo": tipo_resuelto,
                        }
                    )

                else:
                    activos_no_resueltos.append(
                        activo_original
                    )

            else:
                activos_no_resueltos.append(
                    activo_original
                )

        duplicados_resueltos = (
            len(activos_resueltos)
            != len(set(activos_resueltos))
        )

        activos_cartera = list(
            dict.fromkeys(activos_resueltos)
        )[:8]

        with p2:
            periodo_portfolio = st.selectbox(
                "Periodo",
                ["1y", "3y", "5y", "10y", "max"],
                index=2,
                key=f"portfolio_period_{ticker}",
            )

        with p3:
            serie_portfolio_ui = st.selectbox(
                "Serie",
                ["Rentabilidad ajustada", "Precio"],
                index=0,
                key=f"portfolio_series_{ticker}",
            )

        with p4:
            capital_portfolio = st.number_input(
                "Capital inicial",
                min_value=100.0,
                max_value=100000000.0,
                value=10000.0,
                step=1000.0,
                key=f"portfolio_capital_{ticker}",
            )

        rf_portfolio = st.number_input(
            "Tasa libre de riesgo anual (%)",
            min_value=0.0,
            max_value=30.0,
            value=0.0,
            step=0.25,
            key=f"portfolio_rf_{ticker}",
        )

        if detalles_resolucion_cartera:
            resoluciones_visibles = []

            for detalle in detalles_resolucion_cartera:
                consulta_detalle = detalle["consulta"]
                ticker_detalle = detalle["ticker"]
                nombre_detalle = detalle["nombre"]
                tipo_detalle = detalle["tipo"]

                resoluciones_visibles.append(
                    f"{consulta_detalle} → "
                    f"{ticker_detalle} · "
                    f"{nombre_detalle} · "
                    f"{tipo_detalle}"
                )

            st.caption(
                "Activos reconocidos · "
                + "  |  ".join(
                    resoluciones_visibles
                )
            )

        if activos_no_resueltos:
            st.warning(
                "FinScope no ha podido identificar: "
                + ", ".join(activos_no_resueltos)
                + ". Revisa el nombre o introduce el ticker."
            )

        if duplicados_resueltos:
            st.warning(
                "Has introducido dos nombres que representan el mismo "
                "activo. FinScope ha eliminado automáticamente el duplicado."
            )

        if len(activos_cartera) < 2:

            st.info(
                "Introduce al menos dos activos para construir la cartera."
            )

        else:

            st.markdown("### Distribución de la cartera")

            columnas_pesos = st.columns(
                min(len(activos_cartera), 4)
            )

            pesos_pct = []

            peso_default = 100.0 / len(
                activos_cartera
            )

            for i, activo in enumerate(
                activos_cartera
            ):

                with columnas_pesos[
                    i % len(columnas_pesos)
                ]:

                    peso = st.number_input(
                        f"{activo} · peso (%)",
                        min_value=0.0,
                        max_value=100.0,
                        value=float(
                            round(peso_default, 2)
                        ),
                        step=1.0,
                        key=(
                            f"portfolio_weight_"
                            f"{ticker}_{activo}"
                        ),
                    )

                    pesos_pct.append(peso)

            suma_pesos = sum(pesos_pct)

            st.caption(
                f"Suma introducida: {suma_pesos:.2f}%. "
                "FinScope normaliza automáticamente los pesos "
                "para que sumen exactamente 100%."
            )

            tipo_serie_portfolio = (
                "ajustada"
                if serie_portfolio_ui
                == "Rentabilidad ajustada"
                else "precio"
            )

            with st.spinner(
                "Calculando cartera y diversificación..."
            ):

                resultado_portfolio = (
                    calcular_cartera_quant(
                        activos_cartera,
                        pesos_pct,
                        periodo=periodo_portfolio,
                        tipo_serie=tipo_serie_portfolio,
                        rf_anual=rf_portfolio / 100.0,
                        capital_inicial=capital_portfolio,
                    )
                )

            error_portfolio = (
                resultado_portfolio.get("error")
            )

            if error_portfolio:

                st.warning(error_portfolio)

            else:

                M = resultado_portfolio[
                    "metricas"
                ]

                st.caption(
                    "Series efectivas · "
                    + " · ".join(
                        f"{a}: "
                        f"{resultado_portfolio['columnas'][a]}"
                        for a in resultado_portfolio[
                            "tickers"
                        ]
                    )
                )

                st.markdown("### Resumen de cartera")

                c1, c2, c3, c4 = st.columns(4)

                with c1:
                    st.metric(
                        "CAGR",
                        (
                            f"{M['cagr']*100:.2f}%"
                            if pd.notna(M.get("cagr"))
                            else "N/D"
                        ),
                    )

                with c2:
                    st.metric(
                        "Volatilidad",
                        (
                            f"{M['volatilidad_anualizada']*100:.2f}%"
                            if pd.notna(
                                M.get(
                                    "volatilidad_anualizada"
                                )
                            )
                            else "N/D"
                        ),
                    )

                with c3:
                    st.metric(
                        "Sharpe",
                        (
                            f"{M['sharpe']:.2f}"
                            if pd.notna(M.get("sharpe"))
                            else "N/D"
                        ),
                    )

                with c4:
                    st.metric(
                        "Máximo drawdown",
                        (
                            f"{M['max_drawdown']*100:.2f}%"
                            if pd.notna(
                                M.get("max_drawdown")
                            )
                            else "N/D"
                        ),
                    )

                st.markdown(
                    "### Evolución histórica · Base 100"
                )

                st.line_chart(
                    resultado_portfolio[
                        "base100"
                    ]
                )

                st.markdown(
                    "### Cartera frente a sus componentes"
                )

                tabla_pm = resultado_portfolio[
                    "comparacion_metricas"
                ].copy()

                for col in [
                    "Peso",
                    "Rentabilidad acumulada",
                    "CAGR",
                    "Volatilidad",
                    "Max Drawdown",
                ]:

                    tabla_pm[col] = tabla_pm[
                        col
                    ].map(
                        lambda x: (
                            "N/D"
                            if pd.isna(x)
                            else f"{x*100:.2f}%"
                        )
                    )

                for col in [
                    "Sharpe",
                    "Sortino",
                ]:

                    tabla_pm[col] = tabla_pm[
                        col
                    ].map(
                        lambda x: (
                            "N/D"
                            if pd.isna(x)
                            else f"{x:.2f}"
                        )
                    )

                st.dataframe(
                    tabla_pm,
                    width="stretch",
                    hide_index=True,
                )

                st.markdown(
                    "### Diversificación"
                )

                d1, d2, d3 = st.columns(3)

                with d1:
                    st.metric(
                        "Volatilidad cartera",
                        f"{resultado_portfolio['volatilidad_matricial']*100:.2f}%",
                    )

                with d2:
                    st.metric(
                        "Volatilidad media ponderada",
                        f"{resultado_portfolio['volatilidad_media_ponderada']*100:.2f}%",
                    )

                with d3:
                    beneficio_div = (
                        resultado_portfolio[
                            "beneficio_diversificacion"
                        ]
                    )

                    st.metric(
                        "Reducción histórica de volatilidad",
                        f"{beneficio_div*100:.2f} p.p.",
                    )

                ratio_div = resultado_portfolio[
                    "ratio_diversificacion"
                ]

                if pd.notna(ratio_div):
                    st.caption(
                        "Ratio de diversificación: "
                        f"{ratio_div:.3f}. "
                        "Compara la volatilidad media ponderada "
                        "de los componentes con la volatilidad "
                        "observada de la cartera."
                    )

                st.markdown(
                    "### Matriz de correlaciones"
                )

                st.dataframe(
                    resultado_portfolio[
                        "correlacion"
                    ].round(3),
                    width="stretch",
                )

                st.markdown(
                    "### Covarianza anualizada"
                )

                st.dataframe(
                    resultado_portfolio[
                        "covarianza_anualizada"
                    ],
                    width="stretch",
                )

                st.markdown(
                    "### Contribución de cada activo"
                )

                contrib = resultado_portfolio[
                    "contribucion"
                ].copy()

                # ------------------------------------------------------------
                # PESO DE CAPITAL VS CONTRIBUCIÓN AL RIESGO
                # ------------------------------------------------------------

                st.markdown(
                    "#### Peso de capital vs contribución al riesgo"
                )

                st.caption(
                    "Compara qué porcentaje del capital representa cada "
                    "posición con qué porcentaje del riesgo total de la "
                    "cartera aporta históricamente. Un activo puede tener "
                    "un peso pequeño en capital y, aun así, concentrar una "
                    "parte importante del riesgo."
                )

                riesgo_visual = (
                    contrib[
                        [
                            "Activo",
                            "Peso",
                            "Contribución riesgo",
                        ]
                    ]
                    .copy()
                    .set_index("Activo")
                    * 100.0
                )

                riesgo_visual = riesgo_visual.rename(
                    columns={
                        "Peso": "Peso en cartera (%)",
                        "Contribución riesgo": "Contribución al riesgo (%)",
                    }
                )

                import altair as alt

                riesgo_visual_largo = (
                    riesgo_visual
                    .reset_index()
                    .melt(
                        id_vars="Activo",
                        var_name="Métrica",
                        value_name="Porcentaje",
                    )
                )

                grafico_riesgo = (
                    alt.Chart(
                        riesgo_visual_largo
                    )
                    .mark_bar()
                    .encode(
                        x=alt.X(
                            "Activo:N",
                            title=None,
                            axis=alt.Axis(
                                labelAngle=0
                            ),
                        ),
                        xOffset="Métrica:N",
                        y=alt.Y(
                            "Porcentaje:Q",
                            title="Porcentaje (%)",
                            scale=alt.Scale(
                                zero=True
                            ),
                        ),
                        color=alt.Color(
                            "Métrica:N",
                            title=None,
                        ),
                        tooltip=[
                            alt.Tooltip(
                                "Activo:N",
                                title="Activo",
                            ),
                            alt.Tooltip(
                                "Métrica:N",
                                title="Métrica",
                            ),
                            alt.Tooltip(
                                "Porcentaje:Q",
                                title="Valor",
                                format=".2f",
                            ),
                        ],
                    )
                    .properties(
                        height=360
                    )
                )

                st.altair_chart(
                    grafico_riesgo,
                    width="stretch",
                )

                # ------------------------------------------------------------
                # LECTURA NUMÉRICA
                # ------------------------------------------------------------

                lectura_riesgo = contrib[
                    [
                        "Activo",
                        "Peso",
                        "Contribución riesgo",
                    ]
                ].copy()

                lectura_riesgo[
                    "Diferencia riesgo - peso"
                ] = (
                    lectura_riesgo[
                        "Contribución riesgo"
                    ]
                    - lectura_riesgo["Peso"]
                )

                lectura_riesgo = lectura_riesgo.rename(
                    columns={
                        "Peso": "Peso en cartera",
                        "Contribución riesgo": "Contribución al riesgo",
                    }
                )

                for col in [
                    "Peso en cartera",
                    "Contribución al riesgo",
                    "Diferencia riesgo - peso",
                ]:
                    lectura_riesgo[col] = lectura_riesgo[
                        col
                    ].map(
                        lambda x: (
                            "N/D"
                            if pd.isna(x)
                            else (
                                f"{x*100:+.2f} p.p."
                                if col == "Diferencia riesgo - peso"
                                else f"{x*100:.2f}%"
                            )
                        )
                    )

                st.dataframe(
                    lectura_riesgo,
                    width="stretch",
                    hide_index=True,
                )

                st.caption(
                    "La diferencia riesgo − peso indica cuánto se desvía "
                    "la aportación histórica al riesgo respecto al peso "
                    "del activo en la cartera. Un valor positivo significa "
                    "que aporta proporcionalmente más riesgo que capital; "
                    "un valor negativo, menos."
                )

                for col in [
                    "Peso",
                    "Contribución riesgo",
                    "Rentabilidad media anualizada",
                    "Contribución rentabilidad",
                ]:

                    contrib[col] = contrib[
                        col
                    ].map(
                        lambda x: (
                            "N/D"
                            if pd.isna(x)
                            else f"{x*100:.2f}%"
                        )
                    )

                contrib[
                    "Contribución volatilidad"
                ] = contrib[
                    "Contribución volatilidad"
                ].map(
                    lambda x: (
                        "N/D"
                        if pd.isna(x)
                        else f"{x*100:.2f} p.p."
                    )
                )

                st.dataframe(
                    contrib,
                    width="stretch",
                    hide_index=True,
                )

                st.markdown(
                    "### Riesgo histórico de la cartera"
                )

                riesgo_portfolio = pd.DataFrame(
                    {
                        "Métrica": [
                            "Rentabilidad acumulada",
                            "CAGR",
                            "Volatilidad anualizada",
                            "Downside deviation",
                            "Sharpe",
                            "Sortino",
                            "Máximo drawdown",
                            "Calmar",
                            "VaR histórico 95%",
                            "CVaR histórico 95%",
                            "VaR histórico 99%",
                            "CVaR histórico 99%",
                            "Sesiones positivas",
                            "Mejor sesión",
                            "Peor sesión",
                            "Skewness",
                            "Kurtosis exceso",
                        ],
                        "Valor": [
                            M.get("rentabilidad_acumulada"),
                            M.get("cagr"),
                            M.get("volatilidad_anualizada"),
                            M.get("downside_deviation"),
                            M.get("sharpe"),
                            M.get("sortino"),
                            M.get("max_drawdown"),
                            M.get("calmar"),
                            M.get("var_95"),
                            M.get("cvar_95"),
                            M.get("var_99"),
                            M.get("cvar_99"),
                            M.get(
                                "porcentaje_sesiones_positivas"
                            ),
                            M.get("mejor_sesion"),
                            M.get("peor_sesion"),
                            M.get("skewness"),
                            M.get("kurtosis_exceso"),
                        ],
                    }
                )

                pct_portfolio = {
                    "Rentabilidad acumulada",
                    "CAGR",
                    "Volatilidad anualizada",
                    "Downside deviation",
                    "Máximo drawdown",
                    "VaR histórico 95%",
                    "CVaR histórico 95%",
                    "VaR histórico 99%",
                    "CVaR histórico 99%",
                    "Sesiones positivas",
                    "Mejor sesión",
                    "Peor sesión",
                }

                riesgo_portfolio[
                    "Valor"
                ] = [
                    (
                        "N/D"
                        if pd.isna(v)
                        else (
                            f"{v*100:.2f}%"
                            if m in pct_portfolio
                            else f"{v:.3f}"
                        )
                    )
                    for m, v in zip(
                        riesgo_portfolio["Métrica"],
                        riesgo_portfolio["Valor"],
                    )
                ]

                st.dataframe(
                    riesgo_portfolio,
                    width="stretch",
                    hide_index=True,
                )

                st.markdown(
                    "### Evolución del capital"
                )

                capital_series = resultado_portfolio[
                    "capital"
                ]

                st.line_chart(
                    capital_series.rename(
                        "Cartera"
                    )
                )

                capital_final = float(
                    capital_series.iloc[-1]
                )

                k1, k2 = st.columns(2)

                with k1:
                    st.metric(
                        "Capital inicial",
                        f"{capital_portfolio:,.2f}",
                    )

                with k2:
                    st.metric(
                        "Capital histórico final",
                        f"{capital_final:,.2f}",
                        f"{capital_final-capital_portfolio:+,.2f}",
                    )

                st.markdown(
                    "### Drawdown de la cartera"
                )

                st.line_chart(
                    M[
                        "serie_drawdown"
                    ].rename("Drawdown")
                )

                with st.expander(
                    "Metodología de Carteras"
                ):

                    st.markdown(
                        """
FinScope calcula primero los rendimientos diarios de cada activo y
solo después los alinea por fechas comunes para construir la cartera.

Los pesos introducidos se normalizan para sumar exactamente 100%.

Esta versión utiliza una **cartera teórica de pesos constantes**:
cada rendimiento diario de la cartera es la suma ponderada de los
rendimientos de sus componentes. Matemáticamente equivale a mantener
los pesos objetivo mediante rebalanceo periódico en cada observación.
No representa una estrategia buy-and-hold sin rebalanceo.

La volatilidad de cartera se verifica también mediante la matriz de
covarianzas:

`σp = √(wᵀΣw)`

La contribución al riesgo muestra qué proporción de la varianza total
procede de cada posición. La suma de las contribuciones relativas al
riesgo es 100%, salvo diferencias mínimas de redondeo.

La reducción histórica de volatilidad compara la media ponderada de
las volatilidades individuales con la volatilidad conjunta de la
cartera. Es una medida descriptiva del efecto de las correlaciones,
no una garantía de diversificación futura.

Las carteras formadas exclusivamente por criptomonedas utilizan 365
periodos por año. Las carteras tradicionales o mixtas utilizan 252
sesiones para las métricas conjuntas calculadas sobre fechas comunes.

Rentabilidad ajustada utiliza `Adj Close` cuando Yahoo Finance lo
proporciona. Precio utiliza `Close`.

No se incluyen comisiones, impuestos, spreads, deslizamiento ni otros
costes de transacción. Los resultados son históricos y no constituyen
una predicción ni una recomendación de inversión.
                        """
                    )

                st.caption(
                    "Fuente de mercado: Yahoo Finance vía yfinance · "
                    f"Observaciones comunes: "
                    f"{resultado_portfolio['observaciones_comunes']} · "
                    f"Anualización: "
                    f"{resultado_portfolio['periodos_anuales']} periodos/año."
                )

    # ============================================================
    # FIN · CARTERAS Y DIVERSIFICACIÓN
    # ============================================================

    with st.expander("Diccionario de métricas"):

        st.caption(
            "Información sobre la definición y las "
            "limitaciones de los indicadores utilizados "
            "por FinScope."
        )

        metricas_mostradas = [
            "precio",
            "capitalizacion",
            "minimo_52_semanas",
            "maximo_52_semanas",
            "ingresos",
            "beneficio_neto",
            "roe",
            "roa",
            "per",
            "price_to_book",
            "crecimiento_ingresos",
            "crecimiento_beneficios",
        ]

        if not datos.get("es_financiera"):
            metricas_mostradas.extend([
                "ebitda",
                "free_cash_flow",
                "margen_bruto",
                "margen_operativo",
                "margen_neto",
                "ev_ebitda",
            ])

        for clave in metricas_mostradas:

            info = METRICAS_INFO.get(clave)

            if not info:
                continue

            st.markdown(
                f"#### {info['nombre']}"
            )

            st.write(
                info["descripcion"]
            )

            c1, c2 = st.columns(2)

            with c1:
                st.caption("TIPO DE DATO")
                st.code(
                    info["tipo"],
                    language=None
                )

            with c2:
                st.caption("CAMPO DE LA FUENTE")
                st.code(
                    info["campo_fuente"],
                    language=None
                )

            st.caption("LIMITACIÓN")

            st.write(
                info["limitacion"]
            )

            st.divider()


    # TRANSPARENCIA DE DATOS

    st.markdown("")

    with st.expander("Fuente y transparencia de los datos"):

        st.markdown("#### Fuente actual")

        st.write(
            "FinScope obtiene actualmente los datos de mercado "
            "y fundamentales mediante la librería `yfinance`, "
            "que proporciona acceso a información distribuida "
            "por Yahoo Finance."
        )

        st.info(
            "Yahoo Finance / yfinance no debe considerarse una "
            "fuente regulatoria primaria. Los datos pueden contener "
            "retrasos, campos ausentes, diferencias de definición "
            "o revisiones posteriores."
        )

        st.markdown("#### Cómo interpreta FinScope los datos")

        st.write(
            "FinScope Insight utiliza únicamente los valores "
            "disponibles en los datos recibidos. Las explicaciones "
            "pretenden aportar contexto financiero y no determinar "
            "automáticamente si una empresa es buena, mala, barata "
            "o cara."
        )

        st.markdown("#### Principales campos utilizados")

        st.markdown(
            """
- **PER:** `trailingPE`
- **Margen operativo:** `operatingMargins`
- **ROE:** `returnOnEquity`
- **Crecimiento de ingresos:** `revenueGrowth`
- **Free Cash Flow:** `freeCashflow`
            """
        )

        st.markdown("#### Datos no disponibles")

        st.write(
            "Cuando la fuente no proporciona una métrica, "
            "FinScope debe tratarla como no disponible. "
            "No se sustituye automáticamente por una estimación."
        )

        st.markdown("#### Importante")

        st.warning(
            "FinScope es una herramienta de análisis y aprendizaje. "
            "Las cifras relevantes deberían contrastarse con los "
            "estados financieros, informes regulatorios y documentación "
            "oficial de la empresa antes de tomar decisiones financieras."
        )




# ============================================================
# FINSCOPE · AJUSTE FINAL POSICIÓN BUSCADOR
# ============================================================

st.markdown(
    """
    <style>

    /*
    El contenedor principal de Streamlit añade espacio vertical
    entre bloques. Reducimos específicamente el espacio anterior
    al buscador principal.
    */

    .main-search-label {
        margin-top: -135px !important;
    }

    /*
    Compensación para pantallas más pequeñas.
    */

    @media (max-width: 1000px) {
        .main-search-label {
            margin-top: -90px !important;
        }
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# FINSCOPE · SUBIR INPUT DEL BUSCADOR PRINCIPAL
# ============================================================

st.markdown(
    """
    <style>

    /*
    El título ya está colocado correctamente.
    Ahora desplazamos también el widget real del buscador.
    */

    div[data-testid="stTextInput"]:has(
        input[placeholder^="Buscar activo ·"]
    ) {
        transform: translateY(-135px) !important;
        margin-bottom: -135px !important;
    }


    @media (max-width: 1000px) {

        div[data-testid="stTextInput"]:has(
            input[placeholder^="Buscar activo ·"]
        ) {
            transform: translateY(-90px) !important;
            margin-bottom: -90px !important;
        }

    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# FINSCOPE · CONTENEDORES DE GRÁFICOS
# ============================================================

st.markdown(
    """
    <style>

    /*
    Integramos completamente los gráficos dentro de su tarjeta.
    Evita que el fondo rectangular interno sobresalga.
    */

    div[data-testid="stVegaLiteChart"] {
        overflow: hidden !important;
        border-radius: 16px !important;
        background: transparent !important;
    }

    div[data-testid="stVegaLiteChart"] > div {
        overflow: hidden !important;
        border-radius: inherit !important;
        background: transparent !important;
    }

    div[data-testid="stVegaLiteChart"] canvas,
    div[data-testid="stVegaLiteChart"] svg {
        max-width: 100% !important;
        border-radius: 16px !important;
    }

    /*
    Contenedor exterior de Streamlit.
    El gráfico nunca puede dibujarse fuera de él.
    */

    div[data-testid="stVegaLiteChart"] {
        max-width: 100% !important;
        box-sizing: border-box !important;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# FINSCOPE · FIX TÍTULO EMPRESA EN MODO CLARO
# ============================================================

if tema == "Claro":
    st.markdown(
        """
        <style>

        /* Nombre de empresa y ticker */
        .company-hero h2 {
            color: #172337 !important;
            -webkit-text-fill-color: #172337 !important;
        }

        /* Sector · industria · país */
        .company-hero p {
            color: #718197 !important;
            -webkit-text-fill-color: #718197 !important;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# FINSCOPE · CONTRASTE COTIZACIÓN EN MODO CLARO
# ============================================================

if tema == "Claro":
    st.markdown(
        """
        <style>

        /* ==================================================
           SELECTOR DE PERIODOS
           ================================================== */

        [data-testid="stSegmentedControl"] button {
            background: #ffffff !important;
            color: #4d6078 !important;

            border-color:
                #d3dde8 !important;
        }

        [data-testid="stSegmentedControl"] button p,
        [data-testid="stSegmentedControl"] button span {
            color: #4d6078 !important;
            -webkit-text-fill-color: #4d6078 !important;
        }

        [data-testid="stSegmentedControl"]
        button[aria-pressed="true"] {
            background: #e9f2fd !important;
            color: #1f65b5 !important;

            border-color:
                #76aee9 !important;
        }

        [data-testid="stSegmentedControl"]
        button[aria-pressed="true"] p,
        [data-testid="stSegmentedControl"]
        button[aria-pressed="true"] span {
            color: #1f65b5 !important;
            -webkit-text-fill-color: #1f65b5 !important;
        }


        /* ==================================================
           TARJETA RESUMEN DE COTIZACIÓN
           ================================================== */

        .quote-summary {
            background: #ffffff !important;

            border-color:
                #d9e3ed !important;
        }

        .quote-summary * {
            -webkit-text-fill-color:
                initial;
        }

        /*
        Precio principal
        */

        .quote-summary h1,
        .quote-summary h2,
        .quote-summary h3,
        .quote-summary strong {
            color: #172337 !important;
            -webkit-text-fill-color: #172337 !important;
        }


        /*
        Nombre de empresa / ticker
        */

        .quote-summary small {
            color: #60748d !important;
            -webkit-text-fill-color: #60748d !important;
        }


        /*
        Variación positiva conserva verde.
        */

        .quote-summary .positive,
        .quote-summary [class*="positive"] {
            color: #15966f !important;
            -webkit-text-fill-color: #15966f !important;
        }


        /*
        Variación negativa conserva rojo.
        */

        .quote-summary .negative,
        .quote-summary [class*="negative"] {
            color: #c94b55 !important;
            -webkit-text-fill-color: #c94b55 !important;
        }


        /* ==================================================
           PANEL DERECHO · PERIODO SELECCIONADO
           ================================================== */

        .quote-summary div,
        .quote-summary span,
        .quote-summary p {
            color: #52667f;
        }

        /*
        Los valores destacados dentro de la tarjeta,
        incluido "1y", deben ser oscuros.
        */

        .quote-summary div strong,
        .quote-summary span strong,
        .quote-summary p strong {
            color: #172337 !important;
            -webkit-text-fill-color: #172337 !important;
        }


        /* ==================================================
           TEXTO SECUNDARIO
           ================================================== */

        .quote-summary {
            color: #52667f !important;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# FINSCOPE · SELECTOR DE PERIODO · MODO CLARO
# ============================================================

if tema == "Claro":
    st.markdown(
        """
        <style>

        /*
        Contenedor completo del selector de periodo.
        */

        div[data-testid="stSegmentedControl"] {
            background: transparent !important;
        }

        div[data-testid="stSegmentedControl"]
        div[role="radiogroup"] {
            background: #ffffff !important;
            border: 1px solid #d5deea !important;
            border-radius: 11px !important;
            overflow: hidden !important;
        }


        /*
        Cada periodo: 1h, 1d, 5d...
        */

        div[data-testid="stSegmentedControl"]
        button {
            background: #ffffff !important;
            color: #40536b !important;
            border-color: #d5deea !important;
        }

        div[data-testid="stSegmentedControl"]
        button * {
            color: #40536b !important;
            -webkit-text-fill-color: #40536b !important;
            opacity: 1 !important;
        }


        /*
        Streamlit puede renderizar las opciones como labels
        en lugar de buttons según la versión.
        */

        div[data-testid="stSegmentedControl"]
        label {
            background: #ffffff !important;
            color: #40536b !important;
            border-color: #d5deea !important;
        }

        div[data-testid="stSegmentedControl"]
        label * {
            color: #40536b !important;
            -webkit-text-fill-color: #40536b !important;
            opacity: 1 !important;
        }


        /*
        Opción seleccionada.
        Cubrimos tanto aria-pressed como radio checked.
        */

        div[data-testid="stSegmentedControl"]
        button[aria-pressed="true"] {
            background: #e7f1fd !important;
            color: #1768bd !important;
            box-shadow:
                inset 0 0 0 1px #72abe7 !important;
        }

        div[data-testid="stSegmentedControl"]
        button[aria-pressed="true"] * {
            color: #1768bd !important;
            -webkit-text-fill-color: #1768bd !important;
        }

        div[data-testid="stSegmentedControl"]
        label:has(input:checked) {
            background: #e7f1fd !important;
            color: #1768bd !important;
            box-shadow:
                inset 0 0 0 1px #72abe7 !important;
        }

        div[data-testid="stSegmentedControl"]
        label:has(input:checked) * {
            color: #1768bd !important;
            -webkit-text-fill-color: #1768bd !important;
            opacity: 1 !important;
        }


        /*
        Hover
        */

        div[data-testid="stSegmentedControl"]
        button:hover,
        div[data-testid="stSegmentedControl"]
        label:hover {
            background: #f1f6fc !important;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# FINSCOPE · SELECTOR DE COTIZACIÓN DEFINITIVO
# ============================================================

st.markdown(
    """
    <style>

    /*
    Selector horizontal de periodo.
    */

    div[data-testid="stRadio"]:has(
        input[value="1h"]
    ) > div[role="radiogroup"] {

        display: inline-flex !important;
        flex-direction: row !important;
        gap: 0 !important;

        padding: 3px !important;

        border-radius: 12px !important;

        width: auto !important;
    }


    div[data-testid="stRadio"]:has(
        input[value="1h"]
    ) > div[role="radiogroup"] label {

        min-width: 58px !important;

        margin: 0 !important;
        padding: 9px 15px !important;

        display: flex !important;
        justify-content: center !important;
        align-items: center !important;

        border-radius: 8px !important;

        cursor: pointer !important;

        transition:
            background .15s ease,
            color .15s ease !important;
    }


    /*
    Ocultar círculo nativo del radio.
    */

    div[data-testid="stRadio"]:has(
        input[value="1h"]
    )
    div[data-testid="stRadio"] {
        gap: 0 !important;
    }

    div[data-testid="stRadio"]:has(
        input[value="1h"]
    )
    label > div:first-child {
        display: none !important;
    }


    /*
    Texto.
    */

    div[data-testid="stRadio"]:has(
        input[value="1h"]
    )
    label p {

        margin: 0 !important;

        font-size: .82rem !important;
        font-weight: 650 !important;

        opacity: 1 !important;
    }


    /*
    MODO OSCURO
    */

    div[data-testid="stRadio"]:has(
        input[value="1h"]
    ) > div[role="radiogroup"] {

        background: #0b1119 !important;
        border: 1px solid #263243 !important;
    }

    div[data-testid="stRadio"]:has(
        input[value="1h"]
    )
    label p {

        color: #9cabc0 !important;
        -webkit-text-fill-color: #9cabc0 !important;
    }

    div[data-testid="stRadio"]:has(
        input[value="1h"]
    )
    label:has(input:checked) {

        background: #182a40 !important;
        box-shadow:
            inset 0 0 0 1px #4e8fd8 !important;
    }

    div[data-testid="stRadio"]:has(
        input[value="1h"]
    )
    label:has(input:checked) p {

        color: #80b9fa !important;
        -webkit-text-fill-color: #80b9fa !important;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


if tema == "Claro":

    st.markdown(
        """
        <style>

        /*
        MODO CLARO
        */

        div[data-testid="stRadio"]:has(
            input[value="1h"]
        ) > div[role="radiogroup"] {

            background: #ffffff !important;

            border:
                1px solid #cbd7e5 !important;

            box-shadow:
                0 4px 14px
                rgba(31, 49, 72, .05) !important;
        }


        div[data-testid="stRadio"]:has(
            input[value="1h"]
        )
        label {

            background:
                transparent !important;
        }


        div[data-testid="stRadio"]:has(
            input[value="1h"]
        )
        label p {

            color: #53677f !important;

            -webkit-text-fill-color:
                #53677f !important;

            opacity: 1 !important;
        }


        div[data-testid="stRadio"]:has(
            input[value="1h"]
        )
        label:hover {

            background:
                #f1f6fc !important;
        }


        div[data-testid="stRadio"]:has(
            input[value="1h"]
        )
        label:has(input:checked) {

            background:
                #e7f1fd !important;

            box-shadow:
                inset 0 0 0 1px
                #5d9fe7 !important;
        }


        div[data-testid="stRadio"]:has(
            input[value="1h"]
        )
        label:has(input:checked) p {

            color:
                #1768bd !important;

            -webkit-text-fill-color:
                #1768bd !important;

            font-weight:
                750 !important;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# FINSCOPE · CORRECCIÓN GLOBAL DE CONTRASTE · MODO CLARO
# ============================================================

st.markdown(
    """
    <style>

    /* INPUTS */
    div[data-baseweb="input"],
    div[data-baseweb="base-input"],
    div[data-testid="stNumberInput"] input,
    div[data-testid="stTextInput"] input {
        background: #ffffff !important;
        color: #172033 !important;
    }

    div[data-testid="stNumberInput"] > div,
    div[data-testid="stTextInput"] > div {
        background: #ffffff !important;
    }

    /* BOTONES +/- DE NUMBER INPUT */
    div[data-testid="stNumberInput"] button {
        background: #f3f7fc !important;
        color: #172033 !important;
        border-color: #d6e2f0 !important;
    }

    div[data-testid="stNumberInput"] button:hover {
        background: #e7f1fd !important;
        color: #1565c0 !important;
    }

    div[data-testid="stNumberInput"] svg {
        fill: currentColor !important;
        color: inherit !important;
    }

    /* EXPANDERS */
    div[data-testid="stExpander"] {
        background: #ffffff !important;
        border-color: #dce5ef !important;
    }

    div[data-testid="stExpander"] details {
        background: #ffffff !important;
    }

    div[data-testid="stExpander"] summary {
        background: #f7f9fc !important;
        color: #172033 !important;
    }

    div[data-testid="stExpander"] summary:hover {
        background: #eaf3ff !important;
        color: #0d47a1 !important;
    }

    div[data-testid="stExpander"] summary * {
        color: inherit !important;
    }

    div[data-testid="stExpander"] summary svg {
        color: #31506f !important;
        fill: currentColor !important;
    }

    /* LABELS */
    div[data-testid="stNumberInput"] label,
    div[data-testid="stTextInput"] label {
        color: #172033 !important;
    }

    /* BOTONES SECUNDARIOS */
    div[data-testid="stButton"] button[kind="secondary"] {
        background: #ffffff !important;
        color: #172033 !important;
        border-color: #cbdbea !important;
    }

    div[data-testid="stButton"] button[kind="secondary"]:hover {
        background: #eaf3ff !important;
        color: #0d47a1 !important;
        border-color: #90b9e8 !important;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# FINSCOPE · SEGMENTED CONTROL · MODO CLARO
# ============================================================

st.markdown(
    """
    <style>

    /* Contenedor */
    div[data-testid="stSegmentedControl"] [data-baseweb="button-group"] {
        background: #ffffff !important;
    }

    /* TODOS los botones */
    div[data-testid="stSegmentedControl"] button {
        background: #ffffff !important;
        color: #172033 !important;
        border-color: #d5e0ec !important;
    }

    div[data-testid="stSegmentedControl"] button * {
        color: #172033 !important;
    }

    /* Hover */
    div[data-testid="stSegmentedControl"] button:hover {
        background: #eaf3ff !important;
        color: #0d47a1 !important;
    }

    div[data-testid="stSegmentedControl"] button:hover * {
        color: #0d47a1 !important;
    }

    /* Opción seleccionada */
    div[data-testid="stSegmentedControl"] button[aria-pressed="true"] {
        background: #eaf3ff !important;
        color: #0d47a1 !important;
        border-color: #4d94e8 !important;
    }

    div[data-testid="stSegmentedControl"] button[aria-pressed="true"] * {
        color: #0d47a1 !important;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# FINSCOPE · LIGHT SYSTEM FINAL V1
# Capa visual definitiva
# ============================================================

st.markdown(
    """
    <style>

    /* ======================================================
       VARIABLES
       ====================================================== */

    :root {
        --fs-bg-final: #f5f7fa;
        --fs-surface-final: #ffffff;
        --fs-surface-soft-final: #f8fafc;
        --fs-surface-blue-final: #eef5fd;

        --fs-text-final: #172033;
        --fs-text-secondary-final: #46566c;
        --fs-muted-final: #718197;

        --fs-border-final: #dce5ef;
        --fs-border-strong-final: #cbd8e6;

        --fs-blue-final: #347fda;
        --fs-blue-dark-final: #0d47a1;
        --fs-blue-soft-final: #eaf3ff;

        --fs-radius-final: 14px;
    }


    /* ======================================================
       APP
       ====================================================== */

    html,
    body,
    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stAppViewBlockContainer"] {
        background: var(--fs-bg-final) !important;
        color: var(--fs-text-final) !important;
    }

    .stApp {
        background-image: none !important;
    }

    .block-container {
        max-width: 1320px !important;
    }


    /* ======================================================
       TEXTO GENERAL
       ====================================================== */

    .stApp h1,
    .stApp h2,
    .stApp h3,
    .stApp h4,
    .stApp h5,
    .stApp h6 {
        color: var(--fs-text-final) !important;
    }

    .stApp p,
    .stApp label {
        color: var(--fs-text-secondary-final);
    }


    /* ======================================================
       SIDEBAR
       ====================================================== */

    section[data-testid="stSidebar"] {
        background: #ffffff !important;
        background-image: none !important;

        border-right:
            1px solid var(--fs-border-final) !important;
    }

    section[data-testid="stSidebar"] > div {
        background: #ffffff !important;
    }

    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span {
        color: var(--fs-text-secondary-final);
    }

    section[data-testid="stSidebar"] .sidebar-brand-name {
        color: var(--fs-text-final) !important;
    }

    section[data-testid="stSidebar"] .sidebar-brand-name span {
        color: var(--fs-blue-final) !important;
    }

    section[data-testid="stSidebar"] .sidebar-brand-sub,
    section[data-testid="stSidebar"] .sidebar-label {
        color: var(--fs-muted-final) !important;
    }

    section[data-testid="stSidebar"] hr {
        border-color: var(--fs-border-final) !important;
    }


    /* ======================================================
       NAVEGACIÓN SIDEBAR
       ====================================================== */

    section[data-testid="stSidebar"]
    div[role="radiogroup"] label {
        background: transparent !important;
        border-radius: 10px !important;
    }

    section[data-testid="stSidebar"]
    div[role="radiogroup"] label:hover {
        background: #f3f7fc !important;
    }

    section[data-testid="stSidebar"]
    div[role="radiogroup"]
    label:has(input:checked) {
        background: var(--fs-blue-soft-final) !important;
    }

    section[data-testid="stSidebar"]
    div[role="radiogroup"]
    label:has(input:checked) p {
        color: var(--fs-blue-dark-final) !important;
        font-weight: 620 !important;
    }


    /* ======================================================
       INPUTS DE TEXTO
       ====================================================== */

    div[data-testid="stTextInput"] input {
        background: #ffffff !important;
        color: var(--fs-text-final) !important;

        border:
            1px solid var(--fs-border-strong-final) !important;

        box-shadow: none !important;
    }

    div[data-testid="stTextInput"] input:hover {
        border-color: #9db8d5 !important;
    }

    div[data-testid="stTextInput"] input:focus {
        background: #ffffff !important;
        border-color: var(--fs-blue-final) !important;

        box-shadow:
            0 0 0 3px
            rgba(52,127,218,.10) !important;
    }

    div[data-testid="stTextInput"] input::placeholder {
        color: #8a98aa !important;
        opacity: 1 !important;
    }

    div[data-testid="stTextInput"] label,
    div[data-testid="stNumberInput"] label {
        color: var(--fs-text-final) !important;
    }


    /* ======================================================
       NUMBER INPUT
       ====================================================== */

    div[data-testid="stNumberInput"] input {
        background: #ffffff !important;
        color: var(--fs-text-final) !important;
        border-color: var(--fs-border-strong-final) !important;
    }

    div[data-testid="stNumberInput"] button {
        background: #f7f9fc !important;
        color: var(--fs-text-secondary-final) !important;
        border-color: var(--fs-border-strong-final) !important;
    }

    div[data-testid="stNumberInput"] button:hover {
        background: var(--fs-blue-soft-final) !important;
        color: var(--fs-blue-dark-final) !important;
    }

    div[data-testid="stNumberInput"] svg {
        color: inherit !important;
        fill: currentColor !important;
    }


    /* ======================================================
       SELECTBOX / MULTISELECT
       ====================================================== */

    div[data-baseweb="select"] > div {
        background: #ffffff !important;
        color: var(--fs-text-final) !important;
        border-color: var(--fs-border-strong-final) !important;
    }

    div[data-baseweb="select"] span,
    div[data-baseweb="select"] div {
        color: var(--fs-text-final);
    }

    div[data-baseweb="popover"],
    div[data-baseweb="menu"],
    ul[role="listbox"] {
        background: #ffffff !important;
        color: var(--fs-text-final) !important;
    }

    li[role="option"] {
        background: #ffffff !important;
        color: var(--fs-text-final) !important;
    }

    li[role="option"]:hover {
        background: var(--fs-blue-soft-final) !important;
        color: var(--fs-blue-dark-final) !important;
    }


    /* ======================================================
       BOTONES
       ====================================================== */

    .stButton > button {
        background: #ffffff !important;
        color: #26384f !important;

        border:
            1px solid var(--fs-border-strong-final) !important;

        box-shadow: none !important;
    }

    .stButton > button:hover {
        background: var(--fs-blue-soft-final) !important;
        color: var(--fs-blue-dark-final) !important;
        border-color: #90b9e8 !important;
    }

    .stButton > button:focus {
        box-shadow:
            0 0 0 3px
            rgba(52,127,218,.10) !important;
    }

    div[data-testid="stButton"]
    button[kind="primary"] {
        background: var(--fs-blue-final) !important;
        color: #ffffff !important;
        border-color: var(--fs-blue-final) !important;
    }

    div[data-testid="stButton"]
    button[kind="primary"]:hover {
        background: #276fc4 !important;
        color: #ffffff !important;
        border-color: #276fc4 !important;
    }


    /* ======================================================
       MÉTRICAS
       ====================================================== */

    div[data-testid="stMetric"] {
        background: #ffffff !important;
        background-image: none !important;

        border:
            1px solid var(--fs-border-final) !important;

        border-radius:
            var(--fs-radius-final) !important;

        box-shadow:
            0 4px 14px
            rgba(35,55,80,.035) !important;
    }

    div[data-testid="stMetricLabel"],
    div[data-testid="stMetricLabel"] * {
        color: var(--fs-muted-final) !important;
    }

    div[data-testid="stMetricValue"],
    div[data-testid="stMetricValue"] * {
        color: var(--fs-text-final) !important;
    }


    /* ======================================================
       EXPANDERS
       ====================================================== */

    div[data-testid="stExpander"] {
        background: #ffffff !important;
        border-color: var(--fs-border-final) !important;
    }

    div[data-testid="stExpander"] details {
        background: #ffffff !important;
    }

    div[data-testid="stExpander"] summary {
        background: var(--fs-surface-soft-final) !important;
        color: var(--fs-text-final) !important;
    }

    div[data-testid="stExpander"] summary:hover {
        background: var(--fs-blue-soft-final) !important;
        color: var(--fs-blue-dark-final) !important;
    }

    div[data-testid="stExpander"] summary * {
        color: inherit !important;
    }

    div[data-testid="stExpander"] summary svg {
        color: #31506f !important;
        fill: currentColor !important;
    }


    /* ======================================================
       SEGMENTED CONTROL
       ====================================================== */

    div[data-testid="stSegmentedControl"]
    [data-baseweb="button-group"] {
        background: #ffffff !important;
    }

    div[data-testid="stSegmentedControl"] button {
        background: #ffffff !important;
        color: var(--fs-text-secondary-final) !important;
        border-color: var(--fs-border-final) !important;
    }

    div[data-testid="stSegmentedControl"] button * {
        color: inherit !important;
        -webkit-text-fill-color: currentColor !important;
    }

    div[data-testid="stSegmentedControl"] button:hover {
        background: var(--fs-blue-soft-final) !important;
        color: var(--fs-blue-dark-final) !important;
    }

    div[data-testid="stSegmentedControl"]
    button[aria-pressed="true"] {
        background: var(--fs-blue-soft-final) !important;
        color: var(--fs-blue-dark-final) !important;
        border-color: #4d94e8 !important;
    }


    /* ======================================================
       RADIO / CHECKBOX
       ====================================================== */

    div[data-testid="stRadio"] label,
    div[data-testid="stCheckbox"] label {
        color: var(--fs-text-secondary-final) !important;
    }


    /* ======================================================
       TABS
       ====================================================== */

    button[data-baseweb="tab"] {
        color: var(--fs-text-secondary-final) !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: var(--fs-blue-dark-final) !important;
    }


    /* ======================================================
       DATAFRAMES / EDITORES
       ====================================================== */

    [data-testid="stDataFrame"],
    [data-testid="stDataEditor"] {
        background: #ffffff !important;
        border-radius: var(--fs-radius-final) !important;
        overflow: hidden !important;
    }


    /* ======================================================
       VEGA / ALTAIR
       ====================================================== */

    div[data-testid="stVegaLiteChart"],
    div[data-testid="stArrowVegaLiteChart"] {
        background: #ffffff !important;
        background-image: none !important;

        border:
            1px solid var(--fs-border-final) !important;

        border-radius:
            var(--fs-radius-final) !important;

        box-shadow:
            0 6px 18px
            rgba(35,55,80,.035) !important;

        overflow: hidden !important;
    }

    div[data-testid="stVegaLiteChart"] > div,
    div[data-testid="stArrowVegaLiteChart"] > div {
        background: transparent !important;
    }


    /* ======================================================
       PLOTLY
       El lienzo Plotly se controla principalmente desde Python,
       pero eliminamos cualquier fondo oscuro exterior.
       ====================================================== */

    div[data-testid="stPlotlyChart"],
    div[data-testid="stPlotlyChart"] > div {
        background: transparent !important;
    }


    /* ======================================================
       COMPANY HERO
       ====================================================== */

    .company-hero {
        background: #ffffff !important;
        background-image: none !important;

        border:
            1px solid var(--fs-border-final) !important;

        box-shadow:
            0 5px 18px
            rgba(35,55,80,.035) !important;
    }

    .company-hero h2 {
        color: var(--fs-text-final) !important;
        -webkit-text-fill-color:
            var(--fs-text-final) !important;
    }

    .company-hero p {
        color: var(--fs-muted-final) !important;
        -webkit-text-fill-color:
            var(--fs-muted-final) !important;
    }


    /* ======================================================
       TARJETAS PERSONALIZADAS
       ====================================================== */

    .metric-card,
    .module-card,
    .result-card,
    .asset-card,
    .portfolio-card,
    .quant-card,
    .analysis-card {
        background: #ffffff !important;
        background-image: none !important;
        border-color: var(--fs-border-final) !important;
    }


    /* ======================================================
       ALERTAS
       ====================================================== */

    [data-testid="stAlert"] {
        border-color: var(--fs-border-final) !important;
    }


    /* ======================================================
       SEPARADORES
       ====================================================== */

    hr {
        border-top-color: var(--fs-border-final) !important;
    }


    /* ======================================================
       TOOLTIPS / POPOVERS
       ====================================================== */

    div[data-baseweb="tooltip"],
    div[role="tooltip"] {
        background: #ffffff !important;
        color: var(--fs-text-final) !important;
        border: 1px solid var(--fs-border-final) !important;
    }


    /* ======================================================
       SCROLLBARS
       ====================================================== */

    ::-webkit-scrollbar-track {
        background: #f5f7fa;
    }

    ::-webkit-scrollbar-thumb {
        background: #cbd7e5;
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #aebfd2;
    }

    </style>
    """,
    unsafe_allow_html=True,
)



# ============================================================
# FINSCOPE · PORTADA PROFESIONAL V2 · ESTILO DEFINITIVO
# ============================================================

st.markdown(
    """
    <style>

    /* ======================================================
       PORTADA
       ====================================================== */

    .fs-home-v2 {
        position: relative;

        width: 100%;
        max-width: 1180px;

        margin:
            1.8rem auto
            1.8rem auto;

        padding:
            4.2rem 4.4rem
            3rem;

        box-sizing: border-box;

        background:
            #ffffff;

        border:
            1px solid #e1e8f0;

        border-radius:
            24px;

        box-shadow:
            0 16px 50px
            rgba(30, 54, 82, .055);

        overflow:
            hidden;
    }


    /* Pequeño detalle azul superior */

    .fs-home-v2::before {
        content: "";

        position: absolute;

        top: 0;
        left: 4.4rem;

        width: 56px;
        height: 3px;

        background:
            #347fda;

        border-radius:
            0 0 4px 4px;
    }


    /* ======================================================
       ETIQUETA SUPERIOR
       ====================================================== */

    .fs-eyebrow {
        display: flex;
        align-items: center;

        gap: 10px;

        margin-bottom:
            2.6rem;

        color:
            #718197;

        font-size:
            .68rem;

        font-weight:
            720;

        letter-spacing:
            .14em;
    }

    .fs-eyebrow > span {
        display: block;

        width: 7px;
        height: 7px;

        flex: 0 0 auto;

        border-radius:
            50%;

        background:
            #347fda;

        box-shadow:
            0 0 0 4px
            rgba(52,127,218,.09);
    }


    /* ======================================================
       TITULAR
       ====================================================== */

    .fs-hero-copy {
        max-width:
            900px;
    }

    .fs-hero-copy h1 {
        margin:
            0 !important;

        padding:
            0 !important;

        color:
            #172337 !important;

        font-size:
            clamp(
                3.3rem,
                5.6vw,
                5.35rem
            ) !important;

        line-height:
            .99 !important;

        font-weight:
            720 !important;

        letter-spacing:
            -.055em !important;
    }

    .fs-hero-copy h1 em {
        color:
            #347fda !important;

        font-style:
            normal !important;

        font-weight:
            inherit !important;

        -webkit-text-fill-color:
            #347fda !important;
    }


    /* ======================================================
       DESCRIPCIÓN
       ====================================================== */

    .fs-hero-copy > p {
        max-width:
            650px;

        margin:
            1.65rem 0
            0 0 !important;

        color:
            #65758a !important;

        font-size:
            1.06rem !important;

        line-height:
            1.65 !important;

        letter-spacing:
            -.012em;
    }


    /* ======================================================
       DIVISOR
       ====================================================== */

    .fs-hero-divider {
        width:
            100%;

        height:
            1px;

        margin:
            3.4rem 0
            1.8rem;

        background:
            #e8edf3;
    }


    /* ======================================================
       TRES ÁREAS
       ====================================================== */

    .fs-hero-grid {
        display:
            grid;

        grid-template-columns:
            repeat(3, 1fr);

        gap:
            0;
    }

    .fs-hero-item {
        display:
            flex;

        align-items:
            flex-start;

        gap:
            15px;

        padding:
            .35rem 2.2rem
            .35rem 0;
    }

    .fs-hero-item:not(:first-child) {
        padding-left:
            2.2rem;

        border-left:
            1px solid #e8edf3;
    }

    .fs-hero-item small {
        margin-top:
            3px;

        color:
            #a0adbd;

        font-size:
            .65rem;

        font-weight:
            720;

        letter-spacing:
            .08em;
    }

    .fs-hero-item div {
        display:
            flex;

        flex-direction:
            column;

        gap:
            5px;
    }

    .fs-hero-item strong {
        color:
            #223249;

        font-size:
            .9rem;

        font-weight:
            720;
    }

    .fs-hero-item span {
        max-width:
            210px;

        color:
            #7b899a;

        font-size:
            .76rem;

        line-height:
            1.5;
    }


    /* ======================================================
       ELIMINAR VISUALMENTE ELEMENTOS DE LA PORTADA V1
       ====================================================== */

    .finscope-home,
    .home-layout,
    .home-copy,
    .home-visual,
    .home-start,
    .home-pills,
    .home-features {
        display:
            none !important;
    }


    /* ======================================================
       CORREGIR DESPLAZAMIENTOS ANTIGUOS DEL BUSCADOR
       ====================================================== */

    .main-search-label {
        margin-top:
            0 !important;

        padding-top:
            0 !important;

        transform:
            none !important;
    }

    div[data-testid="stTextInput"]:has(
        input[placeholder^="Buscar activo ·"]
    ) {
        transform:
            none !important;

        margin:
            0 0
            1.4rem 0 !important;

        width:
            100% !important;

        max-width:
            none !important;
    }


    /* ======================================================
       BUSCADOR
       ====================================================== */

    div[data-testid="stTextInput"]:has(
        input[placeholder^="Buscar activo ·"]
    ) input {

        min-height:
            62px !important;

        padding:
            0 20px !important;

        background:
            #ffffff !important;

        color:
            #172337 !important;

        border:
            1px solid #ccd9e6 !important;

        border-radius:
            14px !important;

        font-size:
            1rem !important;

        box-shadow:
            0 7px 22px
            rgba(32,58,87,.045) !important;
    }

    div[data-testid="stTextInput"]:has(
        input[placeholder^="Buscar activo ·"]
    ) input:hover {
        border-color:
            #a7bdd5 !important;
    }

    div[data-testid="stTextInput"]:has(
        input[placeholder^="Buscar activo ·"]
    ) input:focus {
        border-color:
            #347fda !important;

        box-shadow:
            0 0 0 4px
            rgba(52,127,218,.08) !important;
    }

    div[data-testid="stTextInput"]:has(
        input[placeholder^="Buscar activo ·"]
    ) input::placeholder {
        color:
            #8b99aa !important;

        opacity:
            1 !important;
    }


    /* ======================================================
       RESPONSIVE
       ====================================================== */

    @media (max-width: 900px) {

        .fs-home-v2 {
            padding:
                3.3rem 2.7rem
                2.6rem;
        }

        .fs-home-v2::before {
            left:
                2.7rem;
        }

        .fs-hero-grid {
            grid-template-columns:
                1fr;
        }

        .fs-hero-item,
        .fs-hero-item:not(:first-child) {
            padding:
                1rem 0;

            border-left:
                0;
        }

        .fs-hero-item:not(:last-child) {
            border-bottom:
                1px solid #edf1f5;
        }

        .main-search-label {
            margin-top:
                0 !important;
        }

        div[data-testid="stTextInput"]:has(
            input[placeholder^="Buscar activo ·"]
        ) {
            transform:
                none !important;

            margin-bottom:
                1.3rem !important;
        }
    }


    @media (max-width: 650px) {

        .fs-home-v2 {
            margin-top:
                .7rem;

            padding:
                2.7rem 1.6rem
                2.1rem;

            border-radius:
                18px;
        }

        .fs-home-v2::before {
            left:
                1.6rem;
        }

        .fs-eyebrow {
            margin-bottom:
                2rem;

            font-size:
                .6rem;
        }

        .fs-hero-copy h1 {
            font-size:
                2.75rem !important;
        }

        .fs-hero-copy > p {
            font-size:
                .94rem !important;
        }

        .fs-hero-divider {
            margin-top:
                2.5rem;
        }

    }

    </style>
    """,
    unsafe_allow_html=True,
)



# ============================================================
# FINSCOPE · PORTADA V2.1 · CHART + SEARCH
# ============================================================

st.markdown(
"""
<style>

/* ==========================================================
   HERO EN DOS COLUMNAS
   ========================================================== */

.fs-hero-top {
    display: grid;

    grid-template-columns:
        minmax(0, 1.18fr)
        minmax(320px, .82fr);

    align-items:
        center;

    gap:
        3.4rem;
}

.fs-hero-copy h1 {
    font-size:
        clamp(
            3rem,
            4.7vw,
            4.65rem
        ) !important;

    line-height:
        1.01 !important;
}


/* ==========================================================
   VISUAL FINANCIERO
   ========================================================== */

.fs-market-visual {
    min-width: 0;

    padding:
        18px 18px
        15px;

    background:
        linear-gradient(
            145deg,
            #fbfdff 0%,
            #f5f9fe 100%
        );

    border:
        1px solid #dce8f4;

    border-radius:
        18px;

    box-shadow:
        0 16px 38px
        rgba(50, 100, 155, .06);
}

.fs-market-head {
    display: flex;
    align-items: center;
    justify-content: space-between;

    padding:
        0 2px
        13px;

    color:
        #8296ac;

    font-size:
        .58rem;

    font-weight:
        750;

    letter-spacing:
        .13em;
}

.fs-market-live {
    display: flex;
    align-items: center;
    gap: 6px;
}

.fs-market-live i {
    width: 6px;
    height: 6px;

    background:
        #67aaf0;

    border-radius:
        50%;

    box-shadow:
        0 0 0 4px
        rgba(103,170,240,.11);
}


/* Gráfico */

.fs-market-chart {
    position: relative;

    height:
        230px;

    overflow:
        hidden;

    background:
        rgba(255,255,255,.58);

    border:
        1px solid #e5edf6;

    border-radius:
        13px;
}

.fs-chart-grid {
    position: absolute;
    inset: 0;

    background-image:
        linear-gradient(
            to right,
            rgba(105,145,185,.075) 1px,
            transparent 1px
        ),
        linear-gradient(
            to bottom,
            rgba(105,145,185,.075) 1px,
            transparent 1px
        );

    background-size:
        20% 25%;
}

.fs-chart-svg {
    position: absolute;

    left: 6%;
    right: 6%;
    bottom: 10%;

    width: 88%;
    height: 76%;

    overflow: visible;
}

.fs-chart-line {
    fill: none;

    stroke:
        #6baaf0;

    stroke-width:
        4;

    stroke-linecap:
        round;

    stroke-linejoin:
        round;

    vector-effect:
        non-scaling-stroke;
}

.fs-chart-area {
    fill:
        url(#fsAreaGradient);
}

.fs-chart-point {
    fill:
        #ffffff;

    stroke:
        #7bb5f2;

    stroke-width:
        3;

    vector-effect:
        non-scaling-stroke;
}

.fs-chart-point-main {
    fill:
        #5d9fe8;

    stroke:
        #ffffff;

    filter:
        drop-shadow(
            0 3px 5px
            rgba(69,132,203,.22)
        );
}


/* Etiquetas flotantes */

.fs-chart-label {
    position: absolute;

    display: flex;
    flex-direction: column;

    padding:
        7px 9px;

    background:
        rgba(255,255,255,.92);

    border:
        1px solid #dce8f4;

    border-radius:
        8px;

    box-shadow:
        0 6px 18px
        rgba(51,89,130,.06);
}

.fs-chart-label span {
    color:
        #8da0b5;

    font-size:
        .48rem;

    font-weight:
        750;

    letter-spacing:
        .1em;
}

.fs-chart-label strong {
    margin-top:
        2px;

    color:
        #497fb9;

    font-size:
        .64rem;

    font-weight:
        700;
}

.fs-label-one {
    left:
        11%;

    top:
        22%;
}

.fs-label-two {
    right:
        10%;

    top:
        42%;
}


/* Pie del gráfico */

.fs-market-footer {
    display: grid;

    grid-template-columns:
        repeat(3, 1fr);

    gap:
        8px;

    padding-top:
        12px;
}

.fs-market-footer div {
    display: flex;
    flex-direction: column;

    min-width: 0;
}

.fs-market-footer small {
    color:
        #9aabba;

    font-size:
        .48rem;

    font-weight:
        750;

    letter-spacing:
        .09em;
}

.fs-market-footer strong {
    margin-top:
        2px;

    color:
        #5d7895;

    font-size:
        .61rem;

    font-weight:
        650;
}


/* ==========================================================
   PORTADA MÁS COMPACTA
   ========================================================== */

.fs-home-v2 {
    padding:
        3.2rem 4.4rem
        2.35rem !important;

    margin-bottom:
        .8rem !important;
}

.fs-eyebrow {
    margin-bottom:
        1.8rem !important;
}

.fs-hero-divider {
    margin:
        2.25rem 0
        1.35rem !important;
}


/* ==========================================================
   SUBIR BUSCADOR
   ========================================================== */

/*
El CSS antiguo lo empujaba artificialmente.
Esta regla gana prioridad al estar al final.
*/

.main-search-label {
    margin-top:
        -1.15rem !important;

    padding-top:
        0 !important;

    transform:
        none !important;
}

div[data-testid="stTextInput"]:has(
    input[placeholder^="Buscar activo ·"]
) {
    transform:
        none !important;

    margin-top:
        -.35rem !important;

    margin-bottom:
        1rem !important;
}


/* Buscador ligeramente más protagonista */

div[data-testid="stTextInput"]:has(
    input[placeholder^="Buscar activo ·"]
) input {

    min-height:
        64px !important;

    border:
        1px solid #b9cce0 !important;

    box-shadow:
        0 8px 26px
        rgba(39,84,132,.055) !important;
}


/* ==========================================================
   RESPONSIVE
   ========================================================== */

@media (max-width: 1050px) {

    .fs-hero-top {
        grid-template-columns:
            1fr;

        gap:
            2.4rem;
    }

    .fs-market-visual {
        max-width:
            650px;
    }

    .fs-market-chart {
        height:
            210px;
    }

}


@media (max-width: 700px) {

    .fs-home-v2 {
        padding:
            2.5rem 1.6rem
            1.9rem !important;
    }

    .fs-hero-copy h1 {
        font-size:
            2.65rem !important;
    }

    .fs-market-chart {
        height:
            175px;
    }

    .fs-market-footer {
        display:
            none;
    }

    .main-search-label {
        margin-top:
            -.4rem !important;
    }

}

</style>
""",
unsafe_allow_html=True,
)



# ============================================================
# FINSCOPE · HOME SEARCH PRIORITY V1
# ============================================================

st.markdown(
"""<style>

/* ----------------------------------------------------------
   HERO: vuelve a una sola columna limpia
   ---------------------------------------------------------- */

.fs-hero-top {
    display: block !important;
}

.fs-hero-copy {
    max-width: 900px !important;
}

.fs-hero-copy h1 {
    max-width: 850px !important;

    font-size:
        clamp(
            3.1rem,
            5vw,
            4.9rem
        ) !important;
}


/* ----------------------------------------------------------
   PORTADA MÁS BAJA
   ---------------------------------------------------------- */

.fs-home-v2 {
    margin:
        1.25rem auto
        0 !important;

    padding:
        2.8rem 4.4rem
        1.8rem !important;

    min-height:
        0 !important;
}

.fs-eyebrow {
    margin-bottom:
        1.55rem !important;
}

.fs-hero-copy > p {
    margin-top:
        1.25rem !important;
}

.fs-hero-divider {
    margin:
        1.8rem 0
        1.15rem !important;
}

.fs-hero-grid {
    margin:
        0 !important;
}

.fs-hero-item {
    padding-top:
        .25rem !important;

    padding-bottom:
        .25rem !important;
}


/* ----------------------------------------------------------
   EL GRÁFICO V2.1 YA NO DEBE OCUPAR ESPACIO
   ---------------------------------------------------------- */

.fs-market-visual {
    display:
        none !important;
}


/* ----------------------------------------------------------
   CLAVE:
   ELIMINAMOS EL ESPACIO DE STREAMLIT ENTRE PORTADA Y BUSCADOR
   ---------------------------------------------------------- */

/*
Streamlit envuelve cada st.markdown/widget en bloques verticales.
Reducimos el gap general de la zona principal, pero sin afectar
al contenido interno de los módulos.
*/

div[data-testid="stMainBlockContainer"] {
    row-gap:
        0 !important;
}


/* Contenedor vertical moderno de Streamlit */

div[data-testid="stMainBlockContainer"]
div[data-testid="stVerticalBlock"] {
    gap:
        .65rem;
}


/* ----------------------------------------------------------
   BUSCADOR PRINCIPAL
   ---------------------------------------------------------- */

/*
Los parches antiguos usaban -135px y transform.
Los anulamos definitivamente.
*/

.main-search-label {
    margin:
        .45rem 0
        .25rem 0 !important;

    padding:
        0 !important;

    transform:
        none !important;
}


/*
El bloque real del buscador se mueve hacia arriba respecto
a cualquier espaciador residual de Streamlit.
*/

div[data-testid="stTextInput"]:has(
    input[placeholder^="Buscar activo ·"]
) {
    position:
        relative !important;

    z-index:
        5 !important;

    transform:
        translateY(-8px) !important;

    margin:
        0 0
        -8px 0 !important;

    padding:
        0 !important;

    width:
        100% !important;

    max-width:
        none !important;
}


/* Input */

div[data-testid="stTextInput"]:has(
    input[placeholder^="Buscar activo ·"]
) input {

    min-height:
        62px !important;

    padding:
        0 20px !important;

    background:
        #ffffff !important;

    color:
        #172337 !important;

    border:
        1px solid #b9cce0 !important;

    border-radius:
        14px !important;

    font-size:
        1rem !important;

    box-shadow:
        0 8px 25px
        rgba(39,84,132,.055) !important;
}


div[data-testid="stTextInput"]:has(
    input[placeholder^="Buscar activo ·"]
) input:focus {

    border-color:
        #347fda !important;

    box-shadow:
        0 0 0 4px
        rgba(52,127,218,.08),
        0 8px 25px
        rgba(39,84,132,.055) !important;
}


/* ----------------------------------------------------------
   RESPONSIVE
   ---------------------------------------------------------- */

@media (max-width: 900px) {

    .fs-home-v2 {
        padding:
            2.6rem 2.7rem
            1.7rem !important;
    }

}


@media (max-width: 700px) {

    .fs-home-v2 {
        margin-top:
            .5rem !important;

        padding:
            2.3rem 1.6rem
            1.5rem !important;
    }

    .fs-hero-copy h1 {
        font-size:
            2.65rem !important;
    }

    .fs-hero-divider {
        margin-top:
            1.5rem !important;
    }

    .main-search-label {
        margin-top:
            .3rem !important;
    }

}

</style>""",
unsafe_allow_html=True,
)



# ============================================================
# FINSCOPE · ANALYSIS UI PROFESSIONAL V1
# ============================================================

st.markdown(
"""<style>

/* ==========================================================
   01 · CABECERA DEL ACTIVO
   ========================================================== */

.company-hero {
    position: relative !important;

    margin:
        1.15rem 0
        1.35rem !important;

    padding:
        2rem 2.15rem !important;

    background:
        #ffffff !important;

    border:
        1px solid #dfe7f0 !important;

    border-radius:
        20px !important;

    box-shadow:
        0 10px 32px
        rgba(35, 63, 94, .045) !important;

    overflow:
        hidden !important;
}


/* detalle superior */

.company-hero::before {
    content: "" !important;

    position: absolute !important;

    top: 0 !important;
    left: 2.15rem !important;

    width: 48px !important;
    height: 3px !important;

    background:
        #4b91df !important;

    border-radius:
        0 0 4px 4px !important;
}


/* Todo el texto de la cabecera */

.company-hero,
.company-hero * {
    color:
        #172337 !important;

    -webkit-text-fill-color:
        initial !important;
}


/* Textos secundarios */

.company-hero p,
.company-hero span,
.company-hero small {
    color:
        #738399 !important;
}


/* Precio destacado */

.company-hero .quote-price,
.company-hero [class*="price"] {
    color:
        #172337 !important;

    font-weight:
        720 !important;
}


/* ==========================================================
   02 · MÉTRICAS RESUMEN
   ========================================================== */

div[data-testid="stMetric"] {
    position:
        relative !important;

    min-height:
        108px !important;

    padding:
        17px 18px !important;

    background:
        #ffffff !important;

    border:
        1px solid #e0e8f0 !important;

    border-radius:
        15px !important;

    box-shadow:
        0 5px 18px
        rgba(35,63,94,.035) !important;

    transition:
        border-color .15s ease,
        box-shadow .15s ease,
        transform .15s ease !important;
}


div[data-testid="stMetric"]:hover {
    border-color:
        #c8d8e8 !important;

    box-shadow:
        0 8px 22px
        rgba(35,63,94,.05) !important;

    transform:
        translateY(-1px);
}


/* etiqueta */

div[data-testid="stMetricLabel"] {
    color:
        #7a899c !important;

    opacity:
        1 !important;

    font-size:
        .72rem !important;

    font-weight:
        650 !important;

    letter-spacing:
        .015em !important;
}


/* valor */

div[data-testid="stMetricValue"] {
    color:
        #1d2d43 !important;

    font-size:
        1.48rem !important;

    font-weight:
        720 !important;

    letter-spacing:
        -.025em !important;
}


/* delta */

div[data-testid="stMetricDelta"] {
    font-size:
        .72rem !important;

    font-weight:
        650 !important;
}


/* ==========================================================
   03 · SEPARADORES
   ========================================================== */

hr {
    margin:
        2.1rem 0 !important;

    border:
        0 !important;

    border-top:
        1px solid #e7edf3 !important;
}


/* ==========================================================
   04 · CABECERA DEL PANEL DE ANÁLISIS
   ========================================================== */

.fs-modules-header {
    margin:
        .35rem 0
        1.25rem !important;

    padding:
        0 !important;
}


.fs-modules-kicker {
    margin-bottom:
        .35rem !important;

    color:
        #4b91df !important;

    font-size:
        .64rem !important;

    font-weight:
        750 !important;

    letter-spacing:
        .13em !important;

    text-transform:
        uppercase !important;
}


.fs-modules-title {
    margin:
        0 !important;

    color:
        #172337 !important;

    font-size:
        1.55rem !important;

    font-weight:
        720 !important;

    letter-spacing:
        -.025em !important;

    line-height:
        1.2 !important;
}


.fs-modules-subtitle {
    max-width:
        700px !important;

    margin:
        .35rem 0 0 !important;

    color:
        #7a899c !important;

    font-size:
        .82rem !important;

    line-height:
        1.55 !important;
}


/* ==========================================================
   05 · BOTONES / TARJETAS DE MÓDULOS
   ========================================================== */

/*
La navegación actual usa botones de Streamlit como tarjetas.
No cambiamos su funcionamiento; solo su presentación.
*/

div[data-testid="stButton"] > button {
    min-height:
        52px;

    background:
        #ffffff !important;

    color:
        #33465e !important;

    border:
        1px solid #dfe7f0 !important;

    border-radius:
        12px !important;

    box-shadow:
        none !important;

    font-weight:
        620 !important;

    transition:
        background .15s ease,
        border-color .15s ease,
        color .15s ease,
        transform .15s ease !important;
}


div[data-testid="stButton"] > button:hover {
    background:
        #f6f9fd !important;

    color:
        #2468b2 !important;

    border-color:
        #b9cee4 !important;

    transform:
        translateY(-1px);
}


div[data-testid="stButton"] > button:focus {
    border-color:
        #4b91df !important;

    box-shadow:
        0 0 0 3px
        rgba(75,145,223,.08) !important;
}


/* ==========================================================
   06 · MÓDULO ACTIVO
   ========================================================== */

.fs-active-module {
    position:
        relative !important;

    margin:
        1.15rem 0
        1.45rem !important;

    padding:
        14px 17px
        14px 20px !important;

    background:
        #f5f9fe !important;

    border:
        1px solid #d8e6f4 !important;

    border-radius:
        12px !important;

    box-shadow:
        none !important;
}


.fs-active-module::before {
    content: "" !important;

    position: absolute !important;

    top: 11px !important;
    bottom: 11px !important;
    left: 0 !important;

    width: 3px !important;

    background:
        #4b91df !important;

    border-radius:
        0 4px 4px 0 !important;
}


.fs-active-module span {
    display:
        block !important;

    margin-bottom:
        2px !important;

    color:
        #8191a4 !important;

    font-size:
        .61rem !important;

    font-weight:
        750 !important;

    letter-spacing:
        .11em !important;
}


.fs-active-module strong {
    color:
        #24415f !important;

    font-size:
        .92rem !important;

    font-weight:
        700 !important;
}


/* ==========================================================
   07 · TÍTULOS DE CADA ANÁLISIS
   ========================================================== */

div[data-testid="stMainBlockContainer"] h2 {
    margin-top:
        1.2rem !important;

    margin-bottom:
        .55rem !important;

    color:
        #172337 !important;

    font-size:
        1.7rem !important;

    font-weight:
        720 !important;

    letter-spacing:
        -.035em !important;
}


div[data-testid="stMainBlockContainer"] h3 {
    margin-top:
        1.05rem !important;

    margin-bottom:
        .5rem !important;

    color:
        #24364d !important;

    font-size:
        1.22rem !important;

    font-weight:
        690 !important;

    letter-spacing:
        -.02em !important;
}


/* captions */

div[data-testid="stCaptionContainer"] {
    color:
        #7b899a !important;

    font-size:
        .76rem !important;

    line-height:
        1.5 !important;
}


/* ==========================================================
   08 · INPUTS / SELECTORES DEL ANÁLISIS
   ========================================================== */

div[data-testid="stSelectbox"] > div,
div[data-testid="stMultiSelect"] > div,
div[data-testid="stNumberInput"] > div,
div[data-testid="stTextInput"] > div {
    border-radius:
        11px !important;
}


div[data-baseweb="select"] > div {
    min-height:
        46px !important;

    background:
        #ffffff !important;

    color:
        #263950 !important;

    border-color:
        #d7e1eb !important;

    border-radius:
        11px !important;

    box-shadow:
        none !important;
}


div[data-baseweb="select"] * {
    color:
        #263950 !important;
}


div[data-testid="stNumberInput"] input,
div[data-testid="stTextInput"] input {
    background:
        #ffffff !important;

    color:
        #263950 !important;

    border-color:
        #d7e1eb !important;

    border-radius:
        11px !important;
}


div[data-testid="stNumberInput"] input:focus,
div[data-testid="stTextInput"] input:focus {
    border-color:
        #4b91df !important;

    box-shadow:
        0 0 0 3px
        rgba(75,145,223,.075) !important;
}


/* Labels */

div[data-testid="stWidgetLabel"] p {
    color:
        #53667c !important;

    font-size:
        .76rem !important;

    font-weight:
        650 !important;
}


/* ==========================================================
   09 · SEGMENTED CONTROL
   ========================================================== */

div[data-testid="stSegmentedControl"] {
    padding:
        4px !important;

    background:
        #f3f6fa !important;

    border:
        1px solid #e0e7ef !important;

    border-radius:
        12px !important;
}


div[data-testid="stSegmentedControl"] button {
    min-height:
        38px !important;

    background:
        transparent !important;

    color:
        #617288 !important;

    border:
        0 !important;

    border-radius:
        9px !important;

    box-shadow:
        none !important;

    font-weight:
        620 !important;
}


div[data-testid="stSegmentedControl"]
button[aria-pressed="true"] {
    background:
        #ffffff !important;

    color:
        #2468b2 !important;

    box-shadow:
        0 2px 7px
        rgba(34,72,112,.08) !important;
}


/* ==========================================================
   10 · EXPANDERS
   ========================================================== */

div[data-testid="stExpander"] {
    margin:
        .65rem 0 !important;

    background:
        #ffffff !important;

    border:
        1px solid #e1e8f0 !important;

    border-radius:
        13px !important;

    overflow:
        hidden !important;

    box-shadow:
        none !important;
}


div[data-testid="stExpander"] details,
div[data-testid="stExpander"] summary {
    background:
        #ffffff !important;

    color:
        #33465e !important;
}


div[data-testid="stExpander"] summary:hover {
    background:
        #f8fafc !important;
}


/* ==========================================================
   11 · TABLAS / DATAFRAMES
   ========================================================== */

div[data-testid="stDataFrame"],
div[data-testid="stDataEditor"] {
    overflow:
        hidden !important;

    background:
        #ffffff !important;

    border:
        1px solid #e1e8f0 !important;

    border-radius:
        13px !important;

    box-shadow:
        none !important;
}


/* ==========================================================
   12 · GRÁFICOS ALTAIR / VEGA
   ========================================================== */

div[data-testid="stVegaLiteChart"] {
    margin:
        .75rem 0
        1.25rem !important;

    padding:
        14px 14px
        8px !important;

    background:
        #ffffff !important;

    border:
        1px solid #e1e8f0 !important;

    border-radius:
        15px !important;

    box-shadow:
        0 5px 20px
        rgba(35,63,94,.035) !important;

    overflow:
        hidden !important;
}


div[data-testid="stVegaLiteChart"] > div {
    background:
        #ffffff !important;
}


/* ==========================================================
   13 · ALERTAS
   ========================================================== */

div[data-testid="stAlert"] {
    border-radius:
        12px !important;

    box-shadow:
        none !important;
}


/* ==========================================================
   14 · RADIO / CHECKBOX
   ========================================================== */

div[data-testid="stRadio"] {
    color:
        #33465e !important;
}


div[data-testid="stCheckbox"] {
    color:
        #33465e !important;
}


/* ==========================================================
   15 · ESPACIADO GENERAL DE LA ZONA DE ANÁLISIS
   ========================================================== */

/*
No tocamos la lógica de st.columns.
Solo hacemos más consistente la separación vertical.
*/

.analysis-section {
    margin:
        1.1rem 0
        1.7rem !important;
}


.analysis-caption {
    color:
        #7a899c !important;

    font-size:
        .76rem !important;

    line-height:
        1.55 !important;
}


/* ==========================================================
   16 · TARJETAS ESPECÍFICAS DE COTIZACIÓN
   ========================================================== */

.quote-period-card {
    background:
        #ffffff !important;

    border:
        1px solid #e1e8f0 !important;

    border-radius:
        14px !important;

    box-shadow:
        none !important;
}


.quote-price {
    color:
        #172337 !important;

    font-weight:
        720 !important;
}


/* ==========================================================
   17 · SCROLLBAR DISCRETO
   ========================================================== */

div[data-testid="stMainBlockContainer"] ::-webkit-scrollbar {
    width:
        8px;

    height:
        8px;
}


div[data-testid="stMainBlockContainer"]
::-webkit-scrollbar-thumb {
    background:
        #cbd7e3;

    border-radius:
        10px;
}


div[data-testid="stMainBlockContainer"]
::-webkit-scrollbar-track {
    background:
        transparent;
}


/* ==========================================================
   18 · RESPONSIVE
   ========================================================== */

@media (max-width: 900px) {

    .company-hero {
        padding:
            1.7rem 1.6rem !important;

        border-radius:
            17px !important;
    }

    .company-hero::before {
        left:
            1.6rem !important;
    }

    div[data-testid="stMetric"] {
        min-height:
            98px !important;
    }

    .fs-modules-title {
        font-size:
            1.4rem !important;
    }

}


@media (max-width: 650px) {

    .company-hero {
        margin-top:
            .6rem !important;

        padding:
            1.5rem 1.25rem !important;
    }

    .company-hero::before {
        left:
            1.25rem !important;
    }

    div[data-testid="stMetric"] {
        padding:
            14px 15px !important;
    }

    div[data-testid="stMetricValue"] {
        font-size:
            1.3rem !important;
    }

    div[data-testid="stMainBlockContainer"] h2 {
        font-size:
            1.5rem !important;
    }

}

</style>""",
unsafe_allow_html=True,
)

