from __future__ import annotations

from typing import Optional
import math

import pandas as pd
import yfinance as yf

from services.financial_data import (
    convertir_serie_historica_a_eur,
    _normalizar_moneda_finscope,
    _inferir_moneda_desde_ticker,
)


PERIODOS_RENDIMIENTO = {
    "1m": {"meses": 1, "cagr": False},
    "3m": {"meses": 3, "cagr": False},
    "6m": {"meses": 6, "cagr": False},
    "1y": {"meses": 12, "cagr": True},
    "3y": {"meses": 36, "cagr": True},
    "5y": {"meses": 60, "cagr": True},
    "10y": {"meses": 120, "cagr": True},
}


def _numero(valor) -> Optional[float]:
    try:
        numero = float(valor)

        if math.isfinite(numero):
            return numero

    except (TypeError, ValueError):
        pass

    return None


def _normalizar_historico(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    if isinstance(df.columns, pd.MultiIndex):
        if len(df.columns.levels) > 1:
            try:
                df.columns = df.columns.get_level_values(0)
            except Exception:
                pass

    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()

    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)

    return df


def obtener_serie_rendimiento(
    ticker: str,
    periodo_descarga: str = "max",
) -> pd.DataFrame:
    """
    Descarga una serie diaria sin ajuste automático y la expresa
    en EUR, moneda base de FinScope.

    Close:
        precio observado sin reinversión de dividendos, convertido
        a EUR con el tipo de cambio histórico de cada fecha.

    Adj Close:
        serie ajustada por eventos corporativos, convertida también
        a EUR con el FX histórico correspondiente a cada fecha.

    Para activos denominados originalmente en EUR no se realiza
    ninguna transformación monetaria.

    La moneda original y el estado de la conversión se conservan
    en DataFrame.attrs para trazabilidad.
    """

    ticker = str(ticker or "").strip().upper()

    if not ticker:
        return pd.DataFrame()

    try:
        instrumento = yf.Ticker(ticker)

        try:
            info = instrumento.fast_info
            moneda_original = getattr(
                info,
                "currency",
                None,
            )
        except Exception:
            moneda_original = None

        if not moneda_original:
            try:
                moneda_original = (
                    instrumento.info.get("currency")
                )
            except Exception:
                moneda_original = None

        if not moneda_original:
            moneda_original = _inferir_moneda_desde_ticker(
                ticker
            )

        moneda_original = _normalizar_moneda_finscope(
            moneda_original
        )

        df = yf.download(
            ticker,
            period=periodo_descarga,
            interval="1d",
            auto_adjust=False,
            actions=False,
            progress=False,
            threads=False,
        )

        df = _normalizar_historico(df)

        columnas = [
            columna
            for columna in ["Close", "Adj Close"]
            if columna in df.columns
        ]

        if "Close" not in columnas:
            return pd.DataFrame()

        df = df[columnas].copy()

        for columna in columnas:
            df[columna] = pd.to_numeric(
                df[columna],
                errors="coerce",
            )

        df = df.dropna(
            subset=["Close"]
        )

        if df.empty:
            return pd.DataFrame()

        # ---------------------------------------------------------
        # CONVERSIÓN HISTÓRICA A EUR
        # ---------------------------------------------------------

        conversion_aplicada = False

        if moneda_original == "EUR":
            conversion_aplicada = True

        elif moneda_original:

            convertidas = {}

            for columna in columnas:

                serie_original = (
                    df[columna]
                    .dropna()
                )

                if serie_original.empty:
                    continue

                serie_eur = convertir_serie_historica_a_eur(
                    serie_original,
                    moneda_original,
                )

                if serie_eur is None or serie_eur.empty:
                    # FinScope no etiqueta como EUR una serie que
                    # no haya podido convertir correctamente.
                    return pd.DataFrame()

                convertidas[columna] = serie_eur

            if "Close" not in convertidas:
                return pd.DataFrame()

            indice_comun = convertidas[
                "Close"
            ].index

            nuevo = pd.DataFrame(
                index=indice_comun
            )

            for columna, serie in convertidas.items():
                nuevo[columna] = (
                    serie.reindex(indice_comun)
                )

            nuevo = nuevo.dropna(
                subset=["Close"]
            )

            df = nuevo
            conversion_aplicada = True

        else:
            # Sin conocer la moneda original no podemos afirmar que
            # la serie está expresada en EUR.
            return pd.DataFrame()

        if isinstance(
            df.index,
            pd.DatetimeIndex,
        ):
            if df.index.tz is not None:
                df.index = (
                    df.index.tz_localize(None)
                )

        df = df[
            ~df.index.duplicated(
                keep="last"
            )
        ].sort_index()

        df.attrs[
            "moneda_original"
        ] = moneda_original

        df.attrs[
            "moneda_base"
        ] = "EUR"

        df.attrs[
            "conversion_eur_aplicada"
        ] = conversion_aplicada

        df.attrs[
            "ticker"
        ] = ticker

        return df

    except Exception:
        return pd.DataFrame()

def _punto_inicio(
    serie: pd.Series,
    fecha_objetivo: pd.Timestamp,
):
    """
    Selecciona el último dato disponible en o antes de la fecha
    objetivo. Si no existe porque el activo todavía no cotizaba,
    devuelve el primer dato disponible y marca historial incompleto.
    """

    serie = serie.dropna().sort_index()

    if serie.empty:
        return None, None, False

    anteriores = serie[
        serie.index <= fecha_objetivo
    ]

    if not anteriores.empty:
        fecha = anteriores.index[-1]
        return fecha, _numero(anteriores.iloc[-1]), True

    fecha = serie.index[0]
    return fecha, _numero(serie.iloc[0]), False


def _rentabilidad(
    inicial: Optional[float],
    final: Optional[float],
) -> Optional[float]:

    if (
        inicial is None
        or final is None
        or inicial <= 0
    ):
        return None

    return (
        (final / inicial) - 1
    ) * 100.0


def _cagr(
    inicial: Optional[float],
    final: Optional[float],
    fecha_inicio,
    fecha_fin,
) -> Optional[float]:

    if (
        inicial is None
        or final is None
        or inicial <= 0
        or final <= 0
        or fecha_inicio is None
        or fecha_fin is None
    ):
        return None

    dias = (
        pd.Timestamp(fecha_fin)
        - pd.Timestamp(fecha_inicio)
    ).days

    if dias <= 0:
        return None

    anos = dias / 365.2425

    if anos < 0.95:
        return None

    return (
        (final / inicial) ** (1 / anos)
        - 1
    ) * 100.0


def calcular_rendimiento_periodo(
    ticker: str,
    periodo: str,
    historico: Optional[pd.DataFrame] = None,
) -> Optional[dict]:
    """
    Calcula rendimiento histórico para un periodo.

    Price return:
        Close inicial -> Close final.

    Total return aproximado:
        Adj Close inicial -> Adj Close final.

    CAGR:
        usa el tiempo efectivo entre ambas fechas.
    """

    if periodo not in PERIODOS_RENDIMIENTO:
        return None

    if historico is None:
        historico = obtener_serie_rendimiento(
            ticker,
            "max",
        )

    historico = _normalizar_historico(
        historico
    )

    if historico.empty or "Close" not in historico.columns:
        return None

    close = historico["Close"].dropna()

    if close.empty:
        return None

    fecha_fin = close.index[-1]
    precio_final = _numero(
        close.iloc[-1]
    )

    meses = PERIODOS_RENDIMIENTO[
        periodo
    ]["meses"]

    fecha_objetivo = (
        pd.Timestamp(fecha_fin)
        - pd.DateOffset(months=meses)
    )

    (
        fecha_inicio,
        precio_inicial,
        periodo_completo,
    ) = _punto_inicio(
        close,
        fecha_objetivo,
    )

    if (
        fecha_inicio is None
        or precio_inicial is None
        or precio_final is None
    ):
        return None

    price_return = _rentabilidad(
        precio_inicial,
        precio_final,
    )

    adj_inicial = None
    adj_final = None
    total_return = None

    if "Adj Close" in historico.columns:
        adj = historico["Adj Close"].dropna()

        if not adj.empty:
            disponibles_inicio = adj[
                adj.index <= fecha_inicio
            ]

            disponibles_fin = adj[
                adj.index <= fecha_fin
            ]

            if (
                not disponibles_inicio.empty
                and not disponibles_fin.empty
            ):
                adj_inicial = _numero(
                    disponibles_inicio.iloc[-1]
                )
                adj_final = _numero(
                    disponibles_fin.iloc[-1]
                )

                total_return = _rentabilidad(
                    adj_inicial,
                    adj_final,
                )

    calcular_cagr = PERIODOS_RENDIMIENTO[
        periodo
    ]["cagr"]

    cagr_precio = None
    cagr_total = None

    if calcular_cagr:
        cagr_precio = _cagr(
            precio_inicial,
            precio_final,
            fecha_inicio,
            fecha_fin,
        )

        if (
            adj_inicial is not None
            and adj_final is not None
        ):
            cagr_total = _cagr(
                adj_inicial,
                adj_final,
                fecha_inicio,
                fecha_fin,
            )

    dias_reales = (
        pd.Timestamp(fecha_fin)
        - pd.Timestamp(fecha_inicio)
    ).days

    return {
        "ticker": str(ticker).upper(),
        "periodo": periodo,
        "meses_objetivo": meses,
        "fecha_objetivo": fecha_objetivo,
        "fecha_inicio": pd.Timestamp(
            fecha_inicio
        ),
        "fecha_fin": pd.Timestamp(
            fecha_fin
        ),
        "dias_reales": dias_reales,
        "periodo_completo": periodo_completo,

        "precio_inicial": precio_inicial,
        "precio_final": precio_final,
        "rentabilidad_precio_pct": price_return,

        "precio_ajustado_inicial": adj_inicial,
        "precio_ajustado_final": adj_final,
        "rentabilidad_total_pct": total_return,

        "cagr_precio_pct": cagr_precio,
        "cagr_total_pct": cagr_total,

        "metodologia_precio": (
            "Close sin ajustar"
        ),
        "metodologia_total": (
            "Adj Close de Yahoo Finance"
        ),
        "fuente": (
            "Yahoo Finance vía yfinance"
        ),
    }


def calcular_rendimientos_historicos(
    ticker: str,
) -> dict:
    """
    Calcula todos los periodos usando una única descarga.
    """

    historico = obtener_serie_rendimiento(
        ticker,
        "max",
    )

    resultados = {}

    if historico.empty:
        return {
            "ticker": str(ticker).upper(),
            "resultados": resultados,
            "fuente": "Yahoo Finance vía yfinance",
        }

    for periodo in PERIODOS_RENDIMIENTO:
        resultado = calcular_rendimiento_periodo(
            ticker,
            periodo,
            historico=historico,
        )

        if resultado is not None:
            resultados[periodo] = resultado

    return {
        "ticker": str(ticker).upper(),
        "resultados": resultados,
        "fuente": "Yahoo Finance vía yfinance",
    }


def evolucion_inversion(
    ticker: str,
    capital_inicial: float = 1000.0,
    periodo: str = "5y",
    incluir_dividendos: bool = True,
) -> Optional[dict]:
    """
    Calcula cuánto habría evolucionado un capital inicial según
    la serie histórica seleccionada.

    No incluye impuestos, comisiones ni spreads.

    Cuando la serie histórica utilizada por FinScope ha sido convertida
    a EUR mediante FX histórico, la rentabilidad expresada en EUR sí
    incorpora el efecto de la divisa frente al euro.
    """

    capital = _numero(
        capital_inicial
    )

    if capital is None or capital <= 0:
        return None

    resultado = calcular_rendimiento_periodo(
        ticker,
        periodo,
    )

    if not resultado:
        return None

    if incluir_dividendos:
        rentabilidad = resultado.get(
            "rentabilidad_total_pct"
        )
        metodologia = "total_return_aproximado"
    else:
        rentabilidad = resultado.get(
            "rentabilidad_precio_pct"
        )
        metodologia = "price_return"

    if rentabilidad is None:
        return None

    capital_final = capital * (
        1 + rentabilidad / 100
    )

    return {
        "ticker": str(ticker).upper(),
        "periodo": periodo,
        "capital_inicial": capital,
        "capital_final": capital_final,
        "rentabilidad_pct": rentabilidad,
        "metodologia": metodologia,
        "fecha_inicio": resultado[
            "fecha_inicio"
        ],
        "fecha_fin": resultado[
            "fecha_fin"
        ],
        "incluye_comisiones": False,
        "incluye_impuestos": False,
        "incluye_efecto_divisa": True,
        "moneda_resultado": "EUR",
        "nota_divisa": (
            "La serie histórica de FinScope está expresada en EUR; "
            "para activos originalmente denominados en otra moneda, "
            "la evolución incorpora el efecto histórico de esa divisa "
            "frente al euro."
        ),
    }

def calcular_benchmark(
    ticker: str,
    benchmark: str = "^GSPC",
    periodo: str = "5y",
    incluir_dividendos: bool = True,
) -> Optional[dict]:
    """
    Compara un activo con un benchmark durante el mismo intervalo.

    Ambos se alinean sobre fechas comunes para evitar comparar
    ventanas temporalmente diferentes.

    La serie normalizada comienza en 100.
    """

    if periodo not in PERIODOS_RENDIMIENTO:
        return None

    activo = obtener_serie_rendimiento(
        ticker,
        "max",
    )

    referencia = obtener_serie_rendimiento(
        benchmark,
        "max",
    )

    if activo.empty or referencia.empty:
        return None

    columna = (
        "Adj Close"
        if incluir_dividendos
        and "Adj Close" in activo.columns
        and "Adj Close" in referencia.columns
        else "Close"
    )

    serie_activo = activo[
        columna
    ].dropna().rename("Activo")

    serie_benchmark = referencia[
        columna
    ].dropna().rename("Benchmark")

    conjunto = pd.concat(
        [
            serie_activo,
            serie_benchmark,
        ],
        axis=1,
        join="inner",
    ).dropna()

    if conjunto.empty:
        return None

    fecha_fin = conjunto.index[-1]

    meses = PERIODOS_RENDIMIENTO[
        periodo
    ]["meses"]

    fecha_objetivo = (
        pd.Timestamp(fecha_fin)
        - pd.DateOffset(months=meses)
    )

    anteriores = conjunto[
        conjunto.index <= fecha_objetivo
    ]

    periodo_completo = True

    if not anteriores.empty:
        fecha_inicio = anteriores.index[-1]
    else:
        fecha_inicio = conjunto.index[0]
        periodo_completo = False

    ventana = conjunto[
        conjunto.index >= fecha_inicio
    ].copy()

    if ventana.empty:
        return None

    inicio_activo = _numero(
        ventana["Activo"].iloc[0]
    )
    final_activo = _numero(
        ventana["Activo"].iloc[-1]
    )

    inicio_benchmark = _numero(
        ventana["Benchmark"].iloc[0]
    )
    final_benchmark = _numero(
        ventana["Benchmark"].iloc[-1]
    )

    retorno_activo = _rentabilidad(
        inicio_activo,
        final_activo,
    )

    retorno_benchmark = _rentabilidad(
        inicio_benchmark,
        final_benchmark,
    )

    if (
        retorno_activo is None
        or retorno_benchmark is None
    ):
        return None

    cagr_activo = _cagr(
        inicio_activo,
        final_activo,
        ventana.index[0],
        ventana.index[-1],
    )

    cagr_benchmark = _cagr(
        inicio_benchmark,
        final_benchmark,
        ventana.index[0],
        ventana.index[-1],
    )

    ventana["Activo_normalizado"] = (
        ventana["Activo"]
        / inicio_activo
        * 100.0
    )

    ventana["Benchmark_normalizado"] = (
        ventana["Benchmark"]
        / inicio_benchmark
        * 100.0
    )

    evolucion_1000_activo = (
        1000.0
        * (
            1
            + retorno_activo / 100.0
        )
    )

    evolucion_1000_benchmark = (
        1000.0
        * (
            1
            + retorno_benchmark / 100.0
        )
    )

    return {
        "ticker": str(ticker).upper(),
        "benchmark": str(benchmark).upper(),
        "periodo": periodo,
        "fecha_inicio": pd.Timestamp(
            ventana.index[0]
        ),
        "fecha_fin": pd.Timestamp(
            ventana.index[-1]
        ),
        "periodo_completo": periodo_completo,
        "metodologia": (
            "total_return_ajustado"
            if columna == "Adj Close"
            else "price_return"
        ),
        "columna": columna,
        "rentabilidad_activo_pct":
            retorno_activo,
        "rentabilidad_benchmark_pct":
            retorno_benchmark,
        "diferencia_rentabilidad_pp":
            retorno_activo
            - retorno_benchmark,
        "cagr_activo_pct":
            cagr_activo,
        "cagr_benchmark_pct":
            cagr_benchmark,
        "diferencia_cagr_pp": (
            cagr_activo
            - cagr_benchmark
            if cagr_activo is not None
            and cagr_benchmark is not None
            else None
        ),
        "capital_final_activo":
            evolucion_1000_activo,
        "capital_final_benchmark":
            evolucion_1000_benchmark,
        "serie_normalizada": ventana[
            [
                "Activo_normalizado",
                "Benchmark_normalizado",
            ]
        ].copy(),
        "fuente":
            "Yahoo Finance vía yfinance",
    }

def calcular_metricas_riesgo(
    ticker: str,
    periodo: str = "5y",
    usar_ajustado: bool = True,
) -> Optional[dict]:
    """
    Calcula métricas históricas de riesgo.

    Volatilidad:
        desviación estándar de rendimientos diarios
        anualizada mediante sqrt(252).

    Maximum Drawdown:
        mayor caída porcentual desde un máximo histórico
        hasta un mínimo posterior dentro del periodo.

    Las métricas describen comportamiento histórico y no
    constituyen una estimación del riesgo futuro.
    """

    if periodo not in PERIODOS_RENDIMIENTO:
        return None

    historico = obtener_serie_rendimiento(
        ticker,
        "max",
    )

    if historico.empty:
        return None

    columna = (
        "Adj Close"
        if usar_ajustado
        and "Adj Close" in historico.columns
        else "Close"
    )

    serie = historico[
        columna
    ].dropna().sort_index()

    if len(serie) < 2:
        return None

    fecha_fin = serie.index[-1]

    meses = PERIODOS_RENDIMIENTO[
        periodo
    ]["meses"]

    fecha_objetivo = (
        pd.Timestamp(fecha_fin)
        - pd.DateOffset(months=meses)
    )

    anteriores = serie[
        serie.index <= fecha_objetivo
    ]

    periodo_completo = True

    if not anteriores.empty:
        fecha_inicio = anteriores.index[-1]
    else:
        fecha_inicio = serie.index[0]
        periodo_completo = False

    ventana = serie[
        serie.index >= fecha_inicio
    ].copy()

    if len(ventana) < 2:
        return None

    rendimientos = (
        ventana
        .pct_change()
        .dropna()
    )

    if rendimientos.empty:
        return None

    volatilidad_diaria = float(
        rendimientos.std(ddof=1)
    )

    volatilidad_anual = (
        volatilidad_diaria
        * (252 ** 0.5)
        * 100.0
    )

    maximos = ventana.cummax()

    drawdown = (
        ventana / maximos - 1.0
    )

    fecha_valle = drawdown.idxmin()

    max_drawdown = float(
        drawdown.loc[fecha_valle]
        * 100.0
    )

    hasta_valle = ventana.loc[
        :fecha_valle
    ]

    if hasta_valle.empty:
        return None

    fecha_pico = hasta_valle.idxmax()

    valor_pico = float(
        ventana.loc[fecha_pico]
    )

    fecha_recuperacion = None

    posteriores = ventana.loc[
        fecha_valle:
    ]

    recuperados = posteriores[
        posteriores >= valor_pico
    ]

    if not recuperados.empty:
        fecha_recuperacion = (
            recuperados.index[0]
        )

    dias_caida = (
        pd.Timestamp(fecha_valle)
        - pd.Timestamp(fecha_pico)
    ).days

    dias_recuperacion = None

    if fecha_recuperacion is not None:
        dias_recuperacion = (
            pd.Timestamp(fecha_recuperacion)
            - pd.Timestamp(fecha_valle)
        ).days

    drawdown_df = pd.DataFrame(
        {
            "Precio": ventana,
            "Máximo histórico": maximos,
            "Drawdown_pct":
                drawdown * 100.0,
        }
    )

    return {
        "ticker": str(ticker).upper(),
        "periodo": periodo,
        "fecha_inicio": pd.Timestamp(
            ventana.index[0]
        ),
        "fecha_fin": pd.Timestamp(
            ventana.index[-1]
        ),
        "periodo_completo":
            periodo_completo,
        "observaciones":
            int(len(rendimientos)),
        "volatilidad_diaria_pct":
            volatilidad_diaria * 100.0,
        "volatilidad_anual_pct":
            volatilidad_anual,
        "max_drawdown_pct":
            max_drawdown,
        "fecha_pico":
            pd.Timestamp(fecha_pico),
        "fecha_valle":
            pd.Timestamp(fecha_valle),
        "fecha_recuperacion": (
            pd.Timestamp(fecha_recuperacion)
            if fecha_recuperacion is not None
            else None
        ),
        "dias_caida":
            int(dias_caida),
        "dias_recuperacion":
            (
                int(dias_recuperacion)
                if dias_recuperacion is not None
                else None
            ),
        "serie_drawdown":
            drawdown_df,
        "metodologia": (
            "Adj Close"
            if columna == "Adj Close"
            else "Close"
        ),
        "sesiones_anuales":
            252,
        "fuente":
            "Yahoo Finance vía yfinance",
    }

def calcular_beta_correlacion(
    ticker: str,
    benchmark: str = "^GSPC",
    periodo: str = "5y",
    usar_ajustado: bool = True,
) -> Optional[dict]:
    """
    Calcula beta y correlación histórica frente a un benchmark.

    Beta:
        cov(R_activo, R_mercado) / var(R_mercado)

    Correlación:
        correlación de Pearson entre rendimientos diarios.

    Se utilizan exclusivamente fechas comunes para activo y benchmark.
    """

    if periodo not in PERIODOS_RENDIMIENTO:
        return None

    activo = obtener_serie_rendimiento(ticker, "max")
    mercado = obtener_serie_rendimiento(benchmark, "max")

    if activo.empty or mercado.empty:
        return None

    columna = (
        "Adj Close"
        if (
            usar_ajustado
            and "Adj Close" in activo.columns
            and "Adj Close" in mercado.columns
        )
        else "Close"
    )

    precios = pd.concat(
        [
            activo[columna].rename("Activo"),
            mercado[columna].rename("Mercado"),
        ],
        axis=1,
        join="inner",
    ).dropna().sort_index()

    if len(precios) < 3:
        return None

    fecha_fin = precios.index[-1]

    meses = PERIODOS_RENDIMIENTO[
        periodo
    ]["meses"]

    fecha_objetivo = (
        pd.Timestamp(fecha_fin)
        - pd.DateOffset(months=meses)
    )

    anteriores = precios[
        precios.index <= fecha_objetivo
    ]

    periodo_completo = True

    if not anteriores.empty:
        fecha_inicio = anteriores.index[-1]
    else:
        fecha_inicio = precios.index[0]
        periodo_completo = False

    ventana = precios[
        precios.index >= fecha_inicio
    ].copy()

    rendimientos = (
        ventana
        .pct_change()
        .dropna()
    )

    if len(rendimientos) < 2:
        return None

    var_mercado = float(
        rendimientos["Mercado"].var(ddof=1)
    )

    if (
        pd.isna(var_mercado)
        or var_mercado <= 0
    ):
        return None

    covarianza = float(
        rendimientos[
            ["Activo", "Mercado"]
        ].cov(ddof=1).loc[
            "Activo", "Mercado"
        ]
    )

    beta = covarianza / var_mercado

    correlacion = float(
        rendimientos[
            "Activo"
        ].corr(
            rendimientos["Mercado"]
        )
    )

    if pd.isna(beta) or pd.isna(correlacion):
        return None

    correlacion = max(
        -1.0,
        min(1.0, correlacion),
    )

    return {
        "ticker": str(ticker).upper(),
        "benchmark": str(benchmark).upper(),
        "periodo": periodo,
        "fecha_inicio": pd.Timestamp(
            ventana.index[0]
        ),
        "fecha_fin": pd.Timestamp(
            ventana.index[-1]
        ),
        "periodo_completo": periodo_completo,
        "observaciones": int(
            len(rendimientos)
        ),
        "beta": float(beta),
        "correlacion": float(
            correlacion
        ),
        "covarianza_diaria": float(
            covarianza
        ),
        "varianza_benchmark_diaria": float(
            var_mercado
        ),
        "metodologia": columna,
        "fuente":
            "Yahoo Finance vía yfinance",
    }

def calcular_rentabilidad_ajustada_riesgo(
    ticker: str,
    periodo: str = "5y",
    tasa_libre_riesgo_pct: float = 0.0,
    usar_ajustado: bool = True,
) -> Optional[dict]:
    """
    Calcula Sharpe y Sortino a partir de rendimientos diarios.

    Sharpe:
        media diaria del exceso de rentabilidad /
        desviación estándar diaria, anualizado por sqrt(252).

    Sortino:
        media diaria del exceso de rentabilidad /
        downside deviation respecto al objetivo diario equivalente
        a la tasa libre de riesgo, anualizado por sqrt(252).

    La tasa libre de riesgo se introduce como porcentaje anual.
    """

    if periodo not in PERIODOS_RENDIMIENTO:
        return None

    historico = obtener_serie_rendimiento(
        ticker,
        "max",
    )

    if historico.empty:
        return None

    columna = (
        "Adj Close"
        if usar_ajustado
        and "Adj Close" in historico.columns
        else "Close"
    )

    serie = (
        historico[columna]
        .dropna()
        .sort_index()
    )

    if len(serie) < 3:
        return None

    fecha_fin = serie.index[-1]

    meses = PERIODOS_RENDIMIENTO[
        periodo
    ]["meses"]

    fecha_objetivo = (
        pd.Timestamp(fecha_fin)
        - pd.DateOffset(months=meses)
    )

    anteriores = serie[
        serie.index <= fecha_objetivo
    ]

    periodo_completo = True

    if not anteriores.empty:
        fecha_inicio = anteriores.index[-1]
    else:
        fecha_inicio = serie.index[0]
        periodo_completo = False

    ventana = serie[
        serie.index >= fecha_inicio
    ].copy()

    rendimientos = (
        ventana
        .pct_change()
        .dropna()
    )

    if len(rendimientos) < 2:
        return None

    sesiones = 252

    tasa_anual = (
        float(tasa_libre_riesgo_pct)
        / 100.0
    )

    # Equivalente diario compuesto.
    tasa_diaria = (
        (1.0 + tasa_anual)
        ** (1.0 / sesiones)
        - 1.0
    )

    exceso = (
        rendimientos
        - tasa_diaria
    )

    media_exceso_diaria = float(
        exceso.mean()
    )

    desviacion_diaria = float(
        rendimientos.std(ddof=1)
    )

    if (
        pd.isna(desviacion_diaria)
        or desviacion_diaria <= 0
    ):
        sharpe = None
    else:
        sharpe = (
            media_exceso_diaria
            / desviacion_diaria
            * (sesiones ** 0.5)
        )

    # Downside deviation:
    # solo penaliza desviaciones por debajo del objetivo diario.
    desviaciones_negativas = (
        rendimientos
        - tasa_diaria
    ).clip(upper=0.0)

    downside_deviation_diaria = float(
        (
            (
                desviaciones_negativas ** 2
            ).mean()
        ) ** 0.5
    )

    if (
        pd.isna(downside_deviation_diaria)
        or downside_deviation_diaria <= 0
    ):
        sortino = None
    else:
        sortino = (
            media_exceso_diaria
            / downside_deviation_diaria
            * (sesiones ** 0.5)
        )

    volatilidad_anual = (
        desviacion_diaria
        * (sesiones ** 0.5)
        * 100.0
    )

    downside_anual = (
        downside_deviation_diaria
        * (sesiones ** 0.5)
        * 100.0
    )

    precio_inicial = float(
        ventana.iloc[0]
    )

    precio_final = float(
        ventana.iloc[-1]
    )

    anos = (
        (
            pd.Timestamp(ventana.index[-1])
            - pd.Timestamp(ventana.index[0])
        ).days
        / 365.2425
    )

    cagr = None

    if (
        precio_inicial > 0
        and precio_final > 0
        and anos > 0
    ):
        cagr = (
            (
                precio_final
                / precio_inicial
            ) ** (1.0 / anos)
            - 1.0
        ) * 100.0

    return {
        "ticker": str(ticker).upper(),
        "periodo": periodo,
        "fecha_inicio": pd.Timestamp(
            ventana.index[0]
        ),
        "fecha_fin": pd.Timestamp(
            ventana.index[-1]
        ),
        "periodo_completo":
            periodo_completo,
        "observaciones":
            int(len(rendimientos)),
        "sharpe": (
            float(sharpe)
            if sharpe is not None
            else None
        ),
        "sortino": (
            float(sortino)
            if sortino is not None
            else None
        ),
        "volatilidad_anual_pct":
            float(volatilidad_anual),
        "downside_deviation_anual_pct":
            float(downside_anual),
        "cagr_pct": (
            float(cagr)
            if cagr is not None
            else None
        ),
        "tasa_libre_riesgo_pct":
            float(tasa_libre_riesgo_pct),
        "tasa_libre_riesgo_diaria":
            float(tasa_diaria),
        "sesiones_anuales":
            sesiones,
        "metodologia":
            columna,
        "fuente":
            "Yahoo Finance vía yfinance",
    }

def calcular_per_historico(
    ticker: str,
    periodo: str = "5y",
) -> Optional[dict]:
    """
    Reconstruye una serie histórica aproximada de PER mediante:
    precio diario Close / EPS TTM asociado a los últimos cuatro
    trimestres disponibles en la fuente.

    IMPORTANTE:
    Yahoo/yfinance expone aquí fechas de periodo contable, no una
    base point-in-time completa de fechas históricas de publicación.
    Por ello esta serie NO debe interpretarse como un PER histórico
    estrictamente libre de look-ahead bias.

    Se utiliza para análisis descriptivo aproximado, no para backtests
    ni para afirmar qué información conocía exactamente el mercado
    en cada fecha histórica.

    Solo se calcula PER cuando EPS TTM > 0.
    """
    import yfinance as yf

    periodos_anos = {
        "1y": 1,
        "3y": 3,
        "5y": 5,
        "10y": 10,
    }

    if periodo not in periodos_anos:
        return None

    tk = yf.Ticker(ticker)

    precios = tk.history(
        period="max",
        interval="1d",
        auto_adjust=False,
        actions=False,
    )

    if precios is None or precios.empty or "Close" not in precios.columns:
        return None

    precios = precios[["Close"]].dropna().sort_index()

    try:
        income = tk.quarterly_income_stmt
    except Exception:
        return None

    if income is None or income.empty:
        return None

    fila_eps = None

    for nombre in ["Diluted EPS", "Basic EPS"]:
        if nombre in income.index:
            fila_eps = nombre
            break

    if fila_eps is None:
        return None

    eps = pd.to_numeric(
        income.loc[fila_eps],
        errors="coerce",
    ).dropna()

    eps.index = pd.to_datetime(
        eps.index,
        errors="coerce",
    )

    eps = eps[
        ~eps.index.isna()
    ].sort_index()

    if len(eps) < 4:
        return None

    eps_ttm = (
        eps
        .rolling(4)
        .sum()
        .dropna()
    )

    if eps_ttm.empty:
        return None

    precios.index = pd.to_datetime(
        precios.index
    )

    if getattr(precios.index, "tz", None) is not None:
        precios.index = precios.index.tz_localize(None)

    fecha_fin = pd.Timestamp(
        precios.index[-1]
    )

    fecha_objetivo = (
        fecha_fin
        - pd.DateOffset(
            years=periodos_anos[periodo]
        )
    )

    precios = precios[
        precios.index >= fecha_objetivo
    ].copy()

    if precios.empty:
        return None

    tabla_precios = precios.reset_index()

    tabla_precios = tabla_precios.rename(
        columns={
            tabla_precios.columns[0]: "fecha",
            "Close": "precio",
        }
    )

    tabla_eps = pd.DataFrame({
        "fecha_eps": eps_ttm.index,
        "eps_ttm": eps_ttm.values,
    })

    tabla_precios["fecha"] = pd.to_datetime(
        tabla_precios["fecha"]
    )

    tabla_eps["fecha_eps"] = pd.to_datetime(
        tabla_eps["fecha_eps"]
    )

    combinado = pd.merge_asof(
        tabla_precios.sort_values("fecha"),
        tabla_eps.sort_values("fecha_eps"),
        left_on="fecha",
        right_on="fecha_eps",
        direction="backward",
    )

    combinado = combinado.dropna(
        subset=["precio", "eps_ttm"]
    )

    combinado = combinado[
        combinado["eps_ttm"] > 0
    ].copy()

    if combinado.empty:
        return None

    combinado["per"] = (
        combinado["precio"]
        / combinado["eps_ttm"]
    )

    combinado = combinado[
        (combinado["per"] > 0)
        & (combinado["per"] < 500)
    ].copy()

    if combinado.empty:
        return None

    serie = combinado[
        ["fecha", "precio", "eps_ttm", "per"]
    ].copy()

    per_actual = float(
        serie["per"].iloc[-1]
    )

    per_media = float(
        serie["per"].mean()
    )

    per_mediana = float(
        serie["per"].median()
    )

    per_min = float(
        serie["per"].min()
    )

    if serie is None or serie.empty:
        return None

    per_max = float(
        serie["per"].max()
    )

    percentil = float(
        (
            serie["per"] <= per_actual
        ).mean()
        * 100.0
    )

    return {
        "ticker": str(ticker).upper(),
        "periodo": periodo,
        "fecha_inicio": pd.Timestamp(
            serie["fecha"].iloc[0]
        ),
        "fecha_fin": pd.Timestamp(
            serie["fecha"].iloc[-1]
        ),
        "per_actual": per_actual,
        "per_media": per_media,
        "per_mediana": per_mediana,
        "per_min": per_min,
        "per_max": per_max,
        "percentil_actual": percentil,
        "observaciones": int(len(serie)),
        "serie": serie,
        "metodologia": (
            "PER histórico aproximado reconstruido por FinScope: "
            "precio diario Close / EPS TTM de los últimos cuatro "
            "trimestres disponibles en Yahoo Finance. Las fechas "
            "del EPS corresponden a periodos contables y no "
            "constituyen una base point-in-time completa de fechas "
            "de publicación."
        ),
        "point_in_time": False,
        "advertencia_point_in_time": (
            "Serie descriptiva aproximada. No debe utilizarse como "
            "evidencia de la información que estaba públicamente "
            "disponible en cada fecha histórica ni para backtesting "
            "sin una fuente point-in-time de resultados."
        ),
        "fuente": "Yahoo Finance vía yfinance",
    }



# ============================================================
# NIVEL 2 · VALORACIÓN DCF
# ============================================================

def calcular_dcf(
    fcf_base,
    crecimiento_pct,
    tasa_descuento_pct,
    crecimiento_terminal_pct,
    efectivo=0.0,
    deuda=0.0,
    acciones=None,
    precio_actual=None,
    anos=5,
):
    """
    DCF convencional sobre Free Cash Flow.

    Proyecta FCF durante varios años y calcula un valor terminal
    mediante crecimiento perpetuo de Gordon.

    No estima automáticamente los supuestos.
    """

    try:
        fcf_base = float(fcf_base)
        crecimiento = float(crecimiento_pct) / 100
        descuento = float(tasa_descuento_pct) / 100
        terminal = float(crecimiento_terminal_pct) / 100
        efectivo = float(efectivo or 0)
        deuda = float(deuda or 0)

        if fcf_base <= 0:
            return {
                "valido": False,
                "motivo": "FCF no positivo",
            }

        if anos < 1:
            return {
                "valido": False,
                "motivo": "Horizonte inválido",
            }

        if descuento <= terminal:
            return {
                "valido": False,
                "motivo": (
                    "La tasa de descuento debe ser superior "
                    "al crecimiento terminal."
                ),
            }

        flujos = []

        fcf = fcf_base
        valor_presente_fcf = 0.0

        for ano in range(1, anos + 1):
            fcf = fcf * (1 + crecimiento)

            factor = (1 + descuento) ** ano
            vp = fcf / factor

            valor_presente_fcf += vp

            flujos.append({
                "Año": ano,
                "FCF proyectado": fcf,
                "Valor presente": vp,
            })

        fcf_terminal = fcf * (1 + terminal)

        valor_terminal = (
            fcf_terminal
            / (descuento - terminal)
        )

        valor_presente_terminal = (
            valor_terminal
            / ((1 + descuento) ** anos)
        )

        enterprise_value = (
            valor_presente_fcf
            + valor_presente_terminal
        )

        equity_value = (
            enterprise_value
            + efectivo
            - deuda
        )

        valor_por_accion = None
        diferencia_pct = None

        if acciones is not None:
            acciones = float(acciones)

            if acciones > 0:
                valor_por_accion = (
                    equity_value / acciones
                )

        if (
            valor_por_accion is not None
            and precio_actual is not None
        ):
            precio_actual = float(precio_actual)

            if precio_actual > 0:
                diferencia_pct = (
                    (valor_por_accion / precio_actual)
                    - 1
                ) * 100

        peso_terminal_pct = (
            valor_presente_terminal
            / enterprise_value
            * 100
            if enterprise_value > 0
            else None
        )

        return {
            "valido": True,
            "fcf_base": fcf_base,
            "crecimiento_pct": crecimiento_pct,
            "tasa_descuento_pct": tasa_descuento_pct,
            "crecimiento_terminal_pct":
                crecimiento_terminal_pct,
            "anos": anos,
            "flujos": flujos,
            "valor_presente_fcf":
                valor_presente_fcf,
            "valor_terminal":
                valor_terminal,
            "valor_presente_terminal":
                valor_presente_terminal,
            "enterprise_value":
                enterprise_value,
            "efectivo": efectivo,
            "deuda": deuda,
            "equity_value": equity_value,
            "acciones": acciones,
            "valor_por_accion":
                valor_por_accion,
            "precio_actual": precio_actual,
            "diferencia_pct":
                diferencia_pct,
            "peso_terminal_pct":
                peso_terminal_pct,
        }

    except Exception as exc:
        return {
            "valido": False,
            "motivo": str(exc),
        }


# ============================================================================
# NIVEL 3 · ESTADÍSTICA CUANTITATIVA DE RENDIMIENTOS
# ============================================================================

def calcular_estadistica_rendimientos(
    historico,
    frecuencia="Diaria",
    usar_ajustado=True,
):
    """
    Estadística descriptiva de rendimientos históricos.
    No realiza predicciones.
    """
    import numpy as np
    import pandas as pd

    vacio = {
        "valido": False,
        "mensaje": "No hay datos suficientes.",
    }

    if historico is None or len(historico) < 3:
        return vacio

    df = historico.copy()

    columna = None

    if usar_ajustado and "Adj Close" in df.columns:
        serie = pd.to_numeric(
            df["Adj Close"],
            errors="coerce",
        )

        if serie.notna().sum() >= 3:
            columna = "Adj Close"

    if columna is None and "Close" in df.columns:
        columna = "Close"

    if columna is None:
        return vacio

    precios = pd.to_numeric(
        df[columna],
        errors="coerce",
    ).dropna()

    precios = precios[precios > 0]

    if len(precios) < 3:
        return vacio

    reglas = {
        "Diaria": None,
        "Semanal": "W-FRI",
        "Mensual": "ME",
        "Anual": "YE",
    }

    if frecuencia not in reglas:
        frecuencia = "Diaria"

    regla = reglas[frecuencia]

    if regla is not None:
        precios = precios.resample(regla).last().dropna()

    rendimientos = precios.pct_change().dropna()

    rendimientos = rendimientos.replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    if len(rendimientos) < 2:
        return vacio

    media = float(rendimientos.mean())
    mediana = float(rendimientos.median())
    desviacion = float(rendimientos.std(ddof=1))

    minimo = float(rendimientos.min())
    maximo = float(rendimientos.max())

    positivos = int((rendimientos > 0).sum())
    negativos = int((rendimientos < 0).sum())
    neutros = int((rendimientos == 0).sum())

    total = int(len(rendimientos))

    percentiles = {
        p: float(rendimientos.quantile(p / 100))
        for p in [1, 5, 25, 50, 75, 95, 99]
    }

    return {
        "valido": True,
        "frecuencia": frecuencia,
        "columna_precio": columna,
        "observaciones": total,
        "inicio": precios.index.min(),
        "fin": precios.index.max(),
        "media": media,
        "mediana": mediana,
        "desviacion": desviacion,
        "minimo": minimo,
        "maximo": maximo,
        "fecha_minimo": rendimientos.idxmin(),
        "fecha_maximo": rendimientos.idxmax(),
        "positivos": positivos,
        "negativos": negativos,
        "neutros": neutros,
        "pct_positivos": positivos / total * 100,
        "pct_negativos": negativos / total * 100,
        "percentiles": percentiles,
        "skewness": float(rendimientos.skew()),
        "curtosis_exceso": float(rendimientos.kurt()),
        "rendimientos": rendimientos,
        "precios": precios,
    }


# ============================================================================
# NIVEL 3 · RIESGO CUANTITATIVO AVANZADO
# ============================================================================

def calcular_riesgo_cuantitativo(
    historico,
    nivel_confianza=95,
    usar_ajustado=True,
    ventana_volatilidad=21,
):
    """
    Métricas históricas de riesgo sobre rendimientos diarios.

    VaR y CVaR se calculan mediante distribución empírica histórica.
    No se supone normalidad y no se realizan predicciones.
    """
    import numpy as np
    import pandas as pd

    vacio = {
        "valido": False,
        "mensaje": "No hay datos suficientes.",
    }

    if historico is None or len(historico) < 30:
        return vacio

    df = historico.copy()

    columna = None

    if usar_ajustado and "Adj Close" in df.columns:
        s = pd.to_numeric(
            df["Adj Close"],
            errors="coerce",
        )

        if s.notna().sum() >= 30:
            columna = "Adj Close"

    if columna is None and "Close" in df.columns:
        columna = "Close"

    if columna is None:
        return vacio

    precios = pd.to_numeric(
        df[columna],
        errors="coerce",
    ).dropna()

    precios = precios[precios > 0]

    rendimientos = precios.pct_change().dropna()

    rendimientos = rendimientos.replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    if len(rendimientos) < 20:
        return vacio

    confianza = float(nivel_confianza)

    if confianza <= 50 or confianza >= 100:
        confianza = 95.0

    alpha = (100.0 - confianza) / 100.0

    # ------------------------------------------------------------
    # VaR histórico
    # ------------------------------------------------------------

    cuantile = float(
        rendimientos.quantile(alpha)
    )

    var_historico = max(0.0, -cuantile)

    cola = rendimientos[
        rendimientos <= cuantile
    ]

    if len(cola):
        cvar_historico = max(
            0.0,
            -float(cola.mean()),
        )
    else:
        cvar_historico = var_historico

    # ------------------------------------------------------------
    # Downside deviation
    # Respecto a rendimiento objetivo 0 diario.
    # ------------------------------------------------------------

    downside = np.minimum(
        rendimientos.to_numpy(dtype=float),
        0.0,
    )

    downside_dev_diaria = float(
        np.sqrt(np.mean(downside ** 2))
    )

    downside_dev_anual = (
        downside_dev_diaria
        * np.sqrt(252.0)
    )

    # ------------------------------------------------------------
    # Semidesviación
    # Solo observaciones por debajo de la media de la muestra.
    # ------------------------------------------------------------

    media = float(rendimientos.mean())

    debajo_media = rendimientos[
        rendimientos < media
    ]

    if len(debajo_media) >= 2:
        semidesviacion = float(
            np.sqrt(
                np.mean(
                    (
                        debajo_media.to_numpy(dtype=float)
                        - media
                    ) ** 2
                )
            )
        )
    else:
        semidesviacion = 0.0

    # ------------------------------------------------------------
    # Grandes caídas
    # ------------------------------------------------------------

    umbrales = {}

    for pct in [2, 3, 5]:
        limite = -pct / 100.0

        eventos = rendimientos[
            rendimientos <= limite
        ]

        cantidad = int(len(eventos))

        umbrales[pct] = {
            "cantidad": cantidad,
            "porcentaje": (
                cantidad
                / len(rendimientos)
                * 100.0
            ),
            "media": (
                float(eventos.mean())
                if cantidad
                else None
            ),
        }

    # ------------------------------------------------------------
    # Volatilidad móvil anualizada
    # ------------------------------------------------------------

    ventana = int(ventana_volatilidad)

    if ventana < 5:
        ventana = 21

    rolling_vol = (
        rendimientos
        .rolling(ventana)
        .std(ddof=1)
        * np.sqrt(252.0)
    ).dropna()

    vol_actual = (
        float(rolling_vol.iloc[-1])
        if len(rolling_vol)
        else None
    )

    vol_media = (
        float(rolling_vol.mean())
        if len(rolling_vol)
        else None
    )

    vol_max = (
        float(rolling_vol.max())
        if len(rolling_vol)
        else None
    )

    fecha_vol_max = (
        rolling_vol.idxmax()
        if len(rolling_vol)
        else None
    )

    # ------------------------------------------------------------
    # Peores sesiones
    # ------------------------------------------------------------

    peores = (
        rendimientos
        .nsmallest(min(10, len(rendimientos)))
    )

    return {
        "valido": True,
        "columna_precio": columna,
        "observaciones": int(len(rendimientos)),
        "inicio": precios.index.min(),
        "fin": precios.index.max(),
        "nivel_confianza": confianza,
        "var_historico": var_historico,
        "cvar_historico": cvar_historico,
        "cuantil_var": cuantile,
        "downside_dev_diaria": downside_dev_diaria,
        "downside_dev_anual": downside_dev_anual,
        "semidesviacion_diaria": semidesviacion,
        "umbrales": umbrales,
        "rolling_vol": rolling_vol,
        "vol_actual": vol_actual,
        "vol_media": vol_media,
        "vol_max": vol_max,
        "fecha_vol_max": fecha_vol_max,
        "peores_sesiones": peores,
        "rendimientos": rendimientos,
    }


# ============================================================================
# NIVEL 3 · ANÁLISIS TEMPORAL
# ============================================================================

def calcular_analisis_temporal(
    historico,
    usar_ajustado=True,
):
    """
    Analiza rendimientos históricos por mes y año,
    estacionalidad descriptiva y rachas mensuales.

    No realiza predicciones.
    """
    import numpy as np
    import pandas as pd

    vacio = {
        "valido": False,
        "mensaje": "No hay datos suficientes.",
    }

    if historico is None or len(historico) < 30:
        return vacio

    df = historico.copy()

    columna = None

    if usar_ajustado and "Adj Close" in df.columns:
        s = pd.to_numeric(
            df["Adj Close"],
            errors="coerce",
        )

        if s.notna().sum() >= 30:
            columna = "Adj Close"

    if columna is None and "Close" in df.columns:
        columna = "Close"

    if columna is None:
        return vacio

    precios = pd.to_numeric(
        df[columna],
        errors="coerce",
    ).dropna()

    precios = precios[precios > 0].sort_index()

    if len(precios) < 30:
        return vacio

    # Último precio disponible de cada mes y año.
    mensual_precios = precios.resample("ME").last().dropna()
    anual_precios = precios.resample("YE").last().dropna()

    mensual = mensual_precios.pct_change().dropna()
    anual = anual_precios.pct_change().dropna()

    if len(mensual) < 2:
        return vacio

    # ------------------------------------------------------------
    # Matriz año × mes
    # ------------------------------------------------------------

    mensual_df = pd.DataFrame(
        {
            "Fecha": mensual.index,
            "Rendimiento": mensual.values,
        }
    )

    mensual_df["Año"] = mensual_df["Fecha"].dt.year
    mensual_df["Mes"] = mensual_df["Fecha"].dt.month

    heatmap = mensual_df.pivot(
        index="Año",
        columns="Mes",
        values="Rendimiento",
    )

    # ------------------------------------------------------------
    # Estadísticas por mes del calendario
    # ------------------------------------------------------------

    por_mes = (
        mensual_df
        .groupby("Mes")["Rendimiento"]
        .agg(["mean", "median", "count"])
    )

    positivos_mes = (
        mensual_df
        .assign(
            Positivo=mensual_df["Rendimiento"] > 0
        )
        .groupby("Mes")["Positivo"]
        .mean()
    )

    por_mes["pct_positivo"] = positivos_mes * 100.0

    # ------------------------------------------------------------
    # Positivos / negativos
    # ------------------------------------------------------------

    meses_positivos = int((mensual > 0).sum())
    meses_negativos = int((mensual < 0).sum())
    meses_neutros = int((mensual == 0).sum())

    total_meses = int(len(mensual))

    # ------------------------------------------------------------
    # Mejores y peores meses
    # ------------------------------------------------------------

    mejores = mensual.nlargest(
        min(5, len(mensual))
    )

    peores = mensual.nsmallest(
        min(5, len(mensual))
    )

    # ------------------------------------------------------------
    # Rachas mensuales
    # ------------------------------------------------------------

    mejor_racha = 0
    peor_racha = 0
    actual_pos = 0
    actual_neg = 0

    inicio_mejor = None
    fin_mejor = None
    inicio_peor = None
    fin_peor = None

    inicio_pos_actual = None
    inicio_neg_actual = None

    for fecha, valor in mensual.items():

        if valor > 0:
            if actual_pos == 0:
                inicio_pos_actual = fecha

            actual_pos += 1
            actual_neg = 0
            inicio_neg_actual = None

            if actual_pos > mejor_racha:
                mejor_racha = actual_pos
                inicio_mejor = inicio_pos_actual
                fin_mejor = fecha

        elif valor < 0:
            if actual_neg == 0:
                inicio_neg_actual = fecha

            actual_neg += 1
            actual_pos = 0
            inicio_pos_actual = None

            if actual_neg > peor_racha:
                peor_racha = actual_neg
                inicio_peor = inicio_neg_actual
                fin_peor = fecha

        else:
            actual_pos = 0
            actual_neg = 0
            inicio_pos_actual = None
            inicio_neg_actual = None

    return {
        "valido": True,
        "columna_precio": columna,
        "inicio": precios.index.min(),
        "fin": precios.index.max(),
        "mensual": mensual,
        "anual": anual,
        "heatmap": heatmap,
        "por_mes": por_mes,
        "meses_positivos": meses_positivos,
        "meses_negativos": meses_negativos,
        "meses_neutros": meses_neutros,
        "total_meses": total_meses,
        "pct_meses_positivos": (
            meses_positivos / total_meses * 100.0
        ),
        "pct_meses_negativos": (
            meses_negativos / total_meses * 100.0
        ),
        "mejores_meses": mejores,
        "peores_meses": peores,
        "racha_positiva": mejor_racha,
        "racha_negativa": peor_racha,
        "racha_positiva_inicio": inicio_mejor,
        "racha_positiva_fin": fin_mejor,
        "racha_negativa_inicio": inicio_peor,
        "racha_negativa_fin": fin_peor,
    }


# ============================================================================
# NIVEL 3 · MONTE CARLO INDIVIDUAL
# ============================================================================

def calcular_monte_carlo(
    historico,
    dias=252,
    simulaciones=5000,
    usar_ajustado=True,
    capital_inicial=10000.0,
    seed=42,
    periodos_anuales=252,
):
    """
    Simulación Monte Carlo mediante movimiento browniano geométrico (GBM).

    Los parámetros se estiman a partir de rendimientos logarítmicos
    históricos diarios. El resultado es una simulación condicionada
    a esos supuestos, no una predicción del precio futuro.
    """
    import numpy as np
    import pandas as pd

    vacio = {
        "valido": False,
        "mensaje": "No hay datos suficientes para la simulación.",
    }

    if historico is None or len(historico) < 60:
        return vacio

    if dias < 1:
        return vacio

    if simulaciones < 100:
        return vacio

    try:
        periodos_anuales = int(periodos_anuales)
    except Exception:
        return vacio

    if periodos_anuales not in (252, 365):
        return vacio

    df = historico.copy()

    columna = None

    if usar_ajustado and "Adj Close" in df.columns:
        serie = pd.to_numeric(
            df["Adj Close"],
            errors="coerce",
        )

        if serie.notna().sum() >= 60:
            columna = "Adj Close"

    if columna is None and "Close" in df.columns:
        columna = "Close"

    if columna is None:
        return vacio

    precios = pd.to_numeric(
        df[columna],
        errors="coerce",
    ).dropna()

    precios = precios[
        np.isfinite(precios) & (precios > 0)
    ].sort_index()

    if len(precios) < 60:
        return vacio

    log_returns = np.log(
        precios / precios.shift(1)
    ).replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    if len(log_returns) < 30:
        return vacio

    mu_log_diario = float(
        log_returns.mean()
    )

    sigma_diaria = float(
        log_returns.std(ddof=1)
    )

    if (
        not np.isfinite(mu_log_diario)
        or not np.isfinite(sigma_diaria)
        or sigma_diaria <= 0
    ):
        return vacio

    precio_inicial = float(
        precios.iloc[-1]
    )

    rng = np.random.default_rng(seed)

    z = rng.standard_normal(
        size=(dias, simulaciones)
    )

    # Si log(S_t/S_t-1) ~ N(mu_log, sigma),
    # acumulamos directamente esos incrementos.
    incrementos = (
        mu_log_diario
        + sigma_diaria * z
    )

    log_acumulado = np.cumsum(
        incrementos,
        axis=0,
    )

    trayectorias = (
        precio_inicial
        * np.exp(log_acumulado)
    )

    finales = trayectorias[-1, :]

    if not np.all(np.isfinite(finales)):
        return vacio

    percentiles = {
        5: float(np.percentile(finales, 5)),
        25: float(np.percentile(finales, 25)),
        50: float(np.percentile(finales, 50)),
        75: float(np.percentile(finales, 75)),
        95: float(np.percentile(finales, 95)),
    }

    media_final = float(
        np.mean(finales)
    )

    mediana_final = float(
        np.median(finales)
    )

    prob_sobre_inicial = float(
        np.mean(finales > precio_inicial)
    )

    prob_bajo_inicial = float(
        np.mean(finales < precio_inicial)
    )

    prob_perdida_10 = float(
        np.mean(
            finales
            <= precio_inicial * 0.90
        )
    )

    prob_ganancia_10 = float(
        np.mean(
            finales
            >= precio_inicial * 1.10
        )
    )

    capital_inicial = float(
        max(capital_inicial, 0.0)
    )

    multiplicadores = (
        finales / precio_inicial
    )

    capital_final = (
        capital_inicial
        * multiplicadores
    )

    capital_percentiles = {
        p: float(
            np.percentile(
                capital_final,
                p,
            )
        )
        for p in [5, 25, 50, 75, 95]
    }

    # Para visualización guardamos como máximo 100 trayectorias.
    n_visual = min(
        100,
        simulaciones,
    )

    indices_visual = np.linspace(
        0,
        simulaciones - 1,
        n_visual,
        dtype=int,
    )

    trayectorias_visual = (
        trayectorias[:, indices_visual]
    )

    return {
        "valido": True,
        "columna_precio": columna,
        "inicio_historico": precios.index.min(),
        "fin_historico": precios.index.max(),
        "observaciones": int(
            len(log_returns)
        ),
        "precio_inicial": precio_inicial,
        "dias": int(dias),
        "simulaciones": int(simulaciones),
        "mu_log_diario": mu_log_diario,
        "sigma_diaria": sigma_diaria,
        "periodos_anuales": int(periodos_anuales),
        "retorno_log_anualizado": (
            mu_log_diario * float(periodos_anuales)
        ),
        "volatilidad_anualizada": (
            sigma_diaria * np.sqrt(float(periodos_anuales))
        ),
        "trayectorias_visual": trayectorias_visual,
        "finales": finales,
        "media_final": media_final,
        "mediana_final": mediana_final,
        "percentiles": percentiles,
        "prob_sobre_inicial": prob_sobre_inicial,
        "prob_bajo_inicial": prob_bajo_inicial,
        "prob_perdida_10": prob_perdida_10,
        "prob_ganancia_10": prob_ganancia_10,
        "capital_inicial": capital_inicial,
        "capital_final": capital_final,
        "capital_percentiles": capital_percentiles,
        "seed": int(seed),
        "modelo": "Movimiento browniano geométrico (GBM)",
    }

# ============================================================
# FINSCOPE · ESCENARIOS Y SENSIBILIDAD
# ============================================================

def calcular_escenarios_sensibilidad(
    precio_inicial,
    capital_inicial,
    horizonte_anios,
    rentabilidad_base_pct,
    volatilidad_anual_pct,
    shock_adverso_pct=None,
    shock_favorable_pct=None,
):
    """
    Construye escenarios deterministas de sensibilidad.

    No estima probabilidades ni predice precios futuros.
    Muestra cómo cambia matemáticamente el resultado cuando
    cambian los supuestos de rentabilidad anual.

    La volatilidad histórica se utiliza para proponer, cuando
    no se especifican shocks manuales, una separación razonable
    entre escenarios alrededor del supuesto base.
    """

    import math

    precio_inicial = float(precio_inicial)
    capital_inicial = float(capital_inicial)
    horizonte_anios = float(horizonte_anios)
    rentabilidad_base_pct = float(rentabilidad_base_pct)
    volatilidad_anual_pct = float(volatilidad_anual_pct)

    if precio_inicial <= 0:
        raise ValueError("El precio inicial debe ser positivo.")

    if capital_inicial <= 0:
        raise ValueError("El capital inicial debe ser positivo.")

    if horizonte_anios <= 0:
        raise ValueError("El horizonte debe ser positivo.")

    if volatilidad_anual_pct < 0:
        raise ValueError("La volatilidad no puede ser negativa.")

    # Si el usuario no especifica shocks, se utiliza una banda
    # basada en una volatilidad anual histórica.
    if shock_adverso_pct is None:
        shock_adverso_pct = -volatilidad_anual_pct

    if shock_favorable_pct is None:
        shock_favorable_pct = volatilidad_anual_pct

    shock_adverso_pct = float(shock_adverso_pct)
    shock_favorable_pct = float(shock_favorable_pct)

    retorno_adverso = rentabilidad_base_pct + shock_adverso_pct
    retorno_base = rentabilidad_base_pct
    retorno_favorable = rentabilidad_base_pct + shock_favorable_pct

    def resultado(retorno_anual_pct):
        tasa = retorno_anual_pct / 100.0

        # Una tasa <= -100% no puede componerse matemáticamente
        # durante horizontes fraccionarios. Se limita a pérdida total.
        if tasa <= -1:
            factor = 0.0
        else:
            factor = (1.0 + tasa) ** horizonte_anios

        precio_final = precio_inicial * factor
        capital_final = capital_inicial * factor
        resultado_capital = capital_final - capital_inicial
        rentabilidad_acumulada_pct = (factor - 1.0) * 100.0

        return {
            "rentabilidad_anual_pct": retorno_anual_pct,
            "factor": factor,
            "precio_final": precio_final,
            "capital_final": capital_final,
            "resultado_capital": resultado_capital,
            "rentabilidad_acumulada_pct": rentabilidad_acumulada_pct,
        }

    escenarios = {
        "Adverso": resultado(retorno_adverso),
        "Base": resultado(retorno_base),
        "Favorable": resultado(retorno_favorable),
    }

    # Matriz alrededor del supuesto base.
    # Cinco niveles simétricos en función de volatilidad.
    multiplicadores = [-1.0, -0.5, 0.0, 0.5, 1.0]

    sensibilidad = []

    for mult in multiplicadores:
        variacion = volatilidad_anual_pct * mult
        retorno = rentabilidad_base_pct + variacion
        res = resultado(retorno)

        sensibilidad.append(
            {
                "multiplicador_volatilidad": mult,
                "variacion_pct": variacion,
                "rentabilidad_anual_pct": retorno,
                "precio_final": res["precio_final"],
                "capital_final": res["capital_final"],
                "rentabilidad_acumulada_pct":
                    res["rentabilidad_acumulada_pct"],
            }
        )

    return {
        "precio_inicial": precio_inicial,
        "capital_inicial": capital_inicial,
        "horizonte_anios": horizonte_anios,
        "rentabilidad_base_pct": rentabilidad_base_pct,
        "volatilidad_anual_pct": volatilidad_anual_pct,
        "shock_adverso_pct": shock_adverso_pct,
        "shock_favorable_pct": shock_favorable_pct,
        "escenarios": escenarios,
        "sensibilidad": sensibilidad,
        "metodologia":
            "Capitalización compuesta bajo escenarios deterministas.",
    }


# =============================================================================
# FINSCOPE · QUANT PRO
# Motor cuantitativo centralizado
# =============================================================================

def _quant_periodos_anuales(ticker: str) -> int:
    """
    Convención de anualización.

    Criptomonedas:
        365 días.

    Resto de activos:
        252 sesiones.

    Para métricas conjuntas entre activos con calendarios diferentes,
    se utiliza posteriormente la frecuencia de las observaciones comunes.
    """
    ticker = str(ticker or "").strip().upper()

    if ticker.endswith("-USD"):
        return 365

    return 252


def _quant_preparar_serie(
    historico: pd.DataFrame,
    tipo_serie: str = "ajustada",
) -> tuple[pd.Series, str]:
    """
    Devuelve una serie limpia de precios.

    tipo_serie:
        'ajustada' -> Adj Close si existe y tiene datos.
        'precio'   -> Close.

    Nunca mezcla ambas columnas dentro de una misma serie.
    """

    if historico is None or historico.empty:
        return pd.Series(dtype=float), "Close"

    columna = "Close"

    if (
        str(tipo_serie).lower() == "ajustada"
        and "Adj Close" in historico.columns
        and historico["Adj Close"].notna().sum() >= 2
    ):
        columna = "Adj Close"

    serie = historico[columna]

    if isinstance(serie, pd.DataFrame):
        serie = serie.iloc[:, 0]

    serie = pd.to_numeric(
        serie,
        errors="coerce",
    ).dropna()

    if isinstance(serie.index, pd.DatetimeIndex):
        if serie.index.tz is not None:
            serie.index = serie.index.tz_localize(None)

    serie = serie[
        ~serie.index.duplicated(keep="last")
    ].sort_index()

    serie = serie[
        serie > 0
    ]

    return serie.astype(float), columna


def _quant_rendimientos(
    precios: pd.Series,
) -> pd.Series:

    if precios is None or len(precios) < 2:
        return pd.Series(dtype=float)

    return (
        precios
        .pct_change(fill_method=None)
        .replace([float("inf"), float("-inf")], float("nan"))
        .dropna()
    )


def _quant_downside_deviation(
    rendimientos: pd.Series,
    periodos_anuales: int,
    objetivo_periodico: float = 0.0,
) -> float | None:
    """
    Downside deviation respecto a un MAR/objetivo.

    sqrt(mean(min(r - MAR, 0)^2)) * sqrt(N)
    """
    import numpy as np

    if rendimientos is None or rendimientos.empty:
        return None

    exceso = (
        rendimientos.astype(float)
        - float(objetivo_periodico)
    )

    downside = np.minimum(
        exceso.to_numpy(),
        0.0,
    )

    valor = (
        np.sqrt(
            np.mean(
                np.square(downside)
            )
        )
        * np.sqrt(periodos_anuales)
    )

    return float(valor)


def _quant_drawdown(
    precios: pd.Series,
) -> dict:
    """
    Calcula drawdown sobre la propia serie del activo.

    No depende del calendario del activo comparado.
    """
    import numpy as np

    resultado = {
        "max_drawdown": None,
        "fecha_pico": None,
        "fecha_valle": None,
        "fecha_recuperacion": None,
        "duracion_dias": None,
        "serie_drawdown": pd.Series(dtype=float),
    }

    if precios is None or len(precios) < 2:
        return resultado

    maximos = precios.cummax()

    dd = (
        precios / maximos
        - 1.0
    )

    valle = dd.idxmin()
    max_dd = float(dd.loc[valle])

    hasta_valle = precios.loc[:valle]

    if hasta_valle.empty:
        return resultado

    pico = hasta_valle.idxmax()
    precio_pico = float(precios.loc[pico])

    despues = precios.loc[valle:]

    recuperados = despues[
        despues >= precio_pico
    ]

    recuperacion = (
        recuperados.index[0]
        if not recuperados.empty
        else None
    )

    fin_duracion = (
        recuperacion
        if recuperacion is not None
        else precios.index[-1]
    )

    try:
        duracion = int(
            (
                pd.Timestamp(fin_duracion)
                - pd.Timestamp(pico)
            ).days
        )
    except Exception:
        duracion = None

    resultado.update(
        {
            "max_drawdown": max_dd,
            "fecha_pico": pico,
            "fecha_valle": valle,
            "fecha_recuperacion": recuperacion,
            "duracion_dias": duracion,
            "serie_drawdown": dd,
        }
    )

    return resultado


def _quant_episodios_drawdown(
    precios: pd.Series,
    max_episodios: int = 5,
) -> pd.DataFrame:
    """
    Extrae episodios de drawdown independientes.

    Un episodio comienza cuando el precio abandona un máximo histórico
    y termina cuando recupera ese máximo.
    """

    if precios is None or len(precios) < 2:
        return pd.DataFrame()

    running_max = precios.cummax()
    dd = precios / running_max - 1.0

    episodios = []

    en_dd = False
    inicio = None
    pico_precio = None
    valle = None
    peor = 0.0

    indices = list(precios.index)

    for i, fecha in enumerate(indices):

        valor_dd = float(dd.loc[fecha])

        if valor_dd < -1e-12 and not en_dd:
            en_dd = True

            if i > 0:
                inicio = indices[i - 1]
            else:
                inicio = fecha

            pico_precio = float(
                precios.loc[inicio]
            )

            valle = fecha
            peor = valor_dd

        elif en_dd:

            if valor_dd < peor:
                peor = valor_dd
                valle = fecha

            if valor_dd >= -1e-12:

                fin = fecha

                episodios.append(
                    {
                        "Pico": inicio,
                        "Valle": valle,
                        "Recuperación": fin,
                        "Drawdown": peor,
                        "Duración días": int(
                            (
                                pd.Timestamp(fin)
                                - pd.Timestamp(inicio)
                            ).days
                        ),
                        "Recuperado": True,
                    }
                )

                en_dd = False
                inicio = None
                valle = None
                peor = 0.0

    if en_dd and inicio is not None:

        fin = indices[-1]

        episodios.append(
            {
                "Pico": inicio,
                "Valle": valle,
                "Recuperación": None,
                "Drawdown": peor,
                "Duración días": int(
                    (
                        pd.Timestamp(fin)
                        - pd.Timestamp(inicio)
                    ).days
                ),
                "Recuperado": False,
            }
        )

    if not episodios:
        return pd.DataFrame()

    df = pd.DataFrame(episodios)

    return (
        df.sort_values(
            "Drawdown",
            ascending=True,
        )
        .head(max_episodios)
        .reset_index(drop=True)
    )


def _quant_metricas_individuales(
    ticker: str,
    precios: pd.Series,
    rf_anual: float = 0.0,
) -> dict:
    """
    Métricas calculadas exclusivamente con la serie propia del activo.
    """

    import numpy as np

    anual = _quant_periodos_anuales(ticker)

    ret = _quant_rendimientos(precios)

    resultado = {
        "ticker": ticker,
        "periodos_anuales": anual,
        "observaciones_precio": len(precios),
        "observaciones_rendimiento": len(ret),
        "fecha_inicio": (
            precios.index[0]
            if len(precios)
            else None
        ),
        "fecha_fin": (
            precios.index[-1]
            if len(precios)
            else None
        ),
        "precio_inicial": (
            float(precios.iloc[0])
            if len(precios)
            else None
        ),
        "precio_final": (
            float(precios.iloc[-1])
            if len(precios)
            else None
        ),
        "rendimientos": ret,
    }

    if len(precios) < 2 or ret.empty:
        return resultado

    p0 = float(precios.iloc[0])
    p1 = float(precios.iloc[-1])

    rent_acum = (
        p1 / p0
        - 1.0
    )

    dias = (
        pd.Timestamp(precios.index[-1])
        - pd.Timestamp(precios.index[0])
    ).days

    anos = dias / 365.2425

    cagr = None

    if anos > 0 and p0 > 0 and p1 > 0:
        cagr = (
            (p1 / p0) ** (1.0 / anos)
            - 1.0
        )

    media_diaria = float(ret.mean())
    mediana_diaria = float(ret.median())

    vol = float(
        ret.std(ddof=1)
        * np.sqrt(anual)
    )

    rf_periodico = (
        (1.0 + float(rf_anual))
        ** (1.0 / anual)
        - 1.0
    )

    exceso = (
        ret
        - rf_periodico
    )

    std_exceso = float(
        exceso.std(ddof=1)
    )

    sharpe = None

    if std_exceso > 0:
        sharpe = float(
            exceso.mean()
            / std_exceso
            * np.sqrt(anual)
        )

    downside = _quant_downside_deviation(
        ret,
        anual,
        rf_periodico,
    )

    exceso_anual = float(
        exceso.mean()
        * anual
    )

    sortino = None

    if downside is not None and downside > 0:
        sortino = (
            exceso_anual
            / downside
        )

    draw = _quant_drawdown(precios)

    max_dd = draw["max_drawdown"]

    calmar = None

    if (
        cagr is not None
        and max_dd is not None
        and abs(max_dd) > 0
    ):
        calmar = (
            cagr
            / abs(max_dd)
        )

    positivos = int(
        (ret > 0).sum()
    )

    negativos = int(
        (ret < 0).sum()
    )

    ceros = int(
        (ret == 0).sum()
    )

    porcentaje_positivo = float(
        (ret > 0).mean()
    )

    mejor = float(ret.max())
    peor = float(ret.min())

    fecha_mejor = ret.idxmax()
    fecha_peor = ret.idxmin()

    skew = (
        float(ret.skew())
        if len(ret) >= 3
        else None
    )

    kurt = (
        float(ret.kurt())
        if len(ret) >= 4
        else None
    )

    q05 = float(
        ret.quantile(0.05)
    )

    q01 = float(
        ret.quantile(0.01)
    )

    cola95 = ret[
        ret <= q05
    ]

    cola99 = ret[
        ret <= q01
    ]

    cvar95_raw = (
        float(cola95.mean())
        if not cola95.empty
        else q05
    )

    cvar99_raw = (
        float(cola99.mean())
        if not cola99.empty
        else q01
    )

    # Se expresan como magnitud positiva de pérdida.
    var95 = max(
        0.0,
        -q05,
    )

    var99 = max(
        0.0,
        -q01,
    )

    cvar95 = max(
        0.0,
        -cvar95_raw,
    )

    cvar99 = max(
        0.0,
        -cvar99_raw,
    )

    ganancias = ret[
        ret > 0
    ].sum()

    perdidas = abs(
        ret[
            ret < 0
        ].sum()
    )

    omega = None

    if perdidas > 0:
        omega = float(
            ganancias
            / perdidas
        )

    downside_cero = _quant_downside_deviation(
        ret,
        anual,
        0.0,
    )

    episodios = _quant_episodios_drawdown(
        precios
    )

    resultado.update(
        {
            "rentabilidad_acumulada": float(rent_acum),
            "cagr": (
                float(cagr)
                if cagr is not None
                else None
            ),
            "media_periodica": media_diaria,
            "mediana_periodica": mediana_diaria,
            "rentabilidad_media_anualizada_aritmetica": float(
                media_diaria * anual
            ),
            "volatilidad_anualizada": vol,
            "downside_deviation": downside_cero,
            "sharpe": sharpe,
            "sortino": sortino,
            "max_drawdown": max_dd,
            "fecha_pico_drawdown": draw["fecha_pico"],
            "fecha_valle_drawdown": draw["fecha_valle"],
            "fecha_recuperacion_drawdown": draw["fecha_recuperacion"],
            "duracion_drawdown_dias": draw["duracion_dias"],
            "serie_drawdown": draw["serie_drawdown"],
            "calmar": calmar,
            "sesiones_positivas": positivos,
            "sesiones_negativas": negativos,
            "sesiones_neutras": ceros,
            "porcentaje_sesiones_positivas": porcentaje_positivo,
            "mejor_sesion": mejor,
            "fecha_mejor_sesion": fecha_mejor,
            "peor_sesion": peor,
            "fecha_peor_sesion": fecha_peor,
            "skewness": skew,
            "kurtosis_exceso": kurt,
            "var_95": var95,
            "var_99": var99,
            "cvar_95": cvar95,
            "cvar_99": cvar99,
            "omega": omega,
            "episodios_drawdown": episodios,
        }
    )

    return resultado


def _quant_horizontes(
    precios: pd.Series,
) -> pd.DataFrame:
    """
    Rentabilidad por horizontes de calendario.

    Busca la observación disponible inmediatamente anterior
    o igual a la fecha objetivo.
    """

    if precios is None or len(precios) < 2:
        return pd.DataFrame()

    fin = pd.Timestamp(
        precios.index[-1]
    )

    horizontes = [
        ("1M", pd.DateOffset(months=1)),
        ("3M", pd.DateOffset(months=3)),
        ("6M", pd.DateOffset(months=6)),
        ("1A", pd.DateOffset(years=1)),
        ("3A", pd.DateOffset(years=3)),
        ("5A", pd.DateOffset(years=5)),
        ("10A", pd.DateOffset(years=10)),
    ]

    filas = []

    for nombre, offset in horizontes:

        objetivo = fin - offset

        candidatos = precios[
            precios.index <= objetivo
        ]

        if candidatos.empty:
            continue

        inicio_fecha = candidatos.index[-1]
        p0 = float(candidatos.iloc[-1])
        p1 = float(precios.iloc[-1])

        if p0 <= 0:
            continue

        rent = (
            p1 / p0
            - 1.0
        )

        dias = (
            fin
            - pd.Timestamp(inicio_fecha)
        ).days

        anos = dias / 365.2425

        cagr = None

        # Para horizontes de al menos un año calendario calculamos
        # CAGR usando el tiempo real transcurrido. No exigimos que
        # dias / 365.2425 sea >= 1 porque, por fines de semana y
        # sesiones bursátiles, un horizonte "1A" puede comenzar
        # ligeramente después de la fecha calendario exacta.
        if nombre in {"1A", "3A", "5A", "10A"} and anos > 0:
            cagr = (
                (p1 / p0)
                ** (1.0 / anos)
                - 1.0
            )

        filas.append(
            {
                "Horizonte": nombre,
                "Fecha inicial": inicio_fecha,
                "Fecha final": fin,
                "Rentabilidad": float(rent),
                "CAGR": (
                    float(cagr)
                    if cagr is not None
                    else None
                ),
            }
        )

    return pd.DataFrame(filas)


def _quant_rolling(
    ticker_a: str,
    ticker_b: str,
    ret_a: pd.Series,
    ret_b: pd.Series,
) -> dict:
    """
    Métricas dinámicas sobre rendimientos previamente calculados.

    La alineación se realiza DESPUÉS de calcular los rendimientos
    individuales.
    """
    import numpy as np

    comunes = pd.concat(
        [
            ret_a.rename("A"),
            ret_b.rename("B"),
        ],
        axis=1,
        join="inner",
    ).dropna()

    if len(comunes) < 30:
        return {
            "rendimientos_comunes": comunes,
            "ventana": None,
            "correlacion_rolling": pd.Series(dtype=float),
            "beta_rolling": pd.Series(dtype=float),
            "volatilidad_a_rolling": pd.Series(dtype=float),
            "volatilidad_b_rolling": pd.Series(dtype=float),
        }

    # Para relaciones conjuntas usamos la frecuencia del calendario
    # efectivamente compartido.
    anual_comun = min(
        _quant_periodos_anuales(ticker_a),
        _quant_periodos_anuales(ticker_b),
    )

    if len(comunes) >= anual_comun:
        ventana = anual_comun
    else:
        ventana = max(
            30,
            min(
                126,
                len(comunes) // 2,
            ),
        )

    corr_roll = (
        comunes["A"]
        .rolling(ventana)
        .corr(comunes["B"])
        .dropna()
    )

    cov_roll = (
        comunes["A"]
        .rolling(ventana)
        .cov(comunes["B"])
    )

    var_b_roll = (
        comunes["B"]
        .rolling(ventana)
        .var()
    )

    beta_roll = (
        cov_roll / var_b_roll
    ).replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    vol_a = (
        comunes["A"]
        .rolling(ventana)
        .std(ddof=1)
        * np.sqrt(anual_comun)
    ).dropna()

    vol_b = (
        comunes["B"]
        .rolling(ventana)
        .std(ddof=1)
        * np.sqrt(anual_comun)
    ).dropna()

    return {
        "rendimientos_comunes": comunes,
        "ventana": ventana,
        "periodos_anuales_comunes": anual_comun,
        "correlacion_rolling": corr_roll,
        "beta_rolling": beta_roll,
        "volatilidad_a_rolling": vol_a,
        "volatilidad_b_rolling": vol_b,
    }


def _quant_metricas_relativas(
    ticker_a: str,
    ticker_b: str,
    ret_a: pd.Series,
    ret_b: pd.Series,
    rf_anual: float = 0.0,
) -> dict:
    """
    Métricas conjuntas.

    IMPORTANTE:
    Los rendimientos se calculan primero de forma independiente.
    Solo después se alinean las fechas comunes.
    """
    import numpy as np

    comunes = pd.concat(
        [
            ret_a.rename("A"),
            ret_b.rename("B"),
        ],
        axis=1,
        join="inner",
    ).dropna()

    resultado = {
        "observaciones_comunes": len(comunes),
        "rendimientos_comunes": comunes,
    }

    if len(comunes) < 3:
        return resultado

    anual = min(
        _quant_periodos_anuales(ticker_a),
        _quant_periodos_anuales(ticker_b),
    )

    a = comunes["A"]
    b = comunes["B"]

    correlacion = float(
        a.corr(b)
    )

    covarianza_periodica = float(
        a.cov(b)
    )

    var_b = float(
        b.var(ddof=1)
    )

    beta = None

    if var_b > 0:
        beta = (
            covarianza_periodica
            / var_b
        )

    r2 = float(
        correlacion ** 2
    )

    rf_periodico = (
        (1.0 + float(rf_anual))
        ** (1.0 / anual)
        - 1.0
    )

    alpha_periodico = None
    alpha_anual = None

    if beta is not None:

        alpha_periodico = float(
            (
                a.mean()
                - rf_periodico
            )
            - beta
            * (
                b.mean()
                - rf_periodico
            )
        )

        # Anualización aritmética explícita.
        alpha_anual = float(
            alpha_periodico
            * anual
        )

    activo = (
        a - b
    )

    tracking_error = float(
        activo.std(ddof=1)
        * np.sqrt(anual)
    )

    information_ratio = None

    if tracking_error > 0:
        information_ratio = float(
            (
                activo.mean()
                * anual
            )
            / tracking_error
        )

    # Capture ratios mensuales mediante composición geométrica.
    precios_rel_a = (
        1.0 + a
    ).cumprod()

    precios_rel_b = (
        1.0 + b
    ).cumprod()

    mensual_a = (
        precios_rel_a
        .resample("ME")
        .last()
        .pct_change(fill_method=None)
        .dropna()
    )

    mensual_b = (
        precios_rel_b
        .resample("ME")
        .last()
        .pct_change(fill_method=None)
        .dropna()
    )

    mensual = pd.concat(
        [
            mensual_a.rename("A"),
            mensual_b.rename("B"),
        ],
        axis=1,
        join="inner",
    ).dropna()

    upside_capture = None
    downside_capture = None

    if not mensual.empty:

        meses_up = mensual[
            mensual["B"] > 0
        ]

        meses_down = mensual[
            mensual["B"] < 0
        ]

        # Capture ratios mensuales geométricos.
        #
        # Se seleccionan por separado los meses en los que B
        # tuvo rentabilidad positiva y negativa.
        #
        # Dentro de cada subconjunto:
        #
        # retorno anualizado =
        #     prod(1 + r) ** (12 / n) - 1
        #
        # Capture ratio =
        #     retorno anualizado A / retorno anualizado B
        #
        # De esta forma se conserva la composición temporal
        # de los rendimientos mensuales.

        def _capture_ratio_geometrico(datos):
            import numpy as np

            if datos is None or datos.empty:
                return None

            n = len(datos)

            if n < 1:
                return None

            factor_a = float(
                np.prod(
                    1.0 + datos["A"].to_numpy(dtype=float)
                )
            )

            factor_b = float(
                np.prod(
                    1.0 + datos["B"].to_numpy(dtype=float)
                )
            )

            if factor_a <= 0 or factor_b <= 0:
                return None

            retorno_a = (
                factor_a ** (12.0 / n)
                - 1.0
            )

            retorno_b = (
                factor_b ** (12.0 / n)
                - 1.0
            )

            if abs(retorno_b) < 1e-15:
                return None

            return float(
                retorno_a / retorno_b
            )

        upside_capture = _capture_ratio_geometrico(
            meses_up
        )

        downside_capture = _capture_ratio_geometrico(
            meses_down
        )

    resultado.update(
        {
            "periodos_anuales_comunes": anual,
            "correlacion": correlacion,
            "covarianza_periodica": covarianza_periodica,
            "beta": (
                float(beta)
                if beta is not None
                else None
            ),
            "r2": r2,
            "alpha_periodico": alpha_periodico,
            "alpha_anualizado": alpha_anual,
            "tracking_error": tracking_error,
            "information_ratio": information_ratio,
            "upside_capture": upside_capture,
            "downside_capture": downside_capture,
        }
    )

    return resultado


def calcular_comparacion_quant_pro(
    ticker_a: str,
    ticker_b: str,
    periodo: str = "5y",
    tipo_serie: str = "ajustada",
    rf_anual: float = 0.0,
    capital_inicial: float = 10000.0,
) -> dict:
    """
    Motor principal FinScope Quant Pro.

    Arquitectura:
        histórico individual
        -> serie individual
        -> rendimientos individuales
        -> métricas individuales
        -> alineación solo para métricas conjuntas

    Esto evita contaminar volatilidad, drawdown, Sharpe o Sortino
    por diferencias de calendario entre dos activos.
    """

    ticker_a = str(
        ticker_a or ""
    ).strip().upper()

    ticker_b = str(
        ticker_b or ""
    ).strip().upper()

    if not ticker_a or not ticker_b:
        return {
            "error": "Ticker no válido."
        }

    if ticker_a == ticker_b:
        return {
            "error": "Los activos deben ser diferentes."
        }

    hist_a = obtener_serie_rendimiento(
        ticker_a,
        periodo,
    )

    hist_b = obtener_serie_rendimiento(
        ticker_b,
        periodo,
    )

    if hist_a is None or hist_a.empty:
        return {
            "error": f"Sin histórico suficiente para {ticker_a}."
        }

    if hist_b is None or hist_b.empty:
        return {
            "error": f"Sin histórico suficiente para {ticker_b}."
        }

    precios_a, columna_a = _quant_preparar_serie(
        hist_a,
        tipo_serie,
    )

    precios_b, columna_b = _quant_preparar_serie(
        hist_b,
        tipo_serie,
    )

    if len(precios_a) < 2 or len(precios_b) < 2:
        return {
            "error": "Histórico insuficiente para calcular métricas."
        }

    metricas_a = _quant_metricas_individuales(
        ticker_a,
        precios_a,
        rf_anual,
    )

    metricas_b = _quant_metricas_individuales(
        ticker_b,
        precios_b,
        rf_anual,
    )

    ret_a = metricas_a[
        "rendimientos"
    ]

    ret_b = metricas_b[
        "rendimientos"
    ]

    relativas = _quant_metricas_relativas(
        ticker_a,
        ticker_b,
        ret_a,
        ret_b,
        rf_anual,
    )

    rolling = _quant_rolling(
        ticker_a,
        ticker_b,
        ret_a,
        ret_b,
    )

    # -------------------------------------------------------------------------
    # Gráfico base 100
    # -------------------------------------------------------------------------

    precios_comunes = pd.concat(
        [
            precios_a.rename(ticker_a),
            precios_b.rename(ticker_b),
        ],
        axis=1,
        join="inner",
    ).dropna()

    base100 = pd.DataFrame()

    if not precios_comunes.empty:

        base100 = (
            precios_comunes
            / precios_comunes.iloc[0]
            * 100.0
        )

    # -------------------------------------------------------------------------
    # Misma inversión inicial
    # -------------------------------------------------------------------------

    inversion = pd.DataFrame()

    if not base100.empty:

        inversion = (
            base100
            / 100.0
            * float(capital_inicial)
        )

    # -------------------------------------------------------------------------
    # Horizontes
    # -------------------------------------------------------------------------

    horizontes_a = _quant_horizontes(
        precios_a
    )

    horizontes_b = _quant_horizontes(
        precios_b
    )

    # -------------------------------------------------------------------------
    # Resumen descriptivo automático
    # -------------------------------------------------------------------------

    observaciones = []

    cagr_a = metricas_a.get("cagr")
    cagr_b = metricas_b.get("cagr")

    vol_a = metricas_a.get(
        "volatilidad_anualizada"
    )

    vol_b = metricas_b.get(
        "volatilidad_anualizada"
    )

    dd_a = metricas_a.get(
        "max_drawdown"
    )

    dd_b = metricas_b.get(
        "max_drawdown"
    )

    if (
        cagr_a is not None
        and cagr_b is not None
    ):

        if cagr_a > cagr_b:
            observaciones.append(
                f"{ticker_a} presenta mayor CAGR en el periodo analizado."
            )
        elif cagr_b > cagr_a:
            observaciones.append(
                f"{ticker_b} presenta mayor CAGR en el periodo analizado."
            )
        else:
            observaciones.append(
                "Ambos activos presentan un CAGR muy similar."
            )

    if (
        vol_a is not None
        and vol_b is not None
    ):

        if vol_a < vol_b:
            observaciones.append(
                f"{ticker_a} muestra menor volatilidad histórica anualizada."
            )
        elif vol_b < vol_a:
            observaciones.append(
                f"{ticker_b} muestra menor volatilidad histórica anualizada."
            )

    if (
        dd_a is not None
        and dd_b is not None
    ):

        if abs(dd_a) < abs(dd_b):
            observaciones.append(
                f"{ticker_a} sufrió un drawdown máximo menos profundo."
            )
        elif abs(dd_b) < abs(dd_a):
            observaciones.append(
                f"{ticker_b} sufrió un drawdown máximo menos profundo."
            )

    corr = relativas.get(
        "correlacion"
    )

    if corr is not None:

        if corr >= 0.8:
            texto_corr = "muy elevada"
        elif corr >= 0.5:
            texto_corr = "moderadamente alta"
        elif corr >= 0.2:
            texto_corr = "positiva moderada"
        elif corr > -0.2:
            texto_corr = "baja"
        elif corr > -0.5:
            texto_corr = "negativa moderada"
        else:
            texto_corr = "negativa elevada"

        observaciones.append(
            f"La correlación histórica entre ambos activos es "
            f"{texto_corr} ({corr:.2f})."
        )

    observaciones.append(
        "Estas observaciones describen el histórico y no constituyen "
        "una predicción ni una recomendación de inversión."
    )

    return {
        "error": None,
        "ticker_a": ticker_a,
        "ticker_b": ticker_b,
        "periodo": periodo,
        "tipo_serie_solicitado": tipo_serie,
        "columna_a": columna_a,
        "columna_b": columna_b,
        "precios_a": precios_a,
        "precios_b": precios_b,
        "metricas_a": metricas_a,
        "metricas_b": metricas_b,
        "relativas": relativas,
        "rolling": rolling,
        "base100": base100,
        "inversion": inversion,
        "capital_inicial": float(capital_inicial),
        "horizontes_a": horizontes_a,
        "horizontes_b": horizontes_b,
        "observaciones": observaciones,
    }



# =============================================================================
# FINSCOPE · CARTERAS Y DIVERSIFICACIÓN
# =============================================================================

def _portfolio_normalizar_pesos(tickers, pesos):
    """Valida y normaliza los pesos de una cartera long-only."""
    import numpy as np

    tickers = [str(t).strip().upper() for t in tickers if str(t).strip()]

    if len(tickers) < 2:
        raise ValueError("La cartera necesita al menos dos activos.")

    if len(tickers) != len(set(tickers)):
        raise ValueError("No puede haber activos duplicados.")

    if len(tickers) != len(pesos):
        raise ValueError("El número de pesos no coincide con el número de activos.")

    w = np.asarray(pesos, dtype=float)

    if not np.all(np.isfinite(w)):
        raise ValueError("Los pesos contienen valores no válidos.")

    if np.any(w < 0):
        raise ValueError("Esta versión de FinScope utiliza carteras long-only.")

    total = float(w.sum())

    if total <= 0:
        raise ValueError("La suma de los pesos debe ser mayor que cero.")

    return tickers, w / total


def _portfolio_periodos_anuales(tickers):
    """
    Convención de anualización de la cartera.

    Si todos los activos son criptomonedas se utiliza 365.
    En carteras mixtas o tradicionales se utiliza 252 porque
    las métricas conjuntas se calculan sobre fechas comunes.
    """
    periodos = [_quant_periodos_anuales(t) for t in tickers]

    if periodos and all(p == 365 for p in periodos):
        return 365

    return 252


def _portfolio_metricas_desde_rendimientos(
    rendimientos,
    riqueza,
    periodos_anuales,
    rf_anual=0.0,
):
    import numpy as np
    import pandas as pd

    r = pd.Series(rendimientos).dropna().astype(float)
    wealth = pd.Series(riqueza).dropna().astype(float)

    if r.empty or len(wealth) < 2:
        return {}

    n = int(periodos_anuales)

    media = float(r.mean())
    vol_periodica = float(r.std(ddof=1))
    vol_anual = (
        vol_periodica * np.sqrt(n)
        if np.isfinite(vol_periodica)
        else np.nan
    )

    rf_periodico = (1.0 + float(rf_anual)) ** (1.0 / n) - 1.0

    sharpe = np.nan

    if vol_periodica > 0:
        sharpe = (
            (media - rf_periodico)
            / vol_periodica
            * np.sqrt(n)
        )

    downside = np.minimum(
        r.to_numpy(dtype=float) - rf_periodico,
        0.0,
    )

    downside_dev = float(
        np.sqrt(np.mean(np.square(downside))) * np.sqrt(n)
    )

    sortino = np.nan

    if downside_dev > 0:
        sortino = (
            (media - rf_periodico) * n
            / downside_dev
        )

    acumulada = float(wealth.iloc[-1] / wealth.iloc[0] - 1.0)

    fecha_inicio = pd.Timestamp(wealth.index[0])
    fecha_fin = pd.Timestamp(wealth.index[-1])

    dias = max((fecha_fin - fecha_inicio).days, 1)
    anos = dias / 365.2425

    cagr = np.nan

    if anos > 0 and wealth.iloc[0] > 0 and wealth.iloc[-1] > 0:
        cagr = float(
            (wealth.iloc[-1] / wealth.iloc[0]) ** (1.0 / anos)
            - 1.0
        )

    running_max = wealth.cummax()
    drawdown = wealth / running_max - 1.0
    max_dd = float(drawdown.min())

    calmar = np.nan

    if max_dd < 0 and np.isfinite(cagr):
        calmar = float(cagr / abs(max_dd))

    q05 = float(r.quantile(0.05))
    q01 = float(r.quantile(0.01))

    tail95 = r[r <= q05]
    tail99 = r[r <= q01]

    var95 = max(0.0, -q05)
    var99 = max(0.0, -q01)

    cvar95 = (
        max(0.0, -float(tail95.mean()))
        if not tail95.empty
        else np.nan
    )

    cvar99 = (
        max(0.0, -float(tail99.mean()))
        if not tail99.empty
        else np.nan
    )

    return {
        "rentabilidad_acumulada": acumulada,
        "cagr": cagr,
        "volatilidad_anualizada": float(vol_anual),
        "downside_deviation": downside_dev,
        "sharpe": float(sharpe) if np.isfinite(sharpe) else np.nan,
        "sortino": float(sortino) if np.isfinite(sortino) else np.nan,
        "max_drawdown": max_dd,
        "calmar": float(calmar) if np.isfinite(calmar) else np.nan,
        "var_95": var95,
        "cvar_95": cvar95,
        "var_99": var99,
        "cvar_99": cvar99,
        "porcentaje_sesiones_positivas": float((r > 0).mean()),
        "mejor_sesion": float(r.max()),
        "peor_sesion": float(r.min()),
        "skewness": float(r.skew()),
        "kurtosis_exceso": float(r.kurt()),
        "serie_drawdown": drawdown,
        "fecha_inicio": fecha_inicio,
        "fecha_fin": fecha_fin,
        "observaciones": len(r),
    }


def calcular_cartera_quant(
    tickers,
    pesos,
    periodo="5y",
    tipo_serie="ajustada",
    rf_anual=0.0,
    capital_inicial=10000.0,
):
    """
    Motor histórico de Carteras y Diversificación de FinScope.

    Modelo de cartera:
    - long-only;
    - pesos normalizados;
    - rendimientos diarios ponderados con pesos constantes;
    - equivale a una cartera teóricamente rebalanceada a los pesos
      objetivo en cada periodo de observación;
    - no incluye comisiones, impuestos, spreads ni deslizamiento.

    Las métricas conjuntas utilizan únicamente observaciones comunes
    entre todos los activos.
    """
    import numpy as np
    import pandas as pd

    try:
        tickers, w = _portfolio_normalizar_pesos(
            tickers,
            pesos,
        )
    except Exception as exc:
        return {"error": str(exc)}

    periodo = str(periodo or "5y").lower()

    mapa_descarga = {
        "1y": "2y",
        "3y": "5y",
        "5y": "10y",
        "10y": "max",
        "max": "max",
    }

    descarga = mapa_descarga.get(periodo, "10y")

    precios_individuales = {}
    columnas = {}

    for ticker in tickers:

        historico = obtener_serie_rendimiento(
            ticker,
            descarga,
        )

        if historico is None or historico.empty:
            return {
                "error": (
                    f"No hay histórico diario suficiente para {ticker}."
                )
            }

        precios, columna = _quant_preparar_serie(
            historico,
            tipo_serie,
        )

        if precios is None or precios.empty:
            return {
                "error": (
                    f"No se ha podido preparar la serie de {ticker}."
                )
            }

        precios_individuales[ticker] = precios
        columnas[ticker] = columna

    # ---------------------------------------------------------------------
    # Recorte temporal exacto
    # ---------------------------------------------------------------------

    fecha_fin_global = min(
        serie.index.max()
        for serie in precios_individuales.values()
    )

    offsets = {
        "1y": pd.DateOffset(years=1),
        "3y": pd.DateOffset(years=3),
        "5y": pd.DateOffset(years=5),
        "10y": pd.DateOffset(years=10),
    }

    if periodo in offsets:
        objetivo = fecha_fin_global - offsets[periodo]

        for ticker in tickers:
            serie = precios_individuales[ticker]
            precios_individuales[ticker] = serie[
                serie.index >= objetivo
            ]

    # ---------------------------------------------------------------------
    # Rendimientos individuales primero
    # ---------------------------------------------------------------------

    rendimientos_individuales = {}

    for ticker in tickers:
        rendimientos_individuales[ticker] = _quant_rendimientos(
            precios_individuales[ticker]
        )

    # ---------------------------------------------------------------------
    # Alineación únicamente para cálculos conjuntos
    # ---------------------------------------------------------------------

    retornos = pd.concat(
        [
            rendimientos_individuales[t].rename(t)
            for t in tickers
        ],
        axis=1,
        join="inner",
    ).dropna()

    if len(retornos) < 30:
        return {
            "error": (
                "No existen suficientes observaciones comunes "
                "para analizar la cartera."
            )
        }

    periodos_anuales = _portfolio_periodos_anuales(
        tickers
    )

    # ---------------------------------------------------------------------
    # Cartera de pesos constantes
    # ---------------------------------------------------------------------

    retorno_cartera = pd.Series(
        retornos.to_numpy(dtype=float) @ w,
        index=retornos.index,
        name="Cartera",
    )

    riqueza = (
        1.0 + retorno_cartera
    ).cumprod()

    # Se añade una base inicial 1 justo antes de la primera rentabilidad
    # únicamente para que la rentabilidad acumulada parta correctamente.
    riqueza_base = pd.concat(
        [
            pd.Series(
                [1.0],
                index=[
                    retornos.index[0]
                    - pd.Timedelta(nanoseconds=1)
                ],
            ),
            riqueza,
        ]
    )

    # Evolución monetaria incluyendo explícitamente el momento
    # inicial anterior a la primera rentabilidad observada.
    # Esto permite que el gráfico empiece exactamente en el capital
    # indicado por el usuario.
    capital = (
        riqueza_base
        * float(capital_inicial)
    ).rename("Cartera")

    metricas = _portfolio_metricas_desde_rendimientos(
        retorno_cartera,
        riqueza_base,
        periodos_anuales,
        rf_anual,
    )

    # ---------------------------------------------------------------------
    # Correlación y covarianza
    # ---------------------------------------------------------------------

    correlacion = retornos.corr()
    cov_periodica = retornos.cov()
    cov_anualizada = cov_periodica * periodos_anuales

    # ---------------------------------------------------------------------
    # Volatilidad matricial independiente
    # ---------------------------------------------------------------------

    vol_matricial = float(
        np.sqrt(
            w.T
            @ cov_anualizada.to_numpy(dtype=float)
            @ w
        )
    )

    # ---------------------------------------------------------------------
    # Contribución al riesgo
    # ---------------------------------------------------------------------

    sigma = cov_anualizada.to_numpy(dtype=float)

    var_cartera = float(
        w.T @ sigma @ w
    )

    contribucion_riesgo = pd.DataFrame(
        {
            "Activo": tickers,
            "Peso": w,
        }
    )

    if var_cartera > 0:

        sigma_w = sigma @ w

        contrib_var = (
            w * sigma_w / var_cartera
        )

        contrib_vol_abs = (
            w * sigma_w / np.sqrt(var_cartera)
        )

        contribucion_riesgo[
            "Contribución riesgo"
        ] = contrib_var

        contribucion_riesgo[
            "Contribución volatilidad"
        ] = contrib_vol_abs

    else:

        contribucion_riesgo[
            "Contribución riesgo"
        ] = np.nan

        contribucion_riesgo[
            "Contribución volatilidad"
        ] = np.nan

    # ---------------------------------------------------------------------
    # Rentabilidad media anualizada por activo y contribución
    # ---------------------------------------------------------------------

    media_anual_activos = (
        retornos.mean()
        * periodos_anuales
    )

    contribucion_rentabilidad = (
        media_anual_activos.to_numpy(dtype=float)
        * w
    )

    contribucion_riesgo[
        "Rentabilidad media anualizada"
    ] = [
        float(media_anual_activos[t])
        for t in tickers
    ]

    contribucion_riesgo[
        "Contribución rentabilidad"
    ] = contribucion_rentabilidad

    # ---------------------------------------------------------------------
    # Beneficio histórico de diversificación
    # ---------------------------------------------------------------------

    vols_individuales = (
        retornos.std(ddof=1)
        * np.sqrt(periodos_anuales)
    )

    vol_media_ponderada = float(
        np.dot(
            w,
            vols_individuales.loc[tickers].to_numpy(dtype=float),
        )
    )

    beneficio_diversificacion = (
        vol_media_ponderada - vol_matricial
    )

    ratio_diversificacion = np.nan

    if vol_matricial > 0:
        ratio_diversificacion = (
            vol_media_ponderada / vol_matricial
        )

    # ---------------------------------------------------------------------
    # Evolución comparable cartera + activos
    # ---------------------------------------------------------------------

    base100_activos = (
        (1.0 + retornos).cumprod()
        * 100.0
    )

    base100_cartera = (
        riqueza * 100.0
    ).rename("Cartera")

    comparacion_base100 = pd.concat(
        [
            base100_cartera,
            base100_activos,
        ],
        axis=1,
    )

    # ---------------------------------------------------------------------
    # Métricas individuales sobre el MISMO calendario común
    # para comparación visual justa.
    # ---------------------------------------------------------------------

    comparacion_metricas = []

    for ticker in tickers:

        r = retornos[ticker]

        wealth = (
            1.0 + r
        ).cumprod()

        wealth_base = pd.concat(
            [
                pd.Series(
                    [1.0],
                    index=[
                        r.index[0]
                        - pd.Timedelta(nanoseconds=1)
                    ],
                ),
                wealth,
            ]
        )

        m = _portfolio_metricas_desde_rendimientos(
            r,
            wealth_base,
            periodos_anuales,
            rf_anual,
        )

        comparacion_metricas.append(
            {
                "Activo": ticker,
                "Peso": float(
                    w[tickers.index(ticker)]
                ),
                "Rentabilidad acumulada": m.get(
                    "rentabilidad_acumulada"
                ),
                "CAGR": m.get("cagr"),
                "Volatilidad": m.get(
                    "volatilidad_anualizada"
                ),
                "Sharpe": m.get("sharpe"),
                "Sortino": m.get("sortino"),
                "Max Drawdown": m.get(
                    "max_drawdown"
                ),
            }
        )

    comparacion_metricas.insert(
        0,
        {
            "Activo": "Cartera",
            "Peso": 1.0,
            "Rentabilidad acumulada": metricas.get(
                "rentabilidad_acumulada"
            ),
            "CAGR": metricas.get("cagr"),
            "Volatilidad": metricas.get(
                "volatilidad_anualizada"
            ),
            "Sharpe": metricas.get("sharpe"),
            "Sortino": metricas.get("sortino"),
            "Max Drawdown": metricas.get(
                "max_drawdown"
            ),
        },
    )

    comparacion_metricas = pd.DataFrame(
        comparacion_metricas
    )

    return {
        "error": None,
        "tickers": tickers,
        "pesos": pd.Series(
            w,
            index=tickers,
            name="Peso",
        ),
        "periodo": periodo,
        "columnas": columnas,
        "periodos_anuales": periodos_anuales,
        "observaciones_comunes": len(retornos),
        "rendimientos": retornos,
        "retorno_cartera": retorno_cartera,
        "metricas": metricas,
        "correlacion": correlacion,
        "covarianza_periodica": cov_periodica,
        "covarianza_anualizada": cov_anualizada,
        "volatilidad_matricial": vol_matricial,
        "contribucion": contribucion_riesgo,
        "volatilidad_media_ponderada": vol_media_ponderada,
        "beneficio_diversificacion": beneficio_diversificacion,
        "ratio_diversificacion": (
            float(ratio_diversificacion)
            if np.isfinite(ratio_diversificacion)
            else np.nan
        ),
        "base100": comparacion_base100,
        "capital": capital,
        "capital_inicial": float(capital_inicial),
        "comparacion_metricas": comparacion_metricas,
    }

