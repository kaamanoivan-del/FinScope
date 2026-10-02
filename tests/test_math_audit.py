import math
import numpy as np
import pandas as pd

from services.investment_analysis import (
    calcular_dcf,
    _quant_horizontes,
    _portfolio_metricas_desde_rendimientos,
)


def assert_close(a, b, tol=1e-10):
    assert math.isfinite(float(a))
    assert abs(float(a) - float(b)) <= tol


def test_dcf_manual():
    resultado = calcular_dcf(
        fcf_base=100.0,
        crecimiento_pct=5.0,
        tasa_descuento_pct=10.0,
        crecimiento_terminal_pct=2.0,
        efectivo=50.0,
        deuda=20.0,
        acciones=10.0,
        anos=5,
    )

    assert resultado["valido"]

    crecimiento = 0.05
    descuento = 0.10
    terminal = 0.02

    fcfs = [
        100.0 * (1.0 + crecimiento) ** t
        for t in range(1, 6)
    ]

    vp_fcfs = sum(
        fcf / (1.0 + descuento) ** t
        for t, fcf in enumerate(fcfs, start=1)
    )

    terminal_value = (
        fcfs[-1] * (1.0 + terminal)
        / (descuento - terminal)
    )

    vp_terminal = terminal_value / (1.0 + descuento) ** 5
    ev = vp_fcfs + vp_terminal
    equity = ev + 50.0 - 20.0
    por_accion = equity / 10.0

    assert_close(resultado["valor_presente_fcf"], vp_fcfs)
    assert_close(resultado["valor_terminal"], terminal_value)
    assert_close(resultado["enterprise_value"], ev)
    assert_close(resultado["equity_value"], equity)
    assert_close(resultado["valor_por_accion"], por_accion)


def test_quant_horizonte_1a_tiene_cagr():
    # Más de un año de histórico para garantizar que existe
    # una observación anterior o igual a la fecha objetivo 1A.
    fechas = pd.date_range(
        "2023-12-20",
        "2025-01-01",
        freq="B",
    )

    precios = pd.Series(
        np.linspace(100.0, 120.0, len(fechas)),
        index=fechas,
    )

    tabla = _quant_horizontes(precios)

    fila = tabla.loc[
        tabla["Horizonte"] == "1A"
    ]

    assert not fila.empty

    fila = fila.iloc[0]

    assert pd.notna(fila["CAGR"])

    fecha_inicio = pd.Timestamp(
        fila["Fecha inicial"]
    )

    fecha_fin = pd.Timestamp(
        fila["Fecha final"]
    )

    p0 = float(
        precios.loc[fecha_inicio]
    )

    p1 = float(
        precios.loc[fecha_fin]
    )

    dias = (
        fecha_fin - fecha_inicio
    ).days

    anos = dias / 365.2425

    esperado = (
        (p1 / p0) ** (1.0 / anos)
        - 1.0
    )

    assert_close(
        fila["CAGR"],
        esperado,
    )


def test_portfolio_volatilidad_manual():
    idx = pd.date_range(
        "2025-01-01",
        periods=6,
        freq="B",
    )

    r = pd.Series(
        [0.01, -0.005, 0.007, 0.002, -0.003],
        index=idx[1:],
    )

    riqueza = pd.concat([
        pd.Series([1.0], index=[idx[0]]),
        (1.0 + r).cumprod(),
    ])

    resultado = _portfolio_metricas_desde_rendimientos(
        r,
        riqueza,
        252,
        0.0,
    )

    vol_esperada = r.std(ddof=1) * np.sqrt(252.0)

    assert_close(
        resultado["volatilidad_anualizada"],
        vol_esperada,
    )

    acumulada_esperada = (
        np.prod(1.0 + r.to_numpy()) - 1.0
    )

    assert_close(
        resultado["rentabilidad_acumulada"],
        acumulada_esperada,
    )


def test_portfolio_sharpe_manual():
    idx = pd.date_range(
        "2025-01-01",
        periods=6,
        freq="B",
    )

    r = pd.Series(
        [0.01, -0.005, 0.007, 0.002, -0.003],
        index=idx[1:],
    )

    riqueza = pd.concat([
        pd.Series([1.0], index=[idx[0]]),
        (1.0 + r).cumprod(),
    ])

    resultado = _portfolio_metricas_desde_rendimientos(
        r,
        riqueza,
        252,
        0.0,
    )

    esperado = (
        r.mean()
        / r.std(ddof=1)
        * np.sqrt(252.0)
    )

    assert_close(
        resultado["sharpe"],
        esperado,
    )


def test_portfolio_drawdown_manual():
    idx = pd.date_range(
        "2025-01-01",
        periods=5,
        freq="B",
    )

    riqueza = pd.Series(
        [1.0, 1.10, 0.88, 0.99, 1.20],
        index=idx,
    )

    r = riqueza.pct_change().dropna()

    resultado = _portfolio_metricas_desde_rendimientos(
        r,
        riqueza,
        252,
        0.0,
    )

    drawdown = riqueza / riqueza.cummax() - 1.0

    assert_close(
        resultado["max_drawdown"],
        drawdown.min(),
    )
