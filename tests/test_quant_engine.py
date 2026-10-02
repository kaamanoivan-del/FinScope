import math
import unittest

import numpy as np
import pandas as pd

from services.investment_analysis import (
    _quant_downside_deviation,
    _quant_drawdown,
    _quant_metricas_individuales,
    _quant_metricas_relativas,
    _quant_horizontes,
)


class TestFinScopeQuantEngine(unittest.TestCase):

    def test_downside_deviation_exacta(self):

        r = pd.Series(
            [0.02, -0.01, -0.03, 0.01],
            dtype=float,
        )

        esperado = (
            math.sqrt(
                (
                    0.0 ** 2
                    + (-0.01) ** 2
                    + (-0.03) ** 2
                    + 0.0 ** 2
                ) / 4
            )
            * math.sqrt(252)
        )

        obtenido = _quant_downside_deviation(
            r,
            252,
            0.0,
        )

        self.assertAlmostEqual(
            obtenido,
            esperado,
            places=12,
        )


    def test_drawdown_conocido(self):

        fechas = pd.date_range(
            "2025-01-01",
            periods=5,
            freq="D",
        )

        precios = pd.Series(
            [
                100.0,
                120.0,
                90.0,
                100.0,
                121.0,
            ],
            index=fechas,
        )

        d = _quant_drawdown(
            precios
        )

        self.assertAlmostEqual(
            d["max_drawdown"],
            -0.25,
            places=12,
        )

        self.assertEqual(
            pd.Timestamp(d["fecha_pico"]),
            fechas[1],
        )

        self.assertEqual(
            pd.Timestamp(d["fecha_valle"]),
            fechas[2],
        )

        self.assertEqual(
            pd.Timestamp(d["fecha_recuperacion"]),
            fechas[4],
        )


    def test_volatilidad_individual(self):

        fechas = pd.bdate_range(
            "2025-01-01",
            periods=121,
        )

        retornos = np.array(
            [
                0.010,
                -0.005,
                0.003,
                -0.002,
            ] * 30,
            dtype=float,
        )

        precios = pd.Series(
            100.0
            * np.cumprod(
                np.concatenate(
                    [
                        [1.0],
                        1.0 + retornos,
                    ]
                )
            ),
            index=fechas,
        )

        m = _quant_metricas_individuales(
            "AAPL",
            precios,
            0.0,
        )

        r = (
            precios
            .pct_change(fill_method=None)
            .dropna()
        )

        esperado = (
            r.std(ddof=1)
            * math.sqrt(252)
        )

        self.assertAlmostEqual(
            m["volatilidad_anualizada"],
            esperado,
            places=12,
        )


    def test_sharpe_formula(self):

        fechas = pd.bdate_range(
            "2025-01-01",
            periods=101,
        )

        retornos = np.array(
            [
                0.010,
                -0.004,
                0.006,
                -0.002,
            ] * 25,
            dtype=float,
        )

        precios = pd.Series(
            100.0
            * np.cumprod(
                np.concatenate(
                    [
                        [1.0],
                        1.0 + retornos,
                    ]
                )
            ),
            index=fechas,
        )

        m = _quant_metricas_individuales(
            "AAPL",
            precios,
            0.0,
        )

        r = (
            precios
            .pct_change(fill_method=None)
            .dropna()
        )

        esperado = (
            r.mean()
            / r.std(ddof=1)
            * math.sqrt(252)
        )

        self.assertAlmostEqual(
            m["sharpe"],
            esperado,
            places=12,
        )


    def test_sortino_formula(self):

        fechas = pd.bdate_range(
            "2025-01-01",
            periods=101,
        )

        retornos = np.array(
            [
                0.010,
                -0.004,
                0.006,
                -0.002,
            ] * 25,
            dtype=float,
        )

        precios = pd.Series(
            100.0
            * np.cumprod(
                np.concatenate(
                    [
                        [1.0],
                        1.0 + retornos,
                    ]
                )
            ),
            index=fechas,
        )

        m = _quant_metricas_individuales(
            "AAPL",
            precios,
            0.0,
        )

        r = (
            precios
            .pct_change(fill_method=None)
            .dropna()
        )

        downside = np.minimum(
            r.to_numpy(),
            0.0,
        )

        downside_anual = (
            np.sqrt(
                np.mean(
                    downside ** 2
                )
            )
            * math.sqrt(252)
        )

        esperado = (
            r.mean()
            * 252
            / downside_anual
        )

        self.assertAlmostEqual(
            m["sortino"],
            esperado,
            places=12,
        )


    def test_correlacion_beta_r2_alpha(self):

        fechas = pd.bdate_range(
            "2025-01-01",
            periods=100,
        )

        b = pd.Series(
            np.linspace(
                -0.02,
                0.02,
                100,
            ),
            index=fechas,
        )

        a = (
            0.001
            + 1.5 * b
        )

        rel = _quant_metricas_relativas(
            "AAPL",
            "MSFT",
            a,
            b,
            0.0,
        )

        self.assertAlmostEqual(
            rel["correlacion"],
            1.0,
            places=12,
        )

        self.assertAlmostEqual(
            rel["beta"],
            1.5,
            places=12,
        )

        self.assertAlmostEqual(
            rel["r2"],
            1.0,
            places=12,
        )

        self.assertAlmostEqual(
            rel["alpha_periodico"],
            0.001,
            places=12,
        )

        self.assertAlmostEqual(
            rel["alpha_anualizado"],
            0.252,
            places=12,
        )


    def test_tracking_error_information_ratio(self):

        fechas = pd.bdate_range(
            "2025-01-01",
            periods=100,
        )

        b = pd.Series(
            np.sin(
                np.linspace(
                    0,
                    8,
                    100,
                )
            )
            * 0.01,
            index=fechas,
        )

        ruido = pd.Series(
            np.cos(
                np.linspace(
                    0,
                    12,
                    100,
                )
            )
            * 0.002,
            index=fechas,
        )

        a = (
            b
            + 0.0004
            + ruido
        )

        rel = _quant_metricas_relativas(
            "AAPL",
            "MSFT",
            a,
            b,
            0.0,
        )

        activo = (
            a - b
        )

        te = (
            activo.std(ddof=1)
            * math.sqrt(252)
        )

        ir = (
            activo.mean()
            * 252
            / te
        )

        self.assertAlmostEqual(
            rel["tracking_error"],
            te,
            places=12,
        )

        self.assertAlmostEqual(
            rel["information_ratio"],
            ir,
            places=12,
        )


    def test_crypto_anualiza_365(self):

        fechas = pd.date_range(
            "2025-01-01",
            periods=101,
            freq="D",
        )

        retornos = np.array(
            [
                0.010,
                -0.006,
                0.004,
                -0.003,
            ] * 25,
            dtype=float,
        )

        precios = pd.Series(
            100.0
            * np.cumprod(
                np.concatenate(
                    [
                        [1.0],
                        1.0 + retornos,
                    ]
                )
            ),
            index=fechas,
        )

        m = _quant_metricas_individuales(
            "BTC-USD",
            precios,
            0.0,
        )

        r = (
            precios
            .pct_change(fill_method=None)
            .dropna()
        )

        esperado = (
            r.std(ddof=1)
            * math.sqrt(365)
        )

        self.assertEqual(
            m["periodos_anuales"],
            365,
        )

        self.assertAlmostEqual(
            m["volatilidad_anualizada"],
            esperado,
            places=12,
        )


    def test_horizontes_ordenados(self):

        fechas = pd.date_range(
            "2015-01-01",
            "2026-01-01",
            freq="D",
        )

        precios = pd.Series(
            np.linspace(
                100.0,
                250.0,
                len(fechas),
            ),
            index=fechas,
        )

        h = _quant_horizontes(
            precios
        )

        self.assertEqual(
            list(h["Horizonte"]),
            [
                "1M",
                "3M",
                "6M",
                "1A",
                "3A",
                "5A",
                "10A",
            ],
        )


    def test_alineacion_no_modifica_serie_individual(self):

        fechas_a = pd.bdate_range(
            "2025-01-01",
            periods=150,
        )

        fechas_b = fechas_a.delete(
            list(
                range(
                    0,
                    150,
                    7,
                )
            )
        )

        r_a = pd.Series(
            np.sin(
                np.linspace(
                    0,
                    15,
                    len(fechas_a),
                )
            )
            * 0.01,
            index=fechas_a,
        )

        r_b = pd.Series(
            np.cos(
                np.linspace(
                    0,
                    15,
                    len(fechas_b),
                )
            )
            * 0.008,
            index=fechas_b,
        )

        comunes = pd.concat(
            [
                r_a.rename("A"),
                r_b.rename("B"),
            ],
            axis=1,
            join="inner",
        ).dropna()

        self.assertEqual(
            len(r_a),
            150,
        )

        self.assertLess(
            len(comunes),
            len(r_a),
        )


if __name__ == "__main__":
    unittest.main(
        verbosity=2
    )
