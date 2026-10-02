import unittest
import numpy as np
import pandas as pd

from services.investment_analysis import (
    _portfolio_normalizar_pesos,
    _portfolio_metricas_desde_rendimientos,
)


class TestFinScopePortfolioEngine(unittest.TestCase):

    def test_pesos_normalizados(self):
        tickers, w = _portfolio_normalizar_pesos(
            ["AAPL", "MSFT", "GOOGL"],
            [50, 30, 20],
        )
        self.assertEqual(
            tickers,
            ["AAPL", "MSFT", "GOOGL"],
        )
        self.assertAlmostEqual(
            float(w.sum()),
            1.0,
            places=12,
        )

    def test_pesos_reescalados(self):
        _, w = _portfolio_normalizar_pesos(
            ["A", "B"],
            [2, 1],
        )
        self.assertAlmostEqual(
            w[0],
            2 / 3,
            places=12,
        )
        self.assertAlmostEqual(
            w[1],
            1 / 3,
            places=12,
        )

    def test_rechaza_peso_negativo(self):
        with self.assertRaises(ValueError):
            _portfolio_normalizar_pesos(
                ["A", "B"],
                [120, -20],
            )

    def test_rechaza_duplicados(self):
        with self.assertRaises(ValueError):
            _portfolio_normalizar_pesos(
                ["AAPL", "AAPL"],
                [50, 50],
            )

    def test_formula_retorno_cartera(self):
        r = pd.DataFrame(
            {
                "A": [0.10, -0.05],
                "B": [0.00, 0.10],
            }
        )
        w = np.array([0.6, 0.4])

        cartera = r.to_numpy() @ w

        self.assertAlmostEqual(
            cartera[0],
            0.06,
            places=12,
        )
        self.assertAlmostEqual(
            cartera[1],
            0.01,
            places=12,
        )

    def test_formula_varianza_cartera(self):
        cov = np.array(
            [
                [0.04, 0.01],
                [0.01, 0.09],
            ]
        )
        w = np.array([0.6, 0.4])

        esperado = float(
            w.T @ cov @ w
        )

        manual = (
            0.6**2 * 0.04
            + 0.4**2 * 0.09
            + 2 * 0.6 * 0.4 * 0.01
        )

        self.assertAlmostEqual(
            esperado,
            manual,
            places=12,
        )

    def test_contribuciones_riesgo_suman_uno(self):
        cov = np.array(
            [
                [0.04, 0.01],
                [0.01, 0.09],
            ]
        )
        w = np.array([0.6, 0.4])

        var = float(
            w.T @ cov @ w
        )

        contrib = (
            w * (cov @ w) / var
        )

        self.assertAlmostEqual(
            float(contrib.sum()),
            1.0,
            places=12,
        )

    def test_diversificacion_reduce_volatilidad(self):
        cov = np.array(
            [
                [0.04, 0.00],
                [0.00, 0.04],
            ]
        )
        w = np.array([0.5, 0.5])

        vol_port = np.sqrt(
            w.T @ cov @ w
        )

        vol_media = 0.20

        self.assertLess(
            vol_port,
            vol_media,
        )

    def test_metricas_sinteticas(self):
        idx = pd.date_range(
            "2024-01-01",
            periods=6,
            freq="D",
        )

        r = pd.Series(
            [0.01, -0.02, 0.03, -0.01, 0.02],
            index=idx[1:],
        )

        wealth = pd.concat(
            [
                pd.Series(
                    [1.0],
                    index=[idx[0]],
                ),
                (1 + r).cumprod(),
            ]
        )

        m = _portfolio_metricas_desde_rendimientos(
            r,
            wealth,
            252,
            0.0,
        )

        self.assertAlmostEqual(
            m["rentabilidad_acumulada"],
            float((1 + r).prod() - 1),
            places=12,
        )

        self.assertAlmostEqual(
            m["volatilidad_anualizada"],
            float(
                r.std(ddof=1)
                * np.sqrt(252)
            ),
            places=12,
        )

    def test_var_cvar_orden(self):
        idx = pd.date_range(
            "2024-01-01",
            periods=101,
            freq="D",
        )

        r = pd.Series(
            np.linspace(-0.10, 0.10, 100),
            index=idx[1:],
        )

        wealth = pd.concat(
            [
                pd.Series(
                    [1.0],
                    index=[idx[0]],
                ),
                (1 + r).cumprod(),
            ]
        )

        m = _portfolio_metricas_desde_rendimientos(
            r,
            wealth,
            252,
            0.0,
        )

        self.assertGreaterEqual(
            m["cvar_95"],
            m["var_95"],
        )

        self.assertGreaterEqual(
            m["var_99"],
            m["var_95"],
        )


if __name__ == "__main__":
    unittest.main()
