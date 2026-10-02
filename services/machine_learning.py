"""
FinScope · Machine Learning V2

Validación temporal walk-forward.

Principios:
- histórico diario expresado en EUR;
- ninguna variable utiliza información futura;
- targets futuros separados por horizonte;
- validación walk-forward con entrenamiento expansivo;
- predicciones fuera de muestra concatenadas;
- regresión y clasificación independientes;
- comparación contra baselines;
- evaluación de calibración probabilística;
- intervalo empírico obtenido exclusivamente de errores OOS;
- reentrenamiento final únicamente después de evaluar el modelo.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)

from services.investment_analysis import obtener_serie_rendimiento


HORIZONTES_ML = {
    1: "1 sesión",
    5: "1 semana",
    20: "1 mes",
    60: "3 meses",
}

FEATURES_ML = [
    "ret_1",
    "ret_2",
    "ret_5",
    "ret_10",
    "ret_20",
    "ret_60",
    "vol_5",
    "vol_10",
    "vol_20",
    "vol_60",
    "mom_5",
    "mom_20",
    "mom_60",
    "dist_ma_5",
    "dist_ma_20",
    "dist_ma_60",
    "drawdown_20",
    "drawdown_60",
    "rango_ret_20",
    "rango_ret_60",
]


def _crear_features(
    precio: pd.Series,
    periodos_anuales: int = 252,
) -> pd.DataFrame:
    try:
        periodos_anuales = int(periodos_anuales)
    except Exception:
        periodos_anuales = 252

    if periodos_anuales not in (252, 365):
        periodos_anuales = 252

    precio = (
        pd.to_numeric(precio, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    df = pd.DataFrame(index=precio.index)
    df["precio"] = precio

    retorno_diario = precio.pct_change(fill_method=None)

    for ventana in (1, 2, 5, 10, 20, 60):
        df[f"ret_{ventana}"] = precio.pct_change(
            ventana,
            fill_method=None,
        )

    for ventana in (5, 10, 20, 60):
        df[f"vol_{ventana}"] = (
            retorno_diario
            .rolling(ventana)
            .std(ddof=1)
            * np.sqrt(float(periodos_anuales))
        )

    for ventana in (5, 20, 60):
        df[f"mom_{ventana}"] = (
            precio / precio.shift(ventana) - 1.0
        )

        media = precio.rolling(ventana).mean()

        df[f"dist_ma_{ventana}"] = (
            precio / media - 1.0
        )

    df["drawdown_20"] = (
        precio / precio.rolling(20).max() - 1.0
    )

    df["drawdown_60"] = (
        precio / precio.rolling(60).max() - 1.0
    )

    df["rango_ret_20"] = (
        retorno_diario.rolling(20).max()
        - retorno_diario.rolling(20).min()
    )

    df["rango_ret_60"] = (
        retorno_diario.rolling(60).max()
        - retorno_diario.rolling(60).min()
    )

    return df.replace([np.inf, -np.inf], np.nan)


def _crear_dataset(
    precio: pd.Series,
    horizonte: int,
    periodos_anuales: int = 252,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    features = _crear_features(
        precio,
        periodos_anuales=periodos_anuales,
    )

    retorno_futuro = (
        precio.shift(-horizonte) / precio - 1.0
    )

    dataset = features.copy()
    dataset["target_retorno"] = retorno_futuro

    # Importante:
    # NaN futuro permanece NaN. No se convierte accidentalmente
    # en clase negativa.
    dataset["target_subida"] = np.where(
        retorno_futuro.notna(),
        (retorno_futuro > 0.0).astype(float),
        np.nan,
    )

    entrenamiento = dataset[
        FEATURES_ML
        + ["target_retorno", "target_subida"]
    ].dropna().copy()

    actual = (
        features[["precio"] + FEATURES_ML]
        .dropna()
        .tail(1)
        .copy()
    )

    return entrenamiento, actual


def _modelo_regresion(
    random_state: int = 42,
) -> RandomForestRegressor:

    return RandomForestRegressor(
        n_estimators=250,
        max_depth=6,
        min_samples_leaf=8,
        max_features=0.70,
        random_state=random_state,
        n_jobs=-1,
    )


def _modelo_clasificacion(
    random_state: int = 42,
) -> RandomForestClassifier:

    return RandomForestClassifier(
        n_estimators=250,
        max_depth=6,
        min_samples_leaf=8,
        max_features=0.70,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )


def _bloques_walk_forward(
    n: int,
    minimo_train: int = 300,
    bloques: int = 5,
):
    """
    Entrenamiento expansivo.

    Cada observación de test ocurre estrictamente después de
    todas las observaciones utilizadas para entrenar ese fold.
    """

    if n <= minimo_train:
        return []

    resto = n - minimo_train
    tam = max(1, resto // bloques)

    salida = []
    inicio_test = minimo_train

    while inicio_test < n:
        fin_test = min(n, inicio_test + tam)

        if fin_test - inicio_test >= 20:
            salida.append(
                (0, inicio_test, inicio_test, fin_test)
            )

        inicio_test = fin_test

    return salida


def _walk_forward(
    dataset: pd.DataFrame,
    random_state: int = 42,
) -> Dict[str, Any]:

    bloques = _bloques_walk_forward(len(dataset))

    if len(bloques) < 3:
        return {
            "error": "No hay suficientes bloques walk-forward."
        }

    reales_reg = []
    pred_reg = []

    reales_cls = []
    pred_cls = []
    prob_cls = []

    baseline_reg = []
    baseline_cls = []
    baseline_prob = []

    indices_oos = []
    detalle_folds = []

    for numero, (
        inicio_train,
        fin_train,
        inicio_test,
        fin_test,
    ) in enumerate(bloques, start=1):

        train = dataset.iloc[
            inicio_train:fin_train
        ].copy()

        test = dataset.iloc[
            inicio_test:fin_test
        ].copy()

        X_train = train[FEATURES_ML]
        X_test = test[FEATURES_ML]

        y_reg_train = train["target_retorno"]
        y_reg_test = test["target_retorno"]

        y_cls_train = (
            train["target_subida"].astype(int)
        )

        y_cls_test = (
            test["target_subida"].astype(int)
        )

        if y_cls_train.nunique() < 2:
            continue

        reg = _modelo_regresion(
            random_state + numero
        )

        cls = _modelo_clasificacion(
            random_state + numero
        )

        reg.fit(X_train, y_reg_train)
        cls.fit(X_train, y_cls_train)

        pr = reg.predict(X_test)
        pc = cls.predict(X_test)
        pp = cls.predict_proba(X_test)[:, 1]

        # Baseline de regresión:
        # mediana conocida hasta ese momento.
        base_reg_val = float(
            y_reg_train.median()
        )

        br = np.full(
            len(test),
            base_reg_val,
            dtype=float,
        )

        # Baseline de clasificación:
        # clase histórica mayoritaria conocida.
        base_cls_val = int(
            float(y_cls_train.mean()) >= 0.5
        )

        bc = np.full(
            len(test),
            base_cls_val,
            dtype=int,
        )

        # Baseline probabilístico estrictamente out-of-sample:
        # utiliza únicamente la prevalencia conocida en TRAIN.
        # Nunca usa la frecuencia observada posteriormente en TEST.
        base_prob_val = float(y_cls_train.mean())

        bp = np.full(
            len(test),
            base_prob_val,
            dtype=float,
        )

        reales_reg.extend(
            y_reg_test.to_numpy(dtype=float)
        )
        pred_reg.extend(
            np.asarray(pr, dtype=float)
        )

        reales_cls.extend(
            y_cls_test.to_numpy(dtype=int)
        )
        pred_cls.extend(
            np.asarray(pc, dtype=int)
        )
        prob_cls.extend(
            np.asarray(pp, dtype=float)
        )

        baseline_reg.extend(br)
        baseline_cls.extend(bc)
        baseline_prob.extend(bp)

        indices_oos.extend(test.index.tolist())

        detalle_folds.append({
            "fold": numero,
            "train": int(len(train)),
            "test": int(len(test)),
            "train_fin": train.index.max(),
            "test_inicio": test.index.min(),
            "test_fin": test.index.max(),
        })

    if not reales_reg:
        return {
            "error": "Walk-forward no produjo predicciones."
        }

    yr = np.asarray(reales_reg, dtype=float)
    pr = np.asarray(pred_reg, dtype=float)
    br = np.asarray(baseline_reg, dtype=float)

    yc = np.asarray(reales_cls, dtype=int)
    pc = np.asarray(pred_cls, dtype=int)
    pp = np.asarray(prob_cls, dtype=float)
    bc = np.asarray(baseline_cls, dtype=int)
    bp = np.asarray(baseline_prob, dtype=float)

    mae = float(mean_absolute_error(yr, pr))
    rmse = float(
        np.sqrt(mean_squared_error(yr, pr))
    )

    baseline_mae = float(
        mean_absolute_error(yr, br)
    )

    baseline_rmse = float(
        np.sqrt(mean_squared_error(yr, br))
    )

    accuracy = float(
        accuracy_score(yc, pc)
    )

    balanced_accuracy = float(
        balanced_accuracy_score(yc, pc)
    )

    baseline_accuracy = float(
        accuracy_score(yc, bc)
    )

    baseline_balanced_accuracy = float(
        balanced_accuracy_score(yc, bc)
    )

    if len(np.unique(yc)) == 2:
        auc = float(
            roc_auc_score(yc, pp)
        )
    else:
        auc = float("nan")

    brier = float(
        brier_score_loss(yc, pp)
    )

    # Baseline probabilístico walk-forward.
    # Cada predicción base procede exclusivamente de la frecuencia
    # observada en el TRAIN correspondiente a su fold.
    brier_baseline = float(
        brier_score_loss(yc, bp)
    )

    errores = yr - pr

    q10, q90 = np.quantile(
        errores,
        [0.10, 0.90],
    )

    return {
        "error": None,
        "oos_n": int(len(yr)),
        "folds": detalle_folds,
        "y_reg": yr,
        "pred_reg": pr,
        "y_cls": yc,
        "pred_cls": pc,
        "prob_cls": pp,
        "errores_reg": errores,
        "error_q10": float(q10),
        "error_q90": float(q90),
        "mae": mae,
        "rmse": rmse,
        "baseline_mae": baseline_mae,
        "baseline_rmse": baseline_rmse,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "baseline_accuracy": baseline_accuracy,
        "baseline_balanced_accuracy": (
            baseline_balanced_accuracy
        ),
        "roc_auc": auc,
        "brier": brier,
        "brier_baseline": brier_baseline,
        "indice_oos": indices_oos,
    }


def _calidad(
    validacion: Dict[str, Any],
) -> Dict[str, Any]:

    mae = float(validacion["mae"])
    baseline_mae = float(
        validacion["baseline_mae"]
    )

    bal = float(
        validacion["balanced_accuracy"]
    )

    auc = float(
        validacion["roc_auc"]
    )

    brier = float(
        validacion["brier"]
    )

    brier_base = float(
        validacion["brier_baseline"]
    )

    if baseline_mae > 0:
        mejora_mae = 1.0 - mae / baseline_mae
    else:
        mejora_mae = float("nan")

    if brier_base > 0:
        mejora_brier = 1.0 - brier / brier_base
    else:
        mejora_brier = float("nan")

    # Score interno de evidencia.
    # No representa una probabilidad.
    puntos = 0

    if math.isfinite(mejora_mae):
        if mejora_mae >= 0.10:
            puntos += 2
        elif mejora_mae >= 0.03:
            puntos += 1

    if math.isfinite(auc):
        if auc >= 0.60:
            puntos += 2
        elif auc >= 0.54:
            puntos += 1

    if math.isfinite(bal):
        if bal >= 0.58:
            puntos += 2
        elif bal >= 0.53:
            puntos += 1

    if math.isfinite(mejora_brier):
        if mejora_brier >= 0.05:
            puntos += 2
        elif mejora_brier > 0:
            puntos += 1

    # Penalizamos explícitamente regresión peor
    # que baseline.
    if (
        math.isfinite(mejora_mae)
        and mejora_mae <= 0
    ):
        puntos -= 2

    # Penalizamos clasificación claramente
    # sin señal.
    if (
        math.isfinite(auc)
        and auc < 0.50
    ):
        puntos -= 1

    if puntos >= 6:
        etiqueta = "alta"
    elif puntos >= 4:
        etiqueta = "moderada"
    elif puntos >= 1:
        etiqueta = "limitada"
    else:
        etiqueta = "muy baja"

    return {
        "etiqueta": etiqueta,
        "score_evidencia": int(puntos),
        "mejora_mae_vs_baseline": float(
            mejora_mae
        ),
        "mejora_brier_vs_baseline": float(
            mejora_brier
        ),
    }


def analizar_horizonte_ml(
    precio: pd.Series,
    horizonte: int,
    random_state: int = 42,
    periodos_anuales: int = 252,
) -> Dict[str, Any]:

    if horizonte not in HORIZONTES_ML:
        return {
            "error": (
                f"Horizonte no soportado: {horizonte}"
            )
        }

    dataset, actual = _crear_dataset(
        precio,
        horizonte,
        periodos_anuales=periodos_anuales,
    )

    if len(dataset) < 400:
        return {
            "error": (
                "Histórico insuficiente: "
                f"{len(dataset)} observaciones."
            )
        }

    if actual.empty:
        return {
            "error": (
                "No se pueden construir variables actuales."
            )
        }

    validacion = _walk_forward(
        dataset,
        random_state=random_state,
    )

    if validacion.get("error"):
        return validacion

    calidad = _calidad(validacion)

    # ------------------------------------------------------
    # MODELOS FINALES
    # ------------------------------------------------------
    # Solo después de obtener métricas OOS.
    # ------------------------------------------------------

    X = dataset[FEATURES_ML]
    y_reg = dataset["target_retorno"]
    y_cls = dataset["target_subida"].astype(int)

    if y_cls.nunique() < 2:
        return {
            "error": (
                "El histórico no contiene ambas clases."
            )
        }

    reg_final = _modelo_regresion(
        random_state + 1000
    )

    cls_final = _modelo_clasificacion(
        random_state + 1000
    )

    reg_final.fit(X, y_reg)
    cls_final.fit(X, y_cls)

    if actual is None or actual.empty:
        return {
            "error": (
                "No hay una observación actual válida "
                "para generar la estimación."
            )
        }

    X_actual = actual[FEATURES_ML]

    retorno_estimado = float(
        reg_final.predict(X_actual)[0]
    )

    probabilidad_subida = float(
        cls_final.predict_proba(X_actual)[0, 1]
    )

    precio_actual = float(
        actual["precio"].iloc[0]
    )

    precio_estimado = (
        precio_actual * (1.0 + retorno_estimado)
    )

    retorno_bajo = (
        retorno_estimado
        + validacion["error_q10"]
    )

    retorno_alto = (
        retorno_estimado
        + validacion["error_q90"]
    )

    precio_bajo = (
        precio_actual * (1.0 + retorno_bajo)
    )

    precio_alto = (
        precio_actual * (1.0 + retorno_alto)
    )

    if precio_bajo > precio_alto:
        precio_bajo, precio_alto = (
            precio_alto,
            precio_bajo,
        )

        retorno_bajo, retorno_alto = (
            retorno_alto,
            retorno_bajo,
        )

    importancia = pd.Series(
        reg_final.feature_importances_,
        index=FEATURES_ML,
    ).sort_values(ascending=False)

    return {
        "error": None,
        "version": "ML V2",
        "validacion": "walk-forward",
        "horizonte_sesiones": int(horizonte),
        "horizonte_nombre": HORIZONTES_ML[
            horizonte
        ],
        "observaciones": int(len(dataset)),
        "observaciones_oos": int(
            validacion["oos_n"]
        ),
        "numero_folds": int(
            len(validacion["folds"])
        ),
        "folds": validacion["folds"],
        "fecha_inicio": dataset.index.min(),
        "fecha_fin": dataset.index.max(),
        "precio_actual_eur": precio_actual,
        "retorno_estimado": retorno_estimado,
        "precio_estimado_eur": precio_estimado,
        "probabilidad_subida": (
            probabilidad_subida
        ),
        "intervalo_retorno_80": (
            float(retorno_bajo),
            float(retorno_alto),
        ),
        "intervalo_precio_80_eur": (
            float(precio_bajo),
            float(precio_alto),
        ),
        "metricas_regresion": {
            "mae": float(
                validacion["mae"]
            ),
            "rmse": float(
                validacion["rmse"]
            ),
            "baseline_mae": float(
                validacion["baseline_mae"]
            ),
            "baseline_rmse": float(
                validacion["baseline_rmse"]
            ),
            "mejora_mae_vs_baseline": (
                calidad[
                    "mejora_mae_vs_baseline"
                ]
            ),
        },
        "metricas_clasificacion": {
            "accuracy": float(
                validacion["accuracy"]
            ),
            "balanced_accuracy": float(
                validacion[
                    "balanced_accuracy"
                ]
            ),
            "roc_auc": float(
                validacion["roc_auc"]
            ),
            "baseline_accuracy": float(
                validacion[
                    "baseline_accuracy"
                ]
            ),
            "baseline_balanced_accuracy": (
                float(
                    validacion[
                        "baseline_balanced_accuracy"
                    ]
                )
            ),
            "brier_score": float(
                validacion["brier"]
            ),
            "baseline_brier_score": float(
                validacion[
                    "brier_baseline"
                ]
            ),
            "mejora_brier_vs_baseline": (
                calidad[
                    "mejora_brier_vs_baseline"
                ]
            ),
        },
        "calidad": calidad["etiqueta"],
        "score_evidencia": calidad[
            "score_evidencia"
        ],
        "features_principales": {
            str(k): float(v)
            for k, v in importancia.head(6).items()
        },
    }


def analizar_machine_learning(
    ticker: str,
    periodo: str = "10y",
    horizontes: Iterable[int] = (
        1,
        5,
        20,
        60,
    ),
) -> Dict[str, Any]:

    ticker = str(ticker or "").strip().upper()

    if not ticker:
        return {
            "error": "Ticker vacío.",
            "ticker": ticker,
            "resultados": {},
        }

    try:
        historico = obtener_serie_rendimiento(
            ticker,
            periodo,
        )
    except Exception as exc:
        return {
            "error": (
                f"Error obteniendo histórico: {exc}"
            ),
            "ticker": ticker,
            "resultados": {},
        }

    if historico is None or historico.empty:
        return {
            "error": "No hay histórico disponible.",
            "ticker": ticker,
            "resultados": {},
        }

    columna = (
        "Adj Close"
        if "Adj Close" in historico.columns
        else "Close"
    )

    precio = (
        pd.to_numeric(
            historico[columna],
            errors="coerce",
        )
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    if len(precio) < 500:
        return {
            "error": (
                "Histórico insuficiente para "
                "Machine Learning: "
                f"{len(precio)} observaciones."
            ),
            "ticker": ticker,
            "resultados": {},
        }

    periodos_anuales = (
        365
        if ticker.endswith("-USD")
        else 252
    )

    resultados = {}

    for horizonte in horizontes:
        h = int(horizonte)

        resultados[h] = analizar_horizonte_ml(
            precio,
            h,
            periodos_anuales=periodos_anuales,
        )

    validos = [
        resultado
        for resultado in resultados.values()
        if not resultado.get("error")
    ]

    return {
        "error": (
            None
            if validos
            else "Ningún horizonte válido."
        ),
        "version": "ML V2",
        "ticker": ticker,
        "moneda": "EUR",
        "columna_precio": columna,
        "observaciones_historicas": int(
            len(precio)
        ),
        "fecha_historico_inicio": (
            precio.index.min()
        ),
        "fecha_historico_fin": (
            precio.index.max()
        ),
        "resultados": resultados,
    }
