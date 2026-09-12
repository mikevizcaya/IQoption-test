import numpy as np
import pandas as pd


JAPY2_PERIOD = 7
JAPY2_MAX_DIFF_PCT = 20.0  # 20% del OPEN, criterio histórico JAPY2


def _lsma_last_7(values):
    """
    LSMA/regresión lineal de 7 valores evaluada en x=7.
    values debe contener exactamente 7 observaciones.
    """
    y = np.asarray(values, dtype=float)
    x = np.arange(1, 8, dtype=float)

    n = 7.0
    sumx = x.sum()          # 28
    sumx2 = (x * x).sum()  # 140
    sumy = y.sum()
    sumxy = (x * y).sum()

    denominator = n * sumx2 - sumx * sumx
    b = (n * sumxy - sumx * sumy) / denominator
    a = (sumy - b * sumx) / n

    return a + b * 7.0


def _wma_last_7(values):
    """
    WMA7 manual con pesos 1..7.
    """
    y = np.asarray(values, dtype=float)
    weights = np.arange(1, 8, dtype=float)
    return float((y * weights).sum() / weights.sum())


def calculate_japy2(df: pd.DataFrame) -> pd.DataFrame:
    """
    Añade al DataFrame los cálculos JAPY2.

    Ventana JAPY2 para OPEN actual:
      close[i-6], close[i-5], ..., close[i-1], open[i]

    Triángulo amarillo:
      1) i >= 6
      2) LSMA_OPEN - OPEN >= 0
      3) DT LSMA <= 20% del OPEN
      4) OPEN > WMA7_OPEN

    También calcula líneas de referencia de 7 periodos:
      - LSMA LOW
      - LSMA CLOSE
      - WMA CLOSE
    para usarlas posteriormente en las gráficas.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy().reset_index(drop=True)

    for col in ["open", "high", "low", "close"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    n = len(out)

    lsma_open = np.full(n, np.nan)
    wma_open = np.full(n, np.nan)
    dt_lsma_pct = np.full(n, np.nan)

    lsma_low = np.full(n, np.nan)
    lsma_close = np.full(n, np.nan)
    wma_close = np.full(n, np.nan)

    triangle = np.zeros(n, dtype=bool)

    closes = out["close"].to_numpy(dtype=float)
    opens = out["open"].to_numpy(dtype=float)
    lows = out["low"].to_numpy(dtype=float)

    for i in range(6, n):
        # JAPY2 anclado al OPEN actual:
        # 6 cierres anteriores + OPEN actual.
        open_window = [
            closes[i - 6],
            closes[i - 5],
            closes[i - 4],
            closes[i - 3],
            closes[i - 2],
            closes[i - 1],
            opens[i],
        ]

        if np.all(np.isfinite(open_window)):
            lo = _lsma_last_7(open_window)
            wo = _wma_last_7(open_window)

            lsma_open[i] = lo
            wma_open[i] = wo

            if np.isfinite(opens[i]) and opens[i] != 0:
                diff = lo - opens[i]
                pct = (diff / opens[i]) * 100.0
                dt_lsma_pct[i] = pct

                triangle[i] = (
                    diff >= 0
                    and pct <= JAPY2_MAX_DIFF_PCT
                    and opens[i] > wo
                )

        # Líneas normales de 7 periodos para el futuro gráfico estándar.
        close_window = closes[i - 6:i + 1]
        low_window = lows[i - 6:i + 1]

        if np.all(np.isfinite(close_window)):
            lsma_close[i] = _lsma_last_7(close_window)
            wma_close[i] = _wma_last_7(close_window)

        if np.all(np.isfinite(low_window)):
            lsma_low[i] = _lsma_last_7(low_window)

    out["lsma_open_7"] = lsma_open
    out["wma_open_7"] = wma_open
    out["dt_lsma_pct"] = dt_lsma_pct
    out["japy2_triangle"] = triangle

    out["lsma_low_7"] = lsma_low
    out["lsma_close_7"] = lsma_close
    out["wma_close_7"] = wma_close

    out["japy2_result"] = ""
    out.loc[
        out["japy2_triangle"] & (out["close"] > out["open"]),
        "japy2_result",
    ] = "verde"
    out.loc[
        out["japy2_triangle"] & (out["close"] < out["open"]),
        "japy2_result",
    ] = "rojo"
    out.loc[
        out["japy2_triangle"] & (out["close"] == out["open"]),
        "japy2_result",
    ] = "doji"

    return out


def japy2_events(df: pd.DataFrame, color: str) -> pd.DataFrame:
    """
    Devuelve únicamente señales JAPY2 del color solicitado.
    """
    analyzed = calculate_japy2(df)
    if analyzed.empty:
        return analyzed

    color = color.strip().lower()
    return analyzed.loc[
        analyzed["japy2_triangle"]
        & (analyzed["japy2_result"] == color)
    ].copy()
