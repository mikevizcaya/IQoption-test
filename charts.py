import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots



def _apply_chart_theme(fig, theme: str = "Blanco"):
    """Aplica fondo blanco o negro sin modificar los cálculos."""
    dark = str(theme).strip().lower() == "negro"
    if dark:
        bg, fg, grid, zero, template = "#000000", "#f2f2f2", "#333333", "#555555", "plotly_dark"
    else:
        bg, fg, grid, zero, template = "#ffffff", "#222222", "#e5e5e5", "#bbbbbb", "plotly_white"

    fig.update_layout(
        template=template,
        paper_bgcolor=bg,
        plot_bgcolor=bg,
        font=dict(color=fg),
    )
    fig.update_xaxes(
        gridcolor=grid,
        zerolinecolor=zero,
        tickfont=dict(color=fg),
        title_font=dict(color=fg),
    )
    fig.update_yaxes(
        gridcolor=grid,
        zerolinecolor=zero,
        tickfont=dict(color=fg),
        title_font=dict(color=fg),
    )
    return fig

def _rma(series: pd.Series, period: int) -> pd.Series:
    """RMA de Wilder con semilla SMA, equivalente al rma() de Lua."""
    s = pd.to_numeric(series, errors="coerce").astype(float)
    out = pd.Series(np.nan, index=s.index, dtype=float)
    valid = s.dropna()
    if len(valid) < period:
        return out

    seed_indices = valid.index[:period]
    seed_index = seed_indices[-1]
    seed = valid.loc[seed_indices].mean()
    out.loc[seed_index] = seed

    alpha = 1.0 / period
    prev = seed
    started = False
    for idx in s.index:
        if idx == seed_index:
            started = True
            continue
        if not started:
            continue
        value = s.loc[idx]
        if pd.isna(value):
            out.loc[idx] = np.nan
            continue
        prev = alpha * value + (1.0 - alpha) * prev
        out.loc[idx] = prev
    return out


def _rma_first(series: pd.Series, period: int) -> pd.Series:
    """
    Variante RMA inicializada con el primer valor válido.
    Equivale al comportamiento recursivo de ewm(alpha=1/n, adjust=False)
    desde el primer dato disponible.
    """
    s = pd.to_numeric(series, errors="coerce").astype(float)
    out = pd.Series(np.nan, index=s.index, dtype=float)

    valid = s.dropna()
    if valid.empty:
        return out

    first_idx = valid.index[0]
    prev = float(valid.iloc[0])
    out.loc[first_idx] = prev
    alpha = 1.0 / period

    started = False
    for idx in s.index:
        if idx == first_idx:
            started = True
            continue
        if not started:
            continue
        value = s.loc[idx]
        if pd.isna(value):
            out.loc[idx] = np.nan
            continue
        prev = alpha * value + (1.0 - alpha) * prev
        out.loc[idx] = prev

    return out


def _rma_rolling(series: pd.Series, period: int) -> pd.Series:
    """
    Variante diagnóstica: SMA móvil pura de `period` valores.
    No pretende ser Wilder; sirve como control.
    """
    s = pd.to_numeric(series, errors="coerce").astype(float)
    return s.rolling(window=period, min_periods=period).mean()


def _rma_zero(series: pd.Series, period: int) -> pd.Series:
    """
    Variante diagnóstica: RMA recursiva iniciada en cero.
    """
    s = pd.to_numeric(series, errors="coerce").astype(float)
    out = pd.Series(np.nan, index=s.index, dtype=float)
    alpha = 1.0 / period
    prev = 0.0
    started = False
    for idx in s.index:
        value = s.loc[idx]
        if pd.isna(value):
            continue
        prev = alpha * value + (1.0 - alpha) * prev
        out.loc[idx] = prev
        started = True
    return out


def _adx_with_rma_variant(
    df: pd.DataFrame,
    rma_func,
    period: int = 7,
    di_period: int = 7,
) -> pd.Series:
    """
    Calcula ADX manteniendo fija la fórmula correcta de Wilder:
      ATR = RMA(TR)
      +DI = 100 * RMA(+DM) / ATR
      -DI = 100 * RMA(-DM) / ATR
      ADX = 100 * RMA(DX)

    Aquí sólo cambia la implementación/semilla de RMA.
    """
    out = df.copy().reset_index(drop=True)

    high = pd.to_numeric(out["high"], errors="coerce").astype(float)
    low = pd.to_numeric(out["low"], errors="coerce").astype(float)
    close = pd.to_numeric(out["close"], errors="coerce").astype(float)

    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)

    up_move = high - prev_high
    down_move = prev_low - low

    pdm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=out.index,
        dtype=float,
    )
    mdm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=out.index,
        dtype=float,
    )

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = rma_func(tr, di_period)
    smoothed_pdm = rma_func(pdm, di_period)
    smoothed_mdm = rma_func(mdm, di_period)

    atr_safe = atr.replace(0, np.nan)
    pdi = 100.0 * smoothed_pdm / atr_safe
    mdi = 100.0 * smoothed_mdm / atr_safe

    denom = (pdi + mdi).replace(0, np.nan)
    dx_ratio = (pdi - mdi).abs() / denom
    adx = 100.0 * rma_func(dx_ratio, period)
    return adx


def build_rma_comparison_table(
    analyzed: pd.DataFrame,
    source_index: int,
    radius: int = 6,
) -> pd.DataFrame:
    """
    Compara varias semillas/suavizados RMA manteniendo todo lo demás igual.
    """
    adx_sma_seed = _adx_with_rma_variant(analyzed, _rma, 7, 7)
    adx_first_seed = _adx_with_rma_variant(analyzed, _rma_first, 7, 7)
    adx_rolling = _adx_with_rma_variant(analyzed, _rma_rolling, 7, 7)
    adx_zero_seed = _adx_with_rma_variant(analyzed, _rma_zero, 7, 7)

    start = max(0, source_index - radius)
    end = min(len(analyzed) - 1, source_index + radius)

    result = pd.DataFrame({
        "Seleccionada": ["←" if idx == source_index else "" for idx in range(start, end + 1)],
        "Hora": pd.to_datetime(
            analyzed.loc[start:end, "datetime_mexico"]
        ).dt.strftime("%H:%M:%S").values,
        "RMA semilla SMA": adx_sma_seed.loc[start:end].round(4).values,
        "RMA primer valor": adx_first_seed.loc[start:end].round(4).values,
        "SMA móvil": adx_rolling.loc[start:end].round(4).values,
        "RMA semilla 0": adx_zero_seed.loc[start:end].round(4).values,
    })
    return result


def build_rma_comparison_chart(
    analyzed: pd.DataFrame,
    source_index: int,
    radius: int = 8,
    theme: str = "Blanco",
):
    """
    Gráfico comparativo de variantes de RMA.
    """
    adx_sma_seed = _adx_with_rma_variant(analyzed, _rma, 7, 7)
    adx_first_seed = _adx_with_rma_variant(analyzed, _rma_first, 7, 7)
    adx_rolling = _adx_with_rma_variant(analyzed, _rma_rolling, 7, 7)
    adx_zero_seed = _adx_with_rma_variant(analyzed, _rma_zero, 7, 7)

    start = max(0, source_index - radius)
    end = min(len(analyzed) - 1, source_index + radius)
    x = pd.to_datetime(analyzed.loc[start:end, "datetime_mexico"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=adx_sma_seed.loc[start:end],
        mode="lines+markers", name="RMA semilla SMA",
        line=dict(width=3)
    ))
    fig.add_trace(go.Scatter(
        x=x, y=adx_first_seed.loc[start:end],
        mode="lines+markers", name="RMA primer valor",
        line=dict(width=2, dash="dash")
    ))
    fig.add_trace(go.Scatter(
        x=x, y=adx_rolling.loc[start:end],
        mode="lines+markers", name="SMA móvil",
        line=dict(width=2, dash="dot")
    ))
    fig.add_trace(go.Scatter(
        x=x, y=adx_zero_seed.loc[start:end],
        mode="lines+markers", name="RMA semilla 0",
        line=dict(width=2, dash="dashdot")
    ))

    if source_index in analyzed.index:
        selected_time = pd.to_datetime(
            analyzed.loc[source_index, "datetime_mexico"]
        )
        fig.add_vline(
            x=selected_time,
            line_width=1.5,
            line_dash="dot",
            line_color="gray",
        )

    fig.add_hline(y=50, line_width=1.2, line_dash="dot", line_color="red")
    fig.add_hline(y=20, line_width=1.2, line_dash="dot", line_color="blue")

    fig.update_yaxes(range=[0, 100], title_text="ADX")
    fig.update_layout(
        title="Prueba de variantes RMA",
        height=520,
        margin=dict(l=30, r=30, t=60, b=30),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
    )
    _apply_chart_theme(fig, theme)
    return fig


def calculate_adx(df: pd.DataFrame, period: int = 7, di_period: int = 7) -> pd.DataFrame:
    """
    ADX/DMS replicando la estructura del script Lua de IQ Option:
      atr = rma(tr, di_period)
      pdi = 100 * rma(pdm / atr, di_period)
      mdi = 100 * rma(mdm / atr, di_period)
      adx = 100 * rma(abs(pdi-mdi)/(pdi+mdi), period)
    """
    out = df.copy().reset_index(drop=True)
    high = pd.to_numeric(out["high"], errors="coerce").astype(float)
    low = pd.to_numeric(out["low"], errors="coerce").astype(float)
    close = pd.to_numeric(out["close"], errors="coerce").astype(float)

    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)

    up_move = high - prev_high
    down_move = prev_low - low

    pdm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=out.index, dtype=float)
    mdm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=out.index, dtype=float)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr = _rma(tr, di_period)
    smoothed_pdm = _rma(pdm, di_period)
    smoothed_mdm = _rma(mdm, di_period)
    atr_safe = atr.replace(0, np.nan)
    pdi = 100.0 * smoothed_pdm / atr_safe
    mdi = 100.0 * smoothed_mdm / atr_safe

    denom = (pdi + mdi).replace(0, np.nan)
    dx_ratio = (pdi - mdi).abs() / denom
    adx = 100.0 * _rma(dx_ratio, period)

    out["di_plus"] = pdi
    out["di_minus"] = mdi
    out["adx"] = adx
    return out


def build_adx_components_table(
    analyzed: pd.DataFrame,
    source_index: int,
    radius: int = 4,
    adx_period: int = 7,
    di_period: int = 7,
) -> pd.DataFrame:
    """
    Desglosa los componentes internos del ADX estilo IQ/Quadcode:
    TR, ATR, +DM, -DM, +DM/ATR, -DM/ATR, +DI, -DI, DX y ADX.

    Sirve para detectar exactamente en qué etapa comienza una discrepancia.
    """
    out = analyzed.copy().reset_index(drop=True)

    high = pd.to_numeric(out["high"], errors="coerce").astype(float)
    low = pd.to_numeric(out["low"], errors="coerce").astype(float)
    close = pd.to_numeric(out["close"], errors="coerce").astype(float)

    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)

    up_move = high - prev_high
    down_move = prev_low - low

    pdm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=out.index,
        dtype=float,
    )
    mdm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=out.index,
        dtype=float,
    )

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = _rma(tr, di_period)
    atr_safe = atr.replace(0, np.nan)

    pdm_over_atr = pdm / atr_safe
    mdm_over_atr = mdm / atr_safe

    pdi = 100.0 * _rma(pdm, di_period) / atr_safe
    mdi = 100.0 * _rma(mdm, di_period) / atr_safe

    denom = (pdi + mdi).replace(0, np.nan)
    dx_ratio = (pdi - mdi).abs() / denom
    dx = 100.0 * dx_ratio
    adx = 100.0 * _rma(dx_ratio, adx_period)

    start = max(0, source_index - radius)
    end = min(len(out) - 1, source_index + radius)

    rows = pd.DataFrame({
        "Seleccionada": ["←" if idx == source_index else "" for idx in range(start, end + 1)],
        "Hora": pd.to_datetime(out.loc[start:end, "datetime_mexico"]).dt.strftime("%H:%M:%S").values,
        "High": high.loc[start:end].values,
        "Low": low.loc[start:end].values,
        "Close": close.loc[start:end].values,
        "TR": tr.loc[start:end].values,
        "ATR": atr.loc[start:end].values,
        "+DM": pdm.loc[start:end].values,
        "-DM": mdm.loc[start:end].values,
        "+DM/ATR": pdm_over_atr.loc[start:end].values,
        "-DM/ATR": mdm_over_atr.loc[start:end].values,
        "+DI": pdi.loc[start:end].values,
        "-DI": mdi.loc[start:end].values,
        "DX": dx.loc[start:end].values,
        "ADX": adx.loc[start:end].values,
    })

    numeric_cols = [
        "High", "Low", "Close", "TR", "ATR", "+DM", "-DM",
        "+DM/ATR", "-DM/ATR", "+DI", "-DI", "DX", "ADX"
    ]
    for col in numeric_cols:
        rows[col] = pd.to_numeric(rows[col], errors="coerce").round(6)

    return rows


def _ema_standard(series: pd.Series, period: int) -> pd.Series:
    """
    EMA estándar con alpha = 2/(n+1), iniciada desde el primer valor válido.
    Se usa sólo como comparación diagnóstica.
    """
    s = pd.to_numeric(series, errors="coerce").astype(float)
    return s.ewm(span=period, adjust=False, min_periods=1).mean()


def _sma(series: pd.Series, period: int) -> pd.Series:
    """SMA móvil simple de `period` valores."""
    s = pd.to_numeric(series, errors="coerce").astype(float)
    return s.rolling(window=period, min_periods=period).mean()


def _compute_dx_series(
    df: pd.DataFrame,
    di_period: int = 7,
) -> pd.Series:
    """
    Calcula DX usando la fórmula corregida:
      ATR = RMA(TR)
      +DI = 100 * RMA(+DM) / ATR
      -DI = 100 * RMA(-DM) / ATR
      DX  = 100 * abs(+DI - -DI) / (+DI + -DI)
    """
    out = df.copy().reset_index(drop=True)

    high = pd.to_numeric(out["high"], errors="coerce").astype(float)
    low = pd.to_numeric(out["low"], errors="coerce").astype(float)
    close = pd.to_numeric(out["close"], errors="coerce").astype(float)

    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)

    up_move = high - prev_high
    down_move = prev_low - low

    pdm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=out.index,
        dtype=float,
    )
    mdm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=out.index,
        dtype=float,
    )

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = _rma(tr, di_period)
    atr_safe = atr.replace(0, np.nan)

    pdi = 100.0 * _rma(pdm, di_period) / atr_safe
    mdi = 100.0 * _rma(mdm, di_period) / atr_safe

    denom = (pdi + mdi).replace(0, np.nan)
    dx = 100.0 * (pdi - mdi).abs() / denom
    return dx


def build_adx_final_stage_table(
    analyzed: pd.DataFrame,
    source_index: int,
    radius: int = 6,
    period: int = 7,
) -> pd.DataFrame:
    """
    Compara distintas formas de transformar DX en la línea final ADX.
    """
    dx = _compute_dx_series(analyzed, di_period=7)

    adx_rma7 = _rma(dx / 100.0, period) * 100.0
    adx_sma7 = _sma(dx, period)
    adx_ema7 = _ema_standard(dx, period)
    adx_rma_first = _rma_first(dx / 100.0, period) * 100.0

    start = max(0, source_index - radius)
    end = min(len(analyzed) - 1, source_index + radius)

    result = pd.DataFrame({
        "Seleccionada": ["←" if idx == source_index else "" for idx in range(start, end + 1)],
        "Hora": pd.to_datetime(
            analyzed.loc[start:end, "datetime_mexico"]
        ).dt.strftime("%H:%M:%S").values,
        "DX sin suavizar": dx.loc[start:end].round(4).values,
        "ADX RMA7": adx_rma7.loc[start:end].round(4).values,
        "ADX SMA7": adx_sma7.loc[start:end].round(4).values,
        "ADX EMA7": adx_ema7.loc[start:end].round(4).values,
        "ADX RMA7 primer valor": adx_rma_first.loc[start:end].round(4).values,
    })
    return result


def build_adx_final_stage_chart(
    analyzed: pd.DataFrame,
    source_index: int,
    radius: int = 8,
    theme: str = "Blanco",
    period: int = 7,
):
    """
    Gráfico comparativo de la última etapa DX → ADX.
    """
    dx = _compute_dx_series(analyzed, di_period=7)

    adx_rma7 = _rma(dx / 100.0, period) * 100.0
    adx_sma7 = _sma(dx, period)
    adx_ema7 = _ema_standard(dx, period)
    adx_rma_first = _rma_first(dx / 100.0, period) * 100.0

    start = max(0, source_index - radius)
    end = min(len(analyzed) - 1, source_index + radius)
    x = pd.to_datetime(analyzed.loc[start:end, "datetime_mexico"])

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x,
        y=dx.loc[start:end],
        mode="lines+markers",
        name="DX sin suavizar",
        line=dict(width=3),
    ))
    fig.add_trace(go.Scatter(
        x=x,
        y=adx_rma7.loc[start:end],
        mode="lines+markers",
        name="ADX RMA7",
        line=dict(width=2),
    ))
    fig.add_trace(go.Scatter(
        x=x,
        y=adx_sma7.loc[start:end],
        mode="lines+markers",
        name="ADX SMA7",
        line=dict(width=2, dash="dash"),
    ))
    fig.add_trace(go.Scatter(
        x=x,
        y=adx_ema7.loc[start:end],
        mode="lines+markers",
        name="ADX EMA7",
        line=dict(width=2, dash="dot"),
    ))
    fig.add_trace(go.Scatter(
        x=x,
        y=adx_rma_first.loc[start:end],
        mode="lines+markers",
        name="ADX RMA7 primer valor",
        line=dict(width=2, dash="dashdot"),
    ))

    if source_index in analyzed.index:
        selected_time = pd.to_datetime(
            analyzed.loc[source_index, "datetime_mexico"]
        )
        fig.add_vline(
            x=selected_time,
            line_width=1.5,
            line_dash="dot",
            line_color="gray",
        )

    fig.add_hline(y=50, line_width=1.2, line_dash="dot", line_color="red")
    fig.add_hline(y=20, line_width=1.2, line_dash="dot", line_color="blue")

    fig.update_yaxes(range=[0, 100], title_text="Valor")
    fig.update_layout(
        title="Prueba final: DX → ADX",
        height=560,
        margin=dict(l=30, r=30, t=60, b=30),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
    )

    _apply_chart_theme(fig, theme)
    return fig


def build_event_window(analyzed: pd.DataFrame, source_index: int, mode: str):
    """
    Construye la ventana para:
      Todo   -> 15 previas + seleccionada + 15 posteriores
      Antes  -> vela anterior como última + 14 previas
      Inicio -> 15 previas + seleccionada truncada a OPEN
      Fin    -> 15 previas + seleccionada completa
    """
    mode = mode.lower().strip()
    n = len(analyzed)

    if mode == "todo":
        start = max(0, source_index - 15)
        end = min(n - 1, source_index + 15)
        window = analyzed.loc[start:end].copy()
        selected_local_idx = source_index
        return window, selected_local_idx

    if mode == "antes":
        end = max(0, source_index - 1)
        start = max(0, end - 14)
        window = analyzed.loc[start:end].copy()
        selected_local_idx = None
        return window, selected_local_idx

    if mode in ("inicio", "fin"):
        start = max(0, source_index - 15)
        end = source_index
        window = analyzed.loc[start:end].copy()

        if mode == "inicio":
            # La vela seleccionada sólo existe hasta OPEN.
            # En este instante OPEN = HIGH = LOW = CLOSE.
            idx = source_index
            open_value = float(window.loc[idx, "open"])
            window.loc[idx, "high"] = open_value
            window.loc[idx, "low"] = open_value
            window.loc[idx, "close"] = open_value

            # En "Inicio" la vela actual sólo existe en OPEN.
            # Para los indicadores normales, CLOSE y LOW valen OPEN
            # en ese instante, por lo que se recalculan sin usar futuro.
            prev_closes = analyzed.loc[idx - 6:idx - 1, "close"].astype(float).tolist()

            if len(prev_closes) == 6:
                current_window = prev_closes + [open_value]

                # Importamos aquí para evitar duplicar las fórmulas.
                from japy2 import _lsma_last_7, _wma_last_7

                window.loc[idx, "lsma_close_7"] = _lsma_last_7(current_window)
                window.loc[idx, "wma_close_7"] = _wma_last_7(current_window)

                # LOW también es OPEN en el instante inicial.
                prev_lows = analyzed.loc[idx - 6:idx - 1, "low"].astype(float).tolist()
                if len(prev_lows) == 6:
                    low_window = prev_lows + [open_value]
                    window.loc[idx, "lsma_low_7"] = _lsma_last_7(low_window)

            # ADX/DI deben reflejar únicamente información disponible hasta OPEN.
            # Recalculamos la serie completa truncada en esta vela modificada.
            hist = analyzed.loc[:idx].copy()
            hist.loc[idx, "high"] = open_value
            hist.loc[idx, "low"] = open_value
            hist.loc[idx, "close"] = open_value

            adx_hist = calculate_adx(hist, period=7, di_period=7)
            for col in ["di_plus", "di_minus", "adx"]:
                if col in adx_hist.columns:
                    window.loc[idx, col] = adx_hist.loc[idx, col]

        selected_local_idx = source_index
        return window, selected_local_idx

    raise ValueError(f"Modo no reconocido: {mode}")



def build_recent_chart(analyzed: pd.DataFrame, asset_label: str, count: int = 30, theme: str = "Blanco"):
    """
    Gráfica inicial antes de seleccionar una señal JAPY2.
    Muestra las últimas `count` velas disponibles con los mismos
    indicadores normales usados en la gráfica del evento.
    """
    # Primero calculamos ADX sobre TODA la historia disponible.
    # Sólo después recortamos las últimas velas para dibujar.
    work = calculate_adx(analyzed, period=7, di_period=7)
    window = work.tail(count).copy()

    x = pd.to_datetime(window["datetime_mexico"])

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.56, 0.44],
    )

    fig.add_trace(
        go.Candlestick(
            x=x,
            open=window["open"],
            high=window["high"],
            low=window["low"],
            close=window["close"],
            name="Velas",
            increasing_line_color="#2ca02c",
            decreasing_line_color="#d62728",
            increasing_fillcolor="#2ca02c",
            decreasing_fillcolor="#d62728",
        ),
        row=1,
        col=1,
    )

    if "lsma_low_7" in window.columns:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=window["lsma_low_7"],
                mode="lines",
                name="LSMA LOW 7",
                line=dict(color="#ff8c00", width=2),
            ),
            row=1,
            col=1,
        )

    if "lsma_close_7" in window.columns:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=window["lsma_close_7"],
                mode="lines",
                name="LSMA CLOSE 7",
                line=dict(color="#00b050", width=2),
            ),
            row=1,
            col=1,
        )

    if "wma_close_7" in window.columns:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=window["wma_close_7"],
                mode="lines",
                name="WMA 7",
                line=dict(color="#1f9cf0", width=2),
            ),
            row=1,
            col=1,
        )

    triangles = window.loc[window["japy2_triangle"] == True]
    if not triangles.empty:
        fig.add_trace(
            go.Scatter(
                x=pd.to_datetime(triangles["datetime_mexico"]),
                y=triangles["low"] * 0.9997,
                mode="markers",
                name="JAPY2",
                marker=dict(
                    symbol="triangle-up",
                    size=11,
                    color="#ffd400",
                    line=dict(width=1, color="#8a6d00"),
                ),
            ),
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=window["di_plus"],
            mode="lines",
            name="DI+",
            line=dict(width=1.5),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=window["di_minus"],
            mode="lines",
            name="DI-",
            line=dict(width=1.5),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=window["adx"],
            mode="lines",
            name="ADX 7",
            line=dict(width=2),
        ),
        row=2,
        col=1,
    )

    fig.add_hline(
        y=50,
        line_width=1.5,
        line_dash="dot",
        line_color="red",
        row=2,
        col=1,
    )
    fig.add_hline(
        y=20,
        line_width=1.5,
        line_dash="dot",
        line_color="blue",
        row=2,
        col=1,
    )

    fig.update_yaxes(range=[0, 100], row=2, col=1, title_text="ADX")
    fig.update_yaxes(title_text="Precio", row=1, col=1)

    fig.update_layout(
        title=f"{asset_label} — Últimas {len(window)} velas",
        height=970,
        margin=dict(l=30, r=30, t=60, b=30),
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
        hovermode="x unified",
        hoverdistance=1000,
        spikedistance=1000,
    )

    # La línea vertical punteada del mouse debe abarcar simultáneamente
    # el panel de velas y el panel ADX.
    fig.update_xaxes(
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikedash="dot",
        spikethickness=1,
        spikecolor="gray",
    )

    _apply_chart_theme(fig, theme)
    return fig



def build_adx_diagnostic_chart(
    analyzed: pd.DataFrame,
    source_index: int,
    timeframe_seconds: int = 60,
    radius: int = 6,
    theme: str = "Blanco",
):
    """
    Compara el ADX calculado en tres alineaciones temporales:
    - Normal
    - Desplazado +1 vela
    - Desplazado -1 vela

    No cambia la fórmula; sólo sirve para comprobar si IQ Option
    ancla visualmente el valor a otra vela.
    """
    work = calculate_adx(analyzed, period=7, di_period=7)

    start = max(0, source_index - radius)
    end = min(len(work) - 1, source_index + radius)
    window = work.loc[start:end].copy()

    x = pd.to_datetime(window["datetime_mexico"])
    adx = window["adx"]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=adx,
            mode="lines+markers",
            name="ADX normal",
            line=dict(width=3),
            marker=dict(size=6),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=adx.shift(1),
            mode="lines+markers",
            name="ADX +1 vela",
            line=dict(width=2, dash="dash"),
            marker=dict(size=5),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=adx.shift(-1),
            mode="lines+markers",
            name="ADX -1 vela",
            line=dict(width=2, dash="dot"),
            marker=dict(size=5),
        )
    )

    if source_index in work.index:
        selected_time = pd.to_datetime(work.loc[source_index, "datetime_mexico"])
        fig.add_vline(
            x=selected_time,
            line_width=1.5,
            line_dash="dot",
            line_color="black",
        )

    fig.add_hline(y=50, line_width=1.2, line_dash="dot", line_color="red")
    fig.add_hline(y=20, line_width=1.2, line_dash="dot", line_color="blue")

    fig.update_yaxes(range=[0, 100], title_text="ADX")
    fig.update_layout(
        title="Diagnóstico de alineación ADX",
        height=520,
        margin=dict(l=30, r=30, t=60, b=30),
        hovermode="x unified",
        hoverdistance=1000,
        spikedistance=1000,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
    )

    fig.update_xaxes(
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikedash="dot",
        spikethickness=1,
        spikecolor="gray",
    )

    _apply_chart_theme(fig, theme)
    return fig


def build_adx_diagnostic_table(
    analyzed: pd.DataFrame,
    source_index: int,
    timeframe_seconds: int = 60,
    radius: int = 4,
) -> pd.DataFrame:
    """
    Tabla vela por vela para comparar alineación temporal.
    """
    work = calculate_adx(analyzed, period=7, di_period=7)

    start = max(0, source_index - radius)
    end = min(len(work) - 1, source_index + radius)
    window = work.loc[start:end].copy()

    open_time = pd.to_datetime(window["datetime_mexico"])
    close_time = open_time + pd.to_timedelta(timeframe_seconds, unit="s")

    result = pd.DataFrame({
        "Hora OPEN": open_time.dt.strftime("%H:%M:%S"),
        "Hora CLOSE": close_time.dt.strftime("%H:%M:%S"),
        "DI+": window["di_plus"].round(4),
        "DI-": window["di_minus"].round(4),
        "ADX normal": window["adx"].round(4),
        "ADX anterior": work["adx"].shift(1).loc[start:end].round(4),
        "ADX siguiente": work["adx"].shift(-1).loc[start:end].round(4),
    })

    result.insert(
        0,
        "Seleccionada",
        ["←" if idx == source_index else "" for idx in window.index],
    )

    return result.reset_index(drop=True)


def build_chart(analyzed: pd.DataFrame, source_index: int, mode: str, asset_label: str, theme: str = "Blanco"):
    # Los indicadores se calculan con TODA la historia disponible.
    # La ventana Todo/Antes/Inicio/Fin es únicamente un recorte visual.
    work = calculate_adx(analyzed, period=7, di_period=7)
    window, selected_idx = build_event_window(work, source_index, mode)

    x = pd.to_datetime(window["datetime_mexico"])

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.56, 0.44],
    )

    fig.add_trace(
        go.Candlestick(
            x=x,
            open=window["open"],
            high=window["high"],
            low=window["low"],
            close=window["close"],
            name="Velas",
            increasing_line_color="#2ca02c",
            decreasing_line_color="#d62728",
            increasing_fillcolor="#2ca02c",
            decreasing_fillcolor="#d62728",
        ),
        row=1,
        col=1,
    )

    # Indicadores normales del gráfico (independientes de JAPY2)
    # LSMA(7), offset 0, fuente LOW   -> naranja
    # LSMA(7), offset 0, fuente CLOSE -> verde
    # WMA(7), fuente CLOSE            -> azul
    if "lsma_low_7" in window.columns:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=window["lsma_low_7"],
                mode="lines",
                name="LSMA LOW 7",
                line=dict(color="#ff8c00", width=2),
            ),
            row=1,
            col=1,
        )

    if "lsma_close_7" in window.columns:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=window["lsma_close_7"],
                mode="lines",
                name="LSMA CLOSE 7",
                line=dict(color="#00b050", width=2),
            ),
            row=1,
            col=1,
        )

    if "wma_close_7" in window.columns:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=window["wma_close_7"],
                mode="lines",
                name="WMA 7",
                line=dict(color="#1f9cf0", width=2),
            ),
            row=1,
            col=1,
        )

    # Triángulos JAPY2 visibles dentro de la ventana
    triangles = window.loc[window.get("japy2_triangle", False) == True]
    if not triangles.empty:
        fig.add_trace(
            go.Scatter(
                x=pd.to_datetime(triangles["datetime_mexico"]),
                y=triangles["low"] * 0.9997,
                mode="markers",
                name="JAPY2",
                marker=dict(
                    symbol="triangle-up",
                    size=11,
                    color="#ffd400",
                    line=dict(width=1, color="#8a6d00"),
                ),
            ),
            row=1,
            col=1,
        )

    # Resalta la vela seleccionada cuando aplica
    if selected_idx is not None and selected_idx in window.index:
        sel_time = pd.to_datetime(window.loc[selected_idx, "datetime_mexico"])
        fig.add_vline(
            x=sel_time,
            line_width=1,
            line_dash="dot",
            line_color="#888888",
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=window["di_plus"],
            mode="lines",
            name="DI+",
            line=dict(width=1.5),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=window["di_minus"],
            mode="lines",
            name="DI-",
            line=dict(width=1.5),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=window["adx"],
            mode="lines",
            name="ADX 7",
            line=dict(width=2),
        ),
        row=2,
        col=1,
    )

    # Niveles visuales del ADX
    fig.add_hline(
        y=50,
        line_width=1.5,
        line_dash="dot",
        line_color="red",
        row=2,
        col=1,
    )
    fig.add_hline(
        y=20,
        line_width=1.5,
        line_dash="dot",
        line_color="blue",
        row=2,
        col=1,
    )

    fig.update_yaxes(range=[0, 100], row=2, col=1, title_text="ADX")
    fig.update_yaxes(title_text="Precio", row=1, col=1)

    fig.update_layout(
        title=f"{asset_label} — {mode}",
        height=970,
        margin=dict(l=30, r=30, t=60, b=30),
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
        hovermode="x unified",
        hoverdistance=1000,
        spikedistance=1000,
    )

    # Línea vertical del mouse atravesando ambos paneles.
    fig.update_xaxes(
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikedash="dot",
        spikethickness=1,
        spikecolor="gray",
    )

    _apply_chart_theme(fig, theme)
    return fig
