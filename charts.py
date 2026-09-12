import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    ADX clásico de Wilder sobre HIGH/LOW/CLOSE.
    Devuelve columnas di_plus, di_minus y adx.
    """
    out = df.copy().reset_index(drop=True)

    high = pd.to_numeric(out["high"], errors="coerce")
    low = pd.to_numeric(out["low"], errors="coerce")
    close = pd.to_numeric(out["close"], errors="coerce")

    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)

    up_move = high - prev_high
    down_move = prev_low - low

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=out.index,
        dtype=float,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=out.index,
        dtype=float,
    )

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Wilder smoothing via EMA alpha=1/period
    atr = tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    plus_sm = plus_dm.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    minus_sm = minus_dm.ewm(alpha=1/period, adjust=False, min_periods=period).mean()

    di_plus = 100 * plus_sm / atr.replace(0, np.nan)
    di_minus = 100 * minus_sm / atr.replace(0, np.nan)

    dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, np.nan)
    adx = dx.ewm(alpha=1/period, adjust=False, min_periods=period).mean()

    out["di_plus"] = di_plus
    out["di_minus"] = di_minus
    out["adx"] = adx
    return out


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

            # MUY IMPORTANTE:
            # En "Inicio", CLOSE todavía vale OPEN. Por lo tanto,
            # LSMA CLOSE y WMA CLOSE deben recalcularse con:
            # 6 cierres anteriores + OPEN actual.
            # Eso hace que en la vela seleccionada:
            # LSMA CLOSE == LSMA OPEN
            # WMA CLOSE  == WMA OPEN
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

            adx_hist = calculate_adx(hist, period=14)
            for col in ["di_plus", "di_minus", "adx"]:
                if col in adx_hist.columns:
                    window.loc[idx, col] = adx_hist.loc[idx, col]

        selected_local_idx = source_index
        return window, selected_local_idx

    raise ValueError(f"Modo no reconocido: {mode}")


def build_chart(analyzed: pd.DataFrame, source_index: int, mode: str, asset_label: str):
    work = calculate_adx(analyzed, period=14)
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

    if "lsma_open_7" in window.columns:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=window["lsma_open_7"],
                mode="lines",
                name="LSMA OPEN 7",
                line=dict(color="#00b050", width=2),
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
                line=dict(color="#ff8c00", width=2),
            ),
            row=1,
            col=1,
        )

    if "wma_open_7" in window.columns:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=window["wma_open_7"],
                mode="lines",
                name="WMA OPEN 7",
                line=dict(color="#1f77b4", width=1.8),
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
                name="WMA CLOSE 7",
                line=dict(color="#6f42c1", width=1.4, dash="dot"),
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
            name="ADX 14",
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
    )

    return fig
