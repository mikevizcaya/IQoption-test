import streamlit as st
import pandas as pd

from iqoption_service import (
    ASSETS,
    TIMEFRAMES,
    CANDLE_OPTIONS,
    connect_iq_option,
    fetch_candles,
)
from japy2 import calculate_japy2

st.set_page_config(
    page_title="Trading Analitic",
    page_icon="📈",
    layout="wide",
)

# ---------------------------
# Session state
# ---------------------------
defaults = {
    "iq": None,
    "connected": False,
    "candles_df": None,
    "japy2_df": None,
    "selected_asset": None,
    "selected_timeframe": 60,
    "selected_count": 45,
    "screen": "download",
    "selected_event_source_index": None,
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


def go_to_analysis():
    if st.session_state.candles_df is not None:
        st.session_state.japy2_df = calculate_japy2(
            st.session_state.candles_df
        )
    st.session_state.screen = "analysis"
    st.rerun()


def back_to_download():
    st.session_state.screen = "download"
    st.rerun()


# ---------------------------
# App header
# ---------------------------
st.title("Trading Analitic")

if st.session_state.screen == "download":
    st.caption("Módulo 1 — conexión a IQ Option y carga de velas en memoria")
    st.subheader("Acceso IQ Option")

    default_email = ""
    try:
        default_email = st.secrets.get("IQ_EMAIL", "")
    except Exception:
        pass

    col1, col2 = st.columns(2)
    with col1:
        email = st.text_input(
            "Email",
            value=default_email,
            placeholder="tu_correo@ejemplo.com",
        )
    with col2:
        password = st.text_input(
            "Password",
            type="password",
            placeholder="••••••••",
        )

    st.divider()

    st.subheader("Selecciona un activo")
    asset_label = st.radio(
        "Activo",
        options=list(ASSETS.keys()),
        horizontal=True,
        label_visibility="collapsed",
    )

    st.subheader("Parámetros de descarga")
    c1, c2 = st.columns(2)

    with c1:
        timeframe_label = st.selectbox(
            "Marco de tiempo",
            options=list(TIMEFRAMES.keys()),
            index=0,
        )

    with c2:
        candle_count = st.selectbox(
            "Número de velas",
            options=CANDLE_OPTIONS,
            index=0,
        )

    st.info(
        "Las velas se cargan directamente desde la API a memoria. "
        "No se crea un CSV intermedio."
    )

    if st.button(
        "Conectar y cargar datos",
        type="primary",
        use_container_width=True,
    ):
        if not email.strip() or not password:
            st.error("Escribe email y password.")
            st.stop()

        with st.spinner("Conectando con IQ Option..."):
            iq, ok, reason = connect_iq_option(email.strip(), password)

        if not ok:
            st.session_state.iq = None
            st.session_state.connected = False
            st.error(f"No fue posible conectar con IQ Option: {reason}")
            st.stop()

        st.session_state.iq = iq
        st.session_state.connected = True

        asset_code = ASSETS[asset_label]
        timeframe_seconds = TIMEFRAMES[timeframe_label]

        with st.spinner(
            f"Cargando {candle_count:,} velas de {asset_label}..."
        ):
            try:
                df = fetch_candles(
                    iq=iq,
                    active=asset_code,
                    interval=timeframe_seconds,
                    count=int(candle_count),
                )
            except Exception as exc:
                st.error(f"Error al obtener velas: {exc}")
                st.stop()

        if df.empty:
            st.error("La API no devolvió velas para esa selección.")
            st.stop()

        st.session_state.candles_df = df
        st.session_state.japy2_df = None
        st.session_state.selected_event_source_index = None
        st.session_state.selected_asset = asset_label
        st.session_state.selected_timeframe = timeframe_seconds
        st.session_state.selected_count = int(candle_count)

        st.success(
            f"Datos cargados: {len(df):,} velas de {asset_label}."
        )

    if st.session_state.candles_df is not None:
        df = st.session_state.candles_df

        st.subheader("Vista previa")
        st.dataframe(
            df.tail(10)[
                ["datetime_mexico", "open", "high", "low", "close", "volume"]
            ],
            use_container_width=True,
            hide_index=True,
        )

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Activo", st.session_state.selected_asset)
        with m2:
            st.metric("Velas cargadas", f"{len(df):,}")
        with m3:
            mins = st.session_state.selected_timeframe // 60
            st.metric("Marco", f"{mins} min")

        st.button(
            "Siguiente →",
            type="primary",
            use_container_width=True,
            on_click=go_to_analysis,
        )

else:
    # ---------------------------
    # MÓDULO 2 — JAPY2
    # ---------------------------
    st.caption("Módulo 2 — análisis JAPY2")
    st.subheader("JAPY2 — Triángulo amarillo")

    df = st.session_state.candles_df
    if df is None:
        st.warning("No hay datos cargados.")
        st.button("← Volver", on_click=back_to_download)
        st.stop()

    if st.session_state.japy2_df is None:
        with st.spinner("Calculando JAPY2..."):
            st.session_state.japy2_df = calculate_japy2(df)

    analyzed = st.session_state.japy2_df

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Activo", st.session_state.selected_asset)
    with c2:
        st.metric("Velas analizadas", f"{len(analyzed):,}")
    with c3:
        total_triangles = int(analyzed["japy2_triangle"].sum())
        st.metric("Triángulos JAPY2", f"{total_triangles:,}")

    st.markdown(
        "**Criterio JAPY2:** LSMA7 y WMA7 calculadas con los seis "
        "cierres anteriores + el OPEN actual; triángulo cuando "
        "`LSMA_OPEN ≥ OPEN`, `DT LSMA ≤ 20%` y `OPEN > WMA_OPEN`."
    )

    color_choice = st.radio(
        "Mostrar señales:",
        ["Verde", "Rojo"],
        horizontal=True,
        index=0,
    )
    wanted = color_choice.lower()

    events = analyzed.loc[
        analyzed["japy2_triangle"]
        & (analyzed["japy2_result"] == wanted)
    ].copy()

    st.write(
        f"**Resultados {color_choice.lower()}: {len(events):,} señales**"
    )

    if events.empty:
        st.warning(
            f"No hay triángulos JAPY2 con vela {color_choice.lower()} "
            "en los datos cargados."
        )
        st.session_state.selected_event_source_index = None
    else:
        table = events[
            [
                "datetime_mexico",
                "dt_lsma_pct",
            ]
        ].copy()

        # Conservamos el índice del DataFrame original para localizar luego
        # la vela y construir Todo / Antes / Inicio / Fin.
        table["source_index"] = events.index
        table = table.rename(
            columns={
                "datetime_mexico": "Hora de apertura",
                "dt_lsma_pct": "DT LSMA (%)",
            }
        )

        display_table = table[
            ["Hora de apertura", "DT LSMA (%)"]
        ].copy()

        st.caption("Selecciona un renglón de la tabla.")

        selection = st.dataframe(
            display_table,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            key=f"japy2_{wanted}_table",
            column_config={
                "Hora de apertura": st.column_config.TextColumn(
                    "Hora de apertura"
                ),
                "DT LSMA (%)": st.column_config.NumberColumn(
                    "DT LSMA (%)",
                    format="%.6f",
                ),
            },
        )

        selected_rows = []
        try:
            selected_rows = selection.selection.rows
        except Exception:
            selected_rows = []

        if selected_rows:
            row_pos = selected_rows[0]
            source_index = int(table.iloc[row_pos]["source_index"])
            st.session_state.selected_event_source_index = source_index
        else:
            # Si todavía no se selecciona nada, usamos la primera señal
            # como vista inicial para que la aplicación nunca quede vacía.
            source_index = int(table.iloc[0]["source_index"])
            st.session_state.selected_event_source_index = source_index

        selected = analyzed.loc[
            st.session_state.selected_event_source_index
        ]

        st.divider()
        st.subheader("Evento seleccionado")

        e1, e2, e3, e4 = st.columns(4)
        with e1:
            st.metric("Hora OPEN", selected["datetime_mexico"])
        with e2:
            st.metric("OPEN", f'{selected["open"]:.6f}')
        with e3:
            st.metric("CLOSE", f'{selected["close"]:.6f}')
        with e4:
            st.metric("DT LSMA", f'{selected["dt_lsma_pct"]:.6f}%')

        st.success(
            f"Señal preparada: vela {wanted} con triángulo amarillo JAPY2."
        )

        # Control que ya queda preparado para el siguiente módulo.
        graph_mode = st.radio(
            "Vista de gráfica",
            ["Todo", "Antes", "Inicio", "Fin"],
            horizontal=True,
            index=0,
            help=(
                "Los cuatro modos ya quedan definidos. "
                "La gráfica se incorpora en el siguiente módulo."
            ),
        )

        st.info(
            f"Modo seleccionado: **{graph_mode}**. "
            "El evento ya está identificado por su posición exacta "
            "dentro de las velas descargadas."
        )

    st.button("← Volver al descargador", on_click=back_to_download)
