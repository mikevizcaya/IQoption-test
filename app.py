import streamlit as st
import pandas as pd

from iqoption_service import (
    ASSETS,
    TIMEFRAMES,
    CANDLE_OPTIONS,
    connect_iq_option,
    fetch_candles,
)

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
    "selected_asset": None,
    "selected_timeframe": 60,
    "selected_count": 45,
    "screen": "download",
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


def go_to_analysis():
    st.session_state.screen = "analysis"
    st.rerun()


def back_to_download():
    st.session_state.screen = "download"
    st.rerun()


# ---------------------------
# App header
# ---------------------------
st.title("Trading Analitic")
st.caption("Módulo 1 — conexión a IQ Option y carga de velas en memoria")

if st.session_state.screen == "download":
    st.subheader("Acceso IQ Option")

    # Nunca incrustamos credenciales dentro del repositorio.
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
        st.session_state.selected_asset = asset_label
        st.session_state.selected_timeframe = timeframe_seconds
        st.session_state.selected_count = int(candle_count)

        st.success(
            f"Datos cargados: {len(df):,} velas de {asset_label}."
        )

    if st.session_state.candles_df is not None:
        df = st.session_state.candles_df

        st.subheader("Vista previa")
        preview = df.tail(10).copy()
        st.dataframe(
            preview[
                ["datetime_mexico", "open", "high", "low", "close", "volume"]
            ],
            use_container_width=True,
            hide_index=True,
        )

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Activo", st.session_state.selected_asset)
        with m2:
            st.metric(
                "Velas cargadas",
                f"{len(df):,}",
            )
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
    st.subheader("Módulo 2 — JAPY2")
    st.success(
        "Los datos ya están en memoria y listos para ser procesados por JAPY2."
    )

    df = st.session_state.candles_df
    if df is None:
        st.warning("No hay datos cargados.")
        st.button("← Volver", on_click=back_to_download)
        st.stop()

    st.write(
        f"**Activo:** {st.session_state.selected_asset}  \n"
        f"**Velas disponibles:** {len(df):,}"
    )

    st.dataframe(
        df.tail(20)[
            ["datetime_mexico", "open", "high", "low", "close", "volume"]
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.info(
        "Aquí conectaremos en el siguiente módulo el análisis JAPY2, "
        "la tabla Verde/Rojo, la selección de eventos y los cuatro modos "
        "de gráfica: Todo, Antes, Inicio y Fin."
    )

    st.button("← Volver", on_click=back_to_download)
