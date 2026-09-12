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
from charts import (
    build_chart,
    build_recent_chart,
    build_adx_diagnostic_chart,
    build_adx_diagnostic_table,
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
    "japy2_df": None,
    "selected_asset": None,
    "selected_timeframe": 60,
    "selected_count": 45,
    "requested_count": 45,
    "warmup_count": 100,
    "screen": "download",
    "selected_event_source_index": None,
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


def go_to_analysis():
    if st.session_state.candles_df is not None:
        full_df = calculate_japy2(st.session_state.candles_df)
        requested_count = st.session_state.get(
            "requested_count",
            st.session_state.get("selected_count", len(full_df)),
        )
        # Toda la historia permanece en memoria para los indicadores.
        # Sólo las últimas N velas participan como universo de análisis JAPY2.
        full_df["analysis_eligible"] = False
        if requested_count > 0:
            full_df.loc[full_df.index[-requested_count:], "analysis_eligible"] = True
        st.session_state.japy2_df = full_df
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
                # Descargamos un bloque adicional de historia para "calentar"
                # LSMA/WMA/ADX. El usuario sigue viendo y analizando únicamente
                # el número de velas que eligió.
                warmup_count = 100
                requested_count = int(candle_count)
                internal_count = requested_count + warmup_count

                df = fetch_candles(
                    iq=iq,
                    active=asset_code,
                    interval=timeframe_seconds,
                    count=internal_count,
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
        st.session_state.selected_count = requested_count
        st.session_state.requested_count = requested_count
        st.session_state.warmup_count = warmup_count

        st.success(
            f"Datos cargados: {requested_count:,} velas para análisis "
            f"+ {warmup_count:,} velas previas de calentamiento interno."
        )

    if st.session_state.candles_df is not None:
        df = st.session_state.candles_df
        requested_count = st.session_state.get("requested_count", len(df))
        visible_df = df.tail(requested_count).copy()

        st.subheader("Vista previa")
        st.dataframe(
            visible_df.tail(10)[
                ["datetime_mexico", "open", "high", "low", "close", "volume"]
            ],
            use_container_width=True,
            hide_index=True,
        )

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Activo", st.session_state.selected_asset)
        with m2:
            st.metric("Velas para análisis", f"{len(visible_df):,}")
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
        eligible_count = int(analyzed.get("analysis_eligible", True).sum()) if "analysis_eligible" in analyzed.columns else len(analyzed)
        st.metric("Velas analizadas", f"{eligible_count:,}")
    with c3:
        eligible_mask = (
            analyzed["analysis_eligible"]
            if "analysis_eligible" in analyzed.columns
            else pd.Series(True, index=analyzed.index)
        )
        total_triangles = int(
            (analyzed["japy2_triangle"] & eligible_mask).sum()
        )
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

    eligible_mask = (
        analyzed["analysis_eligible"]
        if "analysis_eligible" in analyzed.columns
        else pd.Series(True, index=analyzed.index)
    )

    events = analyzed.loc[
        eligible_mask
        & analyzed["japy2_triangle"]
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
            # Antes de elegir una señal NO seleccionamos automáticamente
            # la primera fila. Mostramos las últimas 30 velas.
            st.session_state.selected_event_source_index = None

        st.divider()

        if st.session_state.selected_event_source_index is None:
            st.subheader("Gráfica — últimas 30 velas")
            st.caption(
                "Vista inicial. Selecciona un renglón de la tabla para "
                "analizar ese evento con Todo / Antes / Inicio / Fin."
            )

            with st.spinner("Construyendo gráfica inicial..."):
                fig = build_recent_chart(
                    analyzed=analyzed,
                    asset_label=st.session_state.selected_asset,
                    count=30,
                )

            st.plotly_chart(
                fig,
                use_container_width=True,
                config={
                    "displaylogo": False,
                    "scrollZoom": True,
                },
            )

        else:
            selected = analyzed.loc[
                st.session_state.selected_event_source_index
            ]

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

            graph_mode = st.radio(
                "Vista de gráfica",
                ["Todo", "Antes", "Inicio", "Fin"],
                horizontal=True,
                index=0,
            )

            st.subheader("Gráfica del evento")

            with st.spinner("Construyendo gráfica..."):
                fig = build_chart(
                    analyzed=analyzed,
                    source_index=st.session_state.selected_event_source_index,
                    mode=graph_mode,
                    asset_label=st.session_state.selected_asset,
                )

            st.plotly_chart(
                fig,
                use_container_width=True,
                config={
                    "displaylogo": False,
                    "scrollZoom": True,
                },
            )

            if graph_mode == "Todo":
                st.caption(
                    "Todo: 15 velas previas + vela seleccionada + 15 posteriores."
                )
            elif graph_mode == "Antes":
                st.caption(
                    "Antes: la vela inmediatamente anterior es la última visible, "
                    "acompañada por 14 velas previas."
                )
            elif graph_mode == "Inicio":
                st.caption(
                    "Inicio: la vela seleccionada se representa únicamente en su OPEN. "
                    "No se usa su HIGH, LOW ni CLOSE."
                )
            elif graph_mode == "Fin":
                st.caption(
                    "Fin: la vela seleccionada se muestra completa hasta su CLOSE."
                )

            st.divider()
            show_adx_diag = st.checkbox(
                "Mostrar diagnóstico de alineación ADX",
                value=False,
                help=(
                    "Compara el mismo ADX en su posición normal, "
                    "desplazado una vela hacia adelante y una hacia atrás."
                ),
            )

            if show_adx_diag:
                st.subheader("Diagnóstico ADX")
                st.caption(
                    "Esta prueba NO modifica el cálculo. Sirve únicamente para "
                    "comprobar si IQ Option dibuja el valor ADX asociado a una "
                    "vela en una posición temporal diferente."
                )

                diag_fig = build_adx_diagnostic_chart(
                    analyzed=analyzed,
                    source_index=st.session_state.selected_event_source_index,
                    timeframe_seconds=st.session_state.selected_timeframe,
                    radius=6,
                )

                st.plotly_chart(
                    diag_fig,
                    use_container_width=True,
                    config={
                        "displaylogo": False,
                        "scrollZoom": True,
                    },
                )

                diag_table = build_adx_diagnostic_table(
                    analyzed=analyzed,
                    source_index=st.session_state.selected_event_source_index,
                    timeframe_seconds=st.session_state.selected_timeframe,
                    radius=4,
                )

                st.dataframe(
                    diag_table,
                    use_container_width=True,
                    hide_index=True,
                )

                st.info(
                    "Para comprobarlo en IQ Option: coloca el cursor en la misma "
                    "hora y compara el valor amarillo con las columnas "
                    "'ADX normal', 'ADX anterior' y 'ADX siguiente'. "
                    "Si una columna coincide de forma sistemática en varias velas, "
                    "habremos identificado un desplazamiento temporal."
                )

    st.button("← Volver al descargador", on_click=back_to_download)
