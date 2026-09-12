import time
from typing import Tuple

import pandas as pd
from iqoptionapi.stable_api import IQ_Option


# Etiqueta mostrada en la interfaz -> código usado por IQ Option.
ASSETS = {
    "EUR/USD (OTC)": "EURUSD-OTC",
    "EUR/GBP (OTC)": "EURGBP-OTC",
    "GBP/JPY (OTC)": "GBPJPY-OTC",
    "USD/JPY (OTC)": "USDJPY-OTC",
    "EUR/USD": "EURUSD",
    "EUR/JPY": "EURJPY",
    "USD/JPY": "USDJPY",
    "US 30": "US30",
}

TIMEFRAMES = {
    "1 minuto": 60,
    "5 minutos": 300,
    "15 minutos": 900,
}

CANDLE_OPTIONS = [45, 60, 120, 180, 10000]


def connect_iq_option(email: str, password: str) -> Tuple[IQ_Option, bool, str]:
    """
    Crea la conexión con IQ Option.
    Devuelve: (objeto_api, conectado, motivo)
    """
    iq = IQ_Option(email, password)
    ok, reason = iq.connect()

    if ok:
        return iq, True, "ok"

    return iq, False, str(reason)


def _normalize_chunk(chunk):
    rows = []
    for candle in chunk or []:
        rows.append(
            {
                "id": candle.get("id"),
                "from": candle.get("from"),
                "to": candle.get("to"),
                "open": candle.get("open"),
                "close": candle.get("close"),
                "low": candle.get("min"),
                "high": candle.get("max"),
                "volume": candle.get("volume"),
            }
        )
    return rows


def fetch_candles(
    iq: IQ_Option,
    active: str,
    interval: int,
    count: int,
    batch_size: int = 1000,
) -> pd.DataFrame:
    """
    Descarga velas hacia memoria y devuelve un DataFrame.
    Para cantidades grandes, pagina hacia atrás en el tiempo.
    """
    if count <= 0:
        raise ValueError("count debe ser mayor que cero.")

    all_rows = []
    end_ts = time.time()
    remaining = int(count)

    while remaining > 0:
        take = min(batch_size, remaining)

        chunk = iq.get_candles(
            active,
            interval,
            take,
            end_ts,
        )

        rows = _normalize_chunk(chunk)
        if not rows:
            break

        all_rows.extend(rows)

        oldest_from = min(
            r["from"] for r in rows
            if r["from"] is not None
        )

        # Pedimos el siguiente bloque justo antes del bloque actual.
        end_ts = oldest_from - 1
        remaining -= len(rows)

        if len(rows) < take:
            break

        # Pausa corta para no martillar la API.
        time.sleep(0.15)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)

    # Elimina duplicados que puedan aparecer al paginar.
    df = (
        df.drop_duplicates(subset=["from"], keep="last")
        .sort_values("from")
        .tail(count)
        .reset_index(drop=True)
    )

    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Tiempo original en UTC
    df["datetime_utc"] = pd.to_datetime(
        df["from"],
        unit="s",
        utc=True,
    )

    # Horario México. En 2026 CDMX permanece UTC-6.
    df["datetime_mexico"] = (
        df["datetime_utc"]
        .dt.tz_convert("America/Mexico_City")
        .dt.strftime("%Y-%m-%d %H:%M:%S")
    )

    # Color útil para módulos posteriores.
    df["candle_color"] = "doji"
    df.loc[df["close"] > df["open"], "candle_color"] = "verde"
    df.loc[df["close"] < df["open"], "candle_color"] = "rojo"

    return df[
        [
            "id",
            "from",
            "to",
            "datetime_utc",
            "datetime_mexico",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "candle_color",
        ]
    ]
