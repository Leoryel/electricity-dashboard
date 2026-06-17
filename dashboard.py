from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
import os
import re
from zoneinfo import ZoneInfo
import xml.etree.ElementTree as ET

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st


TIMESTAMP_FORMAT = "%d/%m/%Y %H:%M"
EXPECTED_INTERVALS_PER_DAY = 96
BRUSSELS_TZ = ZoneInfo("Europe/Brussels")
ENTSOE_API_URL = "https://web-api.tp.entsoe.eu/api"
BELGIUM_BIDDING_ZONE_EIC = "10YBE----------2"
PROFILE_VIEW_OPTIONS = {
    "Year": "year",
    "Quarter": "quarter",
    "Season": "season",
    "Month": "month",
    "Day of week": "day_of_week",
}
SEASON_BY_MONTH = {
    12: "Winter",
    1: "Winter",
    2: "Winter",
    3: "Spring",
    4: "Spring",
    5: "Spring",
    6: "Summer",
    7: "Summer",
    8: "Summer",
    9: "Autumn",
    10: "Autumn",
    11: "Autumn",
}


@dataclass(frozen=True)
class PreparedConsumptionData:
    data: pd.DataFrame
    timestamp_column: str
    consumption_column: str
    invalid_timestamp_count: int
    invalid_consumption_count: int


@dataclass(frozen=True)
class DayAheadPriceData:
    prices: pd.DataFrame
    raw_resolution: str
    publication_currency: str


def all_quarter_hour_slots() -> pd.DataFrame:
    slots = pd.date_range("2000-01-01", periods=EXPECTED_INTERVALS_PER_DAY, freq="15min")
    return pd.DataFrame(
        {
            "time_slot": slots.strftime("%H:%M"),
            "slot_order": range(EXPECTED_INTERVALS_PER_DAY),
        }
    )


def period_definitions(view: str) -> pd.DataFrame:
    if view == "year":
        labels = ["Year"]
    elif view == "quarter":
        labels = [f"Q{quarter}" for quarter in range(1, 5)]
    elif view == "season":
        labels = ["Winter", "Spring", "Summer", "Autumn"]
    elif view == "month":
        labels = [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ]
    elif view == "day_of_week":
        labels = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
    else:
        raise ValueError(f"Unknown profile view: {view}")

    return pd.DataFrame(
        {
            "period_label": labels,
            "period_order": range(len(labels)),
        }
    )


def add_profile_period(
    data: pd.DataFrame, view: str, timestamp_column: str = "timestamp"
) -> pd.DataFrame:
    working_data = data.copy()
    timestamps = working_data[timestamp_column]

    if view == "year":
        working_data["period_label"] = "Year"
    elif view == "quarter":
        working_data["period_label"] = "Q" + timestamps.dt.quarter.astype(str)
    elif view == "season":
        working_data["period_label"] = timestamps.dt.month.map(SEASON_BY_MONTH)
    elif view == "month":
        working_data["period_label"] = timestamps.dt.month_name()
    elif view == "day_of_week":
        working_data["period_label"] = timestamps.dt.day_name()
    else:
        raise ValueError(f"Unknown profile view: {view}")

    return working_data


def parse_timestamps(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce")

    parsed = pd.to_datetime(series, format=TIMESTAMP_FORMAT, errors="coerce")
    if parsed.notna().any():
        return parsed

    return pd.to_datetime(series, dayfirst=True, errors="coerce")


def parse_consumption(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    cleaned = series.astype(str).str.strip().str.replace(",", ".", regex=False)
    return pd.to_numeric(cleaned, errors="coerce")


def format_price_metric(value: float) -> str:
    if pd.isna(value):
        return "n/a"

    return f"{value:.2f} EUR/MWh"


def format_percentage_metric(value: float) -> str:
    if pd.isna(value):
        return "n/a"

    return f"{value:+.2f}%"


def get_saved_entsoe_token() -> str:
    try:
        saved_token = st.secrets.get("ENTSOE_API_TOKEN", "")
    except Exception:
        saved_token = ""

    return str(saved_token or os.getenv("ENTSOE_API_TOKEN", "")).strip()


def prepare_consumption_data(raw_data: pd.DataFrame) -> PreparedConsumptionData:
    if raw_data.empty:
        raise ValueError("The uploaded Excel file is empty.")

    if raw_data.shape[1] < 2:
        raise ValueError("The uploaded Excel file must contain at least two columns.")

    timestamp_column = str(raw_data.columns[0])
    consumption_column = str(raw_data.columns[1])

    working_data = raw_data.iloc[:, :2].copy()
    working_data.columns = ["timestamp", "consumption_kwh"]

    working_data["timestamp"] = parse_timestamps(working_data["timestamp"])
    working_data["consumption_kwh"] = parse_consumption(working_data["consumption_kwh"])

    invalid_timestamp_count = int(working_data["timestamp"].isna().sum())
    invalid_consumption_count = int(working_data["consumption_kwh"].isna().sum())

    working_data = working_data.dropna(subset=["timestamp", "consumption_kwh"]).copy()
    if working_data.empty:
        raise ValueError("No valid timestamp and kWh rows were found in the uploaded file.")

    working_data["time_slot"] = working_data["timestamp"].dt.strftime("%H:%M")
    working_data["date"] = working_data["timestamp"].dt.date

    return PreparedConsumptionData(
        data=working_data,
        timestamp_column=timestamp_column,
        consumption_column=consumption_column,
        invalid_timestamp_count=invalid_timestamp_count,
        invalid_consumption_count=invalid_consumption_count,
    )


def build_average_profile(data: pd.DataFrame, view: str) -> pd.DataFrame:
    data = add_profile_period(data, view)
    profile = (
        data.groupby(["period_label", "time_slot"], as_index=False)
        .agg(
            average_kwh=("consumption_kwh", "mean"),
            observations=("consumption_kwh", "size"),
        )
    )

    grid = period_definitions(view).merge(all_quarter_hour_slots(), how="cross")
    return grid.merge(profile, on=["period_label", "time_slot"], how="left")


def render_summary(prepared_data: PreparedConsumptionData, raw_column_count: int) -> None:
    data = prepared_data.data
    start_date = data["timestamp"].min().strftime("%d/%m/%Y %H:%M")
    end_date = data["timestamp"].max().strftime("%d/%m/%Y %H:%M")

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Valid rows", f"{len(data):,}")
    col_b.metric("Days", f"{data['date'].nunique():,}")
    col_c.metric("Start", start_date)
    col_d.metric("End", end_date)

    st.caption(
        "Detected columns: "
        f"{prepared_data.timestamp_column} -> timestamp, "
        f"{prepared_data.consumption_column} -> kWh"
    )

    if raw_column_count > 2:
        st.warning(
            f"The file contains {raw_column_count} columns. Only the first two columns were used."
        )

    if prepared_data.invalid_timestamp_count:
        st.warning(
            f"{prepared_data.invalid_timestamp_count:,} row(s) had an invalid timestamp and were excluded."
        )

    if prepared_data.invalid_consumption_count:
        st.warning(
            f"{prepared_data.invalid_consumption_count:,} row(s) had an invalid kWh value and were excluded."
        )


def render_missing_interval_warnings(profile: pd.DataFrame) -> None:
    interval_counts = profile.groupby("period_label", sort=False)["average_kwh"].apply(
        lambda values: int(values.notna().sum())
    )

    incomplete_periods = interval_counts[interval_counts < EXPECTED_INTERVALS_PER_DAY]
    if incomplete_periods.empty:
        return

    details = ", ".join(
        f"{period}: {count}/{EXPECTED_INTERVALS_PER_DAY}"
        for period, count in incomplete_periods.items()
    )
    st.warning(f"Some profiles have missing quarter-hour intervals: {details}.")


def render_profile_chart(profile: pd.DataFrame, view_label: str) -> None:
    render_missing_interval_warnings(profile)

    fig = px.line(
        profile,
        x="time_slot",
        y="average_kwh",
        color="period_label",
        category_orders={
            "period_label": profile.sort_values("period_order")["period_label"].unique().tolist()
        },
        markers=True,
        labels={
            "time_slot": "Time of day",
            "average_kwh": "Average consumption (kWh)",
            "period_label": view_label,
        },
        title=f"Average Daily Electricity Consumption Profile by {view_label}",
    )
    fig.update_traces(connectgaps=False, hovertemplate="%{x}<br>%{y:.3f} kWh<extra></extra>")
    fig.update_layout(
        height=560,
        margin=dict(l=20, r=20, t=70, b=20),
        xaxis=dict(
            tickmode="array",
            tickvals=profile.loc[profile["slot_order"] % 8 == 0, "time_slot"],
            tickangle=0,
        ),
        yaxis=dict(rangemode="tozero"),
    )

    st.plotly_chart(fig, use_container_width=True)


def format_entsoe_period(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y%m%d%H%M")


def parse_iso_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def parse_iso_duration(value: str) -> timedelta:
    match = re.fullmatch(r"PT(?:(\d+)H)?(?:(\d+)M)?", value)
    if not match:
        raise ValueError(f"Unsupported ENTSO-E time resolution: {value}")

    hours = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    return timedelta(hours=hours, minutes=minutes)


def fetch_entsoe_day_ahead_prices(
    security_token: str, start_date: date, end_date: date
) -> DayAheadPriceData:
    local_start = datetime.combine(start_date, time.min, tzinfo=BRUSSELS_TZ)
    local_end = datetime.combine(end_date + timedelta(days=1), time.min, tzinfo=BRUSSELS_TZ)
    params = {
        "securityToken": security_token.strip(),
        "documentType": "A44",
        "in_Domain": BELGIUM_BIDDING_ZONE_EIC,
        "out_Domain": BELGIUM_BIDDING_ZONE_EIC,
        "periodStart": format_entsoe_period(local_start),
        "periodEnd": format_entsoe_period(local_end),
    }

    response = requests.get(ENTSOE_API_URL, params=params, timeout=60)
    response.raise_for_status()

    root = ET.fromstring(response.content)
    if root.tag.endswith("Acknowledgement_MarketDocument"):
        reason = root.findtext(".//{*}Reason/{*}text") or "ENTSO-E returned an acknowledgement instead of prices."
        raise ValueError(reason)

    rows = []
    raw_resolutions = set()
    currency = root.findtext(".//{*}currency_Unit.name") or "EUR"

    for period in root.findall(".//{*}Period"):
        start_text = period.findtext(".//{*}timeInterval/{*}start")
        resolution_text = period.findtext("{*}resolution")
        if not start_text or not resolution_text:
            continue

        period_start = parse_iso_datetime(start_text)
        resolution = parse_iso_duration(resolution_text)
        raw_resolutions.add(resolution_text)

        for point in period.findall("{*}Point"):
            position_text = point.findtext("{*}position")
            price_text = point.findtext("{*}price.amount")
            if not position_text or price_text is None:
                continue

            mtu_start = period_start + (int(position_text) - 1) * resolution
            mtu_end = mtu_start + resolution
            price = float(price_text)

            slot_start = mtu_start
            while slot_start < mtu_end:
                slot_end = min(slot_start + timedelta(minutes=15), mtu_end)
                rows.append(
                    {
                        "datetime": slot_start.astimezone(BRUSSELS_TZ),
                        "delivery_date": slot_start.astimezone(BRUSSELS_TZ).date(),
                        "time_slot": slot_start.astimezone(BRUSSELS_TZ).strftime("%H:%M"),
                        "mtu_start": slot_start.astimezone(BRUSSELS_TZ).strftime("%H:%M"),
                        "mtu_end": slot_end.astimezone(BRUSSELS_TZ).strftime("%H:%M"),
                        "price_eur_mwh": price,
                        "source_resolution": resolution_text,
                    }
                )
                slot_start = slot_end

    if not rows:
        raise ValueError("No Belgian day-ahead price points were returned for the selected period.")

    prices = pd.DataFrame(rows).sort_values("datetime")
    prices["datetime"] = pd.to_datetime(prices["datetime"], utc=True).dt.tz_convert(BRUSSELS_TZ)
    raw_resolution = ", ".join(sorted(raw_resolutions)) or "unknown"
    return DayAheadPriceData(prices=prices, raw_resolution=raw_resolution, publication_currency=currency)


def build_average_price_profile(prices: pd.DataFrame, view: str) -> pd.DataFrame:
    prices = add_profile_period(prices, view, timestamp_column="datetime")
    profile = (
        prices.groupby(["period_label", "time_slot"], as_index=False)
        .agg(
            average_price_eur_mwh=("price_eur_mwh", "mean"),
            observations=("price_eur_mwh", "size"),
        )
    )

    grid = period_definitions(view).merge(all_quarter_hour_slots(), how="cross")
    return grid.merge(profile, on=["period_label", "time_slot"], how="left")


def build_combined_average_profile(
    consumption: pd.DataFrame,
    prices: pd.DataFrame,
    consumption_start_date: date,
    consumption_end_date: date,
    price_start_date: date,
    price_end_date: date,
) -> pd.DataFrame:
    consumption_period = consumption[
        (consumption["date"] >= consumption_start_date)
        & (consumption["date"] <= consumption_end_date)
    ]
    prices_period = prices[
        (prices["delivery_date"] >= price_start_date)
        & (prices["delivery_date"] <= price_end_date)
    ]

    consumption_profile = (
        consumption_period.groupby("time_slot", as_index=False)
        .agg(
            average_kwh=("consumption_kwh", "mean"),
            consumption_observations=("consumption_kwh", "size"),
        )
    )
    price_profile = (
        prices_period.groupby("time_slot", as_index=False)
        .agg(
            average_price_eur_mwh=("price_eur_mwh", "mean"),
            price_observations=("price_eur_mwh", "size"),
        )
    )

    profile = all_quarter_hour_slots().merge(consumption_profile, on="time_slot", how="left")
    return profile.merge(price_profile, on="time_slot", how="left")


def calculate_profile_weighted_price_metrics(profile: pd.DataFrame) -> tuple[float, float, float]:
    arithmetic_average = profile["average_price_eur_mwh"].mean()
    matched_profile = profile.dropna(subset=["average_kwh", "average_price_eur_mwh"])
    total_consumption = matched_profile["average_kwh"].sum()
    if matched_profile.empty or total_consumption == 0:
        return arithmetic_average, float("nan"), float("nan")

    weighted_average = (
        matched_profile["average_kwh"] * matched_profile["average_price_eur_mwh"]
    ).sum() / total_consumption

    if pd.isna(arithmetic_average) or arithmetic_average == 0:
        percentage_difference = float("nan")
    else:
        percentage_difference = ((weighted_average - arithmetic_average) / arithmetic_average) * 100

    return arithmetic_average, weighted_average, percentage_difference


def build_daily_spreads(prices: pd.DataFrame) -> pd.DataFrame:
    def ranked_position_spread(day_prices: pd.Series, rank: int) -> float | None:
        clean_prices = day_prices.dropna().sort_values()
        if len(clean_prices) < rank * 2:
            return None

        ranked_lowest = clean_prices.iloc[rank - 1]
        ranked_highest = clean_prices.iloc[-rank]
        return float(ranked_highest - ranked_lowest)

    def ranked_average_spread(day_prices: pd.Series, quarter_hours: int) -> float | None:
        clean_prices = day_prices.dropna().sort_values()
        if len(clean_prices) < quarter_hours * 2:
            return None

        lowest_average = clean_prices.head(quarter_hours).mean()
        highest_average = clean_prices.tail(quarter_hours).mean()
        return float(highest_average - lowest_average)

    return (
        prices.groupby("delivery_date", as_index=False)
        .agg(
            max_price=("price_eur_mwh", "max"),
            min_price=("price_eur_mwh", "min"),
            four_hour_spread_eur_mwh=(
                "price_eur_mwh",
                lambda values: ranked_position_spread(values, 16),
            ),
            two_hour_spread_eur_mwh=(
                "price_eur_mwh",
                lambda values: ranked_average_spread(values, 8),
            ),
            one_hour_spread_eur_mwh=(
                "price_eur_mwh",
                lambda values: ranked_average_spread(values, 4),
            ),
            intervals=("price_eur_mwh", "size"),
        )
        .assign(max_spread_eur_mwh=lambda data: data["max_price"] - data["min_price"])
    )


def build_negative_hours_evolution(prices: pd.DataFrame) -> pd.DataFrame:
    daily_negative_hours = (
        prices.assign(is_negative=prices["price_eur_mwh"] < 0)
        .groupby("delivery_date", as_index=False)
        .agg(negative_intervals=("is_negative", "sum"))
    )
    daily_negative_hours["negative_hours"] = daily_negative_hours["negative_intervals"] / 4
    daily_negative_hours["cumulative_negative_hours"] = daily_negative_hours[
        "negative_hours"
    ].cumsum()
    return daily_negative_hours


def build_price_category_distribution(prices: pd.DataFrame) -> pd.DataFrame:
    if prices.empty:
        return pd.DataFrame(
            columns=["price_category", "hours", "percentage_of_period"]
        )

    edges = [
        float("-inf"),
        -500,
        -200,
        -100,
        -80,
        -60,
        -40,
        -20,
        0,
        20,
        40,
        60,
        80,
        100,
        200,
        500,
        float("inf"),
    ]
    labels = [
        "Below -500 EUR/MWh",
        "-500 to -200 EUR/MWh",
        "-200 to -100 EUR/MWh",
        "-100 to -80 EUR/MWh",
        "-80 to -60 EUR/MWh",
        "-60 to -40 EUR/MWh",
        "-40 to -20 EUR/MWh",
        "-20 to 0 EUR/MWh",
        "0 to 20 EUR/MWh",
        "20 to 40 EUR/MWh",
        "40 to 60 EUR/MWh",
        "60 to 80 EUR/MWh",
        "80 to 100 EUR/MWh",
        "100 to 200 EUR/MWh",
        "200 to 500 EUR/MWh",
        "500 EUR/MWh and above",
    ]
    categories = pd.cut(
        prices["price_eur_mwh"],
        bins=edges,
        labels=labels,
        right=False,
        include_lowest=True,
    )
    interval_counts = categories.value_counts(sort=False)
    total_hours = len(prices) / 4

    distribution = pd.DataFrame(
        {
            "price_category": interval_counts.index.astype(str),
            "hours": interval_counts.values / 4,
        }
    )
    distribution["percentage_of_period"] = (
        distribution["hours"] / total_hours * 100 if total_hours else 0
    )
    return distribution


def render_negative_hours_chart(negative_hours: pd.DataFrame) -> None:
    fig = px.line(
        negative_hours,
        x="delivery_date",
        y="cumulative_negative_hours",
        markers=True,
        labels={
            "delivery_date": "Date",
            "cumulative_negative_hours": "Cumulative negative hours",
        },
        title="Cumulative Negative Price Hours",
    )
    fig.update_traces(
        hovertemplate="%{x|%d/%m/%Y}<br>%{y:.2f} cumulative hours<extra></extra>"
    )
    fig.update_layout(
        height=420,
        margin=dict(l=20, r=20, t=70, b=20),
        yaxis=dict(rangemode="tozero"),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_price_missing_interval_warnings(profile: pd.DataFrame) -> None:
    interval_counts = profile.groupby("period_label", sort=False)["average_price_eur_mwh"].apply(
        lambda values: int(values.notna().sum())
    )

    incomplete_periods = interval_counts[interval_counts < EXPECTED_INTERVALS_PER_DAY]
    if incomplete_periods.empty:
        return

    details = ", ".join(
        f"{period}: {count}/{EXPECTED_INTERVALS_PER_DAY}"
        for period, count in incomplete_periods.items()
    )
    st.warning(f"Some price profiles have missing quarter-hour intervals: {details}.")


def render_price_profile_chart(
    profile: pd.DataFrame, start_date: date, end_date: date, view_label: str
) -> None:
    render_price_missing_interval_warnings(profile)

    fig = px.line(
        profile,
        x="time_slot",
        y="average_price_eur_mwh",
        color="period_label",
        category_orders={
            "period_label": profile.sort_values("period_order")["period_label"].unique().tolist()
        },
        markers=True,
        labels={
            "time_slot": "Time of day",
            "average_price_eur_mwh": "Average day-ahead price (EUR/MWh)",
            "period_label": view_label,
        },
        title=(
            f"Average Belgian Day-Ahead Spot Price Profile by {view_label} "
            f"- {start_date:%d/%m/%Y} to {end_date:%d/%m/%Y}"
        ),
    )
    fig.update_traces(hovertemplate="%{x}<br>%{y:.2f} EUR/MWh<extra></extra>")
    fig.update_layout(
        height=560,
        margin=dict(l=20, r=20, t=70, b=20),
        xaxis=dict(
            tickmode="array",
            tickvals=profile.loc[profile["slot_order"] % 8 == 0, "time_slot"],
            tickangle=0,
        ),
        yaxis=dict(zeroline=True),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_combined_profile_chart(
    profile: pd.DataFrame,
    consumption_start_date: date,
    consumption_end_date: date,
    price_start_date: date,
    price_end_date: date,
) -> None:
    consumption_intervals = int(profile["average_kwh"].notna().sum())
    price_intervals = int(profile["average_price_eur_mwh"].notna().sum())

    if consumption_intervals < EXPECTED_INTERVALS_PER_DAY:
        st.warning(
            f"Consumption data covers {consumption_intervals} of "
            f"{EXPECTED_INTERVALS_PER_DAY} quarter-hour intervals."
        )

    if price_intervals < EXPECTED_INTERVALS_PER_DAY:
        st.warning(
            f"Spot price data covers {price_intervals} of "
            f"{EXPECTED_INTERVALS_PER_DAY} quarter-hour intervals."
        )

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=profile["time_slot"],
            y=profile["average_kwh"],
            name="Average consumption",
            mode="lines+markers",
            hovertemplate="%{x}<br>%{y:.3f} kWh<extra></extra>",
            connectgaps=False,
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=profile["time_slot"],
            y=profile["average_price_eur_mwh"],
            name="Average spot price",
            mode="lines+markers",
            hovertemplate="%{x}<br>%{y:.2f} EUR/MWh<extra></extra>",
            connectgaps=False,
        ),
        secondary_y=True,
    )
    fig.update_layout(
        title=(
            "Average Consumption and Belgian Day-Ahead Spot Price Profile<br>"
            f"Consumption: {consumption_start_date:%d/%m/%Y} to {consumption_end_date:%d/%m/%Y} | "
            f"Spot prices: {price_start_date:%d/%m/%Y} to {price_end_date:%d/%m/%Y}"
        ),
        height=580,
        margin=dict(l=20, r=20, t=70, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(
            title="Time of day",
            tickmode="array",
            tickvals=profile.loc[profile["slot_order"] % 8 == 0, "time_slot"],
            tickangle=0,
        ),
    )
    fig.update_yaxes(title_text="Average consumption (kWh)", rangemode="tozero", secondary_y=False)
    fig.update_yaxes(title_text="Average day-ahead price (EUR/MWh)", secondary_y=True)

    st.plotly_chart(fig, use_container_width=True)


def render_consumption_dashboard() -> None:
    st.header("Average Consumption Profile")

    uploaded_file = st.file_uploader(
        "Excel consumption file",
        type=["xlsx"],
        accept_multiple_files=False,
    )

    if uploaded_file is None:
        st.info("Upload an Excel file with timestamp and kWh columns to generate the profile.")
        return

    try:
        raw_data = pd.read_excel(uploaded_file, sheet_name=0, engine="openpyxl")
        prepared_data = prepare_consumption_data(raw_data)
    except ValueError as error:
        st.error(str(error))
        return
    except Exception as error:
        st.error(f"The uploaded file could not be processed: {error}")
        return

    st.session_state["prepared_consumption_data"] = prepared_data
    st.session_state["consumption_raw_column_count"] = raw_data.shape[1]

    render_summary(prepared_data, raw_data.shape[1])

    view_label = st.selectbox(
        "Profile view",
        options=list(PROFILE_VIEW_OPTIONS.keys()),
        index=0,
        key="consumption_profile_view",
    )
    profile = build_average_profile(prepared_data.data, PROFILE_VIEW_OPTIONS[view_label])
    render_profile_chart(profile, view_label)

    with st.expander("Average profile data"):
        table = profile[["period_label", "time_slot", "average_kwh", "observations"]].copy()
        table["average_kwh"] = table["average_kwh"].round(4)
        st.dataframe(table, use_container_width=True, hide_index=True)


def render_day_ahead_price_dashboard() -> None:
    st.header("Belgian Day-Ahead Spot Prices")
    st.caption(
        "Belgian SDAC day-ahead prices are published officially on ENTSO-E Transparency Platform, "
        "not as a dedicated Elia Open Data API dataset."
    )

    default_end_date = datetime.now(BRUSSELS_TZ).date()
    default_start_date = default_end_date - timedelta(days=7)

    col_a, col_b, col_c = st.columns([1, 1, 2])
    start_date = col_a.date_input(
        "Start date",
        value=default_start_date,
    )
    end_date = col_b.date_input(
        "End date",
        value=default_end_date,
    )
    saved_security_token = get_saved_entsoe_token()
    manual_security_token = col_c.text_input(
        "ENTSO-E API security token",
        type="password",
        help="Leave empty to use the saved local token.",
    )
    security_token = manual_security_token.strip() or saved_security_token

    if not security_token:
        st.info("Enter an ENTSO-E API token to retrieve Belgian day-ahead prices.")
        return

    if saved_security_token and not manual_security_token:
        st.caption("Using saved local ENTSO-E API token.")

    if start_date > end_date:
        st.error("The start date must be before or equal to the end date.")
        return

    try:
        price_data = fetch_entsoe_day_ahead_prices(security_token, start_date, end_date)
    except requests.HTTPError as error:
        st.error(f"ENTSO-E API request failed: {error}")
        return
    except ValueError as error:
        st.error(str(error))
        return
    except Exception as error:
        st.error(f"Could not retrieve day-ahead prices: {error}")
        return

    st.session_state["day_ahead_price_data"] = price_data
    st.session_state["day_ahead_start_date"] = start_date
    st.session_state["day_ahead_end_date"] = end_date

    prices = price_data.prices
    view_label = st.selectbox(
        "Profile view",
        options=list(PROFILE_VIEW_OPTIONS.keys()),
        index=0,
        key="price_profile_view",
    )
    profile = build_average_price_profile(prices, PROFILE_VIEW_OPTIONS[view_label])
    daily_spreads = build_daily_spreads(prices)
    period_days = (end_date - start_date).days + 1

    average_max_spread = daily_spreads["max_spread_eur_mwh"].mean()
    average_four_hour_spread = daily_spreads["four_hour_spread_eur_mwh"].mean()
    average_two_hour_spread = daily_spreads["two_hour_spread_eur_mwh"].mean()
    average_one_hour_spread = daily_spreads["one_hour_spread_eur_mwh"].mean()
    negative_hours = build_negative_hours_evolution(prices)
    price_distribution = build_price_category_distribution(prices)

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Days", f"{period_days:,}")
    col_b.metric("Raw intervals", f"{len(prices):,}")
    col_c.metric("Average", format_price_metric(prices["price_eur_mwh"].mean()))
    col_d.metric(
        "Min / Max",
        f"{prices['price_eur_mwh'].min():.2f} / {prices['price_eur_mwh'].max():.2f}",
    )

    col_a, col_b, col_c, col_d, col_e = st.columns(5)
    col_a.metric("Average max spread", format_price_metric(average_max_spread))
    col_b.metric("Average 4h spread", format_price_metric(average_four_hour_spread))
    col_c.metric("Average 2h spread", format_price_metric(average_two_hour_spread))
    col_d.metric("Average 1h spread", format_price_metric(average_one_hour_spread))
    col_e.metric("Profile intervals", f"{profile['average_price_eur_mwh'].notna().sum():,}")

    if price_data.raw_resolution != "PT15M":
        st.warning(
            "ENTSO-E returned source resolution "
            f"{price_data.raw_resolution}; values were expanded to 15-minute rows for display."
        )

    render_price_profile_chart(profile, start_date, end_date, view_label)
    render_negative_hours_chart(negative_hours)

    st.subheader("Spot Price Distribution")
    distribution_table = price_distribution.copy()
    distribution_table["hours"] = distribution_table["hours"].round(2)
    distribution_table["percentage_of_period"] = distribution_table[
        "percentage_of_period"
    ].round(2)
    distribution_table.columns = [
        "Price category",
        "Hours",
        "% of selected period",
    ]
    st.dataframe(distribution_table, use_container_width=True, hide_index=True)

    with st.expander("Average spot price profile data"):
        table = profile[
            ["period_label", "time_slot", "average_price_eur_mwh", "observations"]
        ].copy()
        table["average_price_eur_mwh"] = table["average_price_eur_mwh"].round(2)
        st.dataframe(table, use_container_width=True, hide_index=True)

    with st.expander("Daily spread data"):
        table = daily_spreads[
            [
                "delivery_date",
                "max_spread_eur_mwh",
                "four_hour_spread_eur_mwh",
                "two_hour_spread_eur_mwh",
                "one_hour_spread_eur_mwh",
                "max_price",
                "min_price",
                "intervals",
            ]
        ].copy()
        table["max_spread_eur_mwh"] = table["max_spread_eur_mwh"].round(2)
        table["four_hour_spread_eur_mwh"] = table["four_hour_spread_eur_mwh"].round(2)
        table["two_hour_spread_eur_mwh"] = table["two_hour_spread_eur_mwh"].round(2)
        table["one_hour_spread_eur_mwh"] = table["one_hour_spread_eur_mwh"].round(2)
        table["max_price"] = table["max_price"].round(2)
        table["min_price"] = table["min_price"].round(2)
        st.dataframe(table, use_container_width=True, hide_index=True)

    with st.expander("Raw day-ahead price data"):
        table = prices[
            ["delivery_date", "mtu_start", "mtu_end", "price_eur_mwh", "source_resolution"]
        ].copy()
        table["price_eur_mwh"] = table["price_eur_mwh"].round(2)
        st.dataframe(table, use_container_width=True, hide_index=True)


def render_combined_dashboard() -> None:
    st.header("Combined Consumption and Spot Price Profile")

    prepared_data = st.session_state.get("prepared_consumption_data")
    price_data = st.session_state.get("day_ahead_price_data")

    if prepared_data is None:
        st.info("Upload a consumption Excel file in the Consumption profile tab first.")
        return

    if price_data is None:
        st.info("Retrieve spot prices in the Day-ahead spot prices tab first.")
        return

    consumption = prepared_data.data
    prices = price_data.prices
    consumption_min_date = consumption["date"].min()
    consumption_max_date = consumption["date"].max()
    price_min_date = prices["delivery_date"].min()
    price_max_date = prices["delivery_date"].max()

    combined_period_key = (
        f"{consumption_min_date.isoformat()}_{consumption_max_date.isoformat()}_"
        f"{price_min_date.isoformat()}_{price_max_date.isoformat()}"
    )
    if st.session_state.get("combined_period_key") != combined_period_key:
        st.session_state["combined_period_key"] = combined_period_key
        st.session_state["combined_consumption_start_date"] = consumption_min_date
        st.session_state["combined_consumption_end_date"] = consumption_max_date
        st.session_state["combined_price_start_date"] = price_min_date
        st.session_state["combined_price_end_date"] = price_max_date

    col_a, col_b, col_c, col_d = st.columns(4)
    consumption_start_date = col_a.date_input(
        "Consumption start date",
        value=consumption_min_date,
        min_value=consumption_min_date,
        max_value=consumption_max_date,
        key="combined_consumption_start_date",
    )
    consumption_end_date = col_b.date_input(
        "Consumption end date",
        value=consumption_max_date,
        min_value=consumption_min_date,
        max_value=consumption_max_date,
        key="combined_consumption_end_date",
    )
    price_start_date = col_c.date_input(
        "Spot price start date",
        value=price_min_date,
        min_value=price_min_date,
        max_value=price_max_date,
        key="combined_price_start_date",
    )
    price_end_date = col_d.date_input(
        "Spot price end date",
        value=price_max_date,
        min_value=price_min_date,
        max_value=price_max_date,
        key="combined_price_end_date",
    )

    if consumption_start_date > consumption_end_date:
        st.error("The consumption start date must be before or equal to the consumption end date.")
        return

    if price_start_date > price_end_date:
        st.error("The spot price start date must be before or equal to the spot price end date.")
        return

    profile = build_combined_average_profile(
        consumption,
        prices,
        consumption_start_date,
        consumption_end_date,
        price_start_date,
        price_end_date,
    )
    (
        arithmetic_price_average,
        weighted_price_average,
        price_difference_percentage,
    ) = calculate_profile_weighted_price_metrics(profile)

    consumption_points = int(profile["average_kwh"].notna().sum())
    price_points = int(profile["average_price_eur_mwh"].notna().sum())
    col_a, col_b, col_c, col_d, col_e, col_f = st.columns(6)
    col_a.metric("Consumption avg", f"{profile['average_kwh'].mean():.3f} kWh")
    col_b.metric("Spot price avg", format_price_metric(arithmetic_price_average))
    col_c.metric("Weighted spot avg", format_price_metric(weighted_price_average))
    col_d.metric("Weighted vs avg", format_percentage_metric(price_difference_percentage))
    col_e.metric("Consumption intervals", f"{consumption_points:,}")
    col_f.metric("Price intervals", f"{price_points:,}")

    matched_profile_points = int(
        profile.dropna(subset=["average_kwh", "average_price_eur_mwh"]).shape[0]
    )
    if matched_profile_points == 0:
        st.warning(
            "No matching 15-minute profile slots were found between consumption and price data, "
            "so the weighted spot price could not be calculated."
        )

    render_combined_profile_chart(
        profile,
        consumption_start_date,
        consumption_end_date,
        price_start_date,
        price_end_date,
    )

    with st.expander("Combined profile data"):
        table = profile[
            [
                "time_slot",
                "average_kwh",
                "average_price_eur_mwh",
                "consumption_observations",
                "price_observations",
            ]
        ].copy()
        table["average_kwh"] = table["average_kwh"].round(4)
        table["average_price_eur_mwh"] = table["average_price_eur_mwh"].round(2)
        st.dataframe(table, use_container_width=True, hide_index=True)



def main() -> None:
    st.set_page_config(
        page_title="Electricity Consumption Dashboard",
        layout="wide",
    )

    st.title("Electricity Consumption Dashboard")

    consumption_tab, price_tab, combined_tab = st.tabs(
        ["Consumption profile", "Day-ahead spot prices", "Combined profile"]
    )

    with consumption_tab:
        render_consumption_dashboard()

    with price_tab:
        render_day_ahead_price_dashboard()

    with combined_tab:
        render_combined_dashboard()


if __name__ == "__main__":
    main()
