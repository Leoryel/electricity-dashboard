# Electricity Dashboard

Streamlit dashboard for analyzing quarter-hourly electricity consumption and Belgian day-ahead spot prices.

## Features

- Upload Excel consumption data with timestamp and kWh columns.
- Build average daily consumption profiles by year, quarter, season, month, or weekday.
- Retrieve Belgian day-ahead spot prices from ENTSO-E.
- Build average spot-price profiles and spread metrics.
- Compare consumption and spot-price profiles on a combined double-axis chart.

## Run Locally

```powershell
py -m pip install -r requirements.txt
py -m streamlit run dashboard.py
```

The app opens at:

```text
http://localhost:8501
```
