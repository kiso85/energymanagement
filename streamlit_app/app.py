import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from prophet import Prophet
from pathlib import Path
import datetime

# -----------------------------------
# Page configuration
# -----------------------------------
st.set_page_config(page_title="EPSEVG Energy Forecast (Prophet)", layout="wide")
st.title("🏫 EPSEVG Energy Consumption Forecast Dashboard (Prophet)")

DATA_DIR = Path(__file__).parent

# -----------------------------------
# 1️⃣ Load data
# -----------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_DIR / "df_daily_processed.csv", parse_dates=True)

    # Detect date and energy columns automatically
    date_col = None
    for c in df.columns:
        if "date" in c.lower():
            date_col = c
            break

    target_col = [c for c in df.columns if "energy" in c.lower()][0]

    # Parse datetime column
    if date_col is not None:
        df["ds"] = pd.to_datetime(df[date_col])
    else:
        df["ds"] = pd.to_datetime(df.iloc[:, 0], errors="coerce")

    df["y"] = df[target_col].astype(float)
    df = df[["ds", "y"]].dropna().sort_values("ds")

    return df

df = load_data()

# -----------------------------------
# 2️⃣ Define holidays (Spain national + school holidays)
# -----------------------------------
def make_holiday_df(start_year=2020, end_year=2025):
    holidays = []
    for year in range(start_year, end_year + 1):
        for d in ["01-01", "01-06", "05-01", "08-15", "10-12", "11-01", "12-06", "12-08", "12-25"]:
            holidays.append({"holiday": "national_holiday", "ds": f"{year}-{d}"})
        # School summer holidays (July–August)
        for m in [7, 8]:
            for day in range(1, 32):
                try:
                    holidays.append({"holiday": "school_summer", "ds": f"{year}-{m:02d}-{day:02d}"})
                except:
                    pass
    return pd.DataFrame(holidays)

holiday_df = make_holiday_df(df["ds"].dt.year.min(), df["ds"].dt.year.max() + 1)

# -----------------------------------
# 3️⃣ Train Prophet model
# -----------------------------------
@st.cache_resource
def train_prophet(df, holidays):
    m = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        holidays=holidays,
        seasonality_mode="multiplicative"
    )
    m.add_seasonality(name="monthly", period=30.5, fourier_order=5)
    m.fit(df)
    return m

model = train_prophet(df, holiday_df)

# -----------------------------------
# 4️⃣ User input: forecast horizon
# -----------------------------------
col1, col2 = st.columns([1, 2])
with col1:
    horizon = st.selectbox("Select forecast horizon (days)", [7, 15, 30, 90], index=2)
with col2:
    st.markdown("**Model:** Prophet · Automatically captures weekly and yearly seasonality + holidays")

# -----------------------------------
# 5️⃣ Forecast
# -----------------------------------
future = model.make_future_dataframe(periods=horizon)
forecast = model.predict(future)

# -----------------------------------
# 6️⃣ Visualization
# -----------------------------------
fig = px.line(
    forecast,
    x="ds",
    y="yhat",
    labels={"ds": "Date", "yhat": "Predicted Energy (kWh)"},
    title=f"EPSEVG Energy Consumption Forecast for Next {horizon} Days (Prophet)"
)
fig.add_scatter(
    x=df["ds"],
    y=df["y"],
    mode="lines",
    name="Historical Energy",
    line=dict(width=2, color="blue")
)
fig.update_traces(line=dict(width=2))
st.plotly_chart(fig, use_container_width=True)

# -----------------------------------
# 7️⃣ Notes
# -----------------------------------
st.caption("""
📊 This dashboard uses **Facebook Prophet** to model EPSEVG daily energy consumption.

- Higher values during weekdays.
- Lower energy consumption during weekends and public/school holidays.
- Prophet automatically captures weekly, yearly, and holiday seasonality patterns.
""")
