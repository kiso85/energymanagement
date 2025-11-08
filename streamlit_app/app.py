
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import plotly.express as px

DATA_DIR = Path("/mnt/data")
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_DIR/"df_daily_processed.csv", index_col=0, parse_dates=True)
    model = joblib.load(DATA_DIR/"rf_energy_model.joblib")
    return df, model

df, model = load_data()

st.set_page_config(page_title="EPSEVG 能耗仪表板", layout="wide")
st.title("EPSEVG 能耗仪表板（2020-2024）")
st.markdown("展示历史日能耗，并使用 RandomForest 预测未来 7/15/30/90 天的能耗（逐步预测）")

# Sidebar controls
st.sidebar.header("设置")
start_date = st.sidebar.date_input("显示开始日期", value=df.index.min().date())
end_date = st.sidebar.date_input("显示结束日期", value=df.index.max().date())

# Filter data
df_view = df.loc[str(start_date):str(end_date)].copy()

st.subheader("历史日能耗总览")
fig = px.line(df_view, x=df_view.index, y='energy_kwh', labels={'x':'日期','energy_kwh':'能耗 (kWh)'})
st.plotly_chart(fig, use_container_width=True)

# Forecasting
st.subheader("未来能耗预测")
horizons = st.multiselect("选择预测天数（天）", [7,15,30,90], default=[7,15,30,90])

def iterative_forecast(model, df, horizon):
    preds = []
    current_df = df.copy()
    for day in range(1, horizon+1):
        next_date = current_df.index.max() + pd.Timedelta(days=1)
        row = {}
        for lag in [1,2,3,7,14,30,60]:
            row[f'lag_{lag}'] = current_df['energy_kwh'].iloc[-lag] if len(current_df)>=lag else current_df['energy_kwh'].iloc[-1]
        row['dayofweek'] = next_date.dayofweek
        row['month'] = next_date.month
        row['dayofyear'] = next_date.dayofyear
        row['roll7_mean'] = current_df['energy_kwh'].rolling(7).mean().iloc[-1]
        row['roll30_mean'] = current_df['energy_kwh'].rolling(30).mean().iloc[-1]
        X = pd.DataFrame([row])
        pred = model.predict(X)[0]
        preds.append((next_date, pred))
        new_row = pd.Series({'energy_kwh':pred,'temp_C':np.nan,'rh_pct':np.nan}, name=next_date)
        current_df = current_df.append(new_row)
    return preds

if len(horizons)==0:
    st.info("请在左侧选择至少一个预测天数（例如 7）")
else:
    all_forecasts = {}
    for h in horizons:
        all_forecasts[h] = iterative_forecast(model, df, h)

    # Show a combined plot overlaying history and forecasts
    for h, preds in all_forecasts.items():
        st.markdown(f"**{h}-天预测**")
        dates = [p[0] for p in preds]
        values = [p[1] for p in preds]
        dfp = pd.DataFrame({'date':dates, 'pred_kwh':values}).set_index('date')
        fig2 = px.line(dfp, x=dfp.index, y='pred_kwh', labels={'x':'日期','pred_kwh':'预测能耗 (kWh)'})
        st.plotly_chart(fig2, use_container_width=True)
        st.table(dfp.round(1))

st.sidebar.markdown("---")
st.sidebar.write("模型信息：RandomForestRegressor。训练范围：2020-01-01 至 2023-12-31")
st.sidebar.write("注意：模型为基线模型（滞后 + 日历特征 + 滚动均值）。可改进：加入天气预测、节假日、建筑使用计划等。")
