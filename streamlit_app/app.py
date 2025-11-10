import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from prophet import Prophet
from pathlib import Path
import datetime

# -----------------------------------
# 页面配置
# -----------------------------------
st.set_page_config(page_title="EPSEVG 能耗预测（Prophet）", layout="wide")
st.title("🏫 EPSEVG 能耗分析与预测 Dashboard（Prophet）")

DATA_DIR = Path(__file__).parent

# -----------------------------------
# 1️⃣ 加载数据
# -----------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_DIR / "df_daily_processed.csv", index_col=0, parse_dates=True)
    # 自动识别能耗列
    target_col = [c for c in df.columns if "energy" in c.lower()][0]
    df = df[[target_col]].rename(columns={target_col: "y"})
    df = df.reset_index().rename(columns={"index": "ds"})
    df = df.sort_values("ds")
    return df

df = load_data()

# -----------------------------------
# 2️⃣ 定义节假日（西班牙通用 + 校园假期）
# -----------------------------------
def make_holiday_df(start_year=2020, end_year=2025):
    holidays = []
    for year in range(start_year, end_year + 1):
        for d in ["01-01", "01-06", "05-01", "08-15", "10-12", "11-01", "12-06", "12-08", "12-25"]:
            holidays.append({"holiday": "national_holiday", "ds": f"{year}-{d}"})
        # 学校假期：7、8月为暑假
        for m in [7, 8]:
            for day in range(1, 32):
                try:
                    holidays.append({"holiday": "school_summer", "ds": f"{year}-{m:02d}-{day:02d}"})
                except:
                    pass
    return pd.DataFrame(holidays)

holiday_df = make_holiday_df(df["ds"].dt.year.min(), df["ds"].dt.year.max() + 1)

# -----------------------------------
# 3️⃣ 模型训练
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
# 4️⃣ 用户输入预测范围
# -----------------------------------
col1, col2 = st.columns([1, 2])
with col1:
    horizon = st.selectbox("选择预测天数", [7, 15, 30, 90], index=2)
with col2:
    st.markdown("模型: **Prophet** · 自动捕捉周末/年度季节性与假期影响")

# -----------------------------------
# 5️⃣ 生成预测
# -----------------------------------
future = model.make_future_dataframe(periods=horizon)
forecast = model.predict(future)

# -----------------------------------
# 6️⃣ 可视化
# -----------------------------------
fig = px.line(
    forecast,
    x="ds",
    y="yhat",
    labels={"ds": "日期", "yhat": "能耗 (kWh)"},
    title=f"EPSEVG 能耗历史与未来 {horizon} 天预测（Prophet 模型）"
)
fig.add_scatter(
    x=df["ds"],
    y=df["y"],
    mode="lines",
    name="历史能耗",
    line=dict(width=2, color="blue")
)
fig.update_traces(line=dict(width=2))
st.plotly_chart(fig, use_container_width=True)

# -----------------------------------
# 7️⃣ 页面说明
# -----------------------------------
st.caption("""
📊 本模型使用 **Facebook Prophet** 自动学习能耗的季节性规律：  
- 周一至周五能耗较高；  
- 周末及节假日较低；  
- 年度周期（如夏季低谷、冬季高峰）自动捕捉。  
""")
