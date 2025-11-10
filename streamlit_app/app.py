import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
import datetime

# =============================
# 路径与缓存设置
# =============================
DATA_DIR = Path(__file__).parent
st.set_page_config(page_title="EPSEVG 能耗仪表盘", layout="wide")

# =============================
# 定义假期与辅助函数
# =============================

def spain_holidays(year):
    """定义西班牙节假日（示例，可扩展）"""
    fixed_holidays = [
        "01-01", "01-06", "04-15", "05-01", "08-15",
        "10-12", "11-01", "12-06", "12-08", "12-25"
    ]
    return [datetime.datetime.strptime(f"{year}-{d}", "%Y-%m-%d").date() for d in fixed_holidays]

def in_school_holiday(date):
    """简单示例：7-8月为暑假，圣诞节假期"""
    if date.month in [7, 8]:
        return True
    if date.month == 12 and date.day >= 20:
        return True
    if date.month == 1 and date.day <= 7:
        return True
    return False

# =============================
# 加载或重训模型
# =============================
@st.cache_data
def load_data_and_model():
    df = pd.read_csv(DATA_DIR / "df_daily_processed.csv", index_col=0, parse_dates=True)

    # 自动识别能耗列名
    target_col = [c for c in df.columns if "energy" in c.lower()][0]
    y = df[target_col]
    features = [c for c in df.columns if c != target_col]
    X = df[features]

    # 尝试加载模型
    model_path = DATA_DIR / "rf_energy_model.joblib"
    try:
        model = joblib.load(model_path)
    except Exception as e:
        st.warning(f"⚠️ 模型加载失败（{e}），正在重新训练...")
        model = RandomForestRegressor(n_estimators=200, random_state=42)
        model.fit(X, y)
        joblib.dump(model, model_path)
    return df, model, features, target_col

# =============================
# 预测函数（已修正：周末与节假日处理）
# =============================
def iterative_forecast(model, df, features, horizon):
    preds = []
    current_df = df.copy()
    last_date = current_df.index.max()

    for day in range(1, horizon + 1):
        next_date = last_date + pd.Timedelta(days=day)
        row = {}

        # 滞后特征
        for lag in [1, 2, 3, 7, 14, 30, 60]:
            lag_col = f"lag_{lag}"
            if lag_col in features:
                row[lag_col] = current_df.iloc[-lag:][features[0]].iloc[-1] if len(current_df) >= lag else current_df.iloc[-1][features[0]]

        # 日历特征
        row["dayofweek"] = next_date.dayofweek
        row["month"] = next_date.month
        row["dayofyear"] = next_date.dayofyear

        # 周末 / 节假日 / 学校假期 / 学期时间
        year_holidays = spain_holidays(next_date.year)
        row["is_weekend"] = int(next_date.dayofweek >= 5)
        row["is_holiday"] = int(next_date.date() in year_holidays)
        row["is_school_holiday"] = int(in_school_holiday(next_date))
        row["is_term_time"] = int(not (row["is_weekend"] or row["is_holiday"] or row["is_school_holiday"]))

        # 滚动均值
        if "roll7_mean" in features:
            row["roll7_mean"] = current_df.iloc[-7:][features[0]].mean()
        if "roll30_mean" in features:
            row["roll30_mean"] = current_df.iloc[-30:][features[0]].mean()

        # 确保所有列都存在
        for f in features:
            if f not in row:
                row[f] = current_df[f].iloc[-1] if f in current_df.columns else 0

        X_pred = pd.DataFrame([row])[features]
        y_pred = model.predict(X_pred)[0]
        preds.append((next_date, y_pred))

        # 将预测结果追加回 df
        new_row = pd.Series({features[0]: y_pred}, name=next_date)
        current_df = pd.concat([current_df, new_row.to_frame().T])

    return pd.DataFrame(preds, columns=["date", "predicted_energy"]).set_index("date")

# =============================
# 页面主体
# =============================
st.title("🏫 EPSEVG 能耗分析与预测 Dashboard")

df, model, features, target_col = load_data_and_model()

col1, col2 = st.columns([1, 2])
with col1:
    horizon = st.selectbox("选择预测天数", [7, 15, 30, 90], index=2)
with col2:
    st.markdown("模型: RandomForestRegressor · 特征: 滞后 + 日历 + 假期")

# 执行预测
preds = iterative_forecast(model, df, features, horizon)

# 合并历史与预测数据
df_view = df[[target_col]].copy()
df_view = pd.concat([df_view, preds.rename(columns={"predicted_energy": target_col})])
df_view["type"] = ["历史"] * len(df) + ["预测"] * len(preds)

# 绘图（不显示表格）
fig = px.line(df_view, x=df_view.index, y=target_col, color="type",
              labels={"x": "日期", target_col: "能耗 (kWh)", "type": "数据类型"},
              title=f"EPSEVG 能耗历史与未来 {horizon} 天预测")
fig.update_traces(line=dict(width=2))
st.plotly_chart(fig, use_container_width=True)

st.caption("📊 工作日能耗较高，周末与节假日应较低。模型基于 RandomForest，使用滞后值与日期特征。")
