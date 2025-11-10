import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
import datetime

st.set_page_config(page_title="EPSEVG 能耗仪表盘", layout="wide")

DATA_DIR = Path(__file__).parent

# =============================
# 节假日与假期规则
# =============================
def spain_holidays(year):
    fixed = ["01-01", "01-06", "05-01", "08-15", "10-12", "11-01", "12-06", "12-08", "12-25"]
    return [datetime.date.fromisoformat(f"{year}-{d}") for d in fixed]

def in_school_holiday(date):
    if date.month in [7, 8]:  # 暑假
        return True
    if (date.month == 12 and date.day >= 20) or (date.month == 1 and date.day <= 7):  # 圣诞假期
        return True
    return False

# =============================
# 数据与模型加载
# =============================
@st.cache_data
def load_data_and_model():
    df = pd.read_csv(DATA_DIR / "df_daily_processed.csv", index_col=0, parse_dates=True)

    # 自动识别能耗列
    target_col = [c for c in df.columns if "energy" in c.lower()][0]
    y = df[target_col]
    features = [c for c in df.columns if c != target_col]
    X = df[features]

    # 尝试加载模型或重新训练
    model_path = DATA_DIR / "rf_energy_model.joblib"
    try:
        model = joblib.load(model_path)
    except Exception:
        st.warning("⚙️ 模型文件不兼容，正在重新训练...")
        model = RandomForestRegressor(n_estimators=200, random_state=42)
        model.fit(X, y)
        joblib.dump(model, model_path)
    return df, model, features, target_col

# =============================
# 修正版预测函数
# =============================
def iterative_forecast(model, df, features, target_col, horizon):
    current_df = df.copy()
    preds = []

    for i in range(1, horizon + 1):
        next_date = current_df.index.max() + pd.Timedelta(days=1)
        row = {}

        # 滞后特征：从能耗列提取
        for lag in [1, 2, 3, 7, 14, 30, 60]:
            val = current_df[target_col].iloc[-lag] if len(current_df) >= lag else current_df[target_col].iloc[-1]
            row[f"lag_{lag}"] = val

        # 日历与假期特征
        row["dayofweek"] = next_date.dayofweek
        row["month"] = next_date.month
        row["dayofyear"] = next_date.dayofyear
        row["is_weekend"] = int(next_date.dayofweek >= 5)
        row["is_holiday"] = int(next_date.date() in spain_holidays(next_date.year))
        row["is_school_holiday"] = int(in_school_holiday(next_date))
        row["is_term_time"] = int(not (row["is_weekend"] or row["is_holiday"] or row["is_school_holiday"]))

        # 滚动均值
        row["roll7_mean"] = current_df[target_col].tail(7).mean()
        row["roll30_mean"] = current_df[target_col].tail(30).mean()

        # 确保所有特征列都存在
        for f in features:
            if f not in row:
                if f in current_df.columns:
                    row[f] = current_df[f].iloc[-1]
                else:
                    row[f] = 0

        X_pred = pd.DataFrame([row])[features]
        pred = model.predict(X_pred)[0]
        preds.append((next_date, pred))

        # 追加预测行
        new_row = pd.Series({target_col: pred}, name=next_date)
        current_df = pd.concat([current_df, new_row.to_frame().T])

    return pd.DataFrame(preds, columns=["date", "prediction"]).set_index("date")

# =============================
# Streamlit 页面
# =============================
st.title("🏫 EPSEVG 能耗分析与预测 Dashboard")

df, model, features, target_col = load_data_and_model()

col1, col2 = st.columns([1, 2])
with col1:
    horizon = st.selectbox("选择预测天数", [7, 15, 30, 90], index=2)
with col2:
    st.markdown("模型: RandomForest · 特征: 滞后 + 日期 + 假期")

pred_df = iterative_forecast(model, df, features, target_col, horizon)

# 合并历史 + 预测
df_plot = pd.concat([
    df[[target_col]].assign(type="历史"),
    pred_df.rename(columns={"prediction": target_col}).assign(type="预测")
])

# 绘制折线图（不显示表格）
fig = px.line(
    df_plot,
    x=df_plot.index,
    y=target_col,
    color="type",
    labels={"x": "日期", target_col: "能耗 (kWh)", "type": "数据类型"},
    title=f"EPSEVG 能耗历史与未来 {horizon} 天预测"
)
fig.update_traces(line=dict(width=2))
st.plotly_chart(fig, use_container_width=True)

st.caption("📊 模型基于滞后值与日期特征。预计工作日能耗较高，周末与节假日较低。")
