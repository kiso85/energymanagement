import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px

# ======================
# 基本配置
# ======================
DATA_DIR = Path(__file__).parent
st.set_page_config(page_title="EPSEVG 能耗仪表板", layout="wide")

st.title("EPSEVG 能耗仪表板（2020-2024）")
st.markdown("展示历史日能耗，并使用 RandomForest 预测未来 7/15/30/90 天的能耗（逐步预测）")


# ======================
# 数据加载与模型训练
# ======================
@st.cache_data
def load_data_and_model():
    df = pd.read_csv(DATA_DIR / "df_daily_processed.csv", index_col=0, parse_dates=True)

    model_path = DATA_DIR / "rf_energy_model.joblib"
    features_path = DATA_DIR / "rf_features.joblib"

    try:
        model = joblib.load(model_path)
        features = joblib.load(features_path)
    except Exception as e:
        st.warning(f"⚠️ 模型加载失败 ({e})，正在重新训练模型...")
        target_col = [c for c in df.columns if "energy" in c.lower()][0]
        features = [c for c in df.columns if c != target_col]
        X = df[features]
        y = df[target_col]
        model = RandomForestRegressor(n_estimators=200, random_state=42)
        model.fit(X, y)
        joblib.dump(model, model_path)
        joblib.dump(features, features_path)

    return df, model, features


df, model, features = load_data_and_model()


# ======================
# 侧边栏筛选日期
# ======================
st.sidebar.header("设置")
start_date = st.sidebar.date_input("显示开始日期", value=df.index.min().date())
end_date = st.sidebar.date_input("显示结束日期", value=df.index.max().date())

# 筛选显示的数据
df_view = df.loc[str(start_date):str(end_date)].copy()


# ======================
# 历史能耗趋势
# ======================
energy_col = [c for c in df_view.columns if "energy" in c.lower()][0]

st.subheader("历史能耗趋势")
fig = px.line(
    df_view,
    x=df_view.index,
    y=energy_col,
    labels={'x': '日期', energy_col: '能耗 (kWh)'},
    title='EPSEVG 能耗趋势（历史数据）'
)
st.plotly_chart(fig, use_container_width=True)


# ======================
# 预测函数
# ======================
def iterative_forecast(model, df, features, horizon):
    preds = []
    current_df = df.copy()

    for day in range(1, horizon + 1):
        next_date = current_df.index.max() + pd.Timedelta(days=1)
        row = {}

        # 滞后特征
        for lag in [1, 2, 3, 7, 14, 30, 60]:
            row[f'lag_{lag}'] = (
                current_df['energy_kwh'].iloc[-lag]
                if len(current_df) >= lag
                else current_df['energy_kwh'].iloc[-1]
            )

        # 日历特征
        row['dayofweek'] = next_date.dayofweek
        row['month'] = next_date.month
        row['dayofyear'] = next_date.dayofyear

        # 滚动平均特征
        row['roll7_mean'] = current_df['energy_kwh'].rolling(7).mean().iloc[-1]
        row['roll30_mean'] = current_df['energy_kwh'].rolling(30).mean().iloc[-1]

        X = pd.DataFrame([row])

        # 🔧 确保列名完全匹配训练特征
        for col in features:
            if col not in X.columns:
                X[col] = 0
        X = X[features]

        # 预测
        pred = model.predict(X)[0]
        preds.append((next_date, pred))

        # 将预测结果加入当前数据集
        new_row = pd.Series({'energy_kwh': pred, 'temp_C': np.nan, 'rh_pct': np.nan}, name=next_date)
        current_df = pd.concat([current_df, new_row.to_frame().T])

    return preds


# ======================
# 预测展示
# ======================
st.subheader("未来能耗预测")

horizons = st.multiselect("选择预测天数（天）", [7, 15, 30, 90], default=[7, 15, 30, 90])

if len(horizons) == 0:
    st.info("请在左侧选择至少一个预测天数（例如 7）")
else:
    all_forecasts = {}
    for h in horizons:
        all_forecasts[h] = iterative_forecast(model, df, features, h)

    # 展示预测结果
    for h, preds in all_forecasts.items():
        st.markdown(f"### 🔹 {h}-天预测结果")
        dates = [p[0] for p in preds]
        values = [p[1] for p in preds]
        dfp = pd.DataFrame({'date': dates, 'pred_kwh': values}).set_index('date')

        fig2 = px.line(dfp, x=dfp.index, y='pred_kwh',
                       labels={'x': '日期', 'pred_kwh': '预测能耗 (kWh)'},
                       title=f'{h}-天能耗预测')
        st.plotly_chart(fig2, use_container_width=True)
        st.table(dfp.round(2))


# ======================
# 侧边栏信息
# ======================
st.sidebar.markdown("---")
st.sidebar.write("模型信息：RandomForestRegressor")
st.sidebar.write("训练范围：2020-01-01 至 2023-12-31")
st.sidebar.write("特征：滞后 + 日历 + 滚动均值")
st.sidebar.write("可改进方向：天气预测、节假日、建筑使用计划等")
