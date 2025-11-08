import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px

# ===============================
# 基本设置
# ===============================
st.set_page_config(page_title="Energy Forecasting", layout="wide")
st.title("🏫 School Energy Forecasting (Spain)")

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

uploaded_file = st.file_uploader("上传你的数据文件 (CSV, 包含 energy_kwh, temp_C, rh_pct)", type=["csv"])

# ===============================
# 加载与训练模型
# ===============================
@st.cache_data
def load_data_and_model(file):
    df = pd.read_csv(file, parse_dates=True)
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    if not isinstance(df.index, pd.DatetimeIndex):
        # 若没有日期索引，尝试解析第一列
        if not pd.api.types.is_datetime64_any_dtype(df.iloc[:, 0]):
            df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], errors='coerce')
        df = df.set_index(df.columns[0])
    df = df.sort_index()

    # ========== 滞后与滚动特征 ==========
    for lag in [1, 2, 3, 7, 14, 30]:
        df[f'lag_energy_{lag}'] = df['energy_kwh'].shift(lag)
    df['roll7_energy'] = df['energy_kwh'].rolling(7, min_periods=1).mean()
    df['roll30_energy'] = df['energy_kwh'].rolling(30, min_periods=1).mean()

    # ========== 日期特征 ==========
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    df['dayofyear'] = df.index.dayofyear
    dow_dummies = pd.get_dummies(df['dayofweek'], prefix='dow')
    df = pd.concat([df, dow_dummies], axis=1)

    # ========== 西班牙节假日 + 学校假期 ==========
    SPAIN_HOLIDAYS = [
        "2025-01-01", "2025-01-06", "2025-03-20", "2025-03-21",
        "2025-05-01", "2025-08-15", "2025-10-12",
        "2025-11-01", "2025-12-06", "2025-12-08", "2025-12-25",
    ]
    SCHOOL_HOLIDAYS = [
        ("2025-12-20", "2026-01-06"),
        ("2025-03-24", "2025-03-30"),
        ("2025-07-01", "2025-08-31"),
    ]

    df['is_weekend'] = df.index.dayofweek >= 5
    spain_holidays = pd.to_datetime(SPAIN_HOLIDAYS)
    df['is_holiday'] = df.index.isin(spain_holidays)

    def in_school_holiday(date):
        for start, end in SCHOOL_HOLIDAYS:
            if pd.Timestamp(start) <= date <= pd.Timestamp(end):
                return True
        return False

    df['is_school_holiday'] = df.index.map(in_school_holiday)
    df['is_term_time'] = ~(df['is_school_holiday'] | df['is_holiday'] | df['is_weekend'])

    df_model = df.dropna().copy()
    target_col = 'energy_kwh'
    features = [c for c in df_model.columns if c != target_col]

    model_path = DATA_DIR / "rf_energy_model.joblib"
    features_path = DATA_DIR / "rf_features.joblib"

    try:
        model = joblib.load(model_path)
        saved_features = joblib.load(features_path)
        if set(saved_features) != set(features):
            raise RuntimeError("特征变化，重新训练中...")
    except Exception as e:
        st.warning(f"⚠️ 模型加载失败 ({e})，正在重新训练模型...")
        X = df_model[features]
        y = df_model[target_col]
        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=5,
            random_state=42
        )
        model.fit(X, y)
        joblib.dump(model, model_path)
        joblib.dump(features, features_path)
        st.success("✅ 模型重新训练完成。")

    return df, model, features


# ===============================
# 预测函数
# ===============================
def iterative_forecast(model, df, horizon, features):
    preds = []
    current_df = df.copy()
    energy_col = 'energy_kwh'

    SPAIN_HOLIDAYS = [
        "2025-01-01", "2025-01-06", "2025-03-20", "2025-03-21",
        "2025-05-01", "2025-08-15", "2025-10-12",
        "2025-11-01", "2025-12-06", "2025-12-08", "2025-12-25",
    ]
    SCHOOL_HOLIDAYS = [
        ("2025-12-20", "2026-01-06"),
        ("2025-03-24", "2025-03-30"),
        ("2025-07-01", "2025-08-31"),
    ]

    def in_school_holiday(date):
        for start, end in SCHOOL_HOLIDAYS:
            if pd.Timestamp(start) <= date <= pd.Timestamp(end):
                return True
        return False

    for day in range(1, horizon + 1):
        next_date = current_df.index.max() + pd.Timedelta(days=1)
        row = {}

        # 滞后特征
        for lag in [1, 2, 3, 7, 14, 30]:
            row[f'lag_energy_{lag}'] = (
                current_df[energy_col].iloc[-lag]
                if len(current_df) >= lag
                else current_df[energy_col].iloc[-1]
            )

        # 滚动特征
        row['roll7_energy'] = current_df[energy_col].rolling(7, min_periods=1).mean().iloc[-1]
        row['roll30_energy'] = current_df[energy_col].rolling(30, min_periods=1).mean().iloc[-1]

        # 日期特征
        row['dayofweek'] = next_date.dayofweek
        row['month'] = next_date.month
        row['dayofyear'] = next_date.dayofyear
        for i in range(7):
            row[f'dow_{i}'] = 1 if next_date.dayofweek == i else 0

        # 节假日与学校假期
        row['is_weekend'] = next_date.dayofweek >= 5
        row['is_holiday'] = next_date in pd.to_datetime(SPAIN_HOLIDAYS)
        row['is_school_holiday'] = in_school_holiday(next_date)
        row['is_term_time'] = not (row['is_weekend'] or row['is_holiday'] or row['is_school_holiday'])

        X = pd.DataFrame([row])[features]
        pred = model.predict(X)[0]
        preds.append((next_date, pred))

        new_row = pd.Series({energy_col: pred}, name=next_date)
        current_df = pd.concat([current_df, new_row.to_frame().T])

    return preds


# ===============================
# 主流程
# ===============================
if uploaded_file is not None:
    df, model, features = load_data_and_model(uploaded_file)

    st.subheader("📊 历史数据预览")
    st.dataframe(df.tail(10))

    horizon = st.slider("预测天数", 7, 90, 30)

    if st.button("开始预测"):
        forecasts = iterative_forecast(model, df, horizon, features)
        forecast_df = pd.DataFrame(forecasts, columns=["date", "pred_energy_kwh"]).set_index("date")

        st.success("✅ 预测完成！")

        # 合并实际与预测
        full = pd.concat([
            df[['energy_kwh']].rename(columns={'energy_kwh': 'actual'}),
            forecast_df.rename(columns={'pred_energy_kwh': 'forecast'})
        ])

        fig = px.line(full, x=full.index, y=full.columns, title="📈 Energy Forecasting (Actual vs Predicted)")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📅 周平均对比（验证周期性是否合理）")
        full['dayofweek'] = full.index.dayofweek
        weekly_avg = full.groupby('dayofweek').mean().rename(index={
            0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'
        })
        fig2 = px.bar(weekly_avg, barmode='group', title="Weekly Pattern (Actual vs Forecast)")
        st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("请先上传包含 `energy_kwh`, `temp_C`, `rh_pct` 的 CSV 文件。")



# ======================
# 侧边栏信息
# ======================
st.sidebar.markdown("---")
st.sidebar.write("模型信息：RandomForestRegressor")
st.sidebar.write("训练范围：2020-01-01 至 2023-12-31")
st.sidebar.write("特征：滞后 + 日历 + 滚动均值")
st.sidebar.write("可改进方向：天气预测、节假日、建筑使用计划等")
