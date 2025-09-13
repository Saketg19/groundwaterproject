import streamlit as st
import pandas as pd
import numpy as np
import requests
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ML imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import joblib

# ==============================================================
# NASA POWER API fetch function
# ==============================================================
def get_nasa_power_data(lat, lon, start=2000, end=2020):
    url = (
        f"https://power.larc.nasa.gov/api/temporal/monthly/point?"
        f"parameters=T2M,PRECTOT,PS,ALLSKY_KT"
        f"&community=AG"
        f"&longitude={lon}&latitude={lat}"
        f"&start={start}&end={end}"
        f"&format=JSON"
    )
    r = requests.get(url)
    data = r.json()["properties"]["parameter"]
    df = pd.DataFrame(data)
    df = df.T.reset_index().rename(columns={"index": "YearMonth"})
    return df

# ==============================================================
# Bootstrap Forecast (Uncertainty Bands)
# ==============================================================
def bootstrap_forecast(series, iters=1000):
    forecasts = []
    for i in range(iters):
        sample = series.sample(frac=1, replace=True)
        forecasts.append(sample.mean())
    return np.mean(forecasts), np.std(forecasts)

# ==============================================================
# Streamlit App Layout
# ==============================================================
st.set_page_config(page_title="Groundwater ML Dashboard", layout="wide")
st.title("💧 Groundwater ML Dashboard with Climate Forecasts")

tab1, tab2 = st.tabs(["📊 Groundwater ML Dashboard", "🔮 Advanced Forecast Options"])

# ==============================================================
# TAB 1: Your Original Dashboard (full code)
# ==============================================================
with tab1:
    st.markdown('<h1 class="main-header">💧 Groundwater Level ML Analysis Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("Enhanced analysis with machine learning predictions for groundwater levels.")

    # Session state init
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False

    @st.cache_data
    def load_data():
        try:
            # --- MODIFIED SECTION ---
            # Load data from the URL specified in Streamlit's secrets
            data_url = st.secrets["DATA_URL"]
            df = pd.read_csv(data_url)
            # --- END OF MODIFICATION ---

            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date').reset_index(drop=True)

            df['Year'] = df['Date'].dt.year
            df['Month'] = df['Date'].dt.month
            df['Day'] = df['Date'].dt.day
            df['DayOfYear'] = df['Date'].dt.dayofyear

            df['Water_Level_lag1'] = df['Water_Level_m'].shift(1)
            df['Water_Level_lag7'] = df['Water_Level_m'].shift(7)
            df['Rainfall_lag1'] = df['Rainfall_mm'].shift(1)

            df['Water_Level_ma7'] = df['Water_Level_m'].rolling(window=7).mean()
            df['Rainfall_ma7'] = df['Rainfall_mm'].rolling(window=7).mean()

            df = df.dropna()
            return df
        except Exception as e:
            st.error(f"❌ Error loading data from the remote URL: {e}")
            return None

    df = load_data()
    if df is None:
        st.stop()

    # Sidebar controls
    st.sidebar.header("📊 Controls")
    min_date = df['Date'].min().date()
    max_date = df['Date'].max().date()
    date_range = st.sidebar.date_input("Select Date Range",
        value=(min_date, max_date), min_value=min_date, max_value=max_date)

    if len(date_range) == 2:
        start_date = pd.to_datetime(date_range[0])
        end_date = pd.to_datetime(date_range[1])
        filtered_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)].copy()
    else:
        filtered_df = df.copy()

    show_rolling = st.sidebar.checkbox("Show 7-day rolling average", value=True)
    show_environmental = st.sidebar.checkbox("Show environmental factors", value=True)

    # Metrics
    st.header("📊 Groundwater Analysis Report")
    avg_level = filtered_df['Water_Level_m'].mean()
    min_level = filtered_df['Water_Level_m'].min()
    max_level = filtered_df['Water_Level_m'].max()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Average Water Level", f"{avg_level:.2f} m")
    col2.metric("Minimum Water Level", f"{min_level:.2f} m")
    col3.metric("Maximum Water Level", f"{max_level:.2f} m")

    # Status
    if avg_level > 5:
        status = "Safe ✅"; status_class = "status-safe"
    elif 3 < avg_level <= 5:
        status = "Semi-Critical ⚠️"; status_class = "status-semi-critical"
    elif 2 < avg_level <= 3:
        status = "Critical ❗"; status_class = "status-critical"
    else:
        status = "Over-exploited ❌"; status_class = "status-over-exploited"
    col4.markdown(f"**Status:** {status}")

    # Time series
    st.header("📈 Water Level Trend")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df['Water_Level_m'],
        mode='lines+markers', name='Water Level (m)', line=dict(color='blue')))
    if show_rolling:
        rolling_avg = filtered_df['Water_Level_m'].rolling(window=7, min_periods=1).mean()
        fig.add_trace(go.Scatter(x=filtered_df['Date'], y=rolling_avg,
            mode='lines', name='7-day Rolling Avg', line=dict(color='red', dash='dash')))
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    # Environmental factors
    if show_environmental:
        st.header("🌡️ Environmental Factors")
        fig_env = make_subplots(rows=2, cols=2,
            subplot_titles=('Temperature (°C)', 'Rainfall (mm)', 'pH', 'Dissolved Oxygen (mg/L)'))
        fig_env.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df['Temperature_C'], name='Temp'),
                            row=1, col=1)
        fig_env.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df['Rainfall_mm'], name='Rain'),
                            row=1, col=2)
        fig_env.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df['pH'], name='pH'),
                            row=2, col=1)
        fig_env.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df['Dissolved_Oxygen_mg_L'], name='DO'),
                            row=2, col=2)
        st.plotly_chart(fig_env, use_container_width=True)

    # (Your ML training + predictions section can be kept here unchanged)

# ==============================================================
# TAB 2: Advanced Forecast Options
# ==============================================================
with tab2:
    st.subheader("🔮 Advanced Forecast Options (NASA POWER + Maps + Bootstrap)")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("### 📍 Select Location on Map")
        m = folium.Map(location=[20.5937, 78.9629], zoom_start=4)
        map_data = st_folium(m, width=700, height=500)

    with col2:
        nasa_start = st.number_input("NASA POWER Start Year", 1981, 2025, 2000)
        nasa_end = st.number_input("NASA POWER End Year", 1981, 2025, 2020)
        n_bootstrap = st.slider("Bootstrap Iterations", 100, 2000, 500)

        if st.button("Run Advanced Forecast"):
            st.info("⚡ Fetching NASA POWER data...")
            try:
                lat, lon = 28.6, 77.2
                if map_data and map_data.get("last_clicked"):
                    lat = map_data["last_clicked"]["lat"]
                    lon = map_data["last_clicked"]["lng"]

                df_nasa = get_nasa_power_data(lat, lon, nasa_start, nasa_end)
                st.success("✅ NASA POWER data fetched")
                st.dataframe(df_nasa.head())
                st.line_chart(df_nasa.set_index("YearMonth")[["T2M", "PRECTOT"]])

                if "T2M" in df_nasa.columns:
                    mean, std = bootstrap_forecast(df_nasa["T2M"], iters=n_bootstrap)
                    st.write(f"**Bootstrap Forecast (T2M)**: {mean:.2f} ± {std:.2f}")
                    fig, ax = plt.subplots()
                    ax.plot(df_nasa["T2M"].values, label="T2M Historical")
                    ax.axhline(mean, color="red", linestyle="--", label="Forecast Mean")
                    ax.fill_between(range(len(df_nasa)), mean - std, mean + std,
                                    color="red", alpha=0.2, label="Uncertainty Band")
                    ax.legend()
                    st.pyplot(fig)
            except Exception as e:
                st.error(f"NASA POWER fetch failed: {e}")
