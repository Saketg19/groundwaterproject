import streamlit as st
import pandas as pd
import numpy as np
import requests
import folium
from streamlit_folium import st_folium
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="Groundwater Level ML Analysis Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------
# APP TABS
# -------------------------
tab1, tab2 = st.tabs(["📊 ML Dashboard", "🔮 Advanced Forecast Options"])

# -------------------------
# TAB 1 – YOUR ORIGINAL DASHBOARD
# -------------------------
with tab1:
    st.title("💧 Groundwater Level ML Analysis Dashboard")
    st.markdown("Enhanced analysis with machine learning predictions for groundwater levels.")

    # ---- Controls (Example) ----
    with st.sidebar:
        st.header("⚙️ Controls")
        date_range = st.date_input("Select Date Range", [])
        show_avg = st.checkbox("Show 7-day rolling average")
        show_env = st.checkbox("Show environmental factors")

    # ---- Dummy Data for Example ----
    dates = pd.date_range("2023-01-01", periods=200, freq="D")
    df = pd.DataFrame({
        "Date": dates,
        "Water_Level_m": 3.0 + 0.01*np.arange(len(dates)) + np.random.normal(0,0.1,len(dates)),
        "Temperature_C": 20 + 10*np.sin(np.linspace(0, 3*np.pi, len(dates))),
        "Rainfall_mm": np.random.gamma(2, 2, len(dates)),
        "pH": 7 + 0.2*np.random.randn(len(dates)),
        "Dissolved_Oxygen_mgL": 8 + 0.5*np.random.randn(len(dates))
    })

    # ---- Summary Report ----
    st.subheader("📑 Groundwater Analysis Report")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Average Water Level", f"{df['Water_Level_m'].mean():.2f} m")
    col2.metric("Min Water Level", f"{df['Water_Level_m'].min():.2f} m")
    col3.metric("Max Water Level", f"{df['Water_Level_m'].max():.2f} m")
    col4.metric("Status", "Semi-Critical ⚠️")

    # ---- Trend Chart ----
    st.subheader("📈 Water Level Trend")
    st.line_chart(df.set_index("Date")["Water_Level_m"])

    # ---- Environmental Factors ----
    if show_env:
        st.subheader("🌡️ Environmental Factors")
        st.line_chart(df.set_index("Date")[["Temperature_C", "Rainfall_mm", "pH", "Dissolved_Oxygen_mgL"]])

    # ---- Model Training ----
    st.subheader("🤖 Machine Learning Model Training")
    model_choice = st.selectbox("Select Model", ["Random Forest", "Gradient Boosting", "Linear Regression"])
    X = df[["Temperature_C", "Rainfall_mm", "pH", "Dissolved_Oxygen_mgL"]]
    y = df["Water_Level_m"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if st.button("Train Model"):
        if model_choice == "Random Forest":
            model = RandomForestRegressor(random_state=42)
        elif model_choice == "Gradient Boosting":
            model = GradientBoostingRegressor(random_state=42)
        else:
            model = LinearRegression()

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        st.success(f"Model trained: {model_choice}")
        st.write("R² Score:", r2_score(y_test, preds))
        st.write("RMSE:", mean_squared_error(y_test, preds, squared=False))

    # ---- Prediction Form ----
    st.subheader("🔮 Make Predictions")
    with st.form("prediction_form"):
        temp = st.number_input("Temperature (°C)", value=25.0)
        rain = st.number_input("Rainfall (mm)", value=10.0)
        ph = st.number_input("pH", value=7.0)
        do = st.number_input("Dissolved Oxygen (mg/L)", value=8.0)
        submit = st.form_submit_button("Predict Water Level")

        if submit:
            if 'model' in locals():
                pred = model.predict([[temp, rain, ph, do]])[0]
                st.success(f"Predicted Water Level: {pred:.2f} m")
            else:
                st.error("Train a model first!")

    # ---- Data Table ----
    st.subheader("📊 Data Table")
    st.dataframe(df)

    st.download_button("Download Data as CSV", df.to_csv(index=False), file_name="groundwater_data.csv")

# -------------------------
# TAB 2 – ADVANCED FORECAST OPTIONS
# -------------------------
with tab2:
    st.title("🔮 Advanced Forecast Options")
    st.markdown("Use NASA POWER, CMIP6 scenarios, and uncertainty estimation for advanced forecasts.")

    # Sidebar controls
    st.sidebar.subheader("Advanced Forecast Options")
    use_nasa = st.sidebar.checkbox("Use NASA POWER historical monthly", value=True)
    nasa_start = st.sidebar.number_input("NASA POWER start year", 2000, 2023, 2000)
    use_cmip6 = st.sidebar.checkbox("Use CMIP6 scenario data (NetCDF required)", value=False)
    use_bootstrap = st.sidebar.checkbox("Use bootstrap uncertainty", value=False)
    bootstrap_iters = st.sidebar.slider("Bootstrap iterations", 10, 200, 50)

    # Google Maps input
    st.subheader("📍 Search or Pick Location")
    place = st.text_input("Enter place name (India)", "Delhi")
    map_center = [23.5, 85.0]  # default center
    m = folium.Map(location=map_center, zoom_start=5)
    folium.Marker(location=map_center, tooltip="Default Location").add_to(m)
    map_data = st_folium(m, width=700, height=400)

    if st.button("Run Advanced Forecast (NASA POWER + CMIP6 + Bootstrap)"):
        st.info("⚡ Running advanced forecast pipeline...")
        # Placeholder: NASA POWER / CMIP6 fetch + bootstrap integration
        # Here you’ll add your actual request & parsing code
        st.success("✅ Forecast complete. Results will appear here.")
