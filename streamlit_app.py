import streamlit as st
import pandas as pd
import numpy as np
import requests
import folium
from streamlit_folium import st_folium
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# ==============================================================
# NASA POWER API fetch function
# ==============================================================
def get_nasa_power_data(lat, lon, start=2000, end=2020):
    """
    Fetch NASA POWER monthly climate data for given coordinates.
    Parameters:
        lat, lon : float
        start, end : int (years)
    Returns:
        pd.DataFrame
    """
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

    # Convert to DataFrame
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
# TAB 1: Your Original Dashboard (placeholder demo)
# ==============================================================
with tab1:
    st.subheader("Groundwater Quality ML Dashboard (Demo Layout)")
    st.write("This tab should contain your original dashboard code.")

    # Example placeholder ML model
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([2.3, 2.9, 3.8, 4.5, 5.1])
    model = LinearRegression().fit(X, y)

    fig, ax = plt.subplots()
    ax.scatter(X, y, label="Data")
    ax.plot(X, model.predict(X), color="red", label="Model")
    ax.legend()
    st.pyplot(fig)


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
        st.markdown("### ⚙️ Parameters")
        nasa_start = st.number_input("NASA POWER Start Year", 1981, 2025, 2000)
        nasa_end = st.number_input("NASA POWER End Year", 1981, 2025, 2020)
        n_bootstrap = st.slider("Bootstrap Iterations", 100, 2000, 500)

        if st.button("Run Advanced Forecast"):
            st.info("⚡ Fetching NASA POWER data...")

            try:
                # Default coords (Delhi) if not clicked
                lat, lon = 28.6, 77.2
                if map_data and map_data.get("last_clicked"):
                    lat = map_data["last_clicked"]["lat"]
                    lon = map_data["last_clicked"]["lng"]

                df_nasa = get_nasa_power_data(lat, lon, nasa_start, nasa_end)

                st.success("✅ NASA POWER data fetched")
                st.dataframe(df_nasa.head())

                # Plot time series
                st.line_chart(df_nasa.set_index("YearMonth")[["T2M", "PRECTOT"]])

                # Bootstrap forecast on temperature
                if "T2M" in df_nasa.columns:
                    mean, std = bootstrap_forecast(df_nasa["T2M"], iters=n_bootstrap)
                    st.write(f"**Bootstrap Forecast (T2M)**: {mean:.2f} ± {std:.2f}")

                    fig, ax = plt.subplots()
                    ax.plot(df_nasa["T2M"].values, label="T2M Historical")
                    ax.axhline(mean, color="red", linestyle="--", label="Forecast Mean")
                    ax.fill_between(
                        range(len(df_nasa)),
                        mean - std,
                        mean + std,
                        color="red",
                        alpha=0.2,
                        label="Uncertainty Band",
                    )
                    ax.legend()
                    st.pyplot(fig)

            except Exception as e:
                st.error(f"NASA POWER fetch failed: {e}")
