# === BEGIN: NASA POWER + Google Maps + CMIP6 scaffold + Bootstrap Uncertainty Extension ===

import requests, io, os
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
import xarray as xr  # for CMIP6 scaffold
import googlemaps
import folium
from streamlit_folium import st_folium
from sklearn.utils import resample
from sklearn.base import clone
import plotly.express as px
import streamlit as st
# -------------------------
# CONFIG (fill keys here)
# -------------------------
GOOGLE_MAPS_API_KEY = st.secrets.get("GOOGLE_MAPS_API_KEY", None)  # or set directly
# NASA POWER does not require a key for basic use

# -------------------------
# NASA POWER: fetch monthly historical (T2M, PRECTOT)
# Endpoint example:
# https://power.larc.nasa.gov/api/temporal/monthly/point?parameters=T2M,PRECTOT&community=AG&longitude=78.9629&latitude=20.5937&start=2010&end=2023&format=JSON
# -------------------------
def fetch_nasa_power_monthly(lat, lon, start_year=2000, end_year=None):
    """Return DataFrame with monthly Date, Temperature_C (T2M), Rainfall_mm (PRECTOT)."""
    if end_year is None:
        end_year = datetime.now().year
    base_url = "https://power.larc.nasa.gov/api/temporal/monthly/point"
    params = {
        "start": start_year,
        "end": end_year,
        "latitude": lat,
        "longitude": lon,
        "community": "AG",
        "parameters": "T2M,PRECTOT",
        "format": "JSON",
        "user": "groundwater_app"
    }
    try:
        r = requests.get(base_url, params=params, timeout=30)
        r.raise_for_status()
        js = r.json()
        # The JSON structure: js['properties']['parameter']['T2M'][YYYYMM] = value
        params_dict = js.get("properties", {}).get("parameter", {})
        t2m = params_dict.get("T2M", {})
        pr = params_dict.get("PRECTOT", {})
        records = []
        # keys like '200001' etc.
        for ym, val in t2m.items():
            year = int(ym[:4])
            month = int(ym[4:])
            date = pd.Timestamp(year=year, month=month, day=15)
            temp = val
            rain = pr.get(ym, np.nan)
            records.append({"Date": date, "Temperature_C": temp, "Rainfall_mm": rain})
        df = pd.DataFrame(records).sort_values("Date").reset_index(drop=True)
        # NASA T2M is mean monthly temp in degC; PRECTOT is monthly total precipitation in mm
        return df
    except Exception as e:
        st.warning(f"NASA POWER fetch failed: {e}")
        return pd.DataFrame(columns=["Date", "Temperature_C", "Rainfall_mm"])

# -------------------------
# Google Maps geocoding (search box) — optional usage
# -------------------------
def google_geocode(place):
    """Return (lat, lon) using Google Geocoding API if key present, else None."""
    if not GOOGLE_MAPS_API_KEY:
        st.info("Google Maps API key not provided - falling back to Nominatim for geocoding.")
        return None
    try:
        gmaps = googlemaps.Client(key=GOOGLE_MAPS_API_KEY)
        res = gmaps.geocode(place, components={"country": "IN"})
        if res:
            loc = res[0]["geometry"]["location"]
            return loc["lat"], loc["lng"]
    except Exception as e:
        st.warning(f"Google geocoding error: {e}")
    return None

# -------------------------
# CMIP6 ingestion scaffold
# -------------------------
def load_cmip6_scenario_example(netcdf_path, variable_name, lat, lon, time_slice=None):
    """
    Scaffold to extract CMIP6 netCDF variable for a specific lat/lon.
    - netcdf_path: path to local NetCDF file (one model / one scenario)
    - variable_name: e.g. 'tas' (near-surface air temp), 'pr' (precipitation)
    - lat/lon: points to sample
    - time_slice: (start_year, end_year)
    Returns xarray DataArray or DataFrame of monthly values.
    
    NOTE: You must provide pre-downloaded CMIP6 NetCDF files for this to work.
    """
    try:
        ds = xr.open_dataset(netcdf_path)
        # nearest neighbor selection
        da = ds[variable_name].sel(lat=lat, lon=lon, method='nearest')
        # convert to pandas monthly if needed
        df = da.to_dataframe(name=variable_name).reset_index()
        if time_slice:
            df = df[(df['time'].dt.year >= time_slice[0]) & (df['time'].dt.year <= time_slice[1])]
        # ensure monthly aggregation
        df['Date'] = pd.to_datetime(df['time'].dt.strftime('%Y-%m-15'))
        df = df.groupby(pd.Grouper(key='Date', freq='M'))[variable_name].mean().reset_index()
        return df
    except Exception as e:
        st.warning(f"CMIP6 load error (check netcdf_path/variable_name): {e}")
        return pd.DataFrame()

# -------------------------
# Bootstrapped uncertainty (train-many approach)
# -------------------------
def bootstrap_prediction(model_class, model_params, X_train, y_train, X_pred, n_boot=50, random_state=42):
    """
    model_class: a sklearn estimator class (not instance), e.g., RandomForestRegressor
    model_params: dict of params to pass when instantiating
    Returns: predictions_mean, lower_pct, upper_pct, all_preds (n_boot x n_preds)
    """
    rng = np.random.RandomState(random_state)
    preds = []
    n = len(X_train)
    for i in range(n_boot):
        idx = rng.randint(0, n, n)  # bootstrap indices with replacement
        X_b = X_train.iloc[idx]
        y_b = y_train.iloc[idx]
        m = model_class(**model_params)
        m.fit(X_b, y_b)
        preds_i = m.predict(X_pred)
        preds.append(preds_i)
    all_preds = np.vstack(preds)  # shape (n_boot, n_pred)
    mean_pred = np.mean(all_preds, axis=0)
    lower = np.percentile(all_preds, 2.5, axis=0)  # 95% CI lower
    upper = np.percentile(all_preds, 97.5, axis=0)
    return mean_pred, lower, upper, all_preds

# -------------------------
# Forecast helper tying everything together
#   - pull NASA POWER historical monthly (if requested)
#   - optionally ingest CMIP6 scenario (scaffold) for projection; else use simple trend + NASA projections
#   - create projected monthly env df for forecast horizon
#   - generate predictions and optionally bootstrapped uncertainty bands
# -------------------------
def create_env_projection_and_forecast(lat, lon,
                                       base_from_nasa=True,
                                       nasa_start_year=2000,
                                       nasa_end_year=None,
                                       use_cmip6=False,
                                       cmip6_paths=None,
                                       horizon_years=5,
                                       annual_trend_temp_pct=0.0,
                                       annual_trend_rain_pct=0.0,
                                       bootstrap=False,
                                       n_boot=50):
    # Step 1: Get historical monthly from NASA POWER to compute monthly climatologies
    if base_from_nasa:
        hist_df = fetch_nasa_power_monthly(lat, lon, start_year=nasa_start_year, end_year=nasa_end_year)
    else:
        hist_df = filtered_df[['Date','Temperature_C','Rainfall_mm','pH','Dissolved_Oxygen_mg_L']].copy()

    # Build monthly means
    hist_df['Month'] = hist_df['Date'].dt.month
    temp_monthly = hist_df.groupby('Month')['Temperature_C'].mean().to_dict() if 'Temperature_C' in hist_df else {}
    rain_monthly = hist_df.groupby('Month')['Rainfall_mm'].mean().to_dict() if 'Rainfall_mm' in hist_df else {}
    pH_mean = hist_df['pH'].mean() if 'pH' in hist_df else 7.0
    do_mean = hist_df['Dissolved_Oxygen_mg_L'].mean() if 'Dissolved_Oxygen_mg_L' in hist_df else 8.0

    base_monthly = {
        "Temperature_C": temp_monthly or 20.0,
        "Rainfall_mm": rain_monthly or 50.0,
        "pH": pH_mean,
        "Dissolved_Oxygen_mg_L": do_mean
    }

    # Step 2: If user selected CMIP6 ingestion, try to construct projected monthly series
    if use_cmip6 and cmip6_paths:
        # Example: for each path in cmip6_paths, load var 'tas' and 'pr', average across models
        proj_list = []
        for path in cmip6_paths:
            # user must supply path and variable mapping
            # This is heavy; sample call uses load_cmip6_scenario_example
            df_cm = load_cmip6_scenario_example(path['netcdf'], path['var'], lat, lon, time_slice=None)
            if not df_cm.empty:
                proj_list.append(df_cm)
        if proj_list:
            # naive average across provided CMIP6 extracts
            merged = pd.concat(proj_list).groupby('Date').mean().reset_index()
            # map CMIP var names to Temperature_C / Rainfall_mm as needed (user must ensure correct conversion)
            projected_env = merged.rename(columns={merged.columns[1]: 'Temperature_C'})  # THIS IS A SCAFFOLD - adapt per dataset
        else:
            projected_env = project_future_env(base_monthly, pd.Timestamp.now(), years=horizon_years,
                                               annual_trend_temp_pct=annual_trend_temp_pct,
                                               annual_trend_rain_pct=annual_trend_rain_pct)
    else:
        # simple projection (seasonal monthly pattern + trend)
        projected_env = project_future_env(base_monthly, pd.Timestamp.now(), years=horizon_years,
                                           annual_trend_temp_pct=annual_trend_temp_pct,
                                           annual_trend_rain_pct=annual_trend_rain_pct)

    # Step 3: Ensure features required by model are present
    sel_feats = st.session_state.get("selected_features", [])
    # add temporal features if requested
    projected_env['Year'] = projected_env['Date'].dt.year
    projected_env['Month'] = projected_env['Date'].dt.month
    projected_env['Day'] = projected_env['Date'].dt.day
    projected_env['DayOfYear'] = projected_env['Date'].dt.dayofyear

    # Fill missing feature columns with last-known or mean from filtered_df
    for f in sel_feats:
        if f not in projected_env.columns:
            if f in filtered_df.columns:
                # for lag features, better to use last known series value
                projected_env[f] = filtered_df[f].iloc[-1]
            else:
                projected_env[f] = 0.0

    # Step 4: Scale features and predict
    model = st.session_state.get("trained_model")
    scaler = st.session_state.get("scaler")
    if model is None or scaler is None:
        raise ValueError("Model and scaler must be present in st.session_state (train first).")

    X_future = projected_env[sel_feats].copy()
    X_scaled = scaler.transform(X_future)

    if bootstrap and hasattr(model, "get_params"):
        # bootstrap by retraining the same model class using its params
        model_class = model.__class__
        model_params = model.get_params()
        # For speed, use a smaller n_boot during development
        # Need original training data (we can use session_state saved X_train/y_train)
        X_train = st.session_state.get("X_train_df")
        y_train = st.session_state.get("y_train_series")
        if X_train is None or y_train is None:
            st.warning("Training data not saved in session_state; cannot bootstrap. Save X_train/y_train in st.session_state during training.")
            preds = model.predict(X_scaled)
            projected_env['Predicted_Water_Level_m'] = preds
            return projected_env, None
        mean_pred, lower, upper, all_preds = bootstrap_prediction(model_class, model_params, X_train, y_train, X_future, n_boot=n_boot)
        projected_env['Predicted_Water_Level_m'] = mean_pred
        projected_env['PI_lower'] = lower
        projected_env['PI_upper'] = upper
        return projected_env, all_preds
    else:
        preds = model.predict(X_scaled)
        projected_env['Predicted_Water_Level_m'] = preds
        return projected_env, None

# -------------------------
# UI: Add options in sidebar for NASA POWER, CMIP6, bootstrap, Google Maps
# -------------------------
st.sidebar.markdown("### 🔭 Advanced Forecast Options")
use_nasa = st.sidebar.checkbox("Use NASA POWER historical monthly (recommended)", value=True)
nasa_start = st.sidebar.number_input("NASA POWER start year", value=2000, min_value=1950, max_value=datetime.now().year-1)
use_cmip6 = st.sidebar.checkbox("Use CMIP6 scenario data (advanced — provide NetCDF)", value=False)
cmip6_upload = None
cmip6_paths = None
if use_cmip6:
    st.sidebar.info("Upload one or more CMIP6 netCDF files (var and path). Format and variables must match scaffold.")
    uploaded = st.sidebar.file_uploader("Upload CMIP6 NetCDF (optional)", accept_multiple_files=True)
    if uploaded:
        # Save uploaded to temp files and build path list. User must later map variable names.
        cmip6_paths = []
        for u in uploaded:
            tmp_path = f"/tmp/{u.name}"
            with open(tmp_path, "wb") as fh:
                fh.write(u.read())
            # Ask user to input variable name mapping per file
            varname = st.sidebar.text_input(f"Variable name in {u.name} (e.g. tas or pr)", value="tas")
            cmip6_paths.append({"netcdf": tmp_path, "var": varname})

bootstrap = st.sidebar.checkbox("Use bootstrap uncertainty (retrain bootstrap models)", value=False)
n_boot = st.sidebar.slider("Bootstrap iterations", min_value=10, max_value=200, value=50, step=10)

# Add Google Maps API key UI
if not GOOGLE_MAPS_API_KEY:
    key_input = st.sidebar.text_input("Optional: Enter Google Maps API key (for better search & place autocomplete)", value="")
    if key_input:
        GOOGLE_MAPS_API_KEY = key_input

# Map search + nicer UI
st.sidebar.markdown("### 📍 Search or pick location")
search_place = st.sidebar.text_input("Enter place name (India) to search")
if st.sidebar.button("Find on Map"):
    coords = None
    if GOOGLE_MAPS_API_KEY:
        coords = google_geocode(search_place)
    if coords is None:
        coords = geocode_place(search_place)  # your earlier Nominatim implementation
    if coords:
        sel_lat, sel_lon = coords
        st.session_state['sel_lat'] = sel_lat
        st.session_state['sel_lon'] = sel_lon
    else:
        st.sidebar.error("Geocoding failed. Try another name or click on the map.")

# show map with selected point if available
map_center = [st.session_state.get('sel_lat', 20.5937), st.session_state.get('sel_lon', 78.9629)]
map_obj = folium.Map(location=map_center, zoom_start=6)
if 'sel_lat' in st.session_state and 'sel_lon' in st.session_state:
    folium.Marker([st.session_state['sel_lat'], st.session_state['sel_lon']], tooltip="Selected location").add_to(map_obj)
st_folium(map_obj, width=700, height=350)

# Button to run full advanced forecast
if st.button("🔬 Run Advanced Forecast (NASA POWER + CMIP6 + Bootstrap)"):
    if 'sel_lat' not in st.session_state:
        st.error("Select or search a location first.")
    elif not st.session_state.get("model_trained", False):
        st.error("Train the model first (use 'Train Model'). Make sure training data X_train/y_train were saved in st.session_state.")
    else:
        # optionally ensure training data saved in session_state by modifying training section to add:
        # st.session_state['X_train_df'] = X_train  and st.session_state['y_train_series'] = y_train
        try:
            proj_df, all_preds = create_env_projection_and_forecast(
                lat=st.session_state['sel_lat'],
                lon=st.session_state['sel_lon'],
                base_from_nasa=use_nasa,
                nasa_start_year=nasa_start,
                use_cmip6=use_cmip6,
                cmip6_paths=cmip6_paths,
                horizon_years=st.sidebar.selectbox("Horizon (years)", [5,10], index=0),
                annual_trend_temp_pct=st.sidebar.slider("Temp annual trend (%)", -1.0, 3.0, 0.2, 0.1),
                annual_trend_rain_pct=st.sidebar.slider("Rain annual trend (%)", -5.0, 10.0, 0.0, 0.5),
                bootstrap=bootstrap,
                n_boot=n_boot
            )
            # Plot with uncertainty if available
            fig = px.line(proj_df, x='Date', y='Predicted_Water_Level_m', title="Advanced Forecasted Groundwater Level")
            if 'PI_lower' in proj_df.columns and 'PI_upper' in proj_df.columns:
                fig.add_traces([
                    px.line(proj_df, x='Date', y='PI_upper').data[0],
                    px.line(proj_df, x='Date', y='PI_lower').data[0]
                ])
                fig.update_traces(selector=dict(name="PI_upper"), line=dict(dash="dot"), showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
            st.subheader("Forecast Table")
            st.dataframe(proj_df[['Date','Temperature_C','Rainfall_mm','pH','Dissolved_Oxygen_mg_L','Predicted_Water_Level_m'] + (['PI_lower','PI_upper'] if 'PI_lower' in proj_df.columns else [])], use_container_width=True)

            # CSV download
            csv = proj_df.to_csv(index=False)
            st.download_button("📥 Download Advanced Forecast CSV", data=csv, file_name=f"advanced_gw_forecast_{st.session_state['sel_lat']:.5f}_{st.session_state['sel_lon']:.5f}.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Advanced forecast failed: {e}")

# === END: NASA POWER + Google Maps + CMIP6 scaffold + Bootstrap Uncertainty Extension ===
