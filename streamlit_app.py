"""
Enhanced Streamlit app for Groundwater Level Analysis with Machine Learning
Combines your existing model logic with proper ML training
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib

# Page configuration
st.set_page_config(
    page_title="Groundwater ML Analysis Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .model-performance {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #ff7f0e;
    }
    .status-safe { color: green; font-weight: bold; }
    .status-semi-critical { color: orange; font-weight: bold; }
    .status-critical { color: red; font-weight: bold; }
    .status-over-exploited { color: darkred; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">💧 Groundwater Level ML Analysis Dashboard</h1>', unsafe_allow_html=True)
st.markdown("Enhanced analysis with machine learning predictions for groundwater levels.")

# Initialize session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
if 'model_type' not in st.session_state:
    st.session_state.model_type = None
if 'feature_importance' not in st.session_state:
    st.session_state.feature_importance = None
if 'model_metrics' not in st.session_state:
    st.session_state.model_metrics = None

# Load data function
@st.cache_data
def load_data():
    """Load and preprocess the groundwater data from a remote URL"""
    try:
        # Load data from the URL specified in Streamlit's secrets
        data_url = st.secrets["DATA_URL"]
        # Use robust settings to handle potential CSV formatting issues
        df = pd.read_csv(data_url, engine='python', on_bad_lines='skip')

        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)

        # Create additional features for better ML performance
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day
        df['DayOfYear'] = df['Date'].dt.dayofyear

        # Lag features for time series prediction
        df['Water_Level_lag1'] = df['Water_Level_m'].shift(1)
        df['Water_Level_lag7'] = df['Water_Level_m'].shift(7)
        df['Rainfall_lag1'] = df['Rainfall_mm'].shift(1)

        # Rolling averages
        df['Water_Level_ma7'] = df['Water_Level_m'].rolling(window=7).mean()
        df['Rainfall_ma7'] = df['Rainfall_mm'].rolling(window=7).mean()

        # Remove rows with NaN values
        df = df.dropna()

        return df
    except Exception as e:
        st.error(f"❌ Error loading data from the remote URL. Make sure the 'DATA_URL' secret is set correctly. Error: {e}")
        return None

# Load data
df = load_data()

if df is None:
    st.stop()

# Sidebar controls
st.sidebar.header("📊 Controls")

# Date range selection
min_date = df['Date'].min().date()
max_date = df['Date'].max().date()
date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# Filter data
if len(date_range) == 2:
    start_date = pd.to_datetime(date_range[0])
    end_date = pd.to_datetime(date_range[1])
    filtered_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)].copy()
else:
    filtered_df = df.copy()

# Analysis options
show_rolling = st.sidebar.checkbox("Show 7-day rolling average", value=True)
show_environmental = st.sidebar.checkbox("Show environmental factors", value=True)

# Main analysis (based on your original code)
st.header("📊 Groundwater Analysis Report")

# Calculate statistics
avg_level = filtered_df['Water_Level_m'].mean()
min_level = filtered_df['Water_Level_m'].min()
max_level = filtered_df['Water_Level_m'].max()

# Display metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Average Water Level", f"{avg_level:.2f} m")

with col2:
    st.metric("Minimum Water Level", f"{min_level:.2f} m")

with col3:
    st.metric("Maximum Water Level", f"{max_level:.2f} m")

with col4:
    # Groundwater status (from your original logic)
    if avg_level > 5:
        status = "Safe ✅"
        status_class = "status-safe"
    elif 3 < avg_level <= 5:
        status = "Semi-Critical ⚠️"
        status_class = "status-semi-critical"
    elif 2 < avg_level <= 3:
        status = "Critical ❗"
        status_class = "status-critical"
    else:
        status = "Over-exploited ❌"
        status_class = "status-over-exploited"

    st.markdown(f'<div class="metric-card"><strong>Status:</strong> <span class="{status_class}">{status}</span></div>', unsafe_allow_html=True)

# Time series plot
st.header("📈 Water Level Trend")

fig = go.Figure()

# Add main water level line
fig.add_trace(go.Scatter(
    x=filtered_df['Date'],
    y=filtered_df['Water_Level_m'],
    mode='lines+markers',
    name='Water Level (m)',
    line=dict(color='blue', width=2),
    marker=dict(size=4)
))

# Add rolling average if requested
if show_rolling:
    rolling_avg = filtered_df['Water_Level_m'].rolling(window=7, min_periods=1).mean()
    fig.add_trace(go.Scatter(
        x=filtered_df['Date'],
        y=rolling_avg,
        mode='lines',
        name='7-day Rolling Average',
        line=dict(color='red', width=3, dash='dash')
    ))

# Add status thresholds
fig.add_hline(y=5, line_dash="dot", line_color="green", annotation_text="Safe (>5m)")
fig.add_hline(y=3, line_dash="dot", line_color="orange", annotation_text="Semi-Critical (3-5m)")
fig.add_hline(y=2, line_dash="dot", line_color="red", annotation_text="Critical (2-3m)")

fig.update_layout(
    title="Groundwater Level Trend (DWLR Data)",
    xaxis_title="Date",
    yaxis_title="Water Level (m)",
    height=500,
    hovermode='x unified'
)

st.plotly_chart(fig, use_container_width=True)

# Environmental factors
if show_environmental:
    st.header("🌡️ Environmental Factors")

    # Create subplots for environmental factors
    fig_env = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Temperature (°C)', 'Rainfall (mm)', 'pH', 'Dissolved Oxygen (mg/L)'),
        vertical_spacing=0.1
    )

    fig_env.add_trace(
        go.Scatter(x=filtered_df['Date'], y=filtered_df['Temperature_C'],
                    mode='lines', name='Temperature', line=dict(color='orange')),
        row=1, col=1
    )
    fig_env.add_trace(
        go.Scatter(x=filtered_df['Date'], y=filtered_df['Rainfall_mm'],
                    mode='lines', name='Rainfall', line=dict(color='blue')),
        row=1, col=2
    )
    fig_env.add_trace(
        go.Scatter(x=filtered_df['Date'], y=filtered_df['pH'],
                    mode='lines', name='pH', line=dict(color='green')),
        row=2, col=1
    )
    fig_env.add_trace(
        go.Scatter(x=filtered_df['Date'], y=filtered_df['Dissolved_Oxygen_mg_L'],
                    mode='lines', name='DO', line=dict(color='purple')),
        row=2, col=2
    )

    fig_env.update_layout(height=600, showlegend=False)
    st.plotly_chart(fig_env, use_container_width=True)

# Machine Learning Section
st.header("🤖 Machine Learning Model Training")

# Model selection
model_type = st.selectbox(
    "Select Model Type",
    [
        "Random Forest", "Gradient Boosting", "Linear Regression", "XGBoost",
        "SVR", "K-Neighbors Regressor", "AdaBoost Regressor", "Lasso"
    ],
    help="Choose the machine learning algorithm for groundwater level prediction"
)

# Feature selection
st.subheader("🔧 Feature Selection")
feature_options = {
    'Environmental': ['Temperature_C', 'Rainfall_mm', 'pH', 'Dissolved_Oxygen_mg_L'],
    'Temporal': ['Year', 'Month', 'Day', 'DayOfYear'],
    'Lag Features': ['Water_Level_lag1', 'Water_Level_lag7', 'Rainfall_lag1'],
    'Rolling Averages': ['Water_Level_ma7', 'Rainfall_ma7']
}

selected_features = []
for category, features in feature_options.items():
    if st.checkbox(f"Use {category} Features", value=True):
        selected_features.extend(features)

if not selected_features:
    st.warning("Please select at least one feature category.")
    st.stop()

# Train/test split
test_size = st.slider("Test Set Size (%)", 10, 40, 20) / 100

# Prepare data for ML
X = filtered_df[selected_features]
y = filtered_df['Water_Level_m']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# Train model
if st.button("🚀 Train Model", type="primary"):
    with st.spinner("Training model..."):
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Select and train model
        if model_type == "Random Forest":
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_type == "Gradient Boosting":
            model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        elif model_type == "XGBoost":
            model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
        elif model_type == "SVR":
            model = SVR(kernel='rbf')
        elif model_type == "K-Neighbors Regressor":
            model = KNeighborsRegressor(n_neighbors=5)
        elif model_type == "AdaBoost Regressor":
            model = AdaBoostRegressor(n_estimators=100, random_state=42)
        elif model_type == "Lasso":
            model = Lasso(random_state=42)
        else: # Linear Regression
            model = LinearRegression()

        # Train model
        model.fit(X_train_scaled, y_train)

        # Make predictions
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)

        # Calculate metrics
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)

        # Cross-validation score
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')

        # Store in session state
        st.session_state.model_trained = True
        st.session_state.trained_model = model
        st.session_state.model_type = model_type
        st.session_state.scaler = scaler
        st.session_state.selected_features = selected_features
        st.session_state.model_metrics = {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }

        # Feature importance - handled differently by models
        st.session_state.feature_importance = None # Reset first
        if hasattr(model, 'feature_importances_'):
            st.session_state.feature_importance = dict(zip(selected_features, model.feature_importances_))
        elif hasattr(model, 'coef_'):
            st.session_state.feature_importance = dict(zip(selected_features, abs(model.coef_)))

        st.success("Model trained successfully! 🎉")

# Display model performance
if st.session_state.model_trained:
    st.subheader("📊 Model Performance")

    metrics = st.session_state.model_metrics

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f'<div class="model-performance"><strong>R² Score (Test)</strong><br>{metrics["test_r2"]:.3f}</div>', unsafe_allow_html=True)

    with col2:
        st.markdown(f'<div class="model-performance"><strong>RMSE (Test)</strong><br>{metrics["test_rmse"]:.3f}</div>', unsafe_allow_html=True)

    with col3:
        st.markdown(f'<div class="model-performance"><strong>MAE (Test)</strong><br>{metrics["test_mae"]:.3f}</div>', unsafe_allow_html=True)

    with col4:
        st.markdown(f'<div class="model-performance"><strong>CV Score</strong><br>{metrics["cv_mean"]:.3f} ± {metrics["cv_std"]:.3f}</div>', unsafe_allow_html=True)

    # Feature importance plot
    if st.session_state.feature_importance:
        st.subheader("🎯 Feature Importance")
        importance_df = pd.DataFrame(
            list(st.session_state.feature_importance.items()),
            columns=['Feature', 'Importance']
        ).sort_values('Importance', ascending=True)

        fig_importance = px.bar(
            importance_df,
            x='Importance',
            y='Feature',
            orientation='h',
            title=f'{st.session_state.model_type} Feature Importance',
            color='Importance',
            color_continuous_scale='viridis'
        )
        fig_importance.update_layout(height=400)
        st.plotly_chart(fig_importance, use_container_width=True)
    else:
        st.info(f"Feature importance is not available for the {st.session_state.model_type} model.")

    # Predictions vs Actual plot
    st.subheader("🔍 Predictions vs Actual")

    # Get test predictions
    X_test_scaled = st.session_state.scaler.transform(X_test)
    y_pred_test = st.session_state.trained_model.predict(X_test_scaled)

    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(
        x=y_test,
        y=y_pred_test,
        mode='markers',
        name='Predictions vs Actual',
        marker=dict(color='blue', size=8, opacity=0.7)
    ))

    # Perfect prediction line
    min_val = min(y_test.min(), y_pred_test.min())
    max_val = max(y_test.max(), y_pred_test.max())
    fig_pred.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Perfect Prediction',
        line=dict(dash='dash', color='red')
    ))

    fig_pred.update_layout(
        title='Model Predictions vs Actual Values',
        xaxis_title='Actual Water Level (m)',
        yaxis_title='Predicted Water Level (m)',
        height=500
    )
    st.plotly_chart(fig_pred, use_container_width=True)

# Prediction Interface
st.header("🔮 Make Predictions")

if st.session_state.model_trained:
    st.subheader("📝 Enter Environmental Conditions")

    col1, col2 = st.columns(2)

    with col1:
        temp = st.number_input("Temperature (°C)", value=20.0, step=0.1)
        rainfall = st.number_input("Rainfall (mm)", value=10.0, step=0.1, min_value=0.0)

    with col2:
        ph = st.number_input("pH", value=7.0, step=0.1, min_value=0.0, max_value=14.0)
        do = st.number_input("Dissolved Oxygen (mg/L)", value=8.0, step=0.1, min_value=0.0)

    # Additional features if selected
    input_data = {}
    if 'Year' in st.session_state.selected_features:
        input_data['Year'] = st.number_input("Year", value=datetime.now().year, min_value=2020, max_value=2030)
    if 'Month' in st.session_state.selected_features:
        input_data['Month'] = st.number_input("Month", value=datetime.now().month, min_value=1, max_value=12)
    if 'Day' in st.session_state.selected_features:
        input_data['Day'] = st.number_input("Day", value=datetime.now().day, min_value=1, max_value=31)

    if st.button("🔮 Predict Water Level", type="primary"):
        # Prepare input data
        for feature in st.session_state.selected_features:
            if feature == 'Temperature_C':
                input_data[feature] = temp
            elif feature == 'Rainfall_mm':
                input_data[feature] = rainfall
            elif feature == 'pH':
                input_data[feature] = ph
            elif feature == 'Dissolved_Oxygen_mg_L':
                input_data[feature] = do
            elif feature not in input_data: # For lag/rolling features not covered above
                # Use the mean of the column from the original filtered data as a placeholder
                input_data[feature] = filtered_df[feature].mean()

        # Convert to DataFrame and scale, ensuring column order is correct
        input_df = pd.DataFrame([input_data])[st.session_state.selected_features]
        input_scaled = st.session_state.scaler.transform(input_df)

        # Make prediction
        prediction = st.session_state.trained_model.predict(input_scaled)[0]

        # Display result
        st.success(f"🎯 **Predicted Water Level: {prediction:.2f} meters**")

        # Status interpretation
        if prediction > 5:
            status = "Safe ✅"
            status_color = "green"
        elif 3 < prediction <= 5:
            status = "Semi-Critical ⚠️"
            status_color = "orange"
        elif 2 < prediction <= 3:
            status = "Critical ❗"
            status_color = "red"
        else:
            status = "Over-exploited ❌"
            status_color = "darkred"

        st.markdown(f'**Status:** <span style="color: {status_color}">{status}</span>', unsafe_allow_html=True)

else:
    st.info("Please train a model first to make predictions.")

# Model Export
st.header("💾 Model Management")

if st.session_state.model_trained:
    if st.button("💾 Export Model"):
        model_data = {
            'model': st.session_state.trained_model,
            'scaler': st.session_state.scaler,
            'features': st.session_state.selected_features,
            'model_type': st.session_state.model_type,
            'metrics': st.session_state.model_metrics
        }

        # Save model
        joblib.dump(model_data, 'groundwater_model.pkl')
        st.success("Model exported as 'groundwater_model.pkl'")

# Data table
st.header("📋 Data Table")
st.dataframe(filtered_df, use_container_width=True)

# Download button
csv = filtered_df.to_csv(index=False)
st.download_button(
    label="📥 Download Data as CSV",
    data=csv,
    file_name=f"groundwater_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
    mime="text/csv"
)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p>💧 Groundwater Level ML Analysis Dashboard</p>
    <p>Built with ❤️ using Streamlit and Scikit-learn</p>
    <p>Enhanced version of your original model with ML capabilities!</p>
</div>
""", unsafe_allow_html=True)

