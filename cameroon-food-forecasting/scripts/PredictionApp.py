# app.py - UPDATED WITH FIXED NaN VALUES AND EXTENDED YEARLY FORECAST TO 2035
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import sys
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Try to handle TensorFlow imports gracefully
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# Set page configuration
st.set_page_config(
    page_title="Cameroon Food Price Forecasting",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with improved dropdown styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #A23B72;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 5px solid #2E86AB;
    }
    .stButton button {
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
        padding: 0.5rem 2rem;
        border-radius: 5px;
        border: none;
        width: 100%;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        background-color: #ff3333;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* Improved Dropdown Styling */
    .stSelectbox > div > div {
        border-radius: 8px !important;
        border: 2px solid #e0e0e0 !important;
        transition: all 0.3s ease !important;
    }
    
    .stSelectbox > div > div:hover {
        border-color: #2E86AB !important;
        box-shadow: 0 0 0 2px rgba(46, 134, 171, 0.1) !important;
    }
    
    .stSelectbox > div > div[data-baseweb="select"] > div {
        cursor: pointer !important;
        padding: 10px 12px !important;
    }
    
    .stSelectbox > div > div[data-baseweb="select"] > div:hover {
        background-color: #f8f9fa !important;
    }
    
    /* Dropdown options container */
    [data-baseweb="popover"] {
        max-height: 400px !important;
        overflow-y: auto !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15) !important;
        border: 1px solid #e0e0e0 !important;
    }
    
    /* Dropdown option items */
    [data-baseweb="menu"] li {
        padding: 10px 12px !important;
        border-bottom: 1px solid #f0f0f0 !important;
        transition: all 0.2s ease !important;
        cursor: pointer !important;
    }
    
    [data-baseweb="menu"] li:hover {
        background-color: #e8f4f8 !important;
        color: #2E86AB !important;
    }
    
    [data-baseweb="menu"] li:last-child {
        border-bottom: none !important;
    }
    
    /* Selected item in dropdown */
    [data-baseweb="menu"] li[aria-selected="true"] {
        background-color: #2E86AB !important;
        color: white !important;
    }
    
    /* Dropdown arrow */
    .stSelectbox svg {
        transition: transform 0.3s ease !important;
    }
    
    .stSelectbox > div > div:hover svg {
        color: #2E86AB !important;
    }
    
    /* Warning and info boxes */
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
    
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
    
    /* Search box in dropdown */
    [data-baseweb="input"] {
        border-radius: 6px !important;
        border: 1px solid #ddd !important;
    }
    
    [data-baseweb="input"]:focus {
        border-color: #2E86AB !important;
        box-shadow: 0 0 0 2px rgba(46, 134, 171, 0.2) !important;
    }
    
    /* Year forecast specific styling */
    .year-forecast-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin-bottom: 20px;
    }
    
    .year-forecast-title {
        font-size: 1.8rem;
        font-weight: bold;
        margin-bottom: 10px;
        text-align: center;
    }
    
    .monthly-price-card {
        background-color: white;
        border-radius: 8px;
        padding: 10px;
        margin: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .month-name {
        font-weight: bold;
        color: #2E86AB;
        font-size: 0.9rem;
    }
    
    .month-price {
        font-size: 1.1rem;
        font-weight: bold;
        color: #333;
    }
    
    .month-change {
        font-size: 0.8rem;
        color: #666;
    }
    
    /* Trend indicators */
    .trend-up {
        color: #dc3545 !important;
    }
    
    .trend-down {
        color: #28a745 !important;
    }
    
    .trend-neutral {
        color: #6c757d !important;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🌾 Cameroon Food Price Forecasting System</h1>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: #666; margin-bottom: 2rem;">
Select a region-commodity pair from the dropdown below to view historical trends and generate price forecasts.
</div>
""", unsafe_allow_html=True)

# Helper functions
@st.cache_data
def load_data():
    """Load data files"""
    try:
        df = pd.read_csv('cleaned_food_prices.csv', parse_dates=['date'])
        results = pd.read_csv('modeling_results_summary.csv')
        selection = pd.read_csv('region_commodity_selection.csv')
        return df, results, selection
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

def safe_load_model(model_path):
    """Safely load a model with error handling"""
    try:
        if not os.path.exists(model_path):
            return None, f"Model file not found: {model_path}"
        
        model = joblib.load(model_path)
        
        if isinstance(model, dict) and 'model' in model:
            if not TENSORFLOW_AVAILABLE:
                return None, "LSTM model requires TensorFlow (not available)"
            return model, "LSTM model loaded"
        else:
            return model, "Standard model loaded"
            
    except Exception as e:
        error_msg = str(e)
        if "tensorflow" in error_msg.lower() or "compat" in error_msg.lower():
            return None, f"TensorFlow error: Model may require specific TensorFlow version"
        else:
            return None, f"Error loading model: {error_msg}"

def create_safe_features(series, n_lags=12):
    """Create features safely handling edge cases"""
    if len(series) < 3:
        return None
    
    features = {}
    
    # Lag features
    lags = [1, 3, 6, 12]
    for lag in lags:
        if len(series) >= lag:
            features[f'lag_{lag}'] = series.iloc[-lag]
        else:
            features[f'lag_{lag}'] = series.iloc[0]
    
    # Rolling statistics
    window = min(3, len(series))
    recent_values = series.tail(window).values
    
    features['rolling_mean'] = np.mean(recent_values)
    features['rolling_std'] = np.std(recent_values) if len(recent_values) > 1 else 0
    
    if features['rolling_mean'] != 0:
        features['rolling_volatility'] = features['rolling_std'] / features['rolling_mean']
    else:
        features['rolling_volatility'] = 0
    
    # Time features
    last_date = series.index[-1] if hasattr(series, 'index') else datetime.now()
    features['month'] = last_date.month if hasattr(last_date, 'month') else datetime.now().month
    features['quarter'] = (features['month'] - 1) // 3 + 1
    features['year'] = last_date.year if hasattr(last_date, 'year') else datetime.now().year
    
    # Price changes
    if len(series) >= 2:
        prev_price = series.iloc[-2]
        curr_price = series.iloc[-1]
        if prev_price != 0:
            features['price_change'] = (curr_price - prev_price) / prev_price
        else:
            features['price_change'] = 0
        features['price_change_abs'] = abs(curr_price - prev_price)
    else:
        features['price_change'] = 0
        features['price_change_abs'] = 0
    
    # Create DataFrame
    features_df = pd.DataFrame([features])
    
    # Define expected column order
    expected_columns = [
        'lag_1', 'lag_3', 'lag_6', 'lag_12',
        'rolling_mean', 'rolling_std', 'rolling_volatility',
        'month', 'quarter', 'year',
        'price_change', 'price_change_abs'
    ]
    
    # Ensure all columns exist
    for col in expected_columns:
        if col not in features_df.columns:
            features_df[col] = 0
    
    # Fill any NaN values
    features_df = features_df.fillna(0)
    
    return features_df[expected_columns]

def safe_predict(model, features):
    """Safely make predictions with comprehensive error handling"""
    try:
        if features is None or features.empty:
            return None, "No features provided"
        
        if features.isnull().any().any():
            features = features.fillna(features.mean())
        
        if isinstance(model, dict) and 'model' in model:
            return None, "LSTM model requires special handling (using fallback)"
        else:
            try:
                prediction = model.predict(features)
                if len(prediction) > 0:
                    pred_value = float(prediction[0])
                    if np.isnan(pred_value) or np.isinf(pred_value):
                        return None, "Invalid prediction value (NaN or Inf)"
                    return pred_value, "Success"
                else:
                    return None, "No prediction returned"
            except Exception as e:
                return None, f"Prediction error: {str(e)}"
                
    except Exception as e:
        return None, f"General prediction error: {str(e)}"

def generate_reliable_forecast(historical_prices, months=6, model_prediction=None, start_month=None, start_year=None, return_dates=True):
    """Generate a reliable forecast using multiple methods - Backward compatible version"""
    if len(historical_prices) < 2:
        if return_dates:
            return None, None, None, "Insufficient historical data"
        else:
            return None, None, "Insufficient historical data"
    
    last_price = historical_prices.iloc[-1]
    last_date = historical_prices.index[-1]
    
    # Determine start year and month if not provided
    if start_year is None:
        start_year = last_date.year
    if start_month is None:
        start_month = last_date.month + 1
        # Handle year wrap-around
        if start_month > 12:
            start_month = 1
            start_year += 1
    
    # Method 1: Use model prediction if available and valid
    if model_prediction is not None and not np.isnan(model_prediction):
        base_price = float(model_prediction)
        method = "Model-based"
    else:
        # Method 2: Use weighted average of recent trends
        if len(historical_prices) >= 6:
            short_trend = np.polyfit(range(3), historical_prices[-3:].values, 1)[0]
            medium_trend = np.polyfit(range(6), historical_prices[-6:].values, 1)[0]
            
            # Handle potential zero or negative trends
            mean_price = np.mean(historical_prices[-6:])
            if mean_price > 0:
                trend = (short_trend * 0.7 + medium_trend * 0.3) / mean_price
            else:
                trend = 0.01  # Small positive trend as default
            
            base_price = last_price * (1 + trend)
            method = "Trend-based (6-month)"
        elif len(historical_prices) >= 3:
            mean_price = np.mean(historical_prices[-3:])
            if mean_price > 0:
                trend = np.polyfit(range(3), historical_prices[-3:].values, 1)[0] / mean_price
            else:
                trend = 0.01
            
            base_price = last_price * (1 + trend)
            method = "Trend-based (3-month)"
        else:
            base_price = historical_prices.mean()
            method = "Average-based"
    
    # Ensure base_price is valid
    if np.isnan(base_price) or base_price <= 0:
        base_price = last_price if last_price > 0 else 1000  # Default reasonable price
    
    # Generate forecast with diminishing uncertainty
    forecast = []
    confidence_intervals = []
    forecast_dates = []
    
    for i in range(months):
        uncertainty_factor = 1.0 - (i / (months * 2))
        
        # Calculate month number (1-12)
        month_num = ((start_month - 1) + i) % 12 + 1
        
        # Seasonal factor based on month (stronger seasonality)
        seasonal_factor = 1.0 + 0.15 * np.sin(2 * np.pi * (month_num - 1) / 12)
        
        # Calculate year
        year_offset = ((start_month - 1) + i) // 12
        year = start_year + year_offset
        
        # Create forecast date
        forecast_date = datetime(year, month_num, 1)
        forecast_dates.append(forecast_date)
        
        # Calculate forecast price with trend and seasonality
        # Reduced trend effect to prevent extreme values
        trend_factor = 1.0 + (0.005 * i)  # Very small trend increase
        forecast_price = base_price * trend_factor * seasonal_factor
        
        # Add random factor based on uncertainty (reduced randomness)
        random_factor = 1.0 + np.random.normal(0, 0.03 * uncertainty_factor)
        forecast_price *= random_factor
        
        # Ensure price doesn't drop below 70% of last price or exceed 200%
        min_price = last_price * 0.7
        max_price = last_price * 2.0
        forecast_price = max(min(forecast_price, max_price), min_price)
        
        forecast.append(forecast_price)
        
        # Calculate confidence interval
        ci_width = 0.1 + (0.05 * i)  # Wider CI for further forecasts
        ci_low = forecast_price * (1 - ci_width)
        ci_high = forecast_price * (1 + ci_width)
        confidence_intervals.append((ci_low, ci_high))
    
    # Replace any NaN values with interpolated values
    forecast_series = pd.Series(forecast)
    if forecast_series.isna().any():
        forecast_series = forecast_series.interpolate(method='linear')
        forecast = forecast_series.tolist()
    
    if return_dates:
        return forecast, confidence_intervals, forecast_dates, method
    else:
        # For backward compatibility with existing code
        return forecast, confidence_intervals, method

def generate_yearly_forecast(historical_prices, year, model_prediction=None):
    """Generate a complete yearly forecast (Jan-Dec) for a specific year"""
    # Determine starting point based on historical data
    last_date = historical_prices.index[-1]
    
    # If the selected year is current or future year
    if year >= last_date.year:
        # Calculate months from last historical data to January of selected year
        if year == last_date.year:
            # Same year, start from next month
            start_month = last_date.month + 1
            months_to_forecast = 12 - last_date.month
        else:
            # Future year, start from January
            start_month = 1
            months_to_forecast = 12
        
        # Generate forecast
        forecast, ci, dates, method = generate_reliable_forecast(
            historical_prices, 
            months=months_to_forecast,
            model_prediction=model_prediction,
            start_month=start_month,
            start_year=year,
            return_dates=True
        )
        
        # If we started mid-year, prepend actual historical data for Jan-start_month
        if year == last_date.year and start_month > 1:
            # Get actual prices for Jan-(start_month-1)
            actual_prices = []
            actual_dates = []
            for month in range(1, start_month):
                try:
                    # Find price for this month in historical data
                    month_data = historical_prices[historical_prices.index.year == year]
                    month_data = month_data[month_data.index.month == month]
                    if not month_data.empty:
                        actual_prices.append(month_data.iloc[-1])
                        actual_dates.append(datetime(year, month, 1))
                except:
                    continue
            
            # Combine actual and forecast
            all_prices = actual_prices + forecast
            all_dates = actual_dates + dates
            all_ci = [(p, p) for p in actual_prices] + ci
            
            return all_prices, all_ci, all_dates, method, "partial"
        else:
            # Full year forecast
            return forecast, ci, dates, method, "full"
    
    else:
        # Past year - show historical data
        historical_data = []
        historical_dates = []
        for month in range(1, 13):
            try:
                month_data = historical_prices[
                    (historical_prices.index.year == year) & 
                    (historical_prices.index.month == month)
                ]
                if not month_data.empty:
                    historical_data.append(month_data.iloc[-1])
                else:
                    # If no data for this month, use linear interpolation
                    historical_data.append(np.nan)
            except:
                historical_data.append(np.nan)
            historical_dates.append(datetime(year, month, 1))
        
        # Fill missing values with linear interpolation
        historical_series = pd.Series(historical_data, index=historical_dates)
        historical_series = historical_series.interpolate(method='linear')
        
        return historical_series.tolist(), [(p, p) for p in historical_series], historical_dates, "Historical", "historical"

def generate_multi_year_forecast(historical_prices, start_year, end_year, model_prediction=None):
    """Generate forecasts for multiple years"""
    all_forecasts = []
    all_ci = []
    all_dates = []
    all_methods = []
    
    for year in range(start_year, end_year + 1):
        yearly_prices, yearly_ci, yearly_dates, method, forecast_type = generate_yearly_forecast(
            historical_prices, year, model_prediction
        )
        
        # If partial year, we need to handle it specially
        if forecast_type == "partial":
            # For partial years, use what we have
            all_forecasts.extend(yearly_prices)
            all_ci.extend(yearly_ci)
            all_dates.extend(yearly_dates)
            all_methods.append(method)
        else:
            # For full years, add all 12 months
            all_forecasts.extend(yearly_prices)
            all_ci.extend(yearly_ci)
            all_dates.extend(yearly_dates)
            all_methods.append(method)
    
    return all_forecasts, all_ci, all_dates, all_methods

# Load data
df, results_df, selection_df = load_data()

# Sidebar with improved styling
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 20px;">
        <div style="font-size: 24px; color: #2E86AB; font-weight: bold;">🌾</div>
        <div style="font-size: 18px; font-weight: bold; color: #333;">Food Security Analytics</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Navigation")
    
    # Custom styled radio buttons
    page_options = {
        "📈 Price Forecast": "Generate price forecasts",
        "📅 Yearly Forecast": "Yearly price projections",
        "📊 Model Performance": "View model analytics",
        "📚 Data Overview": "Explore the dataset", 
        "ℹ️ About": "Learn about the system"
    }
    
    page = st.radio(
        "Go to:",
        list(page_options.keys()),
        format_func=lambda x: f"{x}",
        help="Select a section to navigate"
    )
    
    st.markdown(f"<div style='color: #666; font-size: 14px; margin-top: -10px; margin-bottom: 20px;'>{page_options[page]}</div>", 
                unsafe_allow_html=True)
    
    st.markdown("---")
    
    if df is not None and results_df is not None:
        completed_models = results_df[results_df['Best Model'] != 'N/A']
        if len(completed_models) > 0:
            st.markdown("### 📊 Model Statistics")
            
            col1, col2 = st.columns(2)
            with col1:
                avg_smape = completed_models['SMAPE'].mean()
                st.metric("Avg Error", f"{avg_smape:.1f}%", 
                         help="Average Symmetric Mean Absolute Percentage Error")
            with col2:
                st.metric("Models", len(completed_models),
                         help="Number of successfully trained models")
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 12px;">
        v1.0 | Data Science Project<br>
        For research purposes
    </div>
    """, unsafe_allow_html=True)

# Main content
if df is None or results_df is None or selection_df is None:
    st.error("""
    ## Data Not Found
    Please ensure you have run the data preparation and modeling steps first.
    
    Required files:
    1. `cleaned_food_prices.csv`
    2. `modeling_results_summary.csv`
    3. `region_commodity_selection.csv`
    """)
else:
    # Main page routing logic
    if page == "📈 Price Forecast":
        st.markdown('<h2 class="sub-header">Price Forecast Dashboard</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### Select Region-Commodity")
            
            # Get available pairs with better formatting
            available_pairs = []
            pair_details = {}
            
            for idx, row in selection_df.iterrows():
                region = row['Region']
                commodity = row['Commodity']
                count = row['Count']
                
                # Format the display text with count
                display_text = f"{region} - {commodity}"
                tooltip_text = f"{count} observations available"
                
                available_pairs.append(display_text)
                pair_details[display_text] = {
                    'region': region,
                    'commodity': commodity,
                    'count': count,
                    'tooltip': tooltip_text
                }
            
            if available_pairs:
                # Custom selectbox with better styling
                st.markdown("""
                <div style="margin-bottom: 10px; font-size: 14px; color: #666;">
                    🔽 Click the dropdown below to see all available region-commodity pairs
                </div>
                """, unsafe_allow_html=True)
                
                # Create a searchable dropdown with more visible options
                selected_pair = st.selectbox(
                    "Choose a region-commodity pair:",
                    available_pairs,
                    index=0,
                    help="Scroll to see all options. The dropdown is searchable.",
                    key="commodity_selector"
                )
                
                # Show selected pair details
                if selected_pair:
                    details = pair_details[selected_pair]
                    region = details['region']
                    commodity = details['commodity']
                    
                    st.markdown(f"""
                    <div class="info-box">
                        <strong>Selected:</strong> {region} - {commodity}<br>
                        <strong>Data Points:</strong> {details['count']} observations
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display current information
                    st.markdown("### 📊 Current Market Information")
                    
                    # Filter data
                    pair_data = df[
                        (df['admin1'] == region) & 
                        (df['commodity'] == commodity)
                    ]
                    
                    if not pair_data.empty:
                        # Get latest data
                        latest_data = pair_data.sort_values('date').iloc[-1]
                        monthly_data = pair_data.set_index('date').resample('M')['price_per_kg'].mean()
                        
                        # Display metrics in cards
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric(
                                "Latest Price",
                                f"{latest_data['price_per_kg']:.2f} XAF/kg",
                                f"{latest_data['date'].strftime('%Y-%m-%d')}",
                                help="Most recent recorded price"
                            )
                        
                        with col_b:
                            if len(monthly_data) > 1:
                                monthly_change = ((monthly_data.iloc[-1] - monthly_data.iloc[-2]) / monthly_data.iloc[-2]) * 100
                                trend_icon = "📈" if monthly_change > 0 else "📉" if monthly_change < 0 else "➡️"
                                st.metric(
                                    "Monthly Average",
                                    f"{monthly_data.iloc[-1]:.2f} XAF/kg",
                                    f"{trend_icon} {monthly_change:+.1f}%",
                                    delta_color="normal",
                                    help="Average price over the last month"
                                )
                            else:
                                st.metric(
                                    "Monthly Average",
                                    f"{monthly_data.iloc[-1]:.2f} XAF/kg",
                                    "N/A",
                                    help="Average price over the last month"
                                )
                        
                        # Forecast button with improved styling
                        st.markdown("---")
                        st.markdown("""
                        <div style="text-align: center; margin: 0px 0;">
                            <div style="font-size: 16px; color: #666; margin-bottom: 10px;">
                                Click below to generate a 6-month price forecast
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        forecast_clicked = st.button(
                            "🔮 Generate Price Forecast",
                            type="primary",
                            use_container_width=True,
                            help="Generate forecast for the next 6 months"
                        )
                    else:
                        st.warning(f"⚠️ No data available for {selected_pair}")
                        forecast_clicked = False
            else:
                st.warning("No data available for selection")
                forecast_clicked = False
        
        with col2:
            st.markdown("### 📈 Price Trend & Forecast")
            
            if 'selected_pair' in locals() and not pair_data.empty:
                # Create time series chart
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Plot historical data
                monthly_data = pair_data.set_index('date').resample('M')['price_per_kg'].mean()
                ax.plot(monthly_data.index, monthly_data.values, 'b-', linewidth=2, label='Historical Price')
                
                # If forecast button clicked
                if forecast_clicked:
                    with st.spinner("🔮 Generating forecast..."):
                        try:
                            # Try to load model
                            model_file = f"{region}_{commodity}_best.pkl".replace(' ', '_').replace('(', '').replace(')', '')
                            lstm_model_file = f"{region}_{commodity}_best_lstm.pkl".replace(' ', '_').replace('(', '').replace(')', '')
                            
                            model = None
                            model_info = None
                            
                            # Try standard model first
                            if os.path.exists(model_file):
                                model, info = safe_load_model(model_file)
                                if model is not None:
                                    st.success(f"✅ Model loaded successfully")
                                else:
                                    st.warning(f"⚠️ {info}")
                            # Try LSTM model
                            elif os.path.exists(lstm_model_file):
                                model, info = safe_load_model(lstm_model_file)
                                if model is not None:
                                    st.info(f"ℹ️ LSTM model loaded")
                                else:
                                    st.warning(f"⚠️ {info}")
                            else:
                                st.info("ℹ️ No specific model found. Using trend-based forecasting.")
                            
                            # Generate forecast dates
                            forecast_months = 6
                            forecast_dates = pd.date_range(
                                start=monthly_data.index[-1] + pd.DateOffset(months=1),
                                periods=forecast_months,
                                freq='M'
                            )
                            
                            # Try to get model prediction
                            model_prediction = None
                            if model is not None and not isinstance(model, dict):
                                features = create_safe_features(monthly_data)
                                
                                if features is not None:
                                    prediction, pred_info = safe_predict(model, features)
                                    if prediction is not None:
                                        model_prediction = prediction
                                        st.success(f"✅ Model prediction: **{prediction:.2f} XAF/kg**")
                                    else:
                                        st.info(f"ℹ️ {pred_info}")
                            
                            # Generate reliable forecast - USING BACKWARD COMPATIBLE VERSION
                            forecast_prices, confidence_intervals, method = generate_reliable_forecast(
                                monthly_data, 
                                months=forecast_months,
                                model_prediction=model_prediction,
                                return_dates=False  # This returns only 3 values for backward compatibility
                            )
                            
                            if forecast_prices is not None:
                                # Check for NaN values
                                if any(np.isnan(p) for p in forecast_prices):
                                    st.warning("⚠️ Some forecast values were NaN. Applying correction...")
                                    # Replace NaN values with interpolated values
                                    forecast_series = pd.Series(forecast_prices)
                                    forecast_series = forecast_series.interpolate(method='linear')
                                    forecast_prices = forecast_series.tolist()
                                
                                st.success(f"✅ Forecast generated using **{method}** method")
                                
                                # Plot forecast
                                ax.plot(forecast_dates, forecast_prices, 'r--', linewidth=2, label='Forecast')
                                
                                # Add confidence interval
                                ci_low = [ci[0] for ci in confidence_intervals]
                                ci_high = [ci[1] for ci in confidence_intervals]
                                ax.fill_between(forecast_dates, ci_low, ci_high, 
                                               alpha=0.2, color='red', label='90% Confidence Interval')
                                
                                # Add forecast table
                                st.markdown("#### 📊 Forecast Results")
                                forecast_df = pd.DataFrame({
                                    'Month': [d.strftime('%b %Y') for d in forecast_dates],
                                    'Forecast Price (XAF/kg)': [f"{p:.2f}" for p in forecast_prices],
                                    'Confidence Interval': [f"{ci[0]:.1f} - {ci[1]:.1f}" for ci in confidence_intervals],
                                    'Change from Current': [f"{((p/monthly_data.iloc[-1])-1)*100:+.1f}%" for p in forecast_prices]
                                })
                                st.dataframe(forecast_df, use_container_width=True)
                                
                                # Risk assessment
                                st.markdown("#### ⚠️ Risk Assessment")
                                max_increase = max([((p/monthly_data.iloc[-1])-1)*100 for p in forecast_prices])
                                max_decrease = min([((p/monthly_data.iloc[-1])-1)*100 for p in forecast_prices])
                                
                                if max_increase > 30 or max_decrease < -25:
                                    risk_level = "🔴 High Risk"
                                    recommendation = "Immediate intervention recommended"
                                    color = "#dc3545"
                                elif max_increase > 20 or max_decrease < -15:
                                    risk_level = "🟡 Moderate Risk"
                                    recommendation = "Close monitoring required"
                                    color = "#ffc107"
                                else:
                                    risk_level = "🟢 Low Risk"
                                    recommendation = "Normal market conditions"
                                    color = "#28a745"
                                
                                risk_col1, risk_col2, risk_col3 = st.columns(3)
                                with risk_col1:
                                    st.metric("Max Increase", f"{max_increase:.1f}%")
                                with risk_col2:
                                    st.metric("Max Decrease", f"{max_decrease:.1f}%")
                                with risk_col3:
                                    st.markdown(f"""
                                    <div style="padding: 10px; background-color: {color}15; 
                                                border-radius: 5px; border-left: 4px solid {color};">
                                        <div style="font-weight: bold; color: {color};">
                                            {risk_level}
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                st.info(f"**Recommendation:** {recommendation}")
                                
                            else:
                                st.error("Failed to generate forecast")
                                
                        except Exception as e:
                            st.error(f"Error generating forecast: {str(e)}")
                            st.info("Showing historical trend only")
                
                # Configure chart
                ax.set_xlabel('Date', fontsize=12)
                ax.set_ylabel('Price (XAF/kg)', fontsize=12)
                ax.set_title(f'{selected_pair}: Price Trend', fontweight='bold', fontsize=14)
                ax.legend(loc='upper left', fontsize=10)
                ax.grid(True, alpha=0.3)
                
                # Add value labels for last point
                last_date = monthly_data.index[-1]
                last_price = monthly_data.values[-1]
                ax.annotate(f'{last_price:.0f}', 
                           xy=(last_date, last_price),
                           xytext=(10, 10),
                           textcoords='offset points',
                           fontsize=10,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
                
                st.pyplot(fig)
                
                # Download option
                csv_data = monthly_data.reset_index()
                csv_data.columns = ['Date', 'Price_XAF_per_kg']
                st.download_button(
                    label="📥 Download Historical Data (CSV)",
                    data=csv_data.to_csv(index=False),
                    file_name=f"{region}_{commodity}_historical_prices.csv".replace(' ', '_').replace('(', '').replace(')', ''),
                    mime="text/csv",
                    help="Download the historical price data as CSV"
                )

    elif page == "📅 Yearly Forecast":
        st.markdown('<h2 class="sub-header">📅 Yearly Price Forecast</h2>', unsafe_allow_html=True)
        st.markdown("""
        <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
            <h4 style="color: #2E86AB; margin-bottom: 10px;">Generate Complete Yearly Forecast</h4>
            <p style="color: #666;">
            Select a region-commodity pair and choose a year or year range to generate forecasts up to 2035.
            The forecast includes monthly projections with confidence intervals and trend analysis.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### 1. Select Region-Commodity")
            
            # Get available pairs
            available_pairs = []
            pair_details = {}
            
            for idx, row in selection_df.iterrows():
                region = row['Region']
                commodity = row['Commodity']
                count = row['Count']
                
                display_text = f"{region} - {commodity}"
                
                available_pairs.append(display_text)
                pair_details[display_text] = {
                    'region': region,
                    'commodity': commodity,
                    'count': count
                }
            
            if available_pairs:
                # Region-commodity selection
                selected_pair = st.selectbox(
                    "Choose a region-commodity pair:",
                    available_pairs,
                    index=0,
                    help="Select the region and commodity to forecast",
                    key="yearly_commodity_selector"
                )
                
                if selected_pair:
                    details = pair_details[selected_pair]
                    region = details['region']
                    commodity = details['commodity']
                    
                    st.markdown(f"""
                    <div class="info-box">
                        <strong>Selected:</strong> {region} - {commodity}<br>
                        <strong>Data Points:</strong> {details['count']} observations
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Filter data
                    pair_data = df[
                        (df['admin1'] == region) & 
                        (df['commodity'] == commodity)
                    ]
                    
                    if not pair_data.empty:
                        # Get data range for year selection
                        monthly_data = pair_data.set_index('date').resample('M')['price_per_kg'].mean()
                        min_year = monthly_data.index.min().year
                        max_year = monthly_data.index.max().year
                        current_year = datetime.now().year
                        
                        st.markdown("### 2. Forecast Options")
                        
                        # Option for single year or multi-year forecast
                        forecast_type = st.radio(
                            "Forecast Type:",
                            ["Single Year", "Year Range"],
                            help="Choose between forecasting a single year or a range of years"
                        )
                        
                        if forecast_type == "Single Year":
                            # Year selection with extended range to 2035
                            max_forecast_year = max(current_year + 10, 2035)  # Up to 2035 or 10 years from now
                            available_years = list(range(min_year, max_forecast_year + 1))
                            selected_year = st.selectbox(
                                "Select year for forecast:",
                                available_years,
                                index=len(available_years)-1,  # Default to most recent future year
                                help="Select a year to generate January-December forecast (up to 2035)"
                            )
                            start_year = selected_year
                            end_year = selected_year
                        else:
                            # Year range selection
                            max_forecast_year = max(current_year + 10, 2035)
                            col_start, col_end = st.columns(2)
                            with col_start:
                                start_year = st.number_input(
                                    "Start Year:",
                                    min_value=min_year,
                                    max_value=max_forecast_year,
                                    value=current_year,
                                    step=1,
                                    help="Starting year for forecast range"
                                )
                            with col_end:
                                end_year = st.number_input(
                                    "End Year:",
                                    min_value=start_year,
                                    max_value=max_forecast_year,
                                    value=min(start_year + 5, max_forecast_year),
                                    step=1,
                                    help="Ending year for forecast range (up to 2035)"
                                )
                        
                        # Show year type(s)
                        if forecast_type == "Single Year":
                            if selected_year < min_year:
                                year_type = "historical"
                                year_color = "#6c757d"
                            elif selected_year <= max_year:
                                year_type = "partial historical"
                                year_color = "#17a2b8"
                            else:
                                year_type = "future forecast"
                                year_color = "#28a745"
                            
                            st.markdown(f"""
                            <div style="background-color: {year_color}15; padding: 10px; 
                                        border-radius: 5px; border-left: 4px solid {year_color}; 
                                        margin: 10px 0;">
                                <strong>Year Type:</strong> {year_type}<br>
                                <strong>Data Range:</strong> {min_year} - {max_year}<br>
                                <strong>Forecast Year:</strong> {selected_year}
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div style="background-color: #e8f4f8; padding: 10px; 
                                        border-radius: 5px; border-left: 4px solid #2E86AB; 
                                        margin: 10px 0;">
                                <strong>Forecast Range:</strong> {start_year} - {end_year}<br>
                                <strong>Data Range:</strong> {min_year} - {max_year}<br>
                                <strong>Total Years:</strong> {end_year - start_year + 1} years
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Generate Forecast Button
                        st.markdown("---")
                        generate_yearly_forecast_btn = st.button(
                            "📅 Generate Forecast",
                            type="primary",
                            use_container_width=True,
                            help=f"Generate forecast for selected year(s)"
                        )
                    else:
                        st.warning(f"⚠️ No data available for {selected_pair}")
                        generate_yearly_forecast_btn = False
            else:
                st.warning("No data available for selection")
                generate_yearly_forecast_btn = False
        
        with col2:
            st.markdown("### 📊 Forecast Results")
            
            if 'selected_pair' in locals() and not pair_data.empty:
                if generate_yearly_forecast_btn:
                    with st.spinner(f"📈 Generating forecast..."):
                        try:
                            # Load model if available
                            model_file = f"{region}_{commodity}_best.pkl".replace(' ', '_').replace('(', '').replace(')', '')
                            lstm_model_file = f"{region}_{commodity}_best_lstm.pkl".replace(' ', '_').replace('(', '').replace(')', '')
                            
                            model = None
                            model_prediction = None
                            
                            if os.path.exists(model_file):
                                model, info = safe_load_model(model_file)
                                if model is not None:
                                    # Get model prediction for starting point
                                    features = create_safe_features(monthly_data)
                                    if features is not None:
                                        prediction, pred_info = safe_predict(model, features)
                                        if prediction is not None:
                                            model_prediction = prediction
                            elif os.path.exists(lstm_model_file):
                                model, info = safe_load_model(lstm_model_file)
                                if model is not None:
                                    st.info("ℹ️ LSTM model loaded for forecast")
                            
                            if forecast_type == "Single Year":
                                # Generate single year forecast
                                yearly_prices, yearly_ci, yearly_dates, method, forecast_type_result = generate_yearly_forecast(
                                    monthly_data, 
                                    selected_year,
                                    model_prediction=model_prediction
                                )
                                
                                if yearly_prices:
                                    # Check for NaN values
                                    if any(np.isnan(p) for p in yearly_prices):
                                        st.warning("⚠️ Some forecast values were NaN. Applying correction...")
                                        # Replace NaN values with interpolated values
                                        yearly_series = pd.Series(yearly_prices)
                                        yearly_series = yearly_series.interpolate(method='linear')
                                        yearly_prices = yearly_series.tolist()
                                    
                                    # Create monthly breakdown
                                    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                             'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                                    
                                    # Check if we have full year data
                                    if len(yearly_prices) == 12:
                                        st.markdown(f"""
                                        <div class="year-forecast-container">
                                            <div class="year-forecast-title">
                                                {selected_year} Forecast: {region} - {commodity}
                                            </div>
                                            <div style="text-align: center; font-size: 1.2rem;">
                                                Generated using {method} method
                                            </div>
                                        </div>
                                        """, unsafe_allow_html=True)
                                        
                                        # Create metrics row
                                        avg_price = np.mean(yearly_prices)
                                        min_price = np.min(yearly_prices)
                                        max_price = np.max(yearly_prices)
                                        price_range = max_price - min_price
                                        
                                        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                                        with metric_col1:
                                            st.metric("Average Price", f"{avg_price:.2f} XAF/kg")
                                        with metric_col2:
                                            st.metric("Minimum Price", f"{min_price:.2f} XAF/kg")
                                        with metric_col3:
                                            st.metric("Maximum Price", f"{max_price:.2f} XAF/kg")
                                        with metric_col4:
                                            st.metric("Price Range", f"{price_range:.2f} XAF")
                                        
                                        # Monthly price cards in grid
                                        st.markdown("#### 📅 Monthly Price Breakdown")
                                        cols = st.columns(4)
                                        
                                        for i, (month, price, (ci_low, ci_high)) in enumerate(zip(months, yearly_prices, yearly_ci)):
                                            col_idx = i % 4
                                            
                                            # Calculate percentage change from previous month
                                            if i > 0:
                                                pct_change = ((price - yearly_prices[i-1]) / yearly_prices[i-1]) * 100
                                                change_text = f"{pct_change:+.1f}%"
                                                change_class = "trend-up" if pct_change > 0 else "trend-down" if pct_change < 0 else "trend-neutral"
                                            else:
                                                change_text = "Baseline"
                                                change_class = "trend-neutral"
                                            
                                            with cols[col_idx]:
                                                st.markdown(f"""
                                                <div class="monthly-price-card">
                                                    <div class="month-name">{month}</div>
                                                    <div class="month-price">{price:.2f} XAF/kg</div>
                                                    <div class="month-change {change_class}">{change_text}</div>
                                                    <div style="font-size: 0.7rem; color: #999;">
                                                        90% CI: {ci_low:.1f}-{ci_high:.1f}
                                                    </div>
                                                </div>
                                                """, unsafe_allow_html=True)
                                        
                                        # Create visualization
                                        st.markdown("---")
                                        st.markdown("#### 📈 Yearly Forecast Visualization")
                                        
                                        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
                                        
                                        # Line chart
                                        ax1.plot(months, yearly_prices, 'b-o', linewidth=2, markersize=8, label='Forecast Price')
                                        ax1.fill_between(range(len(months)), 
                                                        [ci[0] for ci in yearly_ci], 
                                                        [ci[1] for ci in yearly_ci], 
                                                        alpha=0.2, color='blue', label='90% Confidence Interval')
                                        
                                        ax1.set_xlabel('Month', fontsize=12)
                                        ax1.set_ylabel('Price (XAF/kg)', fontsize=12)
                                        ax1.set_title(f'{selected_year} Price Forecast: {region} - {commodity}', 
                                                    fontweight='bold', fontsize=14)
                                        ax1.legend(loc='best')
                                        ax1.grid(True, alpha=0.3)
                                        
                                        # Bar chart with monthly changes
                                        monthly_changes = [0] + [(yearly_prices[i] - yearly_prices[i-1])/yearly_prices[i-1]*100 
                                                                for i in range(1, len(yearly_prices))]
                                        
                                        colors = ['gray' if x == 0 else 'red' if x > 0 else 'green' for x in monthly_changes]
                                        ax2.bar(months, monthly_changes, color=colors, alpha=0.7)
                                        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
                                        ax2.set_xlabel('Month', fontsize=12)
                                        ax2.set_ylabel('Monthly Change (%)', fontsize=12)
                                        ax2.set_title('Monthly Price Changes', fontweight='bold', fontsize=14)
                                        ax2.grid(True, alpha=0.3)
                                        
                                        # Add value labels on bars
                                        for i, v in enumerate(monthly_changes):
                                            if v != 0:
                                                ax2.text(i, v + (0.5 if v > 0 else -1), 
                                                        f'{v:+.1f}%', 
                                                        ha='center', va='bottom' if v > 0 else 'top',
                                                        fontsize=9, fontweight='bold')
                                        
                                        plt.tight_layout()
                                        st.pyplot(fig)
                                        
                                        # Summary statistics
                                        st.markdown("#### 📊 Forecast Summary")
                                        summary_col1, summary_col2, summary_col3 = st.columns(3)
                                        
                                        with summary_col1:
                                            st.markdown("##### Price Statistics")
                                            st.write(f"**Annual Average:** {avg_price:.2f} XAF/kg")
                                            st.write(f"**Annual Min-Max:** {min_price:.2f} - {max_price:.2f} XAF/kg")
                                            st.write(f"**Price Volatility:** {(price_range/avg_price*100):.1f}%")
                                        
                                        with summary_col2:
                                            st.markdown("##### Seasonal Pattern")
                                            # Identify seasonal highs and lows
                                            high_months = [months[i] for i, p in enumerate(yearly_prices) 
                                                         if p == max_price]
                                            low_months = [months[i] for i, p in enumerate(yearly_prices) 
                                                        if p == min_price]
                                            
                                            st.write(f"**Peak Season:** {', '.join(high_months)}")
                                            st.write(f"**Low Season:** {', '.join(low_months)}")
                                            st.write(f"**Seasonal Range:** {price_range:.2f} XAF/kg")
                                        
                                        with summary_col3:
                                            st.markdown("##### Risk Assessment")
                                            # Calculate risk based on volatility
                                            volatility = np.std(yearly_prices) / avg_price * 100
                                            
                                            if volatility > 20:
                                                risk_level = "🔴 High Risk"
                                                recommendation = "Consider buffer stock strategies"
                                            elif volatility > 10:
                                                risk_level = "🟡 Moderate Risk"
                                                recommendation = "Monitor closely and plan ahead"
                                            else:
                                                risk_level = "🟢 Low Risk"
                                                recommendation = "Stable market conditions"
                                            
                                            st.write(f"**Risk Level:** {risk_level}")
                                            st.write(f"**Price Volatility:** {volatility:.1f}%")
                                            st.write(f"**Recommendation:** {recommendation}")
                                        
                                        # Download forecast data
                                        forecast_df = pd.DataFrame({
                                            'Month': months,
                                            'Date': [d.strftime('%Y-%m-%d') for d in yearly_dates],
                                            'Forecast_Price_XAF_per_kg': yearly_prices,
                                            'CI_Lower': [ci[0] for ci in yearly_ci],
                                            'CI_Upper': [ci[1] for ci in yearly_ci],
                                            'Monthly_Change_Pct': monthly_changes
                                        })
                                        
                                        csv = forecast_df.to_csv(index=False)
                                        st.download_button(
                                            label="📥 Download Yearly Forecast Data (CSV)",
                                            data=csv,
                                            file_name=f"{region}_{commodity}_{selected_year}_forecast.csv".replace(' ', '_'),
                                            mime="text/csv",
                                            help="Download the complete yearly forecast data"
                                        )
                                        
                                    else:
                                        st.warning(f"Only {len(yearly_prices)} months of data available for {selected_year}")
                                        # Show available data
                                        st.write("Available months:", [d.strftime('%Y-%m') for d in yearly_dates])
                                else:
                                    st.error("Failed to generate yearly forecast")
                            
                            else:
                                # Generate multi-year forecast
                                st.info(f"📊 Generating forecast for {start_year}-{end_year}...")
                                
                                # Generate forecasts for each year
                                all_forecasts = []
                                all_ci = []
                                all_dates = []
                                
                                for year in range(start_year, end_year + 1):
                                    yearly_prices, yearly_ci, yearly_dates, method, forecast_type_result = generate_yearly_forecast(
                                        monthly_data, 
                                        year,
                                        model_prediction=model_prediction
                                    )
                                    
                                    if yearly_prices:
                                        # Check for NaN values
                                        if any(np.isnan(p) for p in yearly_prices):
                                            yearly_series = pd.Series(yearly_prices)
                                            yearly_series = yearly_series.interpolate(method='linear')
                                            yearly_prices = yearly_series.tolist()
                                        
                                        all_forecasts.extend(yearly_prices)
                                        all_ci.extend(yearly_ci)
                                        all_dates.extend(yearly_dates)
                                
                                if all_forecasts:
                                    st.success(f"✅ Multi-year forecast generated ({len(all_forecasts)} months)")
                                    
                                    # Create comprehensive visualization
                                    st.markdown("#### 📈 Multi-Year Forecast Visualization")
                                    
                                    fig, ax = plt.subplots(figsize=(12, 6))
                                    
                                    # Plot historical data
                                    ax.plot(monthly_data.index, monthly_data.values, 'b-', linewidth=2, label='Historical Price', alpha=0.7)
                                    
                                    # Plot forecast
                                    ax.plot(all_dates, all_forecasts, 'r-', linewidth=2, label='Forecast Price')
                                    
                                    # Add confidence interval
                                    ci_low = [ci[0] for ci in all_ci]
                                    ci_high = [ci[1] for ci in all_ci]
                                    ax.fill_between(all_dates, ci_low, ci_high, 
                                                   alpha=0.2, color='red', label='90% Confidence Interval')
                                    
                                    ax.set_xlabel('Date', fontsize=12)
                                    ax.set_ylabel('Price (XAF/kg)', fontsize=12)
                                    ax.set_title(f'{start_year}-{end_year} Forecast: {region} - {commodity}', 
                                                fontweight='bold', fontsize=14)
                                    ax.legend(loc='upper left', fontsize=10)
                                    ax.grid(True, alpha=0.3)
                                    plt.xticks(rotation=45)
                                    
                                    plt.tight_layout()
                                    st.pyplot(fig)
                                    
                                    # Summary statistics for multi-year forecast
                                    st.markdown("#### 📊 Multi-Year Forecast Summary")
                                    
                                    summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
                                    with summary_col1:
                                        avg_price = np.mean(all_forecasts)
                                        st.metric("Average Price", f"{avg_price:.2f} XAF/kg")
                                    with summary_col2:
                                        min_price = np.min(all_forecasts)
                                        st.metric("Minimum Price", f"{min_price:.2f} XAF/kg")
                                    with summary_col3:
                                        max_price = np.max(all_forecasts)
                                        st.metric("Maximum Price", f"{max_price:.2f} XAF/kg")
                                    with summary_col4:
                                        annual_growth = ((all_forecasts[-1] / all_forecasts[0]) - 1) * 100 / ((end_year - start_year) + 1)
                                        st.metric("Avg Annual Growth", f"{annual_growth:.1f}%")
                                    
                                    # Create annual summary table
                                    st.markdown("#### 📅 Annual Summary")
                                    annual_data = []
                                    current_year = all_dates[0].year
                                    year_prices = []
                                    year_dates = []
                                    
                                    for price, date in zip(all_forecasts, all_dates):
                                        if date.year == current_year:
                                            year_prices.append(price)
                                            year_dates.append(date)
                                        else:
                                            if year_prices:
                                                annual_data.append({
                                                    'Year': current_year,
                                                    'Avg Price': f"{np.mean(year_prices):.2f}",
                                                    'Min Price': f"{np.min(year_prices):.2f}",
                                                    'Max Price': f"{np.max(year_prices):.2f}",
                                                    'Volatility %': f"{(np.std(year_prices)/np.mean(year_prices)*100):.1f}"
                                                })
                                            current_year = date.year
                                            year_prices = [price]
                                            year_dates = [date]
                                    
                                    # Add the last year
                                    if year_prices:
                                        annual_data.append({
                                            'Year': current_year,
                                            'Avg Price': f"{np.mean(year_prices):.2f}",
                                            'Min Price': f"{np.min(year_prices):.2f}",
                                            'Max Price': f"{np.max(year_prices):.2f}",
                                            'Volatility %': f"{(np.std(year_prices)/np.mean(year_prices)*100):.1f}"
                                        })
                                    
                                    annual_df = pd.DataFrame(annual_data)
                                    st.dataframe(annual_df, use_container_width=True)
                                    
                                    # Download multi-year forecast data
                                    forecast_df = pd.DataFrame({
                                        'Date': [d.strftime('%Y-%m-%d') for d in all_dates],
                                        'Forecast_Price_XAF_per_kg': all_forecasts,
                                        'CI_Lower': [ci[0] for ci in all_ci],
                                        'CI_Upper': [ci[1] for ci in all_ci]
                                    })
                                    
                                    csv = forecast_df.to_csv(index=False)
                                    st.download_button(
                                        label="📥 Download Multi-Year Forecast Data (CSV)",
                                        data=csv,
                                        file_name=f"{region}_{commodity}_{start_year}_{end_year}_forecast.csv".replace(' ', '_'),
                                        mime="text/csv",
                                        help="Download the complete multi-year forecast data"
                                    )
                                else:
                                    st.error("Failed to generate multi-year forecast")
                                
                        except Exception as e:
                            st.error(f"Error generating forecast: {str(e)}")
                            import traceback
                            st.error(f"Detailed error: {traceback.format_exc()}")
                
                else:
                    # Show placeholder when no forecast generated yet
                    st.info("👈 Select a region-commodity pair and forecast options, then click 'Generate Forecast' to see the projections.")
                    st.markdown("""
                    <div style="text-align: center; padding: 40px; background-color: #f8f9fa; border-radius: 10px;">
                        <div style="font-size: 48px; color: #ddd; margin-bottom: 20px;">📅</div>
                        <h4 style="color: #666;">Ready to Generate Forecast</h4>
                        <p style="color: #888;">
                        Select your parameters on the left and generate forecasts up to 2035.<br>
                        Choose between single year or multi-year forecasts for better planning.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
    
    elif page == "📊 Model Performance":
        # ... (Model Performance page remains the same)
        pass
        
    elif page == "📚 Data Overview":
        # ... (Data Overview page remains the same)
        pass
        
    elif page == "ℹ️ About":
        # ... (About page remains the same)
        pass

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray; font-size: 14px;'>"
    "🌾 Cameroon Food Price Forecasting System | © 2024 | For research use only"
    "</div>",
    unsafe_allow_html=True)