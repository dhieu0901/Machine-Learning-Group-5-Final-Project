
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from feature_pipeline import create_features_new as create_features
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Group 5 Hanoi Weather Hub",
    page_icon="☀️",
    layout="wide"
)

# --- CONSTANTS ---
PLOT_COLORS = {'forecast': "#1547eb", 'actual': "#FF934F"}

FEATURE_DESCRIPTIONS = {
    # Core & High Importance Features
    "temp": "The average daily temperature (°C). This is the primary target variable and the most important feature for the model.",
    "cos_day_of_year": "Cosine transformation of the day of the year (1-365). Paired with sine, it captures the main seasonal cycle smoothly.",
    "temp_ewm_14": "14-day Exponentially Weighted Moving Average of temperature. Represents the medium-term temperature trend, giving more weight to recent days.",
    "temp_ewm_30": "30-day Exponentially Weighted Moving Average of temperature. Represents the long-term temperature trend.",
    "winddir_cos": "Cosine transformation of the wind direction (in degrees). Helps the model understand the East-West component of the wind.",
    "temp_diff_1": "Difference between today's and yesterday's temperature. Indicates the short-term rate of temperature change.",
    "season_Winter": "A binary flag (1 if the season is Winter, 0 otherwise). Captures distinct winter weather patterns.",
    "temp_ewm_7": "7-day Exponentially Weighted Moving Average of temperature. Represents the short-term temperature trend.",
    "windspeed": "The average wind speed for the day (km/h). Higher wind can lower perceived temperature.",
    "windgust": "The highest instantaneous wind speed recorded during the day (km/h). Indicates the intensity of wind.",
    "solarradiation": "The amount of solar energy received per unit area. A primary driver of heat.",
    "wind_vector_ew": "The East-West vector component of the wind, calculated from wind speed and direction.",
    "temp_roll_min_7": "The minimum temperature recorded over the past 7 days.",
    "temp_lag_1": "The average temperature from the previous day.",
    "season_Summer": "A binary flag (1 if the season is Summer, 0 otherwise).",
    "temp_roll_min_3": "The minimum temperature recorded over the past 3 days.",
    "humidity": "The relative humidity of the air (%). High humidity can make it feel warmer.",
    "visibility": "The distance at which objects can be clearly seen (km). Often related to humidity and pollution.",
    "windspeed_diff_1": "Difference between today's and yesterday's wind speed.",
    "precip_ewm_30": "30-day Exponentially Weighted Moving Average of precipitation.",
    "temp_range": "The difference between the maximum (tempmax) and minimum (tempmin) temperature of the day.",
    "interaction_lag1_summer": "An interaction term between yesterday's temperature and the Summer season. Models how heat persists differently in summer.",
    "temp_resid_yearly": "The residual (irregular noise) component after removing trend and seasonality from the temperature series using STL.",
    "humidity_diff_1": "Difference between today's and yesterday's humidity.",
    "windspeed_diff_7": "The difference between today's wind speed and the wind speed from 7 days ago.",
    "precipcover": "The percentage of the day during which precipitation was recorded.",
    "precip_diff_1": "Difference between today's and yesterday's precipitation amount.",
    "sin_day_of_year": "Sine transformation of the day of the year. Paired with cosine to pinpoint the exact time of year.",
    "season_Spring": "A binary flag (1 if the season is Spring, 0 otherwise).",
    "season_Fall": "A binary flag (1 if the season is Fall, 0 otherwise).",
    "temp_trend_yearly": "The long-term trend component of the temperature (e.g., gradual warming over years), isolated using STL.",
    "temp_diff_7": "The difference in temperature between today and 7 days ago.",
    "cloudcover_roll_std_30": "Standard deviation of cloud cover over the past 30 days, indicating cloudiness stability.",
    "cloudcover_ewm_3": "3-day Exponentially Weighted Moving Average of cloud cover.",
    "precip_diff_7": "The difference in precipitation between today and 7 days ago.",
    "precip": "The amount of precipitation (rain) recorded (mm).",
    "windspeed_lag_8": "Wind speed from 8 days ago.",
    "cloudcover_ewm_7": "7-day Exponentially Weighted Moving Average of cloud cover.",
    "precip_roll_mean_30": "The average precipitation over the past 30 days.",
    "windspeed_ewm_30": "30-day Exponentially Weighted Moving Average of wind speed.",
    "humidity_diff_7": "The difference in humidity between today and 7 days ago.",
    "windspeed_roll_min_30": "The minimum wind speed recorded in the past 30 days.",
    "humidity_ewm_30": "30-day Exponentially Weighted Moving Average of humidity.",
    "humidity_ewm_14": "14-day Exponentially Weighted Moving Average of humidity.",
    "windspeed_roll_mean_30": "The average wind speed over the past 30 days.",
    "temp_roll_std_30": "Standard deviation of temperature over the past 30 days.",
    "moonphase_cos": "Cosine transformation of the lunar phase (0-1). Captures the cyclical effect of the moon.",
    "cloudcover_diff_7": "The difference in cloud cover between today and 7 days ago.",
    "precip_roll_mean_3": "The average precipitation over the past 3 days.",
    "icon_rain": "A binary flag indicating if the weather icon for the day was 'rain'.",
    "precip_lag_5": "Precipitation amount from 5 days ago.",
    "cloudcover_diff_1": "Difference between today's and yesterday's cloud cover.",
    "fft_sin_freq_1": "A cyclical feature from Fourier analysis, capturing a dominant underlying cycle in the temperature data's residuals.",
    "humidity_lag_9": "Humidity from 9 days ago.",
    "humidity_roll_std_7": "Standard deviation of humidity over the past 7 days.",
    "windspeed_lag_9": "Wind speed from 9 days ago.",
    "precip_roll_min_3": "The minimum precipitation recorded in the past 3 days.",
    "precip_lag_4": "Precipitation amount from 4 days ago.",
    "windspeed_roll_std_14": "Standard deviation of wind speed over the past 14 days.",
    "temp_roll_std_7": "Standard deviation of temperature over the past 7 days.",
    "windspeed_roll_min_3": "The minimum wind speed recorded in the past 3 days.",
    "year": "The year of the observation. Can capture very long-term trends.",
    "precip_roll_min_7": "The minimum precipitation recorded in the past 7 days.",
    "temp_roll_std_3": "Standard deviation of temperature over the past 3 days.",
    "moonphase_sin": "Sine transformation of the lunar phase. Paired with cosine.",
    "fft_cos_freq_2": "A feature capturing the second most dominant underlying cycle from Fourier analysis.",
    "cloudcover_roll_std_3": "Standard deviation of cloud cover over the past 3 days.",
    "windspeed_roll_min_14": "The minimum wind speed recorded in the past 14 days.",
    "icon_cloudy": "A binary flag for the 'cloudy' weather icon.",
    "interaction_wind_clearsky_effect": "Interaction between wind speed and clear skies (1 - cloud cover). Models the wind chill effect.",
    "day_of_week": "The day of the week (0=Monday, 6=Sunday).",
    "icon_partly-cloudy-day": "A binary flag for the 'partly-cloudy-day' weather icon.",
    "Partially cloudy": "A binary flag indicating if the text condition was 'Partially cloudy'.",
    "icon_clear-day": "A binary flag for the 'clear-day' weather icon.",
    "Clear": "A binary flag indicating if the text condition was 'Clear'.",
    "precipprob": "The probability of precipitation occurring (%). Had low importance, likely because other precip features were more informative.",
    "fft_cos_freq_1": "Paired with fft_sin_freq_1 to pinpoint the phase of the dominant cycle.",
    "fft_sin_freq_2": "Paired with fft_cos_freq_2 for the second dominant cycle.",
    "interaction_lag1_weekend": "Interaction between yesterday's temperature and whether it is a weekend.",
    "precip_roll_min_30": "The minimum precipitation recorded in the past 30 days.",
    "day_of_month": "The day of the month (1-31).",
    "is_holiday": "A binary flag (1 if it's a public holiday in Vietnam, 0 otherwise).",
    "precip_roll_min_14": "The minimum precipitation recorded in the past 14 days.",
    "windspeed_lag_7": "Wind speed from 7 days ago.",
    "windspeed_lag_3": "Wind speed from 3 days ago.",
    "humidity_roll_std_3": "Standard deviation of humidity over the past 3 days.",
    "humidity_roll_std_30": "Standard deviation of humidity over the past 30 days.",
    "windspeed_roll_std_3": "Standard deviation of wind speed over the past 3 days.",
    "windspeed_roll_std_7": "Standard deviation of wind speed over the past 7 days.",
    "windspeed_roll_max_30": "The maximum wind speed recorded in the past 30 days.",
    "humidity_ewm_3": "3-day Exponentially Weighted Moving Average of humidity.",
    "precip_roll_max_3": "The maximum precipitation recorded in the past 3 days.",
    "precip_lag_3": "Precipitation amount from 3 days ago."
}

# --- 2. CUSTOM CSS FOR STYLING ---
def load_css():
    st.markdown("""
    <style>
    /* ===== Background gradient ===== */
    [data-testid="stAppViewContainer"] {
    background-image: linear-gradient(to bottom, #d4e1da 20%, #005aa7 100%) !important;
    background-attachment: fixed !important;
    background-size: cover !important;
    }
    /* ===== Global text ===== */
    .stApp, h1, h2, h3, p, label {
    color: #0A1931;
    }
    /* ===== Tabs ===== */
    button[data-baseweb="tab"][aria-selected="true"] {
    background-color: #FFFFFF !important;
    color: #005aa7 !important;
    padding: 10px 16px !important;
    border-radius: 10px 10px 0 0 !important;
    font-weight: 700 !important;
    border-bottom: 3px solid #e53935 !important;
    }
    button[data-baseweb="tab"][aria-selected="false"] {
    background-color: rgba(255,255,255,0.3) !important;
    color: #284270 !important;
    padding: 10px 16px !important;
    }
    /* ===== Metrics: Forecast titles, main numbers, and Actual (delta) ===== */
    [data-testid="stMetricLabel"] p,
    div[data-testid="stMetricValue"] {
    font-weight: 700 !important;
    text-shadow: 1px 1px 3px rgba(0,0,0,0.3);
    }
    div[data-testid="stMetricLabel"] p:contains("Forecast for") {
    color: #ffffff !important;
    font-size: 2.2rem !important;
    font-weight: 800 !important;
    text-shadow: 1px 1px 3px rgba(0,0,0,0.3);
    background: none !important;
    border: none !important;
    }
    div[data-testid="stMetricValue"] {
    color: #fefefe !important;
    font-size: 1.9rem !important;
    text-shadow: 1px 1px 6px rgba(0,0,0,0.5);
    }
    div[data-testid="stMetricLabel"] p {
    color: #dce9ff !important;
    font-weight: 700 !important;
    letter-spacing: 0.3px;
    }
    div[data-testid="stMetricLabel"] p {
    color: #e8f1ff !important;
    font-size: 1.4rem !important;
    font-weight: 700 !important;
    text-shadow: none !important;
    background: rgba(0, 0, 0, 0.15);
    padding: 4px 8px;
    border-radius: 6px;
    }
    div[data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-size: 1.8rem !important;
    text-shadow: 0 0 3px rgba(0,0,0,0.3);
    background: rgba(0,0,0,0.15);
    border-radius: 6px;
    padding: 6px 10px;
    }
    div[data-testid="stMetricValue"] {
    color: #fefefe !important;
    font-size: 1.4rem !important;
    text-shadow: 1px 1px 6px rgba(0,0,0,0.5);
    }
    div[data-testid="stMetricLabel"] p {
    color: #dce9ff !important;
    font-weight: 700 !important;
    letter-spacing: 0.3px;
    }
    div[data-testid="stMetricValue"],
    div[data-testid="stMetricLabel"] p {
    filter: drop-shadow(0 0 3px rgba(255,255,255,0.4));
    }
    /* ===== Custom feature cards (Tab 2) ===== */
    .custom-card-transparent {
    background: linear-gradient(145deg, rgba(255,255,255,0.12), rgba(0,0,0,0.25));
    border: 1px solid rgba(255,255,255,0.25);
    border-radius: 12px;
    padding: 14px;
    text-align: center;
    transition: all 0.25s ease;
    box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    width: 95%;
    margin: 0 auto;
    }
    .custom-card-transparent:hover {
    transform: translateY(-3px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.25);
    filter: brightness(1.1);
    }
    .custom-card-transparent h2 {
    color: #fefefe !important;
    font-size: 1.6rem !important;
    font-weight: 700 !important;
    text-shadow: 0 0 4px rgba(255,255,255,0.25);
    margin-top: 10px;
    }
    .feature-tag-final {
    background-color: #1c3d70 !important;
    padding: 5px 10px;
    border-radius: 8px;
    font-size: 0.9rem;
    font-weight: 600;
    color: #e3eeff !important;
    text-shadow: 0 0 2px rgba(255,255,255,0.15);
    display: inline-block;
    margin-bottom: 3px;
    }
    /* ===== Restored missing pieces ===== */
    div[data-testid="stExpander"] {
    background-color: rgba(255, 255, 255, 0.85) !important;
    border-radius: 12px !important;
    border: 1px solid rgba(0, 90, 167, 0.25) !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1) !important;
    margin-top: 2rem;
    }
    div.st-emotion-cache-1e5k5x7 {
    background-color: rgba(230, 240, 255, 0.5) !important;
    border-radius: 15px !important;
    padding: 1rem 0.5rem !important;
    border: 1px solid rgba(255, 255, 255, 0.3);
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    margin: 1rem 0;
    }
    div.st-emotion-cache-1e5k5x7 div[data-testid="stMetricValue"] {
    color: white !important;
    text-shadow: 1px 1px 3px rgba(0,0,0,0.4);
    }
    div.st-emotion-cache-1e5k5x7 div[data-testid="stMetricLabel"] p {
    color: #0A1931 !important;
    font-weight: 600 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
    border-bottom: 2px solid rgba(229,57,53,0.9) !important;
    }
    div[data-testid="stMetricLabel"] p:contains("Forecast for") {
    color: #ffffff !important;
    font-size: 2.2rem !important;
    font-weight: 800 !important;
    text-shadow: 1px 1px 3px rgba(0,0,0,0.3);
    background: none !important;
    border: none !important;
    }
    div[data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-size: 2.2rem !important;
    font-weight: 800 !important;
    text-shadow: 2px 2px 5px rgba(0,0,0,0.4);
    background: none !important;
    border: none !important;
    padding: 0 !important;
    }
    [data-testid="stMetricDelta"] {
    font-size: 1.2rem !important;
    font-weight: 700 !important;
    color: #ffea7a !important;
    text-shadow: 1px 1px 3px rgba(0,0,0,0.4);
    }
    div[data-testid="stMetricLabel"] p {
    font-size: 1.5rem !important;
    font-weight: 700 !important;
    color: #06285a !important;
    text-shadow: none !important;
    background: none !important;
    }
    div[data-testid="stMetricValue"],
    div[data-testid="stMetricLabel"] p {
    filter: none !important;
    }
    /* ===== Custom Styling for Date INPUT FIELD ONLY ===== */
    div[data-testid="stDateInput"] input {
    background-color: #f0f2f6 !important;
    color: #0A1931 !important;
    border: 0.5px solid #cccccc !important;
    border-radius: 4px;
    padding: 8px 10px;
    box-shadow: inset 0 1px 2px rgba(75, 98, 138);
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }
    div[data-testid="stDateInput"] button {
    display: none !important;
    }
    div[data-testid="stDateInput"] input:hover,
    div[data-testid="stDateInput"] input:focus {
    border-color: #1547eb !important;
    box-shadow: inset 0 1px 1px rgba(0,0,0,0.1), 0 0 0 2px rgba(21, 71, 235, 0.2);
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. DATA & MODEL LOADING ---
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load('artifacts/best_daily_model.joblib')
        selected_features = joblib.load('artifacts/final_model_features.joblib')
        # <<< THAY ĐỔI: Tải cả X_test và y_test >>>
        X_test = pd.read_csv('data/X_test.csv', index_col='datetime', parse_dates=True)
        y_test = pd.read_csv('data/y_test.csv', index_col='datetime', parse_dates=True)
        feature_importances = pd.read_csv('data/final_feature_importances.csv')
    except FileNotFoundError as e:
        st.error(f"Error loading artifacts: {e}. Please ensure X_test.csv, y_test.csv, etc. exist.")
        return None, None, None, None, None
    return model, selected_features, X_test, y_test, feature_importances

@st.cache_data
def process_full_data(_selected_features):
    raw_df = pd.read_csv('data/Hanoi Daily 10 years.csv')
    df_featured = create_features(raw_df)
    horizons = [1, 2, 3, 4, 5]
    for h in horizons:
        df_featured[f'target_temp_t+{h}'] = df_featured['temp'].shift(-h)
    df_final_clean = df_featured.dropna(subset=_selected_features)
    return df_final_clean
# --- 4. MAIN APP LOGIC ---
load_css()
st.title("☀️ Group 5 Hanoi Weather Hub")
st.write("An interactive dashboard to forecast Hanoi's temperature for the next 5 days.")

model, selected_features, X_test, y_test, feature_importances = load_artifacts()
if model is None:
    st.stop()

with st.spinner("Processing full historical data..."):
    processed_df = process_full_data(selected_features)

if processed_df is None:
    st.stop()

col1, col2 = st.columns([1, 2])
with col1:
    selected_date = st.date_input(
        label="Select a date to forecast from:",
        value=processed_df.index.max(),
        min_value=processed_df.index.min(),
        max_value=processed_df.index.max(),
        format="YYYY-MM-DD"
    )

tab1, tab2, tab3 = st.tabs(["📈 Interactive Forecast", "🧠 Forecast Deep Dive", "📊 Model Performance"])

# --- TAB 1 --- 
with tab1:
    selected_date_ts = pd.Timestamp(selected_date)
    st.header(f"5-Day Forecast from {selected_date_ts.strftime('%Y-%m-%d')}")
    
    input_data = processed_df.loc[[selected_date_ts]]
    input_features = input_data[selected_features]
    prediction = model.predict(input_features)[0]

    forecast_dates = pd.date_range(start=selected_date_ts + pd.Timedelta(days=1), periods=5, freq='D')
    actual_values = [processed_df.loc[selected_date_ts, f'target_temp_t+{i+1}'] for i in range(5)]

    pred_cols = st.columns(5)
    for i, date in enumerate(forecast_dates):
        actual_val = actual_values[i]
        delta_text = f"Actual: {actual_val:.1f}°C" if pd.notna(actual_val) else "Actual: --°C"
        pred_cols[i].metric(label=f"Forecast for {date.strftime('%b %d')}", value=f"{prediction[i]:.1f}°C", delta=delta_text, delta_color="off")

    st.subheader("Visualizations")
    st.markdown("#### Historical Context: Past 30 Days")
    hist_data = processed_df.loc[selected_date_ts - pd.Timedelta(days=30):selected_date_ts, 'temp']
    fig_hist = go.Figure(go.Scatter(x=hist_data.index, y=hist_data, mode='lines', name='Past 30 Days', line=dict(color='#636EFA')))
    fig_hist.update_layout(title={'text': "<b>Actual Temperature - Past 30 Days</b>", 'x': 0.5}, paper_bgcolor="#fafbfc", plot_bgcolor="#e5ecf6")
    components.html(fig_hist.to_html(include_plotlyjs='cdn'), height=450)

    is_partial_forecast = any(pd.isna(v) for v in actual_values)

    if not is_partial_forecast:
        st.markdown("#### Forecast vs. Actual Comparison")
        fig_comp = go.Figure()
        fig_comp.add_trace(go.Scatter(x=forecast_dates, y=prediction, mode='lines+markers', name='5-Day Forecast', line=dict(color=PLOT_COLORS['forecast'])))
        fig_comp.add_trace(go.Scatter(x=forecast_dates, y=actual_values, mode='lines+markers', name='5-Day Actual', line=dict(color=PLOT_COLORS['actual'])))
        fig_comp.update_layout(title={'text': "<b>5-Day Forecast vs. Actual Temperature</b>", 'x': 0.5}, paper_bgcolor='#fafbfc', plot_bgcolor='#e5ecf6')
        components.html(fig_comp.to_html(include_plotlyjs='cdn'), height=450)
# --- TAB 2 --- 
with tab2:
    st.header("What were the most important factors for this forecast?")
    st.write(f"For the forecast made on **{selected_date.strftime('%Y-%m-%d')}**, the model paid most attention to these factors:")
    
    top_5_features = feature_importances.head(5)['Feature'].tolist()
    key_factor_cols = st.columns(len(top_5_features))
    
    for i, feature in enumerate(top_5_features):
        value = processed_df.loc[pd.Timestamp(selected_date), feature]
        with key_factor_cols[i]:
            st.markdown(f"""
            <div class="custom-card-transparent">
                <span class="feature-tag-final">{feature}</span>
                <h2>{value:.2f}</h2>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    st.subheader("Overall Feature Importance")
    top_10_df = feature_importances.head(10)
    fig_imp = go.Figure(go.Bar(x=top_10_df['Importance'], y=top_10_df['Feature'], orientation='h', marker_color='#005aa7'))
    fig_imp.update_layout(
        title={'text': '<b>Top 10 Feature Importance (Model-Based)</b>', 'x': 0.5},
        xaxis_title='Importance_Mean',
        yaxis={'autorange': "reversed"},
        margin=dict(l=150, r=20, t=50, b=70), paper_bgcolor='#fafbfc', plot_bgcolor='#e5ecf6'
    )
    components.html(fig_imp.to_html(include_plotlyjs='cdn'), height=520)

    st.markdown("<br>", unsafe_allow_html=True)
    
    st.header("Feature Glossary")
    with st.expander("Click to learn about all model features", expanded=False):
        glossary_df = pd.DataFrame(
            FEATURE_DESCRIPTIONS.items(), 
            columns=['Feature', 'Description']
        ).sort_values(by='Feature').reset_index(drop=True)
        html_table = "<table class='glossary-table'><tr><th>Feature</th><th>Description</th></tr>"
        for _, row in glossary_df.iterrows():
            if row['Feature'] in selected_features:
                html_table += f"<tr><td><b>{row['Feature']}</b></td><td>{row['Description']}</td></tr>"
        html_table += "</table>"
        st.markdown(html_table, unsafe_allow_html=True)

# --- TAB 3  ---
with tab3:
    st.header("Model Performance on the Entire Test Set")

    X_test_aligned = X_test[selected_features]
    
    y_pred_test = model.predict(X_test_aligned)

    macro_r2 = r2_score(y_test, y_pred_test)
    macro_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    macro_mae = mean_absolute_error(y_test, y_pred_test)
    
    m_cols = st.columns(3)
    
    m_cols[0].metric("Average R2 Score", f"{macro_r2:.4f}")
    m_cols[1].metric("Average RMSE", f"{macro_rmse:.4f}°C")
    m_cols[2].metric("Average MAE", f"{macro_mae:.4f}°C")
    
    st.subheader("Prediction vs. Actual Values")
    y_test_flat = y_test.values.flatten()
    y_pred_flat = y_pred_test.flatten()
    
    fig_scatter = go.Figure()
    fig_scatter.add_trace(go.Scatter(x=y_test_flat, y=y_pred_flat, mode='markers', marker=dict(color="rgba(73, 82, 199, 0.6)"), name='Predictions'))
    fig_scatter.add_trace(go.Scatter(x=[y_test_flat.min(), y_test_flat.max()], y=[y_test_flat.min(), y_test_flat.max()], mode='lines', line=dict(color='#EF553B', dash='dash'), name='Perfect Prediction Line'))
    fig_scatter.update_layout(title={'text': "<b>How well do predictions match actual values?</b>", 'x': 0.5}, xaxis_title="Actual Temperature (°C)", yaxis_title="Predicted Temperature (°C)", paper_bgcolor='#fafbfc', plot_bgcolor='#e5ecf6')
    components.html(fig_scatter.to_html(include_plotlyjs='cdn'), height=600)