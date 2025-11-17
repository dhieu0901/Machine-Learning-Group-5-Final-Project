---
title: Hanoi Weather Intelligence Hub 🌡️
emoji: 🌤️
colorFrom: blue
colorTo: red
sdk: streamlit
app_file: app.py
pinned: false
license: mit
short_description: 5-day Hanoi temperature forecast with CatBoost + full XAI & backtesting
sdk_version: 1.41.0
python_version: 3.10
tags:
  - weather-forecast
  - catboost
  - streamlit
  - shap
  - time-series
---

# 🌤️ Hanoi Weather Intelligence Hub
**Group 5 - DSEB65A**

An interactive web application built with Streamlit that forecasts the average daily temperature in Hanoi for the next 5 days using a CatBoost machine learning model, featuring full model explainability (SHAP), backtesting mode, and live forecasting.

![Banner](https://cdn-uploads.huggingface.co/production/uploads/67aade8adc35713e0dc4d838/LdoMAAc46llJhXYKPz8ic.png)

---

## 📋 Table of Contents

- [Project Objectives](#project-objectives)
- [Key Features](#key-features)
- [The 9-Step ML Workflow](#the-9-step-ml-workflow)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Installation & Local Setup](#installation--local-setup)
- [How to Run](#how-to-run)
- [Team Members](#team-members)

---

## 🎯 Project Objectives

Build a complete end-to-end machine learning pipeline to accurately forecast Hanoi's average daily temperature for the next 5 days. The project covers the entire ML lifecycle:

- Data collection & cleaning (10+ years of weather data)
- Advanced feature engineering (lagged variables, rolling statistics, STL decomposition, FFT, holidays, etc.)
- Training a MultiOutput CatBoost Regressor
- Rigorous evaluation and deployment as an interactive web application with full model interpretability (XAI)

---

## ✨ Key Features

- **5-day temperature forecast** with high accuracy (R² ≈ 0.84 on test set)
- **Backtest Mode**: Select any past date → see both predictions and actual temperatures for direct comparison
- **Live Forecast Mode**: Real future forecast (today + next 5 days)
- **Interactive visualizations (Plotly)**:
  - Historical context (last 30 days)
  - Clear forecast vs actual overlay
- **Model Explainability (SHAP)**:
  - Top 5 features impacting each specific prediction
  - Global top 10 feature importance
  - Expandable detailed glossary of all 93 features
- **Performance dashboard**: R², RMSE, MAE on test set

---

## 🔬 The 9-Step ML Workflow

To ensure a systematic and reproducible approach, this project follows a 9-step machine learning (ML) workflow designed to move from raw data collection to model deployment and evaluation.

### Step 1 – Data Collection
The process begins with gathering 10 years of historical daily weather data for Hanoi from the *Visual Crossing Weather API*. This step ensures sufficient temporal coverage to capture long-term temperature patterns and climate variability.

### Step 2 – Data Understanding
In this stage, the dataset structure and meaning of all 33 features are explored. Statistical summaries and visualizations are used to analyze the target variable (temperature) and to examine relationships among features such as humidity, precipitation, and wind speed.

### Step 3 – Data Processing
Raw data is cleaned and standardized. Missing values are handled appropriately, data types are validated, and correlations are computed to identify redundant or irrelevant variables. The dataset is then normalized to prepare for efficient model training.

### Step 4 – Feature Engineering
This stage focuses on constructing new features to enhance predictive performance. Temporal features such as lag values, rolling means, and day–month indicators are created to capture sequential dependencies. The target is defined as temperature forecasting for the next five days.

### Step 5 – Model Training and Hyperparameter Tuning
Various ML models are trained, evaluated, and fine-tuned using frameworks such as *Optuna* for hyperparameter optimization. Performance metrics like **RMSE**, **MAPE**, and **R²** are employed to assess and interpret model quality in this forecasting context.

### Step 6 – User Interface Development
A user-friendly demo application is built using **Streamlit** (deployed on **Hugging Face**) to visualize predictions and enable interactive testing. This interface bridges the technical model and real-world usability.

### Step 7 – Model Monitoring and Retraining
As weather patterns evolve, model performance may degrade over time. This step involves tracking prediction errors and defining retraining strategies to maintain reliability in continuous deployment.

### Step 8 – Extension with Hourly Data
To evaluate the benefits of higher-resolution inputs, the entire pipeline is rerun using hourly data. This allows comparison between daily and hourly forecasting performance.

### Step 9 – Deployment Optimization with ONNX
Finally, the model is exported using the **ONNX** format to improve deployment efficiency and cross-platform compatibility, ensuring fast inference in production environments.

---

## 🛠️ Technologies Used

**Machine Learning & Data Processing**
- Python 3.10+
- Pandas, NumPy
- Scikit-learn
- CatBoost
- SHAP (explainability)
- Statsmodels, SciPy

**Frontend & Deployment**
- Streamlit (interactive web app)
- Plotly (interactive charts)
- Hugging Face Spaces (hosting)
- ONNX (deployment optimization)

---

## 📁 Project Structure

```text
.
├── app/                      
│   ├── artifacts/            # Trained models & feature lists
│   │   ├── best_daily_model.joblib
│   │   └── final_model_features.joblib
│   ├── data/                 # Test data
│   │   ├── Hanoi Daily 10 years.csv
│   │   ├── X_test.csv
│   │   ├── final_feature_importances.csv
│   │   └── y_test.csv
│   ├── .gitattributes
│   ├── app.py                # Main script
│   ├── feature_pipeline.py   # Feature engineering pipeline
│   └── requirements.txt      # Dependencies
│
├── data/                     # Raw data
│   ├── Hanoi Daily 10 years.csv
│   └── hanoi_weather_data_hourly.csv
│
├── models/                   
│   ├── daily/
│   │   ├── Best Daily Model Hyperparams.joblib
│   │   ├── Daily Model Final.ipynb
│   │   └── Daily Preprocessing Final.ipynb
│   └── hourly/
│       ├── Best Hourly Model Hyperparams.joblib
│       ├── Hourly Model Final.ipynb
│       └── Hourly Preprocessing Final.ipynb
│
├── notebooks/                # Experimental, analysis, and optimization notebooks
│   ├── Model Retraining.ipynb
│   └── ONNX.ipynb
│
├── README.md                 
└── requirements.txt          
```

## ⚙️ Setup & Installation

Follow these steps to set up the project environment and run the application locally.

**Prerequisites:**
- Python 3.9+

**Steps:**

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/dhieu0901/Machine-Learning-Group-5-Final-Project.git
    cd Machine-Learning-Group-5-Final-Project
    ```

2.  **Create and activate a virtual environment:**

    -   **On macOS/Linux:**
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```
    -   **On Windows:**
        ```bash
        python -m venv venv
        .\venv\Scripts\activate
        ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

---

## 🚀 How to Run

Once the setup is complete, you can launch the Streamlit application with a single command.

Make sure you are in the root directory of the project (`Machine-Learning-Group-5-Final-Project/`).

```bash
streamlit run app.py
```

The application will open in a new tab in your web browser.

---

## 👥 Group Members

- [Bui Chau Anh]
- [Tran Tuan Anh]
- [Nguyen Duong Hieu]
- [Dang Nhat Huy]
- [Nguyen Dai Quan]
