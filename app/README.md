---
title: ML DSEB Group5 Hanoi Weather Hub
emoji: 🔥
colorFrom: red
colorTo: red
sdk: streamlit
app_port: 8501
tags:
- streamlit
app_file: app.py
pinned: false
short_description: Streamlit template space
sdk_version: 1.51.0
---


# ☀️ Hanoi Weather Intelligence Hub

**A Project by Group 5 - DSEB65A**

An interactive web application built with Streamlit to forecast Hanoi's temperature for the next 5 days using a CatBoost machine learning model.

![ảnh](https://cdn-uploads.huggingface.co/production/uploads/67aade8adc35713e0dc4d838/LdoMAAc46llJhXYKPz8ic.png)

---

## 📋 Table of Contents

- [Project Goal](#-project-goal)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Setup & Installation](#-setup--installation)
- [How to Run](#-how-to-run)
- [Group Members](#-group-members)

---

## 🎯 Project Goal

The primary objective of this project is to build an end-to-end machine learning application that forecasts the daily average temperature in Hanoi, Vietnam, for the next 5 days. The project covers the complete ML lifecycle, from data collection and feature engineering to model training, evaluation, and deployment as an interactive web app.

---

## ✨ Features

- **5-Day Temperature Forecast**: Provides temperature predictions for the next 5 days based on historical data.
- **Interactive Visualizations**:
    - **Historical Context**: Displays the actual temperature for the 30 days leading up to the forecast date.
    - **Forecast vs. Actual**: Compares the model's 5-day forecast against the real temperatures (for past dates).
- **Model Explainability (XAI)**:
    - **Key Factors**: Shows the top 5 most influential features for any given prediction.
    - **Overall Importance**: Visualizes the top 10 most important features for the model overall.
    - **Full Feature Glossary**: An expandable section detailing all 46 features used by the model.
- **Performance Dashboard**: Displays key regression metrics (R² Score, RMSE, MAE) evaluated on the test set.
- **Live vs. Backtest Modes**:
    - **Backtest Mode**: When selecting a past date, the app shows both predicted and actual values for comparison.
    - **Live Forecast Mode**: When selecting the latest available date, the app provides a true forecast for the future.

---

## 🛠️ Tech Stack

- **Backend & Modeling**:
    - **Python**: Core programming language.
    - **Pandas**: Data manipulation and analysis.
    - **Scikit-learn**: Data preprocessing and model evaluation.
    - **CatBoost**: The gradient boosting model used for forecasting.
    - **Statsmodels & Scipy**: For advanced feature engineering (STL decomposition, FFT).
- **Frontend & Visualization**:
    - **Streamlit**: For building the interactive web application.
    - **Plotly**: For creating interactive charts and visualizations.

---

## 📂 Project Structure

```
|
|-- artifacts/
|   |-- best_daily_model.joblib         # The final trained MultiOutput CatBoost model.
|   |-- final_model_features.joblib     # List of all feature names the model was trained on.
|
|-- data/
|   |-- Hanoi Daily 10 years.csv      # The raw, original 10-year weather dataset.
|   |-- final_feature_importances.csv # Model-based feature importance scores, used in the 'Deep Dive' tab.
|   |-- X_test.csv                      # The exact test set features saved from the notebook for 100% reproducible evaluation.
|   |-- y_test.csv                      # The exact test set labels saved from the notebook.
|
|-- app.py                              # The main Streamlit application script.
|-- feature_pipeline.py                 # Contains the function for all data cleaning and feature engineering.
|-- requirements.txt                    # A list of all required Python packages to run the project.
|-- README.md                           # This file.
```

---

## ⚙️ Setup & Installation

Follow these steps to set up the project environment and run the application locally.

**Prerequisites:**
- Python 3.9+

**Steps:**

1.  **Clone the repository:**
    ```bash
    git clone [link-to-your-github-repo]
    cd ML_DSEB_Group5_Hanoi_weather_hub
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

Make sure you are in the root directory of the project (`ML_DSEB_Group5_Hanoi_weather_hub/`).

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