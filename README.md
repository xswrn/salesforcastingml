# AI-Powered Sales Forecasting System


🚀 Live Demo: https://your-app-link.onrender.com

📊 Dataset : https://github.com/upgini/upgini/raw/main/notebooks/train.csv.zip


## Overview

This project is an AI-powered sales forecasting application that predicts future product sales using Machine Learning and Natural Language Processing.

Users can either manually select a store, item, and forecast date or interact with the application using natural language queries such as:

> "Predict sales for Store 10 Item 5 on 31 January 2018"

The system extracts forecasting parameters using an LLM and generates predictions using a trained CatBoost model.

---

## Features

* Sales forecasting using CatBoost Regressor
* Natural language forecasting requests
* Interactive Streamlit dashboard
* Historical sales visualization
* AI-assisted parameter extraction
* Feature engineering with lag and rolling statistics
* Real-time forecast generation

---

## Tech Stack

### Machine Learning

* CatBoost Regressor
* Pandas
* NumPy
* Scikit-learn

### Frontend

* Streamlit
* Plotly

### AI Integration

* LLM API (Groq/OpenAI-compatible)
* OpenAI SDK

---

## Dataset

The project uses a historical retail sales dataset containing:

* Store ID
* Item ID
* Date
* Daily Sales

Data spans multiple stores and products across several years.

---

## Feature Engineering

The model uses time-series features including:

* Store
* Item
* Month
* Year
* Day
* Day of Week
* Weekend Indicator
* Lag 1
* Lag 7
* Lag 30
* Rolling Mean 7
* Rolling Mean 30

These features help capture seasonality, trends, and recent sales behavior.

---

## Model Performance

The CatBoost model was trained on historical sales data and evaluated using Root Mean Squared Error (RMSE).

The model learns temporal sales patterns and generates next-day sales forecasts based on engineered features.

---

## Running the Project

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configure Environment Variables

Create a `.env` file:

```env
LLM_API_KEY=your_api_key
LLM_MODEL=llama-3.3-70b-versatile
```

### Start the Application

```bash
streamlit run app.py
```

---

## Example Queries

* Predict sales for Store 10 Item 5 on 2018-01-31
* Forecast Item 12 in Store 3 for February 10, 2018
* Estimate sales for Store 7 Item 22 next week

---

## Future Improvements

* Multi-day recursive forecasting
* Inventory optimization recommendations
* Demand anomaly detection
* Forecast explanation using LLMs
* Cloud deployment and monitoring

---

## Author

Swaran

Electronics and Communication Engineering (ECE)

Interested in AI, Machine Learning, and Embedded Systems.
