# Time Series Model Deployment with FastAPI

This repository contains a Time Series forecasting project, including the exploratory data analysis, model building, and deployment using **FastAPI** and **Uvicorn**.

## Project Overview

The goal of this project is to analyze time series data (Sales data), make it stationary, train a forecasting model, and serve predictions via a RESTful API.

### 1. Exploratory Data Analysis & Modeling (`TS_final_project.ipynb`)
The Jupyter Notebook contains the data preprocessing and model development steps:
* **Data Loading & Preparation:** Loads the time series data from `sales.csv` and sets the date as the index.
* **Stationarity Checks:** Uses the Augmented Dickey-Fuller (ADF) test to check for stationarity.
* **Differencing:** Applies first and second-order differencing. The data becomes stationary after the second difference (p-value: 3.48e-15).
* **Autocorrelation Analysis:** Plots ACF and PACF to determine the appropriate parameters for the ARIMA/SARIMA model.
* **Model Serialization:** The final trained model is exported as a pickle file (`tsmodel.pkl`).

### 2. API Deployment (`app.py` / `main.py`)
The application is deployed using **FastAPI**. It loads the pre-trained `tsmodel.pkl` model and provides endpoints to interact with the model and generate future predictions based on user-input dates.

---

## File Structure
- `TS_final_project.ipynb`: The Jupyter Notebook containing data analysis, stationarity tests, and model training.
- `app.py` (or `main.py`): The FastAPI application script to serve the model.
- `tsmodel.pkl`: The serialized machine learning model. *(Note: Ensure this is tracked or added to `.gitignore` depending on its size).*
- `sales.csv`: The dataset used for training *(add this to the repo if it's public)*.

---

## Requirements

To run this project, you need the following Python libraries:

```bash
numpy
pandas
matplotlib
statsmodels
fastapi
uvicorn


** How to Run the API **

**Clone the repository:**
git clone [https://github.com/sidharthapanda2005/TimeSeries_project_Forecasting.git](https://github.com/sidharthapanda2005/TimeSeries_project_Forecasting.git)
cd your-repo-name

**You can install them via pip:**
**pip install numpy pandas matplotlib statsmodels fastapi uvicorn**

**Update the model path:**
In your FastAPI script, ensure the path to tsmodel.pkl is correct for your local machine or server.
pickle_in = open("tsmodel.pkl", "rb")

**Start the FastAPI Server:**
**Run the following command in your terminal:**

Bash
**python app.py**

**Access the API:**
The API will be hosted at http://127.0.0.1:9000. You can access the automatic interactive API documentation provided by FastAPI (Swagger UI) at:
http://127.0.0.1:9000/docs

**API Endpoints**
1. Root Endpoint (GET /)
Checks if the API is running.

Response: {"Deployment": "Hello and Welcome to 5 Minutes Engineering"}

2. Predict Endpoint (POST /predict)
Generates a time series prediction for a specific given date.

Parameters (Query):

year (int): The year to predict.

month (int): The month to predict.

day (int): The day to predict.

Response: {"prediction": [predicted_value]}

