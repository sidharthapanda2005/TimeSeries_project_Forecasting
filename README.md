# 📈 Time Series Sales Forecasting using SARIMAX

A complete **Time Series Forecasting** project that predicts future monthly sales using the **SARIMAX (Seasonal AutoRegressive Integrated Moving Average with Exogenous Variables)** model. This project demonstrates the complete workflow from data preprocessing, stationarity testing, model building, forecasting, and deployment using **FastAPI**.

---

# 📂 Project Structure

```
.
├── TS_final_project.ipynb      # Complete Machine Learning workflow
├── sales.csv                   # Historical sales dataset
├── times.py                    # FastAPI deployment script
├── tsmodel.pkl                 # Trained SARIMAX model (generated after training)
├── requirements.txt
└── README.md
```

---

# 🚀 Project Overview

The objective of this project is to forecast future sales based on historical monthly sales data.

The notebook covers the entire Time Series forecasting pipeline:

- Importing libraries
- Loading dataset
- Data preprocessing
- Converting Date column
- Setting Date as Index
- Stationarity testing
- Differencing
- Model Selection
- Training SARIMAX Model
- Forecasting Future Sales
- Model Deployment using FastAPI

---

# 📊 Dataset

The dataset contains two columns:

| Column | Description |
|----------|-------------|
| Date | Monthly sales date |
| Amount | Sales amount |

Example

| Date | Amount |
|-------|---------|
| 2000-01-01 | 266.0 |
| 2000-02-01 | 145.9 |

---

# 🛠 Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Statsmodels
- FastAPI
- Uvicorn
- Pickle

---

# 📖 Step-by-Step Notebook Explanation

---

## 1. Import Libraries

```python
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sms
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
```

### Explanation

### NumPy

Used for numerical computations.

```python
import numpy as np
```

---

### Matplotlib

Used for plotting graphs.

```python
import matplotlib.pyplot as plt
```

---

### Statsmodels

Provides statistical and Time Series models like ARIMA and SARIMAX.

```python
import statsmodels.api as sms
```

---

### Pandas

Used for handling datasets.

```python
import pandas as pd
```

---

### Ignore Warning Messages

```python
warnings.filterwarnings("ignore")
```

This hides unnecessary warning messages during execution.

---

# 2. Load Dataset

```python
data = pd.read_csv('sales.csv')
```

### Explanation

Reads the CSV file into a Pandas DataFrame.

---

# 3. View Dataset

```python
data.head()
```

Displays the first five rows.

---

```python
data.tail()
```

Displays the last five rows.

---

# 4. Convert Date Column

```python
data['Date'] = pd.to_datetime(data['Date'])
```

### Why?

Machine Learning models cannot understand dates stored as strings.

This converts them into datetime format.

---

# 5. Set Date as Index

```python
data.set_index('Date', inplace=True)
```

### Why?

Time Series models require the Date column as the index.

Result

```
Date
2000-01-01
2000-02-01
...
```

---

# 6. Visualize Sales

```python
data.plot()
```

### Purpose

Shows the sales trend over time.

This helps identify

- Trend
- Seasonality
- Increasing/Decreasing pattern

---

# 7. Stationarity Test (ADF Test)

Import

```python
from statsmodels.tsa.stattools import adfuller
```

Function

```python
def adf_test(d):
    result = adfuller(d)

    print("P value ->", result[1])

    if result[1] <= 0.05:
        print("Reject the null hypothesis. Data is stationary")
    else:
        print("Accept the null hypothesis. Data is not stationary")
```

### Explanation

The Augmented Dickey-Fuller (ADF) test checks whether the data is stationary.

Decision Rule

- P-value < 0.05 → Stationary
- P-value > 0.05 → Non-stationary

---

Run the test

```python
adf_test(data['Amount'])
```

---

# 8. First Differencing

```python
data['First Difference'] = data['Amount'] - data['Amount'].shift(1)
```

### Explanation

Removes trend by subtracting each value from its previous value.

Formula

```
Difference = Current Value − Previous Value
```

---

Again check stationarity

```python
adf_test(data['First Difference'].dropna())
```

---

# 9. Second Differencing

```python
data['Second Difference'] = data['First Difference'] - data['First Difference'].shift(1)
```

If the first difference isn't enough, apply differencing again.

---

# 10. Build SARIMAX Model

The notebook trains a **SARIMAX** model on the stationary time series data.

SARIMAX combines:

- Auto Regression (AR)
- Differencing (I)
- Moving Average (MA)
- Seasonal Components

This makes it suitable for forecasting seasonal sales data.

---

# 11. Train Model

```python
model = smodel.fit()
```

### Explanation

Fits the SARIMAX model to the historical sales data by learning the underlying trend, seasonality, and noise.

---

# 12. Predict Existing Data

```python
predicted = model.predict(
    start=datetime(2003,1,1),
    end=datetime(2008,12,1)
)
```

### Explanation

Generates predictions for historical dates to compare model performance against actual values.

---

# 13. Compare Predictions

```python
train_data['Predicted'] = predicted
```

Adds the predictions to the dataset.

---

# 14. Plot Actual vs Predicted

```python
train_data[['Amount','Predicted']].plot(figsize=(10,5))
```

This graph compares:

- Actual Sales
- Predicted Sales

A closer overlap indicates better model performance.

---

# 15. Forecast Future Sales

```python
predicted1 = model.predict(
    start=datetime(2009,1,1),
    end=datetime(2009,12,1)
)
```

Forecasts future monthly sales beyond the training period.

---

# 🌐 FastAPI Deployment

The project also includes a deployment script (`times.py`) to expose the trained model as a REST API.

## Features

- Loads the trained model using Pickle
- Provides a welcome endpoint
- Accepts year, month, and day as inputs
- Returns the predicted value

### API Endpoints

#### Home

```
GET /
```

Response

```json
{
  "Deployment": "Hello and Welcome to 5 Minutes Engineering"
}
```

---

#### Prediction

```
POST /predict
```

Parameters

| Parameter | Type |
|------------|------|
| year | Integer |
| month | Integer |
| day | Integer |

Returns

```json
{
    "prediction": value
}
```

---

# ▶️ How to Run

## 1. Clone Repository

```bash
git clone https://github.com/your-username/your-repository.git
```

---

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 3. Run Notebook

Open

```
TS_final_project.ipynb
```

Run all cells sequentially.

---

## 4. Run API

```bash
python times.py
```

or

```bash
uvicorn times:app --reload
```

---

# 📈 Workflow

```
Sales Dataset
      │
      ▼
Load Data
      │
      ▼
Convert Date
      │
      ▼
Set Date Index
      │
      ▼
Visualize Data
      │
      ▼
ADF Stationarity Test
      │
      ▼
Differencing
      │
      ▼
Train SARIMAX Model
      │
      ▼
Forecast Future Sales
      │
      ▼
Deploy using FastAPI
```

---

# 📌 Key Learning Outcomes

- Time Series Data Analysis
- Stationarity Testing (ADF)
- Differencing Techniques
- SARIMAX Modeling
- Sales Forecasting
- Model Evaluation
- FastAPI Deployment
- REST API Development
- Model Serialization using Pickle

---

# 📄 License

This project is intended for educational purposes and can be freely modified for learning and experimentation.

---

# 👨‍💻 Author

**Shridhar Mankar**

If you found this project helpful, consider giving the repository a ⭐ on GitHub.
