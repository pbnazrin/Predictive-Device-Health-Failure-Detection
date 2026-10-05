# Predictive Device Health & Failure Detection

An end-to-end machine learning project that predicts whether a device is likely to experience a failure within the next 7 days based on device health and telemetry data.

> **Note:** This project uses synthetic device telemetry data created for learning and portfolio purposes.

---

## 📌 Problem Statement

Device failures can lead to downtime, poor user experience, and increased maintenance effort.

The goal of this project is to build a machine learning model that can identify devices at higher risk of failure within the next 7 days using telemetry and device health indicators.

---

## 🎯 Objective

Predict:

```text
Failure_Within_7_Days
```

where:

* `0` → Device is not expected to fail within 7 days
* `1` → Device is expected to fail within 7 days

The model produces both:

* Failure probability
* Binary failure prediction

---

## 📊 Dataset

The project uses a synthetic dataset containing:

* **12,020 device records**
* **19 original columns**
* Device health and telemetry information
* Binary failure target

### Example Features

* Battery Health
* Battery Temperature
* CPU Usage
* Memory Usage
* Storage Usage
* Network Signal
* App Crash Count
* Reboot Count
* Error Count
* Data Usage
* Days Since Last Update
* Last Check-in Hours
* Operating System
* OS Version
* Manufacturer
* Device Model
* Network Type

`Device_ID` is treated as an identifier and is excluded from model training.

---

## 🔎 Exploratory Data Analysis

The EDA investigated:

* Missing values
* Duplicate records
* Feature distributions
* Correlations with device failure
* Failure rates across feature ranges
* Relationships between device health indicators
* Potential outliers
* Potential target leakage

Some notable patterns observed in the synthetic dataset:

* Lower battery health was associated with higher failure rates.
* Higher CPU and memory usage were associated with higher failure rates.
* Longer periods since the last update were associated with higher failure rates.
* Longer last-check-in intervals were associated with higher failure rates.
* Devices with simultaneously high CPU and memory usage showed substantially higher failure rates.

These relationships are observations from the synthetic dataset and should not be interpreted as real-world device failure behavior without validation on real data.

---

## 🏗️ Machine Learning Workflow

```text
Raw Data
   ↓
Exploratory Data Analysis
   ↓
Train/Test Split
   ↓
Missing Value Imputation
   ↓
Categorical Encoding
   ↓
Feature Scaling
   ↓
Model Training
   ↓
Hyperparameter Tuning
   ↓
Model Evaluation
   ↓
Threshold Analysis
   ↓
Final Prediction Pipeline
```

---

## 🧹 Data Preprocessing

### Numerical Features

Missing numerical values are handled using median imputation.

Standard scaling is applied to numerical features.

### Categorical Features

Missing categorical values are handled using the most frequent value.

Categorical variables are converted using one-hot encoding.

```python
OneHotEncoder(handle_unknown="ignore")
```

Using `handle_unknown="ignore"` allows the model to process previously unseen categorical values during prediction.

### Data Leakage Prevention

The dataset is split into training and testing sets **before preprocessing**.

The preprocessing steps are fitted only on the training data and then applied to the test data.

---

## 🤖 Models

Two models were evaluated.

### Logistic Regression — Baseline

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 82.49% |
| Precision | 73.96% |
| Recall    | 58.07% |
| F1 Score  | 65.06% |
| ROC-AUC   | 86.50% |

Logistic Regression was used as the baseline model.

### Random Forest

A Random Forest classifier was then trained and tuned using `RandomizedSearchCV` with 3-fold cross-validation.

The tuning objective was F1 score.

---

## 🎚️ Threshold Analysis

The default classification threshold of 0.50 was evaluated along with lower thresholds.

| Threshold | Precision |    Recall |        F1 |
| --------: | --------: | --------: | --------: |
|      0.50 |     76.5% |     39.6% |     52.1% |
|      0.45 |     70.9% |     50.2% |     58.8% |
|      0.40 |     65.0% |     59.9% |     62.3% |
|  **0.35** | **58.8%** | **69.3%** | **63.6%** |
|      0.30 |     53.6% |     77.6% |     63.4% |

A threshold of **0.35** is currently used as the experimental operating point.

The lower threshold increases recall, allowing the system to identify more potentially failing devices, while also increasing false positives.

> Threshold selection should be validated using a separate validation set or cross-validation before production deployment.

---

## 💾 Saved Model

The final trained model is stored in:

```text
models/device_failure_model.joblib
```

The prediction threshold is stored in:

```text
models/failure_threshold.joblib
```

The saved model contains the preprocessing pipeline together with the trained Random Forest classifier.

---

## 🔮 Making Predictions

Predictions can be generated using:

```bash
python src/predict.py
```

The prediction pipeline accepts raw device telemetry and returns:

```text
Failure Probability
Failure Within 7 Days
```

Example:

```python
{
    "failure_probability": 0.72,
    "failure_within_7_days": 1
}
```

---

## 📁 Project Structure

```text
predictive-device-health/
│
├── data/
│   ├── raw/
│   │   └── device_telemetry.csv
│   └── processed/
│
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_model_experiments.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── models/
│   ├── device_failure_model.joblib
│   └── failure_threshold.joblib
│
├── tests/
│
├── app/
│   └── main.py
│
├── requirements.txt
├── Dockerfile
├── .gitignore
└── README.md
```

---

## 🛠️ Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Joblib
* Jupyter Notebook

---

## 🚀 Setup

Clone the repository:

```bash
git clone https://github.com/<your-username>/predictive-device-health.git
cd predictive-device-health
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run Prediction

From the project root:

```bash
python src/predict.py
```

---

## 🔭 Future Improvements

Potential future improvements include:

* Cross-validation based threshold selection
* SHAP-based model explainability
* FastAPI prediction service
* Docker deployment
* Model monitoring
* Automated retraining
* Real-world device telemetry validation
* Cloud deployment

---

## 📚 Learning Outcomes

This project demonstrates practical experience with:

* Exploratory Data Analysis
* Feature engineering
* Missing value handling
* Categorical encoding
* Feature scaling
* Train/test splitting
* Data leakage prevention
* Logistic Regression
* Random Forest
* Hyperparameter tuning
* Precision, Recall and F1 evaluation
* ROC-AUC
* Classification threshold optimization
* Scikit-learn pipelines

---
