# Customer Churn Prediction System

An end-to-end Machine Learning pipeline to predict customer churn in telecommunications using supervised classification algorithms.

## Project Overview
Customer churn happens when customers stop doing business with a company. Identifying churn risks early allows retention teams to intervene proactively. This project loads, cleans, preprocesses, and trains multiple classification models to benchmark and identify the top predictor.

## Dataset
- **Source:** Kaggle Telco Customer Churn Dataset (IBM)
- **Target:** `Churn` (Yes = 1, No = 0)
- **Features:** Demographics, account information, and subscribed services.

## Project Structure
```text
customer-churn-prediction/
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── models/
│   └── best_churn_model.pkl
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── eda.py
│   ├── preprocess.py
│   ├── train.py
│   └── evaluate.py
├── .gitignore
├── requirements.txt
├── main.py
└── README.md

