#  Fraud Detection — Machine Learning Project

A machine learning project focused on detecting fraudulent financial transactions using XGBoost, with a business-driven approach that prioritizes minimizing undetected fraud over false alarms.

---

## Business Objective

Financial fraud causes billions in losses every year. The goal of this project is to build a classification model capable of identifying fraudulent transactions in real time, with a focus on **maximizing Recall**, because in a fraud detection context, missing a fraudulent transaction is far more costly than flagging a legitimate one for review.

---

##  Final Results

The final tuned XGBoost model outperformed the baseline across the most critical metric:

| Model | Precision | Recall | F1-Score | Accuracy |
|---|---|---|---|---|
| Baseline XGBoost | 0.94 | 0.92 | 0.93 | 0.99 |
| **Tuned XGBoost** | **0.87** | **0.98** | **0.92** | **0.99** |

**Key outcome:** The tuned model detects **98 out of every 100 fraudulent transactions**, reducing undetected fraud from 8 to just 2 cases per 100, a critical improvement for a real-world fraud prevention system.

> Hyperparameter optimization was performed using **Optuna** (Bayesian search, 100 trials) with Stratified K-Fold cross-validation.

---

##  Tech Stack

- **Language:** Python
- **Modeling:** XGBoost, Scikit-learn
- **Hyperparameter Tuning:** Optuna
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Model Persistence:** Joblib

---

##  Project Structure

```
fraud-detection/
│
├── raw data/                        # Original dataset
│
├── data/
│   ├── processed/                   # Cleaned dataset
│   ├── train dataset/               # X_train.csv, y_train.csv
│   └── test dataset/                # X_test.csv, y_test.csv
│
├── models/                          # Trained model artifacts (see note below)
│   ├── encoder.pkl
│   ├── scaler.pkl
│   ├── optuna_study.pkl
│   └── xgboost_final_model.pkl
│
└── notebooks/
    ├── 01_EDA_cleaning.ipynb        # Exploratory data analysis & data cleaning
    ├── 02_feature_eng_encoding.ipynb # Feature engineering & encoding pipeline
    ├── 03_baseline.ipynb            # Baseline model (Logistic Regression)
    ├── 04_XGBoost.ipynb             # XGBoost initial model
    └── 05_hyperparameter_tuning.ipynb  # Optuna tuning 
```

> **Note on data:** The `data/` folders are not included in this repository due to file size. The dataset used is a synthetic fraud detection dataset.

> **Note on models:** The `.pkl` files are not included in the repository. To generate them, run the notebooks in order — each notebook saves its artifacts automatically to the `models/` folder.

---

## How to Run This Project Locally

**1. Clone the repository**
```bash
git clone https://github.com/olgambencomo/fraud-detection.git
cd fraud-detection
```

**2. Install dependencies**
```bash
pip install pandas numpy xgboost scikit-learn optuna mlflow matplotlib seaborn joblib
```

**3. Run the notebooks in order**
```
01_EDA_cleaning.ipynb
02_feature_eng_encoding.ipynb
03_baseline.ipynb
04_XGBoost.ipynb
05_hyperparameter_tuning.ipynb
```

**4. Load the final model**
```python
import joblib
model = joblib.load('models/xgboost_final_model.pkl')
```

---

##  Author

**Olga Bencomo**  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/olgambencomo/)
