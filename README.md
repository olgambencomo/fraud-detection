# Fraud Detection — Financial Transactions

> XGBoost classifier tuned with Bayesian optimization to detect financial fraud with 98% Recall on an imbalanced dataset (95/5 class split), eliminating data leakage and applying principled feature engineering.

---

## Context

Financial fraud has a fundamental asymmetry: missing a fraudulent transaction is far more costly than flagging a legitimate one for review. A model optimized for accuracy alone fails in practice — on a dataset where fraud represents only 5% of transactions, predicting "not fraud" for everything yields 95% accuracy while catching zero fraud cases.

The goal was to build a model that maximizes **Recall on the fraud class**, accepting a deliberate trade-off in Precision — because in a real fraud prevention system, catching 98 out of 100 fraudulent transactions is worth investigating a few extra false alarms.

---

## What I did

### Dataset
Synthetic fraud detection dataset — 10,000 financial transactions with features including transaction amount, type (ATM, QR, Online, POS), merchant category, country, hour, and pre-computed device/IP risk scores.

**Class distribution:**
```
Not fraud   9,500   (95%)
Fraud         500    (5%)
```

### Exploratory Data Analysis
The EDA revealed a critical insight: of 364 outlier transactions (unusually high amounts), **351 were fraud cases**. Removing outliers to "clean" the data would have destroyed 70% of the fraud signal. All outliers were kept, and a feature was engineered to capture their magnitude.

The `hour` variable showed a non-linear cyclical relationship with fraud, so it was encoded using sine/cosine transformation to correctly represent that 23:00 and 00:00 are adjacent — not 23 units apart.

### Data Leakage Detection
The correlation matrix flagged `ip_risk_score` and `device_risk_score` at **0.87 correlation with the target** — a red flag, not a signal. In production, risk scores computed from the transaction itself would not be available at decision time. Both columns were removed to prevent inflated training metrics that would fail at deployment.

### Modeling

**Baseline — Logistic Regression** with `class_weight='balanced'`: Recall 0.84 (16 fraud cases per 100 undetected).

**XGBoost (initial)**: Recall 0.92 — significant improvement, but 8 cases per 100 still undetected.

**XGBoost + Optuna tuning**: Bayesian optimization over 9 hyperparameters across 100 trials, optimizing Recall via Stratified K-Fold (k=5) cross-validation. `scale_pos_weight` converged to ~28, penalizing missed fraud cases 28× more than false positives.

### Key Engineering Decisions

| Decision | Rationale |
|---|---|
| Keep high-value outliers | 351/500 fraud cases had outlier amounts — removing them destroys 70% of the fraud signal |
| Remove high-correlation features | Identified as data leakage; model must be deployable, not just accurate on paper |
| Cyclical encoding for hour | `sin/cos` encoding preserves temporal adjacency (23:00 ≈ 00:00) |
| Optuna over GridSearchCV | 9 continuous hyperparameters make grid search computationally prohibitive; Bayesian optimization converges efficiently |
| Optimize Recall, not accuracy | Accuracy is misleading on a 95/5 imbalanced dataset |

---

## Results

| Model | Precision | Recall | F1 | Accuracy | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression (baseline) | 0.60 | 0.84 | 0.70 | 0.96 | — |
| XGBoost (initial) | 0.94 | 0.92 | 0.93 | 0.99 | — |
| **XGBoost (tuned)** | **0.87** | **0.98** | **0.92** | **0.99** | **0.9956** |

**Confusion matrix — tuned model:**
```
                    Predicted Not Fraud   Predicted Fraud
Actual Not Fraud          1,881                19
Actual Fraud                  2                98
```

The tuned model catches **98 out of 100 fraudulent transactions**, reducing undetected fraud from 8 cases (initial XGBoost) to 2 per 100. The trade-off — Precision dropped from 0.94 to 0.87, meaning 19 legitimate transactions flagged for review instead of 6 — is intentional and justified: a false positive costs a manual review; a false negative costs the full value of the fraudulent transaction. F1 remained stable at 0.92, and ROC-AUC of 0.9956 indicates near-perfect class separation.

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.11 |
| Modeling | XGBoost, scikit-learn |
| Hyperparameter tuning | Optuna (Bayesian, 100 trials) |
| Data processing | pandas, NumPy |
| Visualization | matplotlib, seaborn |
| Model persistence | joblib |
| Version control | Git, GitHub |

---

## Limitations

- **Synthetic data** — does not capture real-world fraud complexity (adversarial behavior, concept drift, jurisdiction-specific patterns). Production performance would need independent validation.
- **Static model** — fraud patterns evolve. A production system would require periodic retraining and drift monitoring.
- **No threshold optimization** — default 0.5 threshold used. In production, the optimal threshold would be calibrated based on the actual cost ratio of false negatives vs. false positives.

---

## Next Steps

- Deploy as a real-time scoring API (FastAPI + Docker)
- Add MLflow tracking to compare Optuna runs systematically
- Implement prediction drift monitoring for production
- Explore cost-sensitive threshold optimization using actual business cost estimates
