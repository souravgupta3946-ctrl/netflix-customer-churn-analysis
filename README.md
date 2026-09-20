# Netflix Customer Churn Analysis

A beginner-friendly, end-to-end churn analysis and machine learning project built with Python.

---

## Dataset Source

> **Kaggle Dataset:**
> [Netflix Customer Churn and Engagement Analytics](https://www.kaggle.com/datasets/zeyadmohamed26/netflix-customer-churn-and-engagement-analytics)
> by Zeyad Mohamed
>
> The dataset contains **5,000 customer records** with 14 features including demographics,
> subscription details, watch behaviour, and a churn label (churned = 1, stayed = 0).

---

## Running Ports

| Service | Port | URL | Description |
|---------|------|-----|-------------|
| **Streamlit Dashboard (Frontend)** | `8501` | http://localhost:8501 | Interactive 7-page web dashboard |
| **Streamlit Network Access** | `8501` | http://192.168.0.122:8501 | Access from other devices on same network |

> The project is a pure Python / Streamlit application.
> There is no separate backend server — the Streamlit process handles both the UI and all data/model logic internally.

---

## Project Structure

```
netflix_churn/
├── data/
│   ├── netflix_customer_churn.csv          # Raw dataset (from Kaggle)
│   └── netflix_churn_clean.csv             # Cleaned dataset (10 values capped)
├── notebooks/
│   ├── churn_analysis.py                   # Original full analysis (11 steps)
│   └── churn_model_training.py             # Analysis + Model training (13 steps)
├── outputs/
│   ├── charts/                             # All PNG charts (21+ files)
│   ├── best_churn_model.pkl                # Trained Gradient Boosting model
│   ├── churn_predictions.csv               # Per-customer churn probability + risk level
│   ├── model_metrics.csv                   # Model comparison table
│   └── train_model_metrics.csv             # Metrics from training script
├── report/
│   ├── Netflix_Churn_Report.md             # Full written report (Markdown)
│   ├── Netflix_Churn_Report.docx           # Full report with charts (Word)
│   └── Netflix_Churn_Visual_Report.html    # Self-contained HTML report
├── app.py                                  # Streamlit dashboard (7 pages)
├── run_dashboard.bat                       # Double-click to launch dashboard (Windows)
├── build_report.py                         # Script to regenerate HTML report
├── build_word_report.py                    # Script to regenerate Word report
├── requirements.txt                        # Python dependencies
└── README.md                               # This file
```

---

## How to Run

### Step 1 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 2 — Run the Analysis + Model Training

From the `netflix_churn/notebooks/` folder:

```bash
python churn_model_training.py
```

This single script does **everything** in 13 steps:

| Part | Steps | What happens |
|------|-------|-------------|
| **Analysis** | 1–4 | Load data, clean, EDA charts, business questions + statistical tests |
| **Model Training** | 5–13 | Feature engineering, pipeline, train/test split, cross-validation, hyperparameter tuning, evaluation, save model + predictions |

**Outputs produced:**
- 12+ EDA charts saved to `outputs/charts/`
- `outputs/best_churn_model.pkl` — trained Gradient Boosting pipeline
- `outputs/churn_predictions.csv` — churn probability + risk level for all 5,000 customers
- `outputs/train_model_metrics.csv` — comparison table for all 4 models

### Step 3 — Launch the Interactive Dashboard

**Option A — Double-click (Windows):**
```
Double-click  run_dashboard.bat
```

**Option B — Command line:**
```bash
# From the netflix_churn/ folder
streamlit run app.py --server.port 8501
```

Then open your browser at:

```
http://localhost:8501
```

> The dashboard loads the freshly trained model and predictions automatically.

---

## Dashboard Pages (http://localhost:8501)

| Page | What you'll see |
|------|----------------|
| 🏠 **Overview** | 5 KPI cards, churn distribution, feature importance, quick facts |
| 📊 **EDA Charts** | All 15+ charts in 4 tabs with plain-English insights |
| ❓ **Business Questions** | 8 Q&A with charts, answers, and significance badges |
| 🤖 **Model Results** | Metrics table, confusion matrix, ROC curve, overfitting check |
| 🔍 **Customer Risk Lookup** | Filter all 5,000 customers, search by ID, download CSV |
| 🔮 **Predict New Customer** | Live form → real-time churn prediction from the trained model |
| 📄 **Full Report** | Rendered Markdown report + download button |

---

## Key Results

| Metric | Value |
|--------|-------|
| Dataset size | 5,000 customers, 14 features |
| Overall churn rate | 50.3% |
| Best model | Gradient Boosting |
| Test Accuracy | 99.2% |
| ROC-AUC | 0.9989 |
| Recall (churned) | 98.8% |
| Revenue at risk / month | $33,010 |
| High-risk customers | 608 (12.2% of base) |

### Top Churn Drivers

1. `avg_watch_time_per_day` — daily viewing habit (63.8% feature importance)
2. `watch_hours` — total engagement
3. `number_of_profiles` — more profiles = more household lock-in
4. `is_inactive_30d` — 30-day inactivity flag (75.1% churn rate)
5. `subscription_type = Basic` — 61.8% churn rate vs 43.7% for Premium

---

## Use the Saved Model on New Customers

```python
import joblib, pandas as pd

# Load the trained pipeline (preprocessor + model)
model = joblib.load('outputs/best_churn_model.pkl')

# Create a new customer record (must include engineered features)
new_customer = pd.DataFrame([{
    'age': 35, 'gender': 'Male', 'subscription_type': 'Basic',
    'watch_hours': 3.5, 'last_login_days': 45, 'region': 'Europe',
    'device': 'Mobile', 'monthly_fee': 8.99, 'payment_method': 'Credit Card',
    'number_of_profiles': 1, 'avg_watch_time_per_day': 0.2,
    'favorite_genre': 'Action',
    # Engineered features
    'is_inactive_30d': 1, 'is_inactive_14d': 1,
    'watch_per_fee': 3.5 / 8.99, 'age_group': '35-45', 'engagement_level': 'Low',
}])

prob = model.predict_proba(new_customer)[0, 1]
risk = 'High' if prob > 0.66 else 'Medium' if prob > 0.33 else 'Low'
print(f"Churn probability: {prob:.1%}  |  Risk level: {risk}")
```

---

## Requirements

- Python 3.9+
- See `requirements.txt` for exact package versions

```
pandas, numpy, matplotlib, seaborn, scikit-learn, joblib, scipy, streamlit, pillow
```

---

## Project Info

| Item | Detail |
|------|--------|
| Dataset | [Kaggle — Netflix Customer Churn and Engagement Analytics](https://www.kaggle.com/datasets/zeyadmohamed26/netflix-customer-churn-and-engagement-analytics) |
| Framework | IBM Skills Build Project |
| Language | Python 3.13 |
| random_state | 42 (all models and splits) |
| Dashboard port | 8501 |
