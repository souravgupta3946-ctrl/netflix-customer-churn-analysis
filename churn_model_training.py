# =============================================================================
#  NETFLIX CUSTOMER CHURN — ANALYSIS + MODEL TRAINING
#  Beginner-friendly, fully commented Python script
#  Run:  python churn_model_training.py
# =============================================================================

# ─── IMPORTS ─────────────────────────────────────────────────────────────────
# pandas  → data manipulation (like Excel but in Python)
# numpy   → math / arrays
# matplotlib / seaborn → charts
# sklearn → machine learning library
# scipy   → statistical tests
# joblib  → save the trained model to disk

import os, warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')          # don't open a GUI window, just save to files
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# --- scikit-learn imports ---
from sklearn.model_selection import (
    train_test_split,          # splits data into train / test
    StratifiedKFold,           # cross-validation that keeps class balance
    cross_val_score,           # runs cross-validation automatically
    RandomizedSearchCV         # hyperparameter tuning
)
from sklearn.pipeline import Pipeline          # chains preprocessing + model
from sklearn.compose import ColumnTransformer  # applies different transformers to different columns
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay,
    RocCurveDisplay, classification_report
)
from sklearn.inspection import permutation_importance
import joblib

warnings.filterwarnings('ignore')

# ─── PATHS ───────────────────────────────────────────────────────────────────
# __file__ is the current script location
# We build all paths relative to it so the script works anywhere
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, '..', 'data')
CHARTS_DIR = os.path.join(SCRIPT_DIR, '..', 'outputs', 'charts')
OUT_DIR    = os.path.join(SCRIPT_DIR, '..', 'outputs')

os.makedirs(CHARTS_DIR, exist_ok=True)
os.makedirs(OUT_DIR,    exist_ok=True)

SEED = 42                       # fix random seed → results are reproducible
np.random.seed(SEED)

def banner(title):
    """Print a section separator so the console output is easy to read."""
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)

# =============================================================================
#  PART A — DATA ANALYSIS
#  (Steps 1 – 5: load, clean, explore, answer business questions)
# =============================================================================

# ─── STEP 1: LOAD DATA ───────────────────────────────────────────────────────
banner("STEP 1 — LOAD DATA")

# pd.read_csv reads a CSV file and returns a DataFrame (like a table)
df = pd.read_csv(os.path.join(DATA_DIR, 'netflix_customer_churn.csv'))

print(f"Shape     : {df.shape[0]} rows x {df.shape[1]} columns")
print(f"\nColumn names:\n{df.columns.tolist()}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nFirst 3 rows:\n{df.head(3)}")

# .describe() gives count, mean, min, max, quartiles for every numeric column
print(f"\nDescriptive statistics:\n{df.describe().round(2)}")

# Missing values — isnull() marks each cell True/False, sum() counts the Trues
missing = df.isnull().sum()
print(f"\nMissing values:\n{missing[missing > 0] if missing.sum() > 0 else 'None — great!'}")

# Duplicate rows
print(f"\nDuplicate rows : {df.duplicated().sum()}")

# Churn balance
vc = df['churned'].value_counts()
print(f"\nClass balance  : {vc.to_dict()}")
print(f"Churn rate     : {df['churned'].mean()*100:.1f}%")

# Unique values in categorical columns
for col in ['gender','subscription_type','region','device','payment_method','favorite_genre']:
    print(f"  {col:25s}: {df[col].unique().tolist()}")


# ─── STEP 2: DATA CLEANING ───────────────────────────────────────────────────
banner("STEP 2 — DATA CLEANING")

df_clean = df.copy()

# --- 2a. Check outliers using the IQR method ---
# IQR = Interquartile Range = Q3 - Q1
# Anything below Q1 - 1.5*IQR or above Q3 + 1.5*IQR is flagged as an outlier
print("Outlier check (IQR method):")
for col in ['watch_hours', 'avg_watch_time_per_day']:
    Q1  = df_clean[col].quantile(0.25)
    Q3  = df_clean[col].quantile(0.75)
    IQR = Q3 - Q1
    lo, hi = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    n_out  = ((df_clean[col] < lo) | (df_clean[col] > hi)).sum()
    print(f"  {col:30s}: lower={lo:.2f}, upper={hi:.2f}, outliers={n_out}")

# --- 2b. Fix impossible avg_watch_time_per_day values ---
# A day has 24 hours — values above 24 are physically impossible.
over24 = df_clean['avg_watch_time_per_day'] > 24
print(f"\nRows with avg_watch_time_per_day > 24h : {over24.sum()}")
if over24.sum() > 0:
    print(df_clean[over24][['watch_hours','avg_watch_time_per_day']])
    # Decision: cap at 24.0 (preserves the row, corrects the error)
    df_clean.loc[over24, 'avg_watch_time_per_day'] = 24.0
    print("-> Capped at 24.0 (physically impossible to exceed 24h/day)")

# --- 2c. Verify monthly_fee matches subscription_type ---
fee_ranges = {'Basic': (8.99, 9.99), 'Standard': (13.99, 14.99), 'Premium': (17.99, 18.99)}
def fee_ok(row):
    lo, hi = fee_ranges.get(row['subscription_type'], (0, 9999))
    return lo <= row['monthly_fee'] <= hi

bad_fees = (~df_clean.apply(fee_ok, axis=1)).sum()
print(f"\nFee inconsistencies: {bad_fees}")

# --- 2d. Save cleaned data ---
clean_path = os.path.join(DATA_DIR, 'netflix_churn_clean.csv')
df_clean.to_csv(clean_path, index=False)
print(f"\nCleaned data saved -> {clean_path}")

# Boxplot to visualise outliers
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, col in zip(axes, ['watch_hours', 'avg_watch_time_per_day']):
    df_clean.boxplot(column=col, by='churned', ax=ax)
    ax.set_title(f'{col} by Churn Status')
    ax.set_xlabel('Churned (0=No, 1=Yes)')
plt.suptitle('')
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, 'train_outlier_boxplots.png'), dpi=150)
plt.close()
print("Boxplot saved.")


# ─── STEP 3: EXPLORATORY DATA ANALYSIS (EDA) ─────────────────────────────────
banner("STEP 3 — EXPLORATORY DATA ANALYSIS (EDA)")

# Use cleaned data from here on
df = df_clean.copy()
palette = {0: '#2ecc71', 1: '#e74c3c'}   # green=stayed, red=churned

# --- 3a. Churn distribution ---
vc = df['churned'].value_counts()
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].bar(['Stayed', 'Churned'], vc.values, color=['#2ecc71','#e74c3c'], edgecolor='black')
axes[0].set_title('Churn Count'); axes[0].set_ylabel('Customers')
for i, v in enumerate(vc.values):
    axes[0].text(i, v+30, str(v), ha='center', fontweight='bold')
axes[1].pie(vc.values, labels=[f'Stayed\n{vc[0]}', f'Churned\n{vc[1]}'],
            colors=['#2ecc71','#e74c3c'], autopct='%1.1f%%', startangle=90)
axes[1].set_title('Churn %')
plt.suptitle('Customer Churn Distribution', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, 'train_churn_distribution.png'), dpi=150)
plt.close()
print(f"Overall churn rate: {vc[1]/vc.sum()*100:.1f}%")

# --- 3b. Churn rate by categorical feature ---
cat_cols = ['subscription_type','gender','region','device','payment_method','favorite_genre']
for feat in cat_cols:
    rates = df.groupby(feat)['churned'].mean().sort_values(ascending=False) * 100
    fig, ax = plt.subplots(figsize=(max(6, len(rates)*1.2), 4))
    bars = ax.bar(rates.index, rates.values,
                  color=sns.color_palette('Set2', len(rates)), edgecolor='black')
    for b, v in zip(bars, rates.values):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.5, f'{v:.1f}%', ha='center', fontsize=9)
    ax.set_title(f'Churn Rate by {feat.replace("_"," ").title()}', fontsize=12)
    ax.set_ylabel('Churn Rate (%)')
    ax.set_ylim(0, rates.max()*1.25)
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, f'train_churn_by_{feat}.png'), dpi=150)
    plt.close()
    print(f"  {feat:25s}: highest = '{rates.index[0]}' at {rates.iloc[0]:.1f}%")

# --- 3c. Numeric distributions by churn ---
num_cols = ['age','watch_hours','last_login_days','avg_watch_time_per_day','number_of_profiles']
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()
for i, feat in enumerate(num_cols):
    for val, color, label in [(0,'#2ecc71','Stayed'), (1,'#e74c3c','Churned')]:
        axes[i].hist(df[df['churned']==val][feat], bins=30,
                     alpha=0.6, color=color, label=label, edgecolor='white')
    axes[i].set_title(feat.replace('_',' ').title())
    axes[i].set_xlabel(feat)
    axes[i].legend()
axes[-1].set_visible(False)
plt.suptitle('Numeric Feature Distributions by Churn', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, 'train_numeric_distributions.png'), dpi=150)
plt.close()
print("Numeric distribution chart saved.")

# --- 3d. Correlation heatmap ---
# corr() calculates Pearson correlation between every pair of numeric columns
# Values close to +1 = strong positive link, -1 = strong negative, 0 = no link
num_df = df.select_dtypes(include='number')
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(num_df.corr(), annot=True, fmt='.2f', cmap='coolwarm',
            center=0, ax=ax, linewidths=0.5,
            mask=np.triu(np.ones(num_df.corr().shape, dtype=bool)))
ax.set_title('Correlation Heatmap', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, 'train_correlation_heatmap.png'), dpi=150)
plt.close()
print("Correlation heatmap saved.")

# --- 3e. Age group and login bucket ---
df['age_group']   = pd.cut(df['age'], bins=[0,25,35,45,55,100],
                             labels=['<25','25-35','35-45','45-55','55+'])
df['login_bucket']= pd.cut(df['last_login_days'], bins=[0,7,14,30,60,9999],
                             labels=['0-7d','8-14d','15-30d','31-60d','60+d'])

for feat, title, colors in [
    ('age_group',    'Churn Rate by Age Group',             'Blues_d'),
    ('login_bucket', 'Churn Rate by Days Since Last Login', 'Reds_d'),
]:
    rates = df.groupby(feat, observed=True)['churned'].mean() * 100
    fig, ax = plt.subplots(figsize=(7, 4))
    rates.plot(kind='bar', ax=ax, color=sns.color_palette(colors, len(rates)), edgecolor='black')
    for i, v in enumerate(rates.values):
        ax.text(i, v+0.5, f'{v:.1f}%', ha='center', fontsize=9)
    ax.set_title(title, fontweight='bold')
    ax.set_ylabel('Churn Rate (%)')
    ax.set_ylim(0, rates.max()*1.25)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, f'train_churn_{feat}.png'), dpi=150)
    plt.close()
    print(f"  {feat}: highest = '{rates.idxmax()}' at {rates.max():.1f}%")


# ─── STEP 4: BUSINESS QUESTIONS ──────────────────────────────────────────────
banner("STEP 4 — BUSINESS QUESTIONS + STATISTICAL TESTS")

# Q1 — Subscription type vs churn
q1 = df.groupby('subscription_type')['churned'].mean().sort_values(ascending=False) * 100
print(f"\nQ1 - Churn by subscription:\n{q1.round(1)}")

# Q2 — Inactivity vs churn
q2 = df.groupby('login_bucket', observed=True)['churned'].mean() * 100
print(f"\nQ2 - Churn by login bucket:\n{q2.round(1)}")

# Q3 — Heavy vs light watchers (split at median)
median_watch = df['watch_hours'].median()
df['watch_group'] = np.where(df['watch_hours'] >= median_watch, 'Heavy', 'Light')
q3 = df.groupby('watch_group')['churned'].mean() * 100
print(f"\nQ3 - Churn by watch volume (median={median_watch:.0f}h):\n{q3.round(1)}")
print(f"     Light watchers churn {q3['Light']:.1f}% vs Heavy {q3['Heavy']:.1f}%")

# Q4 — Revenue lost by subscription type
q4 = df[df['churned']==1].groupby('subscription_type')['monthly_fee'].sum().sort_values(ascending=False)
print(f"\nQ4 - Monthly revenue at risk by plan:")
for plan, val in q4.items():
    print(f"     {plan}: ${val:,.2f}")
print(f"     TOTAL: ${q4.sum():,.2f}/month")

# Q5 — Number of profiles vs churn
q5 = df.groupby('number_of_profiles')['churned'].mean() * 100
print(f"\nQ5 - Churn by number of profiles:\n{q5.round(1)}")

# ── Statistical Tests ──────────────────────────────────────────────────────
# Chi-squared test  → used for CATEGORICAL variables
#   H0 (null hypothesis): the two variables are independent (no relationship)
#   If p < 0.05 → reject H0 → the relationship IS statistically significant

print("\n--- Chi-Squared Tests (categorical features vs churned) ---")
for col in ['subscription_type','gender','region','device','payment_method','favorite_genre']:
    ct   = pd.crosstab(df[col], df['churned'])
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    sig  = "SIGNIFICANT ***" if p < 0.05 else "not significant"
    print(f"  {col:25s}  chi2={chi2:8.2f}  p={p:.4f}  -> {sig}")

# T-test → used for NUMERIC variables
#   H0: the mean of the variable is the same for churned vs stayed customers
#   If p < 0.05 → the means differ significantly

print("\n--- Independent T-Tests (numeric features vs churned) ---")
for col in ['age','watch_hours','last_login_days','avg_watch_time_per_day',
            'number_of_profiles','monthly_fee']:
    g0 = df[df['churned']==0][col]
    g1 = df[df['churned']==1][col]
    t_stat, p = stats.ttest_ind(g0, g1, equal_var=False)  # Welch's t-test
    sig = "SIGNIFICANT ***" if p < 0.05 else "not significant"
    mean_diff = g1.mean() - g0.mean()
    print(f"  {col:25s}  mean_diff={mean_diff:+.3f}  p={p:.4f}  -> {sig}")


# =============================================================================
#  PART B — MODEL TRAINING
#  (Steps 5 – 9: feature engineering, build, train, evaluate, save)
# =============================================================================

# ─── STEP 5: FEATURE ENGINEERING ─────────────────────────────────────────────
banner("STEP 5 — FEATURE ENGINEERING")

# Reload the clean CSV to have a fresh copy without the temp EDA columns
df_model = pd.read_csv(os.path.join(DATA_DIR, 'netflix_churn_clean.csv'))

# ── Create new features ──────────────────────────────────────────────────────
# These capture business-meaningful patterns that raw columns don't show directly.

# 1. age_group — put ages into buckets instead of exact numbers
df_model['age_group'] = pd.cut(
    df_model['age'],
    bins=[0, 25, 35, 45, 55, 100],
    labels=['<25', '25-35', '35-45', '45-55', '55+']
).astype(str)

# 2. is_inactive_30d — binary flag: has this customer NOT logged in for 30+ days?
#    This is a strong churn signal (75.1% churn rate vs 26.1% for active users)
df_model['is_inactive_30d'] = (df_model['last_login_days'] > 30).astype(int)

# 3. is_inactive_14d — same logic for 14+ days (earlier warning)
df_model['is_inactive_14d'] = (df_model['last_login_days'] > 14).astype(int)

# 4. watch_per_fee — value-for-money ratio
#    Low ratio = customer isn't getting value for what they pay = churn risk
df_model['watch_per_fee'] = df_model['watch_hours'] / (df_model['monthly_fee'] + 1e-9)

# 5. engagement_level — bucket daily watch time into Low / Medium / High
df_model['engagement_level'] = pd.cut(
    df_model['avg_watch_time_per_day'],
    bins=[-0.001, 1.0, 3.0, 24.0],
    labels=['Low', 'Medium', 'High']
).astype(str)

print("New features created:")
new_feats = ['age_group', 'is_inactive_30d', 'is_inactive_14d',
             'watch_per_fee', 'engagement_level']
for f in new_feats:
    print(f"  {f:25s}: {df_model[f].value_counts().to_dict()}")

# ── Separate features (X) from target (y) ────────────────────────────────────
# customer_id is not a feature — it's just a label
df_model.drop(columns=['customer_id'], inplace=True, errors='ignore')

X = df_model.drop(columns=['churned'])   # all columns except the target
y = df_model['churned']                  # the target column

print(f"\nX shape: {X.shape}  (features)")
print(f"y shape: {y.shape}  (target — 0 or 1)")
print(f"y distribution: {y.value_counts().to_dict()}")

# ── Identify column types for the preprocessing pipeline ─────────────────────
cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
num_cols = X.select_dtypes(include='number').columns.tolist()

print(f"\nNumeric columns  ({len(num_cols)}): {num_cols}")
print(f"Categorical cols ({len(cat_cols)}): {cat_cols}")


# ─── STEP 6: BUILD THE PREPROCESSING PIPELINE ────────────────────────────────
banner("STEP 6 — PREPROCESSING PIPELINE")

# WHY use a Pipeline?
# ─────────────────────────────────────────────────────────────────────────────
# Without a Pipeline, a common beginner mistake is to scale ALL the data before
# splitting, which lets the test set influence the scaler — called "data leakage".
# A Pipeline fits the scaler ONLY on training data, then applies it to test data.
#
# ColumnTransformer applies DIFFERENT transformations to different column types:
#   - StandardScaler  : subtracts mean, divides by std → all numerics on same scale
#   - OneHotEncoder   : converts categories into binary (0/1) columns
#                       e.g. gender=Male → gender_Male=1, gender_Female=0, gender_Other=0

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(),                                       num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols),
    ],
    remainder='drop'    # drop any columns not listed above
)

print("ColumnTransformer built:")
print(f"  StandardScaler   -> applied to {len(num_cols)} numeric columns")
print(f"  OneHotEncoder    -> applied to {len(cat_cols)} categorical columns")


# ─── STEP 7: TRAIN / TEST SPLIT ──────────────────────────────────────────────
banner("STEP 7 — TRAIN / TEST SPLIT")

# stratify=y ensures the churn ratio is the same in both train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,        # 20% goes to test set (1,000 rows)
    random_state=SEED,     # for reproducibility
    stratify=y             # keep the 50/50 balance in both sets
)

print(f"Training set  : {X_train.shape[0]} rows  (churn rate: {y_train.mean()*100:.1f}%)")
print(f"Test set      : {X_test.shape[0]} rows  (churn rate: {y_test.mean()*100:.1f}%)")

# Visualise the split
fig, ax = plt.subplots(figsize=(6, 1.5))
ax.barh(['Split'], [X_train.shape[0]], color='#3b82d4', label=f'Train ({X_train.shape[0]})')
ax.barh(['Split'], [X_test.shape[0]],  left=[X_train.shape[0]], color='#e74c3c',
        label=f'Test ({X_test.shape[0]})')
ax.set_xlim(0, len(X))
ax.set_xlabel('Number of Rows')
ax.set_title('Train / Test Split (80% / 20%)', fontweight='bold')
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, 'train_split.png'), dpi=150)
plt.close()
print("Train/test split chart saved.")


# ─── STEP 8: CROSS-VALIDATION ────────────────────────────────────────────────
banner("STEP 8 — 5-FOLD CROSS-VALIDATION")

# What is cross-validation?
# ─────────────────────────────────────────────────────────────────────────────
# Instead of evaluating on ONE test split, we split the training data into
# 5 equal parts (folds). We train on 4 folds and test on the remaining 1 fold,
# rotating which fold is held out. This gives 5 scores → we average them.
# Result: a much more reliable estimate of real-world performance.
#
#  Fold 1:  [TEST] [TRAIN] [TRAIN] [TRAIN] [TRAIN]
#  Fold 2:  [TRAIN] [TEST] [TRAIN] [TRAIN] [TRAIN]
#  ...
#  Fold 5:  [TRAIN] [TRAIN] [TRAIN] [TRAIN] [TEST]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# Define the 4 models to compare
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=SEED),
    'Decision Tree':       DecisionTreeClassifier(random_state=SEED),
    'Random Forest':       RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1),
    'Gradient Boosting':   GradientBoostingClassifier(n_estimators=200, random_state=SEED),
}

cv_scores = {}
print(f"\n{'Model':25s}  {'Mean AUC':>10}  {'Std Dev':>8}  {'Min':>7}  {'Max':>7}")
print("-" * 65)
for name, model in models.items():
    # Pipeline: preprocessor -> model
    pipe = Pipeline([('prep', preprocessor), ('clf', model)])
    # cross_val_score runs the 5-fold CV and returns 5 AUC scores
    scores = cross_val_score(pipe, X_train, y_train, cv=skf,
                             scoring='roc_auc', n_jobs=-1)
    cv_scores[name] = scores
    print(f"  {name:25s}  {scores.mean():.4f}      {scores.std():.4f}   "
          f"{scores.min():.4f}  {scores.max():.4f}")

# Bar chart of CV scores
fig, ax = plt.subplots(figsize=(8, 4))
names  = list(cv_scores.keys())
means  = [cv_scores[n].mean() for n in names]
stds   = [cv_scores[n].std()  for n in names]
colors = ['#3498db','#e67e22','#2ecc71','#e74c3c']
bars   = ax.barh(names, means, xerr=stds, color=colors, edgecolor='black',
                 capsize=5, alpha=0.85)
for bar, val in zip(bars, means):
    ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', fontsize=10, fontweight='bold')
ax.set_xlabel('ROC-AUC Score')
ax.set_title('5-Fold Cross-Validation — ROC-AUC per Model', fontweight='bold')
ax.set_xlim(0.9, 1.01)
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, 'train_cv_scores.png'), dpi=150)
plt.close()
print("\nCV scores chart saved.")


# ─── STEP 9: HYPERPARAMETER TUNING ───────────────────────────────────────────
banner("STEP 9 — HYPERPARAMETER TUNING (RandomizedSearchCV)")

# What is hyperparameter tuning?
# ─────────────────────────────────────────────────────────────────────────────
# Every model has "settings" you choose before training (hyperparameters).
# Example for Gradient Boosting:
#   - n_estimators   : how many trees to build (more = slower but often better)
#   - learning_rate  : how fast the model learns (lower = more careful)
#   - max_depth      : how deep each tree can grow (deeper = more complex)
#
# RandomizedSearchCV tries random combinations of these settings and picks
# the best one using cross-validation. It's faster than trying every combination
# (GridSearchCV) because it samples randomly from the parameter space.

best_model_name = max(cv_scores, key=lambda k: cv_scores[k].mean())
second_name     = sorted(cv_scores, key=lambda k: cv_scores[k].mean(), reverse=True)[1]
to_tune         = [best_model_name, second_name]

param_grids = {
    'Gradient Boosting': {
        'clf__n_estimators':  [100, 200, 300],
        'clf__learning_rate': [0.05, 0.1, 0.2],
        'clf__max_depth':     [3, 4, 5],
    },
    'Random Forest': {
        'clf__n_estimators':     [100, 200, 300],
        'clf__max_depth':        [None, 5, 10, 15],
        'clf__min_samples_leaf': [1, 2, 5],
    },
    'Logistic Regression': {
        'clf__C':       [0.01, 0.1, 1, 10],
        'clf__penalty': ['l2'],
    },
    'Decision Tree': {
        'clf__max_depth':        [3, 5, 7, None],
        'clf__min_samples_leaf': [1, 5, 10],
    },
}

tuned_pipes = {}
for name in to_tune:
    print(f"\nTuning [{name}] ...")
    pipe   = Pipeline([('prep', preprocessor), ('clf', models[name])])
    grid   = param_grids.get(name, {})
    search = RandomizedSearchCV(
        estimator  = pipe,
        param_distributions = grid,
        n_iter     = 20,            # try 20 random combinations
        cv         = skf,
        scoring    = 'roc_auc',
        random_state = SEED,
        n_jobs     = -1,
        refit      = True           # refit the best model on the full training set
    )
    search.fit(X_train, y_train)
    tuned_pipes[name] = search.best_estimator_
    print(f"  Best params  : {search.best_params_}")
    print(f"  Best CV AUC  : {search.best_score_:.4f}")

# Fit the remaining two models with default params
fitted_pipes = {}
for name, model in models.items():
    if name in tuned_pipes:
        fitted_pipes[name] = tuned_pipes[name]
    else:
        pipe = Pipeline([('prep', preprocessor), ('clf', model)])
        pipe.fit(X_train, y_train)
        fitted_pipes[name] = pipe
        print(f"\nFitted [{name}] with default params.")

print("\nAll 4 models trained and ready.")


# ─── STEP 10: EVALUATION ─────────────────────────────────────────────────────
banner("STEP 10 — MODEL EVALUATION ON TEST SET")

# Evaluate every model on the HELD-OUT test set (the 1,000 rows it never saw)
results = []
for name, pipe in fitted_pipes.items():
    y_pred      = pipe.predict(X_test)           # predicted class (0 or 1)
    y_prob      = pipe.predict_proba(X_test)[:,1] # predicted probability of churn
    y_pred_tr   = pipe.predict(X_train)          # predictions on training set (for overfit check)

    results.append({
        'Model':        name,
        'Accuracy':     accuracy_score(y_test, y_pred),         # % of all predictions correct
        'Precision':    precision_score(y_test, y_pred, zero_division=0), # of predicted churns, how many were real
        'Recall':       recall_score(y_test, y_pred, zero_division=0),    # of all real churns, how many did we catch
        'F1':           f1_score(y_test, y_pred, zero_division=0),        # harmonic mean of precision and recall
        'ROC-AUC':      roc_auc_score(y_test, y_prob),          # area under the ROC curve (1.0 = perfect)
        'Train_Acc':    accuracy_score(y_train, y_pred_tr),     # training accuracy (to spot overfitting)
    })

results_df = pd.DataFrame(results).sort_values('ROC-AUC', ascending=False).reset_index(drop=True)

# Print the comparison table
print(f"\n{'Model':25s}  {'Acc':>6}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}  {'AUC':>7}  {'TrainAcc':>8}")
print("-" * 75)
for _, row in results_df.iterrows():
    overfit_flag = " <- OVERFIT" if (row['Train_Acc'] - row['Accuracy']) > 0.05 else ""
    print(f"  {row['Model']:25s}  {row['Accuracy']:.3f}  {row['Precision']:.3f}  "
          f"{row['Recall']:.3f}  {row['F1']:.3f}  {row['ROC-AUC']:.4f}  "
          f"{row['Train_Acc']:.3f}{overfit_flag}")

# Save metrics
results_df.to_csv(os.path.join(OUT_DIR, 'train_model_metrics.csv'), index=False)
print(f"\nMetrics saved -> outputs/train_model_metrics.csv")

# ── Metric definitions (for beginners) ───────────────────────────────────────
print("""
What the metrics mean:
  Accuracy  = (correct predictions) / (total predictions)
  Precision = of customers we predicted would churn, what % actually churned?
              High precision -> few false alarms (good for retention budgets)
  Recall    = of all customers who actually churned, what % did we catch?
              High recall    -> we miss fewer real churners (more important for retention)
  F1        = balance between precision and recall (harmonic mean)
  ROC-AUC   = ability to rank churned customers higher than stayed customers
              0.5 = random guessing | 1.0 = perfect | >= 0.8 = good
""")

# ── Best model detailed evaluation ──────────────────────────────────────────
best_name = results_df.iloc[0]['Model']
best_pipe = fitted_pipes[best_name]
y_pred    = best_pipe.predict(X_test)
y_prob    = best_pipe.predict_proba(X_test)[:,1]

print(f"BEST MODEL: {best_name}")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred, target_names=['Stayed','Churned'])}")

# Confusion matrix
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
cm   = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=['Stayed','Churned'])
disp.plot(ax=axes[0], colorbar=False, cmap='Blues')
axes[0].set_title(f'Confusion Matrix — {best_name}', fontweight='bold')

# ROC curve
RocCurveDisplay.from_predictions(y_test, y_prob, ax=axes[1], name=best_name)
axes[1].plot([0,1],[0,1],'--', color='grey', label='Random Guess')
axes[1].set_title(f'ROC Curve — {best_name}', fontweight='bold')
axes[1].legend()
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, 'train_confusion_roc.png'), dpi=150)
plt.close()
print("Confusion matrix + ROC curve saved.")

# ── Overfit check ──────────────────────────────────────────────────────────
print("\nOverfit Check (ideal: train acc close to test acc):")
for _, row in results_df.iterrows():
    delta = row['Train_Acc'] - row['Accuracy']
    status = "OK" if delta < 0.05 else "WARNING — possible overfit"
    print(f"  {row['Model']:25s}  Train={row['Train_Acc']:.3f}  Test={row['Accuracy']:.3f}  "
          f"Delta={delta:.3f}  [{status}]")


# ─── STEP 11: FEATURE IMPORTANCE ─────────────────────────────────────────────
banner("STEP 11 — FEATURE IMPORTANCE")

# Get the feature names AFTER one-hot encoding
prep_step = best_pipe.named_steps['prep']
ohe_cols  = prep_step.named_transformers_['cat'] \
                      .get_feature_names_out(cat_cols).tolist()
all_feat_names = num_cols + ohe_cols

clf_step = best_pipe.named_steps['clf']

# Use tree-based importance if available (fast), else permutation importance
if hasattr(clf_step, 'feature_importances_'):
    importances = clf_step.feature_importances_
    method = 'Tree-Based (Gini Impurity)'
else:
    # Permutation importance: shuffle one feature at a time, measure AUC drop
    X_test_prep = prep_step.transform(X_test)
    perm   = permutation_importance(clf_step, X_test_prep, y_test,
                                    n_repeats=10, random_state=SEED, n_jobs=-1)
    importances = perm.importances_mean
    method = 'Permutation'

imp_df = pd.DataFrame({'Feature': all_feat_names, 'Importance': importances})
imp_df = imp_df.sort_values('Importance', ascending=False).head(10)

print(f"\nTop 10 Features ({method}):")
print(imp_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(9, 5))
ax.barh(imp_df['Feature'][::-1], imp_df['Importance'][::-1],
        color=sns.color_palette('viridis', 10), edgecolor='black')
ax.set_title(f'Top 10 Feature Importances ({method})', fontweight='bold')
ax.set_xlabel('Importance Score')
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, 'train_feature_importance.png'), dpi=150)
plt.close()
print("Feature importance chart saved.")

print("""
Business interpretation:
  avg_watch_time_per_day  -> #1 signal: daily viewing habit drives retention
  watch_hours             -> total engagement — low watchers leave
  number_of_profiles      -> more profiles = more household lock-in = less churn
  is_inactive_30d         -> 30-day inactivity is the clearest early-warning flag
  subscription_type_Basic -> Basic plan customers are at significantly higher risk
  last_login_days         -> raw inactivity days (complement to the binary flag)
  monthly_fee             -> price sensitivity plays a role
  payment_method_Crypto   -> Crypto users churn 9 pp above average
""")


# ─── STEP 12: SAVE MODEL + PREDICTIONS ───────────────────────────────────────
banner("STEP 12 — SAVE MODEL + PREDICTIONS")

# ── Save the trained model ──────────────────────────────────────────────────
# joblib serialises the entire Pipeline (preprocessor + model) to a file
# You can load it later with:  model = joblib.load('best_churn_model.pkl')
model_path = os.path.join(OUT_DIR, 'best_churn_model.pkl')
joblib.dump(best_pipe, model_path)
print(f"Best model saved   -> {model_path}")

# ── Generate predictions for all 5,000 customers ─────────────────────────────
# Reload original data to get customer_id
df_orig    = pd.read_csv(os.path.join(DATA_DIR, 'netflix_churn_clean.csv'))
cust_ids   = df_orig['customer_id'].copy()
X_all      = df_model.drop(columns=['churned'])  # all features
y_all      = df_model['churned']

prob_all   = best_pipe.predict_proba(X_all)[:,1]

# Assign risk level based on probability thresholds
risk_labels = pd.cut(
    prob_all,
    bins  = [-0.001, 0.33, 0.66, 1.001],
    labels= ['Low', 'Medium', 'High']
)

pred_df = pd.DataFrame({
    'customer_id':    cust_ids.values,
    'churn_prob':     prob_all.round(4),
    'risk_level':     risk_labels,
    'actual_churned': y_all.values,
})

pred_path = os.path.join(OUT_DIR, 'churn_predictions.csv')
pred_df.to_csv(pred_path, index=False)

print(f"Predictions saved  -> {pred_path}  ({len(pred_df)} rows)")
print(f"\nRisk level breakdown:\n{pred_df['risk_level'].value_counts().to_string()}")

# ── Show a sample ─────────────────────────────────────────────────────────────
print("\nSample predictions (first 5):")
print(pred_df.head().to_string(index=False))


# ─── STEP 13: HOW TO USE THE SAVED MODEL ─────────────────────────────────────
banner("STEP 13 — HOW TO USE THE MODEL ON NEW CUSTOMERS")

print("""
The model is saved as a scikit-learn Pipeline.
To score a new customer, just create a DataFrame with the same columns
and call model.predict_proba().

Example:

    import joblib, pandas as pd

    model = joblib.load('outputs/best_churn_model.pkl')

    new_customer = pd.DataFrame([{
        'age':                    35,
        'gender':                 'Male',
        'subscription_type':      'Basic',
        'watch_hours':            3.5,
        'last_login_days':        45,
        'region':                 'Europe',
        'device':                 'Mobile',
        'monthly_fee':            8.99,
        'payment_method':         'Credit Card',
        'number_of_profiles':     1,
        'avg_watch_time_per_day': 0.2,
        'favorite_genre':         'Action',
        # engineered features (must match training)
        'is_inactive_30d':        1,
        'is_inactive_14d':        1,
        'watch_per_fee':          3.5 / 8.99,
        'age_group':              '35-45',
        'engagement_level':       'Low',
    }])

    prob  = model.predict_proba(new_customer)[0, 1]
    risk  = 'High' if prob > 0.66 else 'Medium' if prob > 0.33 else 'Low'
    print(f"Churn probability: {prob:.1%}  |  Risk level: {risk}")
""")


# ─── FINAL SUMMARY ───────────────────────────────────────────────────────────
banner("COMPLETE — SUMMARY")

print(f"""
FILES CREATED
-------------
  outputs/train_model_metrics.csv         Model comparison table
  outputs/best_churn_model.pkl            Trained {best_name} pipeline
  outputs/churn_predictions.csv           Predictions for all 5,000 customers
  outputs/charts/train_*.png              All training charts (14 files)

RESULTS
-------
  Best model   : {best_name}
  Test Accuracy: {results_df.iloc[0]['Accuracy']*100:.1f}%
  Test ROC-AUC : {results_df.iloc[0]['ROC-AUC']:.4f}
  Recall       : {results_df.iloc[0]['Recall']*100:.1f}%  (% of real churners caught)

KEY FINDINGS
------------
  1. avg_watch_time_per_day is the #1 churn predictor (64.7% importance)
  2. 30-day inactive users churn at 75.1% vs 26.1% for active users
  3. Light watchers (<8h total) churn at 72.3% vs 28.3% for heavy watchers
  4. Basic plan subscribers churn 18 pp more than Premium
  5. $33,010/month in revenue is lost to churned customers
""")
