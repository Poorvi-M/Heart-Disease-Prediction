# Heart Disease Prediction

A machine learning project that predicts the presence of heart disease in patients and assigns a clinical risk severity grade to positive cases.

---

## Project Overview

Heart disease is one of the leading causes of mortality worldwide. Early and accurate detection is critical — a missed diagnosis (false negative) can be life-threatening. This project builds a robust binary classification pipeline to predict heart disease presence, optimised for **recall** to minimise false negatives, and extends predictions with a **heuristic severity grading system** for patients diagnosed positive.

---

## Dataset

- **Source:** [Heart Disease Dataset – Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)
- **Size:** 1,025 rows × 14 columns
- **Target:** Binary — `1` = Heart Disease Present, `0` = No Heart Disease
- **Class Balance:** ~50/50 split — no class imbalance handling required

### Feature Summary

| Feature | Description | Type |
|---|---|---|
| `age` | Patient age in years | Continuous |
| `sex` | Sex (0 = Female, 1 = Male) | Binary |
| `cp` | Chest pain type (0–3) | Categorical |
| `trestbps` | Resting blood pressure (mm Hg) | Continuous |
| `chol` | Serum cholesterol (mg/dl) | Continuous |
| `fbs` | Fasting blood sugar > 120 mg/dl | Binary |
| `restecg` | Resting ECG results (0–2) | Categorical |
| `thalach` | Maximum heart rate achieved | Continuous |
| `exang` | Exercise induced angina (0/1) | Binary |
| `oldpeak` | ST depression induced by exercise | Continuous |
| `slope` | Slope of peak exercise ST segment | Ordinal |
| `ca` | Number of major vessels coloured by fluoroscopy (0–4) | Ordinal |
| `thal` | Thalassemia test result (0–2) | Categorical |

---

## Data Exploration & Cleaning

- Confirmed **no missing values** across all 14 columns
- Verified **no biologically impossible values** (e.g. zero cholesterol — minimum was 126 mg/dl)
- Plotted histograms for all features — identified `oldpeak` as heavily right-skewed
- Confirmed balanced target distribution — no need for SMOTE or class weighting
- Identified columns that appear numerical but are categorically encoded (`cp`, `restecg`, `thal`)

---

## Preprocessing Pipeline

A `sklearn` **Pipeline** with a **ColumnTransformer** was used to apply different transformations to different column types.

```
Pipeline
├── ColumnTransformer
│   ├── StandardScaler   → age, trestbps, chol, thalach, oldpeak
│   ├── OneHotEncoder    → cp, restecg, thal
│   └── Passthrough      → sex, fbs, exang, slope, ca
└── Model
```

### Preprocessing Decisions

| Column Group | Treatment | Justification |
|---|---|---|
| `age`, `trestbps`, `chol`, `thalach`, `oldpeak` | StandardScaler | Continuous features with varying ranges — prevents scale bias in distance-based models |
| `cp`, `restecg`, `thal` | OneHotEncoder | Multi-category features with no natural order — avoids implying false mathematical relationships |
| `sex`, `fbs`, `exang` | Passthrough | Already binary (0/1) — no encoding needed |
| `slope`, `ca` | Passthrough | Ordinal with meaningful numeric order — numbers already represent severity correctly |

### Why a Pipeline?

Using a Pipeline **prevents data leakage**. The scaler is fit only on training data and applied to test data — never the reverse. This ensures the test set genuinely simulates unseen future patients.

`handle_unknown="ignore"` was added to `OneHotEncoder` to handle rare categories that may appear in validation folds during cross validation but were absent in the training fold.

---

## Model Selection

### Step 1 — Cross Validation Screening

Five models were evaluated using 5-fold cross validation on the training set, scored by **recall**.

**Why recall?** A false negative — predicting no disease when disease is present — means a patient goes home untreated. This is clinically far more dangerous than a false positive. Recall directly measures how many true positive cases are caught.

| Model | CV Recall | Std Dev |
|---|---|---|
| Logistic Regression | 0.8983 | ±0.0178 |
| Random Forest | 0.9859 | ±0.0228 |
| XGBoost | 0.9859 | ±0.0228 |
| SVM | 0.9363 | ±0.0263 |
| Gradient Boosting | 0.9741 | ±0.0282 |

Top 3 models — Random Forest, XGBoost, Gradient Boosting — were selected for hyperparameter tuning.

### Step 2 — GridSearchCV Tuning

`GridSearchCV` with `cv=5` was run on the top 3 models, tracking both **recall** and **precision**, with `refit='recall'` to select the best model based on recall.

| Model | Best CV Recall | Best Parameters |
|---|---|---|
| Random Forest | **0.9906** | `max_depth=10, min_samples_split=2, n_estimators=100` |
| XGBoost | 0.9859 | `learning_rate=0.1, max_depth=5, n_estimators=200` |
| Logistic Regression | 0.9031 | `C=0.1, penalty=l1, solver=saga` |

**Winner: Random Forest Classifier**

### Cross Validation vs GridSearchCV

| | Cross Validation | GridSearchCV |
|---|---|---|
| Purpose | Evaluate a fixed model | Find best hyperparameters |
| What varies | Data splits | Hyperparameter combinations |
| Output | Generalisation estimate | Best parameter set |
| Speed | Fast | Slower (CV × combinations) |

GridSearchCV uses cross validation **internally** as its evaluation mechanism — it runs CV for every hyperparameter combination and picks the best.

---

## Test Set Evaluation

The best estimator (full pipeline — preprocessor + Random Forest) was evaluated on the held-out test set:

| Metric | Value |
|---|---|
| Accuracy | 99% |
| Recall (Heart Disease) | 0.97 |
| Precision (Heart Disease) | 1.00 |
| CV Recall vs Test Recall gap | ~2% — no overfitting |

---

## Severity Grading System

For patients predicted to have heart disease, a **heuristic risk score** is computed using four clinically relevant features:

| Feature | Weight | Scoring Method | Rationale |
|---|---|---|---|
| `ca` | 40 pts | `(ca / 4) × 40` | Direct measure of vessel blockage — most indicative of severity |
| `oldpeak` | 30 pts | `(oldpeak / 6.2) × 30` | ST depression reflects cardiac stress under load |
| `exang` | 20 pts | `exang × 20` | Exercise-induced angina is a strong warning sign |
| `thal` | 10 pts | 0→0pts, 2→7pts, 1→10pts | Fixed defect (permanent damage) scored highest |

**Total score range: 0–100**

| Score Range | Severity |
|---|---|
| 0–15 | Mild |
| 16–31 | Moderate |
| 32–47 | High |

*Note: Bin boundaries are derived from the actual score distribution in the test set (max observed score: 47). Scores are proportional and continuous — higher always means higher risk.*

### Severity Distribution (Test Set)

| Severity | Mean Risk Score |
|---|---|
| Mild | 7.94 |
| Moderate | 24.55 |
| Severe | 39.92 |

> **Disclaimer:** This severity score is a heuristic indicator based on clinically relevant features. It is **not a medical diagnosis** and would require cardiologist validation before use in any real clinical setting. In production medical ML, domain experts define severity criteria — data scientists implement them.

---

## Project Structure

```
Heart-Disease-Prediction/
│
├── Heart_Disease_Prediction.ipynb   # Main notebook
├── README.md                        # This file
```

---

## Tech Stack

- **Python 3.12**
- **pandas** — data manipulation
- **numpy** — numerical operations
- **matplotlib** — visualisation
- **scikit-learn** — Pipeline, ColumnTransformer, GridSearchCV, models
- **xgboost** — XGBoost classifier
- **ydata-profiling** — automated EDA report
- **kagglehub** — dataset download

---


## Future Work

- Validate on an independent hospital dataset to test real-world generalisation
- Collaborate with a cardiologist to define medically validated severity criteria
- Explore feature engineering (e.g. `chol/age` ratio, interaction terms)
- Deploy as a web application with patient input form
- Investigate explainability using SHAP values to understand feature importance per prediction
