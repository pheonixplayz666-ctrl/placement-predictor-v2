# Placement Predictor v2

Predicts whether a student will be placed, and if so, at what salary — trained on a
100,000-row student placement dataset (16 features: academics, skills, and activity metrics).

**[Live Demo](#)** *(add your Streamlit Cloud / HF Spaces link here after deploying)*

## Why v2, not just "a placement predictor"

Two things this project is built around, not bolted onto afterward:

1. **Leakage-safe design.** The dataset's salary column only exists for placed students —
   training a regression model on the full dataset (imputing salary for unplaced students)
   would silently leak the placement outcome into the salary model. This project splits
   classification and regression into **entirely separate pipelines**, with the regression
   model trained only on the placed subset. See [`src/data_pipeline.py`](src/data_pipeline.py)
   for the full reasoning, documented inline.

2. **Honest reporting over inflated numbers.** The classification model plateaus at
   ROC-AUC 0.67 — the available features (skills, academics, activity counts) genuinely
   don't fully determine placement outcomes, and that's reported as-is rather than
   massaged. The app's "Model Performance" tab shows this transparently. The salary
   regression model, by contrast, performs well (R² = 0.79) — skills and academics are
   much stronger predictors of *compensation given placement* than of placement itself.

## Results

| Task | Metric | Score |
|---|---|---|
| Classification (placed / not placed) | F1 | 0.814 |
| | ROC-AUC | 0.674 |
| | Recall | 0.981 |
| | Precision | 0.695 |
| Regression (salary, placed subset only) | MAE | ₹0.95 LPA |
| | RMSE | ₹1.19 LPA |
| | R² | 0.788 |

Top predictive features for placement: **DSA score, coding skills, CGPA, internships,
project count** — extracurriculars and open-source contributions carry almost no signal
in this dataset, which itself is a useful, slightly counterintuitive finding.

## Tech Stack

- **Modeling:** LightGBM, tuned with Optuna (25-trial Bayesian search, 3-fold CV)
- **Explainability:** SHAP (per-prediction and global feature importance)
- **App:** Streamlit — live prediction form + model performance dashboard
- **Data:** Pandas, scikit-learn (ordinal encoding, stratified splitting)

## Project Structure

```
placement-predictor-v2/
├── app/
│   └── app.py                 # Streamlit app (predictions + performance dashboard)
├── data/
│   ├── raw/                   # Original dataset
│   └── processed/             # Cached train/test splits (regenerated, not committed)
├── models/                    # Trained models, encoders, SHAP explainers (committed — small, ~4MB)
├── src/
│   ├── data_pipeline.py       # Leakage-safe preprocessing and splitting
│   └── train.py                # Optuna-tuned LightGBM training + SHAP + metrics
└── requirements.txt
```

## Running Locally

```bash
pip install -r requirements.txt

# Regenerate processed splits from raw data
python src/data_pipeline.py

# Train both models (takes a few minutes — 25 Optuna trials each)
python src/train.py

# Launch the app
streamlit run app/app.py
```

## What I'd improve with more time

- Compare LightGBM against CatBoost as a documented baseline (currently LightGBM-only
  to keep the training loop fast)
- Add an LLM layer that turns the SHAP output into a natural-language explanation
  ("Your placement odds are driven mainly by your DSA score, which is below the
  placed-student average...")
- Increase Optuna trials from 25 to 100+ for a final tuning pass
