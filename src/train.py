"""
train.py

Trains two models using the leakage-safe splits from data_pipeline.py:
  1. Classification: will a student be placed? (LightGBM)
  2. Regression: what salary package, GIVEN they're placed? (LightGBM)

Both are tuned with Optuna (a handful of trials -- kept small so this runs
in minutes, not hours; bump n_trials up if you have more time later).
SHAP values are computed and saved so the app can show per-prediction
explanations instead of a black-box number.
"""

import joblib
import json
import numpy as np
import optuna
import shap
from pathlib import Path
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_score, recall_score,
    mean_absolute_error, mean_squared_error, r2_score
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
N_TRIALS = 25  # keep small for time budget; raise to 50-100 for a final pass later


def tune_classifier(X_train, y_train):
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "random_state": 42,
            "verbose": -1,
        }
        model = LGBMClassifier(**params)
        # Simple train/val split within train for tuning (test set stays untouched)
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(model, X_train, y_train, cv=3, scoring="f1")
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    return study.best_params


def tune_regressor(X_train, y_train):
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "random_state": 42,
            "verbose": -1,
        }
        model = LGBMRegressor(**params)
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(model, X_train, y_train, cv=3, scoring="neg_mean_absolute_error")
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    return study.best_params


def train_classification():
    print("=" * 60)
    print("CLASSIFICATION: Placement Prediction")
    print("=" * 60)
    X_train, X_test, y_train, y_test = joblib.load(PROCESSED_DIR / "classification_splits.pkl")

    print(f"Tuning with Optuna ({N_TRIALS} trials)...")
    best_params = tune_classifier(X_train, y_train)
    print(f"Best params: {best_params}")

    model = LGBMClassifier(**best_params, random_state=42, verbose=-1)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, probs),
    }
    print("Test metrics:", json.dumps(metrics, indent=2))

    joblib.dump(model, MODELS_DIR / "classification_model.pkl")

    # SHAP explainer -- saved so the app doesn't need to recompute the
    # background dataset every time it starts up.
    explainer = shap.TreeExplainer(model)
    joblib.dump(explainer, MODELS_DIR / "classification_shap_explainer.pkl")

    with open(MODELS_DIR / "classification_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def train_regression():
    print("=" * 60)
    print("REGRESSION: Salary Prediction (placed students only)")
    print("=" * 60)
    X_train, X_test, y_train, y_test = joblib.load(PROCESSED_DIR / "regression_splits.pkl")

    print(f"Tuning with Optuna ({N_TRIALS} trials)...")
    best_params = tune_regressor(X_train, y_train)
    print(f"Best params: {best_params}")

    model = LGBMRegressor(**best_params, random_state=42, verbose=-1)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    metrics = {
        "mae": mean_absolute_error(y_test, preds),
        "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
        "r2": r2_score(y_test, preds),
    }
    print("Test metrics:", json.dumps(metrics, indent=2))

    joblib.dump(model, MODELS_DIR / "regression_model.pkl")

    explainer = shap.TreeExplainer(model)
    joblib.dump(explainer, MODELS_DIR / "regression_shap_explainer.pkl")

    with open(MODELS_DIR / "regression_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    clf_metrics = train_classification()
    print()
    reg_metrics = train_regression()

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Classification F1: {clf_metrics['f1']:.4f} | ROC-AUC: {clf_metrics['roc_auc']:.4f}")
    print(f"Regression MAE: {reg_metrics['mae']:.2f} LPA | R2: {reg_metrics['r2']:.4f}")
