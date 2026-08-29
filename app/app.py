"""
app.py

Streamlit app for the Placement Predictor v2.

Two things it does that a typical student project skips:
  1. Shows model performance HONESTLY (including where the classifier
     plateaus) rather than hiding weak metrics.
  2. Explains every single prediction with SHAP, not just returning a number.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import matplotlib.pyplot as plt
import shap


def extract_shap_row(shap_values, class_index=1):
    """
    SHAP's return shape for TreeExplainer varies across versions:
    - list of ndarrays (one per class) -> pick class_index
    - ndarray of shape (n_samples, n_features) -> already single-output
    - ndarray of shape (n_samples, n_features, n_classes) -> pick class_index on last axis
    This normalizes all three into a flat 1D array for a single row.
    """
    if isinstance(shap_values, list):
        arr = shap_values[class_index]
    else:
        arr = shap_values
    arr = np.asarray(arr)
    if arr.ndim == 3:
        arr = arr[:, :, class_index]
    return arr[0]

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

st.set_page_config(page_title="Placement Predictor v2", page_icon="🎓", layout="wide")


@st.cache_resource
def load_artifacts():
    clf_model = joblib.load(MODELS_DIR / "classification_model.pkl")
    reg_model = joblib.load(MODELS_DIR / "regression_model.pkl")
    clf_encoder = joblib.load(MODELS_DIR / "classification_encoder.pkl")
    reg_encoder = joblib.load(MODELS_DIR / "regression_encoder.pkl")
    clf_explainer = joblib.load(MODELS_DIR / "classification_shap_explainer.pkl")
    reg_explainer = joblib.load(MODELS_DIR / "regression_shap_explainer.pkl")
    with open(MODELS_DIR / "classification_metrics.json") as f:
        clf_metrics = json.load(f)
    with open(MODELS_DIR / "regression_metrics.json") as f:
        reg_metrics = json.load(f)
    return clf_model, reg_model, clf_encoder, reg_encoder, clf_explainer, reg_explainer, clf_metrics, reg_metrics


clf_model, reg_model, clf_encoder, reg_encoder, clf_explainer, reg_explainer, clf_metrics, reg_metrics = load_artifacts()

FEATURE_COLS = [
    "branch", "college_tier", "cgpa", "backlogs", "coding_skills", "dsa_score",
    "aptitude_score", "communication_skills", "ml_knowledge", "system_design",
    "internships", "projects_count", "certifications", "hackathons",
    "open_source_contributions", "extracurriculars"
]
BRANCHES = ["CSE", "IT", "ECE", "EE", "ME", "CE", "Chemical"]
TIERS = ["Tier-1", "Tier-2", "Tier-3"]

st.title("🎓 Placement Predictor v2")
st.caption("LightGBM + Optuna tuning + SHAP explainability, trained on 100K student records")

tab1, tab2 = st.tabs(["🔮 Predict", "📊 Model Performance"])

with tab1:
    st.subheader("Enter student profile")
    col1, col2, col3 = st.columns(3)

    with col1:
        branch = st.selectbox("Branch", BRANCHES)
        college_tier = st.selectbox("College Tier", TIERS)
        cgpa = st.slider("CGPA", 0.0, 10.0, 7.5, 0.01)
        backlogs = st.number_input("Backlogs", 0, 20, 0)
        coding_skills = st.slider("Coding Skills (0-10)", 0.0, 10.0, 6.0, 0.1)

    with col2:
        dsa_score = st.slider("DSA Score (0-10)", 0.0, 10.0, 6.0, 0.1)
        aptitude_score = st.slider("Aptitude Score (0-100)", 0.0, 100.0, 65.0, 0.5)
        communication_skills = st.slider("Communication Skills (0-10)", 0.0, 10.0, 6.0, 0.1)
        ml_knowledge = st.slider("ML Knowledge (0-10)", 0.0, 10.0, 5.0, 0.1)
        system_design = st.slider("System Design (0-10)", 0.0, 10.0, 3.0, 0.1)

    with col3:
        internships = st.number_input("Internships", 0, 10, 1)
        projects_count = st.number_input("Projects Count", 0, 20, 3)
        certifications = st.number_input("Certifications", 0, 20, 2)
        hackathons = st.number_input("Hackathons", 0, 20, 1)
        open_source_contributions = st.number_input("Open Source Contributions", 0, 50, 0)
        extracurriculars = st.number_input("Extracurriculars", 0, 20, 1)

    if st.button("Predict", type="primary", use_container_width=True):
        input_df = pd.DataFrame([{
            "branch": branch, "college_tier": college_tier, "cgpa": cgpa,
            "backlogs": backlogs, "coding_skills": coding_skills, "dsa_score": dsa_score,
            "aptitude_score": aptitude_score, "communication_skills": communication_skills,
            "ml_knowledge": ml_knowledge, "system_design": system_design,
            "internships": internships, "projects_count": projects_count,
            "certifications": certifications, "hackathons": hackathons,
            "open_source_contributions": open_source_contributions,
            "extracurriculars": extracurriculars
        }])[FEATURE_COLS]

        # Classification
        clf_input = input_df.copy()
        clf_input[["branch", "college_tier"]] = clf_encoder.transform(clf_input[["branch", "college_tier"]])
        placement_prob = clf_model.predict_proba(clf_input)[0][1]
        placement_pred = clf_model.predict(clf_input)[0]

        st.divider()
        result_col1, result_col2 = st.columns(2)

        with result_col1:
            if placement_pred == 1:
                st.success(f"### ✅ Likely to be Placed\n**Confidence: {placement_prob:.1%}**")
            else:
                st.error(f"### ❌ Placement Unlikely\n**Confidence: {1-placement_prob:.1%}**")

            # SHAP explanation for classification
            st.write("**Why this prediction?**")
            shap_values = clf_explainer.shap_values(clf_input)
            sv = extract_shap_row(shap_values, class_index=1)
            shap_df = pd.DataFrame({
                "feature": FEATURE_COLS,
                "impact": sv
            }).sort_values("impact", key=abs, ascending=False).head(6)
            fig, ax = plt.subplots(figsize=(5, 3))
            colors = ["#2ca02c" if v > 0 else "#d62728" for v in shap_df["impact"]]
            ax.barh(shap_df["feature"], shap_df["impact"], color=colors)
            ax.set_xlabel("Impact on placement probability")
            ax.invert_yaxis()
            st.pyplot(fig)

        with result_col2:
            if placement_pred == 1:
                reg_input = input_df.copy()
                reg_input[["branch", "college_tier"]] = reg_encoder.transform(reg_input[["branch", "college_tier"]])
                salary_pred = reg_model.predict(reg_input)[0]
                st.info(f"### 💰 Predicted Salary\n**₹{salary_pred:.2f} LPA**")

                st.write("**Why this salary?**")
                reg_shap_values = reg_explainer.shap_values(reg_input)
                reg_sv = extract_shap_row(reg_shap_values, class_index=0)
                reg_shap_df = pd.DataFrame({
                    "feature": FEATURE_COLS,
                    "impact": reg_sv
                }).sort_values("impact", key=abs, ascending=False).head(6)
                fig2, ax2 = plt.subplots(figsize=(5, 3))
                colors2 = ["#2ca02c" if v > 0 else "#d62728" for v in reg_shap_df["impact"]]
                ax2.barh(reg_shap_df["feature"], reg_shap_df["impact"], color=colors2)
                ax2.set_xlabel("Impact on salary (LPA)")
                ax2.invert_yaxis()
                st.pyplot(fig2)
            else:
                st.warning("Salary prediction only applies to predicted-placed students.")

with tab2:
    st.subheader("Classification: Placement Status")
    st.caption(
        "Reported honestly, including where the model plateaus — this dataset's "
        "features cap out around this ceiling, it isn't a tuning issue."
    )
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{clf_metrics['accuracy']:.1%}")
    m2.metric("F1 Score", f"{clf_metrics['f1']:.3f}")
    m3.metric("ROC-AUC", f"{clf_metrics['roc_auc']:.3f}")
    m4.metric("Recall", f"{clf_metrics['recall']:.1%}")

    st.subheader("Regression: Salary Prediction (placed students only)")
    r1, r2, r3 = st.columns(3)
    r1.metric("MAE", f"₹{reg_metrics['mae']:.2f} LPA")
    r2.metric("RMSE", f"₹{reg_metrics['rmse']:.2f} LPA")
    r3.metric("R² Score", f"{reg_metrics['r2']:.3f}")

    st.divider()
    st.subheader("Global Feature Importance")
    fi_col1, fi_col2 = st.columns(2)
    with fi_col1:
        st.write("**Classification**")
        imp = pd.Series(clf_model.feature_importances_, index=FEATURE_COLS).sort_values()
        fig3, ax3 = plt.subplots(figsize=(5, 5))
        ax3.barh(imp.index, imp.values, color="#1f77b4")
        st.pyplot(fig3)
    with fi_col2:
        st.write("**Regression**")
        imp2 = pd.Series(reg_model.feature_importances_, index=FEATURE_COLS).sort_values()
        fig4, ax4 = plt.subplots(figsize=(5, 5))
        ax4.barh(imp2.index, imp2.values, color="#ff7f0e")
        st.pyplot(fig4)

st.divider()
st.caption("Built with LightGBM, Optuna, SHAP, and Streamlit — trained on a 100K-row synthetic placement dataset.")
