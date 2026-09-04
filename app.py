"""
app.py -- Interactive churn prediction demo.

Enter a customer's details, get a live churn probability, an expected
value-at-risk estimate, and a SHAP breakdown of exactly why the model
scored them that way.

Run:
    python train_and_save_model.py   # once, to create model.pkl
    streamlit run app.py
"""

import json

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st

st.set_page_config(page_title="Bank Churn Predictor", page_icon="🏦", layout="centered")


@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    with open("feature_columns.json") as f:
        feature_columns = json.load(f)
    with open("reference_stats.json") as f:
        stats = json.load(f)
    explainer = shap.TreeExplainer(model)
    return model, feature_columns, stats, explainer


model, feature_columns, stats, explainer = load_model()

st.title("🏦 Bank Customer Churn Predictor")


# --------------------------------------------------------------------------
# Input form
# --------------------------------------------------------------------------
with st.form("customer_form"):
    col1, col2 = st.columns(2)

    with col1:
        credit_score = st.slider(
            "Credit Score", stats["CreditScore"]["min"], stats["CreditScore"]["max"],
            stats["CreditScore"]["mean"],
        )
        age = st.slider("Age", stats["Age"]["min"], stats["Age"]["max"], 35)
        tenure = st.slider(
            "Tenure (years with bank)", stats["Tenure"]["min"], stats["Tenure"]["max"],
            stats["Tenure"]["mean"],
        )
        balance = st.number_input(
            "Account Balance ($)", min_value=0.0,
            max_value=float(stats["Balance"]["max"]), value=float(stats["Balance"]["mean"]),
            step=1000.0,
        )
        estimated_salary = st.number_input(
            "Estimated Salary ($)", min_value=0.0,
            max_value=float(stats["EstimatedSalary"]["max"]),
            value=float(stats["EstimatedSalary"]["mean"]), step=1000.0,
        )

    with col2:
        geography = st.selectbox("Geography", stats["Geography"])
        gender = st.selectbox("Gender", ["Female", "Male"])
        num_products = st.selectbox("Number of Products", stats["NumOfProducts"])
        has_cr_card = st.selectbox("Has Credit Card?", ["Yes", "No"]) == "Yes"
        is_active_member = st.selectbox("Is Active Member?", ["Yes", "No"]) == "Yes"

    submitted = st.form_submit_button("Predict churn risk", use_container_width=True)

# --------------------------------------------------------------------------
# Build the feature row EXACTLY the way the training pipeline does, then
# predict + explain
# --------------------------------------------------------------------------
if submitted:
    age_group = pd.cut(
        [age], bins=[17, 30, 40, 50, 60, 100],
        labels=["18-30", "31-40", "41-50", "51-60", "60+"],
    )[0]

    row = {col: 0 for col in feature_columns}
    row["CreditScore"] = credit_score
    row["Gender"] = 1 if gender == "Male" else 0  # matches LabelEncoder: Male=1, Female=0
    row["Age"] = age
    row["Tenure"] = tenure
    row["Balance"] = balance
    row["NumOfProducts"] = num_products
    row["HasCrCard"] = int(has_cr_card)
    row["IsActiveMember"] = int(is_active_member)
    row["EstimatedSalary"] = estimated_salary
    row["ZeroBalance"] = int(balance == 0)
    if geography == "Germany":
        row["Geography_Germany"] = 1
    elif geography == "Spain":
        row["Geography_Spain"] = 1
    # France is the dropped baseline category -- both flags stay 0
    age_group_col = f"AgeGroup_{age_group}"
    if age_group_col in row:
        row[age_group_col] = 1
    # "18-30" is the dropped baseline category -- all AgeGroup flags stay 0

    X_input = pd.DataFrame([row])[feature_columns]  # enforce exact training column order

    proba = model.predict_proba(X_input)[0, 1]
    expected_value_at_risk = proba * balance

    st.divider()

    risk_col, evar_col = st.columns(2)
    with risk_col:
        st.metric("Predicted churn probability", f"{proba:.1%}")
        if proba >= 0.5:
            st.error("High risk — flag for retention outreach")
        elif proba >= 0.25:
            st.warning("Moderate risk — worth monitoring")
        else:
            st.success("Low risk")
    with evar_col:
        st.metric("Expected value at risk", f"${expected_value_at_risk:,.0f}")
        st.caption("= churn probability × account balance")

    st.subheader("Why the model scored this customer this way")
    shap_values = explainer(X_input)
    shap_churn = shap_values[:, :, 1]

    contributions = pd.Series(
        shap_churn.values[0], index=feature_columns
    ).sort_values(key=abs, ascending=False)

    st.write("Top factors pushing this prediction up or down:")
    for feature, value in contributions.head(6).items():
        direction = "increases" if value > 0 else "decreases"
        st.write(f"- **{feature}** ({row[feature]}): {direction} churn risk (SHAP = {value:+.3f})")

    with st.expander("See full SHAP waterfall plot"):
        import matplotlib.pyplot as plt

        fig = plt.figure()
        shap.plots.waterfall(shap_churn[0], show=False)
        st.pyplot(fig)

st.divider()
