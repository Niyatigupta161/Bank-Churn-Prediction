"""
train_and_save_model.py -- reproduces the exact training pipeline from the
notebook and saves everything the Streamlit app needs: the trained model,
the exact training feature order, and reference stats for the input widgets.

Run this once before running the app:
    python train_and_save_model.py
"""

import json

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("Churn_Modelling.csv")
df_model = df.drop(columns=["RowNumber", "CustomerId", "Surname"])
df_model["AgeGroup"] = pd.cut(
    df_model["Age"], bins=[17, 30, 40, 50, 60, 100],
    labels=["18-30", "31-40", "41-50", "51-60", "60+"],
)
df_model["ZeroBalance"] = (df_model["Balance"] == 0).astype(int)

df_enc = pd.get_dummies(df_model, columns=["Geography", "AgeGroup"], drop_first=True)
df_enc["Gender"] = LabelEncoder().fit_transform(df_enc["Gender"])  # Male=1, Female=0

X = df_enc.drop(columns=["Exited"])
y = df_enc["Exited"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

rf_model = RandomForestClassifier(
    n_estimators=300, max_depth=8, class_weight="balanced", random_state=42
)
rf_model.fit(X_train, y_train)

# Save the model
joblib.dump(rf_model, "model.pkl")

# Save the exact column order the model expects -- the app must build its
# input row with these columns, in this order, or predictions will be wrong.
feature_columns = list(X.columns)
with open("feature_columns.json", "w") as f:
    json.dump(feature_columns, f, indent=2)

# Save reasonable min/max/mean for each raw input, to set sensible slider
# ranges and defaults in the app.
reference_stats = {
    "CreditScore": {"min": int(df["CreditScore"].min()), "max": int(df["CreditScore"].max()), "mean": int(df["CreditScore"].mean())},
    "Age": {"min": int(df["Age"].min()), "max": int(df["Age"].max()), "mean": int(df["Age"].mean())},
    "Tenure": {"min": int(df["Tenure"].min()), "max": int(df["Tenure"].max()), "mean": int(df["Tenure"].mean())},
    "Balance": {"min": float(df["Balance"].min()), "max": float(df["Balance"].max()), "mean": float(df["Balance"].mean())},
    "EstimatedSalary": {"min": float(df["EstimatedSalary"].min()), "max": float(df["EstimatedSalary"].max()), "mean": float(df["EstimatedSalary"].mean())},
    "NumOfProducts": sorted(df["NumOfProducts"].unique().tolist()),
    "Geography": sorted(df["Geography"].unique().tolist()),
}
with open("reference_stats.json", "w") as f:
    json.dump(reference_stats, f, indent=2)

print("Saved model.pkl, feature_columns.json, reference_stats.json")
print(f"Trained on {len(X_train)} rows, held out {len(X_test)} for testing.")
print(f"Feature columns ({len(feature_columns)}): {feature_columns}")
