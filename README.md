# Bank Customer Churn Prediction

An end-to-end churn prediction project on 10,000 bank customer records: EDA,
feature engineering, model comparison, SHAP-based explainability, a
business-impact translation of the key finding, an interactive Streamlit
demo, and a Tableau dashboard.

## Problem statement

Predict which bank customers are likely to churn (close their account), and
go beyond a bare accuracy score to explain *why* the model flags a given
customer and *what it's worth* to the business to act on that.

## Dataset

(Churn_Modelling.csv) — 10,000 customers, 14 columns
(credit score, geography, gender, age, tenure, balance, number of products,
credit card / active-member flags, estimated salary, and the churn label
`Exited`). No missing values or duplicates.

## Approach

1. **EDA** — churn by geography, gender, age, balance; found churn is not
   evenly distributed (e.g. ~20% overall churn rate, but sharply higher for
   specific segments — see Key Finding below).
2. **Feature engineering** — `AgeGroup` (non-linear age binning, since churn
   risk doesn't rise linearly with age) and `ZeroBalance` (a flag for
   customers holding $0 balance, a distinct behavioral segment).
3. **Encoding** — one-hot encoding for `Geography` / `AgeGroup`, label
   encoding for `Gender`.
4. **Modeling** — Logistic Regression and Random Forest, both with
   `class_weight='balanced'` to address the ~20% churn class imbalance.
   Random Forest performed best.
5. **Explainability** — SHAP (TreeExplainer) on top of the native
   feature-importance chart, to get directional, per-prediction explanations
   instead of just a global importance score.
6. **Business impact** — translated the strongest EDA finding into an
   estimated dollar figure, and built a probability × value prioritization
   score for retention outreach.


## Key finding

Customers with **3+ products churn at ~82–100%**, versus **7.6%** for
customers with exactly 2 products — a non-monotonic, clearly abnormal
pattern that looks like an over-selling problem (customers being pushed into
more products than they want, then leaving). Full EDA and both models'
metrics are in the notebook.

## Explainability (SHAP)

Native Random Forest feature importance is biased toward high-cardinality,
continuous features and gives no sense of *direction*. SHAP confirms `Age`
and `NumOfProducts` as the top two drivers under both methods (a good
consistency check), while revealing that `IsActiveMember` and
`Geography_Germany` matter more than the native chart alone suggested. The
notebook includes global (bar, beeswarm) and per-customer (waterfall) SHAP
plots.

## Business impact

Using account balance as a data-grounded proxy for customer value (the
dataset has no separate profit/revenue field — stated explicitly as an
assumption, not fact), the over-selling pattern above translates to an
estimated **~$22M in excess account balance** lost beyond what the 2-product
baseline churn rate would predict. A prioritized retention list — sorting
customers by `churn_probability × balance` ("expected value at risk") rather
than probability alone — is also built, matching how a bank would actually
triage limited retention outreach. See the notebook for the full
calculation and caveats.




## Repository structure

```
.
├── Bank_churn_v2.ipynb              # full analysis: EDA, models, SHAP, business impact
├── Churn_Modelling.csv              # raw dataset
├── churn_predictions_for_tableau.csv # test-set predictions + expected_value_at_risk
├── bank_churn_dashboard.twb         # Tableau workbook
├── streamlit_app/
│   ├── app.py                       # interactive prediction + SHAP demo
│   ├── train_and_save_model.py      # trains and persists the model
│   ├── requirements.txt
│   └── Churn_Modelling.csv          # local copy so the app is self-contained
└── README.md
```

## Assumptions and caveats

- Account **balance is used as a proxy for customer value**, not actual
  profit — a real deployment would need the bank's own margin/revenue
  figures to turn this into a true revenue estimate.
- The dataset is a well-known public Kaggle dataset (10,000 rows) rather
  than proprietary bank data — findings are illustrative of a real analysis
  workflow, not a claim about any specific bank's customers.
- SHAP was computed on the test set with `TreeExplainer`, exact and fast for
  tree-based models like Random Forest.

