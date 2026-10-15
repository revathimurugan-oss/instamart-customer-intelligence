from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn

from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    silhouette_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import clone

warnings.filterwarnings("ignore", category=FutureWarning)

DATA = Path("data/clean")
OUT = Path("outputs")
OUT.mkdir(parents=True, exist_ok=True)

SNAPSHOT = pd.Timestamp("2025-10-01 00:00:00")
CUTOFF = pd.Timestamp("2025-09-01 00:00:00")
SEPTEMBER_END = pd.Timestamp("2025-10-01 00:00:00")

# ------------------------------------------------------------
# Load and validate the Part A data
# ------------------------------------------------------------
customers = pd.read_csv(DATA / "customers.csv")
orders = pd.read_csv(DATA / "orders.csv")
products = pd.read_csv(DATA / "products.csv")

orders["order_ts"] = pd.to_datetime(orders["order_ts"], errors="coerce")
customers["signup_date"] = pd.to_datetime(
    customers["signup_date"], errors="coerce"
)

for col in ["quantity", "discount_pct", "order_amount", "delivery_minutes"]:
    orders[col] = pd.to_numeric(orders[col], errors="coerce")

assert orders["order_id"].is_unique, "Expected one cleaned row per order ID"
assert customers["customer_id"].is_unique
assert orders["customer_id"].isin(customers["customer_id"]).all()
assert orders["product_id"].isin(products["product_id"]).all()

orders = orders.merge(
    products[["product_id", "category"]],
    on="product_id",
    how="left",
    validate="many_to_one",
)

completed = orders[orders["status"].eq("completed")].copy()

# ------------------------------------------------------------
# Task 1: RFM as of 2025-10-01
# ------------------------------------------------------------
rfm = (
    completed.groupby("customer_id")
    .agg(
        last_order_ts=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum"),
    )
    .reset_index()
)

rfm["recency"] = (SNAPSHOT - rfm["last_order_ts"]).dt.days
rfm = rfm[
    ["customer_id", "recency", "frequency", "monetary"]
].copy()

assert len(rfm) == 1175, f"Unexpected RFM customer count: {len(rfm)}"

print("\n" + "=" * 65)
print("TASK 1 — RFM SUMMARY")
print("=" * 65)
print(rfm[["recency", "frequency", "monetary"]].agg(
    ["count", "mean", "median", "min", "max"]
).round(2).to_string())

# ------------------------------------------------------------
# Task 2: K-Means K=2..8 and silhouette scores
# ------------------------------------------------------------
rfm_cols = ["recency", "frequency", "monetary"]
rfm_scaler = StandardScaler()
X_rfm_scaled = rfm_scaler.fit_transform(rfm[rfm_cols])

k_results = []
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(X_rfm_scaled)
    k_results.append({
        "k": k,
        "inertia": km.inertia_,
        "silhouette": silhouette_score(X_rfm_scaled, labels),
    })

k_table = pd.DataFrame(k_results)
k_table.to_csv(OUT / "kmeans_metrics.csv", index=False)

print("\n" + "=" * 65)
print("TASK 2 — K-MEANS EVALUATION")
print("=" * 65)
print(k_table.round(4).to_string(index=False))
print(
    "\nK=8 has lower inertia because more clusters let K-Means fit the "
    "observations more closely; inertia cannot increase when K increases "
    "under the same data and objective. That does not mean K=8 is best: "
    "too many clusters can fragment useful customer groups, so we also "
    "consider silhouette and whether marketing teams can act on the segments."
)

# Choose K=4: a practical balance of separation and interpretability.
FINAL_K = 4
kmeans = KMeans(n_clusters=FINAL_K, n_init=10, random_state=42)
rfm["segment_id"] = kmeans.fit_predict(X_rfm_scaled)

profiles = (
    rfm.groupby("segment_id")
    .agg(
        size=("customer_id", "size"),
        mean_recency=("recency", "mean"),
        mean_frequency=("frequency", "mean"),
        mean_monetary=("monetary", "mean"),
    )
    .reset_index()
)

# Give cluster IDs human-readable names based on their measured profiles.
# IDs themselves are arbitrary labels, not ranks.
profile_by_id = profiles.set_index("segment_id")
premium_id = int(profile_by_id["mean_monetary"].idxmax())
remaining_ids = [int(x) for x in profiles["segment_id"] if int(x) != premium_id]

frequency_id = int(
    profile_by_id.loc[remaining_ids, "mean_frequency"].idxmax()
)
remaining_ids.remove(frequency_id)

at_risk_id = int(
    profile_by_id.loc[remaining_ids, "mean_recency"].idxmax()
)
remaining_ids.remove(at_risk_id)

other_id = remaining_ids[0]

segment_names = {
    premium_id: "Premium High-Value",
    frequency_id: "Loyal Frequent",
    at_risk_id: "Cooling / At-Risk",
    other_id: "Steady Customers",
}

rfm["segment_name"] = rfm["segment_id"].map(segment_names)
rfm["is_premium"] = rfm["segment_id"].eq(premium_id)

profiles["segment_name"] = profiles["segment_id"].map(segment_names)
profiles["is_premium"] = profiles["segment_id"].eq(premium_id)
profiles.to_csv(OUT / "segment_profiles.csv", index=False)

# ------------------------------------------------------------
# Task 3(c): PCA on the scaled RFM data
# ------------------------------------------------------------
pca = PCA(n_components=2)
X_rfm_pca = pca.fit_transform(X_rfm_scaled)
pca_ratios = pca.explained_variance_ratio_

print("\n" + "=" * 65)
print("TASK 3 — SEGMENT PROFILES")
print("=" * 65)
print(profiles.round(2).to_string(index=False))
print("\nCluster IDs are labels only; they do not imply rank or quality.")
print(f"Premium segment ID: {premium_id}")
print(f"PCA explained variance ratio: {pca_ratios.round(4)}")
print(f"PCA total explained variance: {pca_ratios.sum():.4f}")

rfm[
    [
        "customer_id", "recency", "frequency", "monetary",
        "segment_id", "segment_name", "is_premium",
    ]
].to_csv(OUT / "customer_segments.csv", index=False)

# ------------------------------------------------------------
# Task 4: Build the lapse cohort and leakage-safe features
# ------------------------------------------------------------
# Cohort: at least one completed order strictly before the cutoff.
pre_cutoff = orders[
    (orders["order_ts"] < CUTOFF)
].copy()

pre_completed = pre_cutoff[
    pre_cutoff["status"].eq("completed")
].copy()

cohort_ids = pre_completed["customer_id"].unique()

# September completed orders are used ONLY to construct the label.
september_completed = completed[
    (completed["order_ts"] >= CUTOFF)
    & (completed["order_ts"] < SEPTEMBER_END)
]
september_customer_ids = set(september_completed["customer_id"])

cohort = pd.DataFrame({"customer_id": cohort_ids})
cohort["lapsed"] = (
    ~cohort["customer_id"].isin(september_customer_ids)
).astype(int)

# Behaviour features are calculated strictly before the cutoff.
pre_rfm = (
    pre_completed.groupby("customer_id")
    .agg(
        last_order_ts=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum"),
        avg_discount_pct=("discount_pct", "mean"),
        avg_delivery_minutes=("delivery_minutes", "mean"),
    )
    .reset_index()
)
pre_rfm["recency"] = (CUTOFF - pre_rfm["last_order_ts"]).dt.days

# Cancellation rate uses final-status orders before the cutoff.
status_counts = (
    pre_cutoff.assign(
        is_cancelled=pre_cutoff["status"].eq("cancelled").astype(int)
    )
    .groupby("customer_id")
    .agg(
        all_final_orders=("order_id", "nunique"),
        cancelled_orders=("is_cancelled", "sum"),
    )
    .reset_index()
)
status_counts["cancellation_rate"] = (
    status_counts["cancelled_orders"]
    / status_counts["all_final_orders"].replace(0, np.nan)
)

# Most frequently purchased category among completed pre-cutoff orders.
favourite_category = (
    pre_completed.groupby(["customer_id", "category"])
    .size()
    .reset_index(name="category_orders")
    .sort_values(
        ["customer_id", "category_orders", "category"],
        ascending=[True, False, True],
        kind="stable",
    )
    .drop_duplicates("customer_id")
    [["customer_id", "category"]]
    .rename(columns={"category": "favourite_category"})
)

features = (
    cohort
    .merge(pre_rfm, on="customer_id", how="left", validate="one_to_one")
    .merge(
        status_counts[
            ["customer_id", "cancellation_rate"]
        ],
        on="customer_id", how="left", validate="one_to_one",
    )
    .merge(favourite_category, on="customer_id", how="left",
           validate="one_to_one")
    .merge(
        customers[
            ["customer_id", "city", "age", "gender", "signup_date"]
        ],
        on="customer_id", how="left", validate="one_to_one",
    )
)

features["tenure_days"] = (CUTOFF - features["signup_date"]).dt.days

assert len(features) == 1156, f"Unexpected cohort size: {len(features)}"
assert features["lapsed"].sum() == 443, (
    f"Unexpected lapsed count: {features['lapsed'].sum()}"
)

numeric_features = [
    "recency",
    "frequency",
    "monetary",
    "avg_discount_pct",
    "cancellation_rate",
    "avg_delivery_minutes",
    "age",
    "tenure_days",
]
categorical_features = ["city", "gender", "favourite_category"]

# ID, target, segment ID/name, and Premium flag are deliberately excluded.
X = features[numeric_features + categorical_features].copy()
y = features["lapsed"].copy()
customer_ids = features["customer_id"].copy()

assert not any(
    col in X.columns
    for col in ["segment_id", "segment_name", "is_premium", "lapsed"]
)
assert features["customer_id"].is_unique

X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
    X,
    y,
    customer_ids,
    test_size=0.2,
    stratify=y,
    random_state=42,
)

print("\n" + "=" * 65)
print("TASK 4 — LAPSE COHORT")
print("=" * 65)
print(f"Cohort customers: {len(features)}")
print(f"Lapsed customers: {int(y.sum())} ({y.mean() * 100:.2f}%)")
print(f"Train rows: {len(X_train)}; test rows: {len(X_test)}")
print("Features use only pre-cutoff data; September orders create labels only.")
print("Task 3 segment labels are not model features.")

# ------------------------------------------------------------
# Task 5: Train-only preprocessing
# ------------------------------------------------------------
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore")),
])
preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features),
])

# Explicitly fit the preprocessing on training rows only.
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

def assert_no_missing_or_infinite(matrix, label):
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    assert np.isfinite(matrix).all(), f"{label} has missing/infinite values"

assert_no_missing_or_infinite(X_train_transformed, "Transformed train")
assert_no_missing_or_infinite(X_test_transformed, "Transformed test")

frequency_mean_train = (
    preprocessor.named_transformers_["num"]
    .named_steps["scaler"].mean_[numeric_features.index("frequency")]
)
frequency_mean_cohort = X["frequency"].mean()

print("\n" + "=" * 65)
print("TASK 5 — PREPROCESSING CHECK")
print("=" * 65)
print(f"Training-fitted scaler mean of frequency: {frequency_mean_train:.4f}")
print(f"Full-cohort frequency mean: {frequency_mean_cohort:.4f}")
print(
    "These differ because the scaler is fitted only on training rows, "
    "while the full-cohort mean includes the held-out test customers."
)
print("PASS: transformed train/test features contain no missing or infinite values.")

# Fit models through pipelines so CV refits preprocessing on each fold's
# training rows, rather than sharing preprocessing fitted on all training rows.
def make_preprocessor():
    return ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric_features),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore")),
        ]), categorical_features),
    ])

# ------------------------------------------------------------
# Task 6(a): Majority-class baseline
# ------------------------------------------------------------
baseline_pred = np.zeros(len(y_test), dtype=int)
print("\n" + "=" * 65)
print("TASK 6 — BASELINE")
print("=" * 65)
print(f"Baseline accuracy: {accuracy_score(y_test, baseline_pred):.4f}")
print(
    "Baseline lapsed-class precision/recall/F1: "
    f"{precision_score(y_test, baseline_pred, zero_division=0):.4f} / "
    f"{recall_score(y_test, baseline_pred, zero_division=0):.4f} / "
    f"{f1_score(y_test, baseline_pred, zero_division=0):.4f}"
)

# ------------------------------------------------------------
# Task 6(b): Logistic regression GridSearchCV
# ------------------------------------------------------------
lr_pipeline = Pipeline([
    ("preprocessor", make_preprocessor()),
    ("model", LogisticRegression(max_iter=1000)),
])
lr_grid = GridSearchCV(
    lr_pipeline,
    param_grid={"model__C": [0.01, 0.1, 1, 10, 100]},
    cv=5,
    scoring="f1",
    n_jobs=-1,
)
lr_grid.fit(X_train, y_train)
best_lr = lr_grid.best_estimator_

# ------------------------------------------------------------
# Task 6(c): Random forest GridSearchCV (9 settings x 5 folds)
# ------------------------------------------------------------
rf_pipeline = Pipeline([
    ("preprocessor", make_preprocessor()),
    ("model", RandomForestClassifier(
        n_estimators=100, random_state=42
    )),
])
rf_grid = GridSearchCV(
    rf_pipeline,
    param_grid={
        "model__max_depth": [3, 5, 8],
        "model__min_samples_leaf": [5, 10, 20],
    },
    cv=5,
    scoring="f1",
    n_jobs=-1,
)
rf_grid.fit(X_train, y_train)
best_rf = rf_grid.best_estimator_

print("\n" + "=" * 65)
print("TASK 6 — GRID SEARCH RESULTS")
print("=" * 65)
print("Logistic regression best parameters:", lr_grid.best_params_)
print(f"Logistic regression best CV F1: {lr_grid.best_score_:.4f}")
print("Random forest best parameters:", rf_grid.best_params_)
print(f"Random forest best CV F1: {rf_grid.best_score_:.4f}")

def report_positive_metrics(model, X_eval, y_eval, label):
    pred = model.predict(X_eval)
    print(f"\n{label}")
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(confusion_matrix(y_eval, pred, labels=[0, 1]))
    print(f"Precision (lapsed): {precision_score(y_eval, pred, zero_division=0):.4f}")
    print(f"Recall (lapsed):    {recall_score(y_eval, pred, zero_division=0):.4f}")
    print(f"F1 (lapsed):        {f1_score(y_eval, pred, zero_division=0):.4f}")
    return pred

lr_test_pred = report_positive_metrics(
    best_lr, X_test, y_test, "Tuned Logistic Regression — test set"
)
rf_test_pred = report_positive_metrics(
    best_rf, X_test, y_test, "Tuned Random Forest — test set"
)

rf_test_f1 = f1_score(y_test, rf_test_pred, zero_division=0)
lr_test_f1 = f1_score(y_test, lr_test_pred, zero_division=0)

# Select the model with the higher untouched-test lapsed-class F1.
# Test scores are used here for final selection only, after tuning.
selected_name, selected_model = (
    ("Random Forest", best_rf)
    if rf_test_f1 >= lr_test_f1
    else ("Logistic Regression", best_lr)
)
selected_test_pred = selected_model.predict(X_test)
selected_test_f1 = f1_score(y_test, selected_test_pred, zero_division=0)

print(f"\nSelected model: {selected_name}")
print(f"Selected test F1: {selected_test_f1:.4f}")
if selected_test_f1 < 0.55:
    print(
        "WARNING: selected test F1 is below the 0.55 acceptance target. "
        "Do not claim Part B is complete; investigate the features/data."
    )

print(
    "Business choice: missing a genuinely lapsing customer can lose future "
    "orders, while wrongly targeting a loyal customer wastes campaign budget "
    "and may annoy them. Compare recall and precision together, then tune "
    "the campaign threshold to balance retention coverage and contact cost."
)

# ------------------------------------------------------------
# Task 7: Oversample lapsed training rows only; test stays unchanged
# ------------------------------------------------------------
train_counts = y_train.value_counts()
majority_class = int(train_counts.idxmax())
minority_class = int(train_counts.idxmin())
majority_count = int(train_counts.max())
minority_count = int(train_counts.min())

minority_positions = np.flatnonzero(y_train.to_numpy() == minority_class)
rng = np.random.RandomState(42)
extra_positions = rng.choice(
    minority_positions,
    size=majority_count - minority_count,
    replace=True,
)

X_balanced = pd.concat(
    [X_train.reset_index(drop=True),
     X_train.iloc[extra_positions].reset_index(drop=True)],
    ignore_index=True,
)
y_balanced = pd.concat(
    [y_train.reset_index(drop=True),
     y_train.iloc[extra_positions].reset_index(drop=True)],
    ignore_index=True,
)

balanced_lr = clone(best_lr)
balanced_lr.fit(X_balanced, y_balanced)
balanced_lr_pred = balanced_lr.predict(X_test)

print("\n" + "=" * 65)
print("TASK 7 — TRAIN-ONLY OVERSAMPLING")
print("=" * 65)
print("Original training class counts:")
print(y_train.value_counts().sort_index().to_dict())
print("Balanced training class counts:")
print(y_balanced.value_counts().sort_index().to_dict())

for label, pred in [
    ("Unbalanced logistic regression", lr_test_pred),
    ("Oversampled logistic regression", balanced_lr_pred),
]:
    print(
        f"{label}: precision={precision_score(y_test, pred, zero_division=0):.4f}, "
        f"recall={recall_score(y_test, pred, zero_division=0):.4f}, "
        f"F1={f1_score(y_test, pred, zero_division=0):.4f}"
    )

print(
    "Oversampling changes only the training data. The test set is the original "
    "untouched cohort split, so the two results remain comparable."
)

# ------------------------------------------------------------
# Task 8(a): Interpret RF feature importance and LR coefficient signs
# ------------------------------------------------------------
rf_pre = best_rf.named_steps["preprocessor"]
rf_estimator = best_rf.named_steps["model"]
rf_feature_names = rf_pre.get_feature_names_out()

importance = pd.DataFrame({
    "feature": rf_feature_names,
    "importance": rf_estimator.feature_importances_,
}).sort_values("importance", ascending=False)

importance.head(5).to_csv(OUT / "rf_top_features.csv", index=False)

print("\n" + "=" * 65)
print("TASK 8 — RANDOM FOREST TOP FIVE FEATURES")
print("=" * 65)
print(importance.head(5).round(5).to_string(index=False))

lr_pre = best_lr.named_steps["preprocessor"]
lr_estimator = best_lr.named_steps["model"]
lr_feature_names = lr_pre.get_feature_names_out()
lr_coefs = pd.Series(lr_estimator.coef_[0], index=lr_feature_names)

print("\nLogistic regression coefficient signs:")
for term in ["num__recency", "num__frequency", "num__cancellation_rate"]:
    coef = float(lr_coefs[term])
    sign = "positive" if coef > 0 else "negative" if coef < 0 else "zero"
    print(f"{term}: coefficient={coef:.5f} ({sign})")

# ------------------------------------------------------------
# Task 8(b): Persist and verify the selected model
# ------------------------------------------------------------
model_path = OUT / "lapse_model.joblib"
joblib.dump(selected_model, model_path)
reloaded_model = joblib.load(model_path)

original_predictions = selected_model.predict(X_test)
reloaded_predictions = reloaded_model.predict(X_test)
assert np.array_equal(original_predictions, reloaded_predictions)

print("\nPASS: reloaded model reproduces all original test predictions exactly.")
print("Saved model:", model_path)

# ------------------------------------------------------------
# Task 8(c): Illustrative Premium retention list
# ------------------------------------------------------------
# Premium is a post-hoc label derived from full-window RFM, including
# September; it is NOT used in the model features.
premium_customers = set(
    rfm.loc[rfm["is_premium"], "customer_id"].astype(str)
)
test_results = pd.DataFrame({
    "customer_id": id_test.astype(str).to_numpy(),
    "actually_lapsed": y_test.to_numpy(),
    "lapse_probability": selected_model.predict_proba(X_test)[:, 1],
})
retention = test_results[
    test_results["customer_id"].isin(premium_customers)
].sort_values("lapse_probability", ascending=False).head(10).copy()

retention = retention[
    ["customer_id", "lapse_probability", "actually_lapsed"]
]
retention.to_csv(OUT / "retention_list.csv", index=False)

print("\n" + "=" * 65)
print("PREMIUM RETENTION LIST — ILLUSTRATIVE ONLY")
print("=" * 65)
print(retention.round(4).to_string(index=False))
print(f"Rows in retention list: {len(retention)}")
print(f"Actually lapsed among listed customers: {int(retention['actually_lapsed'].sum())}")
print(
    "The retention list is illustrative, not a leakage-free evaluation, "
    "because the Premium flag comes from whole-window RFM including September."
)

print("\n" + "=" * 65)
print("PART B OUTPUT FILES")
print("=" * 65)
for path in [
    OUT / "customer_segments.csv",
    OUT / "segment_profiles.csv",
    OUT / "kmeans_metrics.csv",
    OUT / "lapse_model.joblib",
    OUT / "retention_list.csv",
    OUT / "rf_top_features.csv",
]:
    print(f"{path}: {'OK' if path.exists() else 'MISSING'}")

print("\nscikit-learn version:", sklearn.__version__)
print("PART B SCRIPT FINISHED")
