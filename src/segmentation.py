"""
src.segmentation
================
Customer Intelligence — RFM segmentation and lapse-risk modelling.

Public API
----------
    run_segmentation(root)   — RFM features + K-Means (K=4) + PCA + profiles
    run_lapse_model(root)    — lapse cohort + GridSearchCV (LR + RF) + persist

Usage
-----
    from src.segmentation import run_segmentation, run_lapse_model
    run_segmentation()
    run_lapse_model()

    # or run both:
    from src.segmentation import run
    run()

    # or as a script:
    python -m src.segmentation
"""

import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, f1_score,
    precision_score, recall_score, silhouette_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)


# ── Path helpers ──────────────────────────────────────────────────────────────
def _root() -> Path:
    return Path(__file__).resolve().parent.parent


def _paths(root: Path | None = None):
    r = root or _root()
    return r / "data" / "clean", r / "outputs"


SNAPSHOT      = pd.Timestamp("2025-10-01")
CUTOFF        = pd.Timestamp("2025-09-01")
SEPTEMBER_END = pd.Timestamp("2025-10-01")

NUMERIC_FEATURES     = ["recency","frequency","monetary","avg_discount_pct",
                         "cancellation_rate","avg_delivery_minutes","age","tenure_days"]
CATEGORICAL_FEATURES = ["city","gender","favourite_category"]


# ── Preprocessing factory ─────────────────────────────────────────────────────
def _make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc",  StandardScaler()),
        ]), NUMERIC_FEATURES),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(drop="first", handle_unknown="ignore")),
        ]), CATEGORICAL_FEATURES),
    ])


# ── Stage 1: RFM + K-Means segmentation ──────────────────────────────────────
def run_segmentation(root: Path | None = None) -> pd.DataFrame:
    """
    Compute RFM features, evaluate K=2…8, fit K=4, name segments,
    run PCA, write outputs.

    Outputs written
    ---------------
    outputs/customer_segments.csv   — RFM + segment_id + segment_name + is_premium
    outputs/segment_profiles.csv    — mean RFM stats per segment
    outputs/kmeans_metrics.csv      — inertia + silhouette for K=2…8

    Returns
    -------
    rfm DataFrame with segment assignments.
    """
    data_dir, out_dir = _paths(root)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── load ──────────────────────────────────────────────────────────────────
    orders    = pd.read_csv(data_dir / "orders.csv")
    orders["order_ts"] = pd.to_datetime(orders["order_ts"], errors="coerce")
    for col in ["quantity","discount_pct","order_amount","delivery_minutes"]:
        orders[col] = pd.to_numeric(orders[col], errors="coerce")

    assert orders["order_id"].is_unique

    completed = orders[orders["status"].eq("completed")].copy()

    # ── Task 1: RFM ──────────────────────────────────────────────────────────
    rfm = (
        completed.groupby("customer_id")
        .agg(last_order_ts=("order_ts","max"),
             frequency=("order_id","nunique"),
             monetary=("order_amount","sum"))
        .reset_index()
    )
    rfm["recency"] = (SNAPSHOT - rfm["last_order_ts"]).dt.days
    rfm = rfm[["customer_id","recency","frequency","monetary"]].copy()

    print("\n=== TASK 1 — RFM SUMMARY ===")
    print(rfm[["recency","frequency","monetary"]].describe().round(2).to_string())

    # ── Task 2: K-Means K=2…8 ────────────────────────────────────────────────
    rfm_cols = ["recency","frequency","monetary"]
    scaler   = StandardScaler()
    X_rfm    = scaler.fit_transform(rfm[rfm_cols])

    k_results = []
    for k in range(2, 9):
        km     = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X_rfm)
        k_results.append({"k":k, "inertia":km.inertia_,
                           "silhouette":silhouette_score(X_rfm, labels)})

    k_table = pd.DataFrame(k_results)
    k_table.to_csv(out_dir / "kmeans_metrics.csv", index=False)
    print("\n=== TASK 2 — K-MEANS EVALUATION ===")
    print(k_table.round(4).to_string(index=False))

    # ── Task 3: fit K=4 + name segments ──────────────────────────────────────
    FINAL_K = 4
    kmeans  = KMeans(n_clusters=FINAL_K, n_init=10, random_state=42)
    rfm["segment_id"] = kmeans.fit_predict(X_rfm)

    profiles = (
        rfm.groupby("segment_id")
        .agg(size=("customer_id","size"),
             mean_recency=("recency","mean"),
             mean_frequency=("frequency","mean"),
             mean_monetary=("monetary","mean"))
        .reset_index()
    )

    pb = profiles.set_index("segment_id")
    premium_id = int(pb["mean_monetary"].idxmax())
    rem = [int(x) for x in profiles["segment_id"] if int(x) != premium_id]
    freq_id = int(pb.loc[rem,"mean_frequency"].idxmax()); rem.remove(freq_id)
    risk_id = int(pb.loc[rem,"mean_recency"].idxmax());   rem.remove(risk_id)
    other_id = rem[0]

    seg_names = {
        premium_id: "Premium High-Value",
        freq_id:    "Loyal Frequent",
        risk_id:    "Cooling / At-Risk",
        other_id:   "Steady Customers",
    }
    rfm["segment_name"] = rfm["segment_id"].map(seg_names)
    rfm["is_premium"]   = rfm["segment_id"].eq(premium_id)
    profiles["segment_name"] = profiles["segment_id"].map(seg_names)
    profiles["is_premium"]   = profiles["segment_id"].eq(premium_id)

    profiles.to_csv(out_dir / "segment_profiles.csv", index=False)

    # ── PCA explained variance ────────────────────────────────────────────────
    pca   = PCA(n_components=2)
    pca.fit_transform(X_rfm)
    print(f"\nPCA explained variance: {pca.explained_variance_ratio_.round(4)}")

    rfm[["customer_id","recency","frequency","monetary",
         "segment_id","segment_name","is_premium"]].to_csv(
        out_dir / "customer_segments.csv", index=False)

    print("\n=== TASK 3 — SEGMENT PROFILES ===")
    print(profiles.round(2).to_string(index=False))

    return rfm


# ── Stage 2: lapse-risk model ─────────────────────────────────────────────────
def run_lapse_model(root: Path | None = None) -> None:
    """
    Build lapse cohort, engineer leakage-safe features, train
    Logistic Regression and Random Forest with GridSearchCV,
    select the best model, persist it.

    Outputs written
    ---------------
    outputs/lapse_model.joblib     — best serialised classifier
    outputs/rf_top_features.csv    — top-5 Random Forest feature importances
    outputs/retention_list.csv     — top-10 premium customers at highest lapse risk
    """
    data_dir, out_dir = _paths(root)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── reload ────────────────────────────────────────────────────────────────
    customers = pd.read_csv(data_dir / "customers.csv")
    customers["signup_date"] = pd.to_datetime(customers["signup_date"], errors="coerce")
    orders = pd.read_csv(data_dir / "orders.csv")
    orders["order_ts"] = pd.to_datetime(orders["order_ts"], errors="coerce")
    products = pd.read_csv(data_dir / "products.csv")
    for col in ["quantity","discount_pct","order_amount","delivery_minutes"]:
        orders[col] = pd.to_numeric(orders[col], errors="coerce")

    orders_cat = orders.merge(products[["product_id","category"]], on="product_id", how="left")
    completed  = orders_cat[orders_cat["status"].eq("completed")].copy()

    # ── label ─────────────────────────────────────────────────────────────────
    pre_cutoff    = orders_cat[orders_cat["order_ts"] < CUTOFF]
    pre_completed = pre_cutoff[pre_cutoff["status"].eq("completed")]
    cohort_ids    = pre_completed["customer_id"].unique()
    sep_completed = completed[(completed["order_ts"] >= CUTOFF) &
                               (completed["order_ts"] < SEPTEMBER_END)]
    sep_ids       = set(sep_completed["customer_id"])

    cohort = pd.DataFrame({"customer_id": cohort_ids})
    cohort["lapsed"] = (~cohort["customer_id"].isin(sep_ids)).astype(int)

    # ── pre-cutoff features ───────────────────────────────────────────────────
    pre_rfm = (
        pre_completed.groupby("customer_id")
        .agg(last_order_ts=("order_ts","max"), frequency=("order_id","nunique"),
             monetary=("order_amount","sum"), avg_discount_pct=("discount_pct","mean"),
             avg_delivery_minutes=("delivery_minutes","mean"))
        .reset_index()
    )
    pre_rfm["recency"] = (CUTOFF - pre_rfm["last_order_ts"]).dt.days

    st_cnt = (
        pre_cutoff
        .assign(is_canc=pre_cutoff["status"].eq("cancelled").astype(int))
        .groupby("customer_id")
        .agg(all_f=("order_id","nunique"), canc=("is_canc","sum"))
        .reset_index()
    )
    st_cnt["cancellation_rate"] = st_cnt["canc"] / st_cnt["all_f"].replace(0, np.nan)

    fav = (
        pre_completed.groupby(["customer_id","category"]).size().reset_index(name="n")
        .sort_values(["customer_id","n","category"], ascending=[True,False,True], kind="stable")
        .drop_duplicates("customer_id")[["customer_id","category"]]
        .rename(columns={"category":"favourite_category"})
    )

    features_df = (
        cohort
        .merge(pre_rfm, on="customer_id", how="left")
        .merge(st_cnt[["customer_id","cancellation_rate"]], on="customer_id", how="left")
        .merge(fav, on="customer_id", how="left")
        .merge(customers[["customer_id","city","age","gender","signup_date"]],
               on="customer_id", how="left")
    )
    features_df["tenure_days"] = (CUTOFF - features_df["signup_date"]).dt.days

    print(f"\n=== TASK 4 — LAPSE COHORT ===")
    print(f"Cohort size : {len(features_df)}")
    print(f"Lapsed      : {features_df['lapsed'].sum()} ({features_df['lapsed'].mean()*100:.1f}%)")

    # ── train / test split ────────────────────────────────────────────────────
    X = features_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = features_df["lapsed"].copy()
    cust_ids = features_df["customer_id"].copy()

    X_tr, X_te, y_tr, y_te, id_tr, id_te = train_test_split(
        X, y, cust_ids, test_size=0.2, stratify=y, random_state=42)

    # ── GridSearchCV ──────────────────────────────────────────────────────────
    lr_pipe = Pipeline([("pre", _make_preprocessor()),
                        ("m",   LogisticRegression(max_iter=1000))])
    lr_grid = GridSearchCV(lr_pipe, {"m__C":[0.01,0.1,1,10,100]},
                           cv=5, scoring="f1", n_jobs=-1)
    lr_grid.fit(X_tr, y_tr)

    rf_pipe = Pipeline([("pre", _make_preprocessor()),
                        ("m",   RandomForestClassifier(n_estimators=100, random_state=42))])
    rf_grid = GridSearchCV(rf_pipe,
                           {"m__max_depth":[3,5,8], "m__min_samples_leaf":[5,10,20]},
                           cv=5, scoring="f1", n_jobs=-1)
    rf_grid.fit(X_tr, y_tr)

    best_lr, best_rf = lr_grid.best_estimator_, rf_grid.best_estimator_
    print(f"\nLR  best params: {lr_grid.best_params_}  CV F1={lr_grid.best_score_:.4f}")
    print(f"RF  best params: {rf_grid.best_params_}  CV F1={rf_grid.best_score_:.4f}")

    def report(model, label):
        pred = model.predict(X_te)
        print(f"\n{label}")
        print(f"  Confusion: {confusion_matrix(y_te, pred, labels=[0,1]).tolist()}")
        print(f"  Precision: {precision_score(y_te,pred,zero_division=0):.4f}")
        print(f"  Recall   : {recall_score(y_te,pred,zero_division=0):.4f}")
        print(f"  F1       : {f1_score(y_te,pred,zero_division=0):.4f}")
        return pred

    lr_pred = report(best_lr, "Logistic Regression — test set")
    rf_pred = report(best_rf, "Random Forest — test set")

    # ── select & persist ──────────────────────────────────────────────────────
    rf_f1 = f1_score(y_te, rf_pred, zero_division=0)
    lr_f1 = f1_score(y_te, lr_pred, zero_division=0)
    sel_model, sel_name = (best_rf,"Random Forest") if rf_f1>=lr_f1 else (best_lr,"Logistic Regression")
    print(f"\nSelected: {sel_name}")

    model_path = out_dir / "lapse_model.joblib"
    joblib.dump(sel_model, model_path)
    assert np.array_equal(sel_model.predict(X_te), joblib.load(model_path).predict(X_te))
    print(f"PASS: model saved & verified → {model_path}")

    # ── feature importance ────────────────────────────────────────────────────
    rf_pre = best_rf.named_steps["pre"]
    rf_est = best_rf.named_steps["m"]
    importance = pd.DataFrame({
        "feature":    rf_pre.get_feature_names_out(),
        "importance": rf_est.feature_importances_,
    }).sort_values("importance", ascending=False)
    importance.head(5).to_csv(out_dir / "rf_top_features.csv", index=False)
    print("\nTop 5 features:")
    print(importance.head(5).round(5).to_string(index=False))

    # ── retention list ────────────────────────────────────────────────────────
    rfm = pd.read_csv(out_dir / "customer_segments.csv")
    premium_ids = set(rfm.loc[rfm["is_premium"].astype(str).str.lower().eq("true"), "customer_id"].astype(str))
    test_res = pd.DataFrame({
        "customer_id"      : id_te.astype(str).to_numpy(),
        "actually_lapsed"  : y_te.to_numpy(),
        "lapse_probability": sel_model.predict_proba(X_te)[:,1],
    })
    retention = (
        test_res[test_res["customer_id"].isin(premium_ids)]
        .sort_values("lapse_probability", ascending=False)
        .head(10)[["customer_id","lapse_probability","actually_lapsed"]]
    )
    retention.to_csv(out_dir / "retention_list.csv", index=False)
    print(f"\nRetention list saved ({len(retention)} rows)")
    print(f"scikit-learn version: {sklearn.__version__}")
    print("PART B SCRIPT FINISHED")


def run(root: Path | None = None) -> None:
    """Run segmentation then lapse model in sequence."""
    run_segmentation(root)
    run_lapse_model(root)


if __name__ == "__main__":
    run()
