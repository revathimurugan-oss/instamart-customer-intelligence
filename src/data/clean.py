"""
src.data.clean
==============
Data cleaning, integrity validation and SQLite loading.

Stages
------
    1. clean()     — normalise raw CSVs → data/clean/
    2. validate()  — assert integrity constraints
    3. load_db()   — write clean tables to data/instamart.db

Usage
-----
    from src.data.clean import clean, validate, load_db
    clean()
    validate()
    load_db()

    # or run all three at once:
    from src.data.clean import run
    run()

    # or as a script:
    python -m src.data.clean
"""

import sqlite3
from pathlib import Path

import pandas as pd


# ── Path helpers ──────────────────────────────────────────────────────────────
def _root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _paths(root: Path | None = None):
    r = root or _root()
    return (
        r / "data" / "raw",
        r / "data" / "clean",
        r / "data" / "instamart.db",
    )


# ── Stage 1: clean ────────────────────────────────────────────────────────────
def clean(root: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Read raw CSVs, apply cleaning rules, write cleaned CSVs to data/clean/.

    Cleaning rules
    --------------
    customers:
        - city   : strip whitespace, title-case
        - age    : coerce to numeric; values outside [0, 120] → NaN
        - gender : map variants (M, m, MALE, Female, f …) → male / female / NaN
        - signup_date : parse to datetime
        - drop exact duplicate customer_ids (keep first)

    products:
        - price_inr : coerce to numeric
        - drop exact duplicate product_ids (keep first)

    orders:
        - drop exact full-row duplicates first
        - coerce order_ts, quantity, discount_pct, order_amount, delivery_minutes
        - negative delivery_minutes → NaN  (physically impossible)
        - resolve partial duplicates: keep row with latest order_ts per order_id

    Returns
    -------
    customers, products, orders DataFrames (also written to data/clean/).
    """
    raw, clean_dir, _ = _paths(root)
    clean_dir.mkdir(parents=True, exist_ok=True)

    # ── customers ──────────────────────────────────────────────────────────
    customers = pd.read_csv(raw / "customers.csv")
    customers["city"]   = customers["city"].astype("string").str.strip().str.title()
    customers["age"]    = pd.to_numeric(customers["age"], errors="coerce")
    customers.loc[~customers["age"].between(0, 120), "age"] = pd.NA
    customers["gender"] = customers["gender"].astype("string").str.strip().str.lower()
    gender_map = {"f":"female","female":"female","m":"male","male":"male"}
    customers["gender"] = customers["gender"].map(gender_map).astype("string")
    customers["signup_date"] = pd.to_datetime(customers["signup_date"], errors="coerce")
    customers = customers.drop_duplicates(subset="customer_id", keep="first")
    customers.to_csv(clean_dir / "customers.csv", index=False)

    # ── products ───────────────────────────────────────────────────────────
    products = pd.read_csv(raw / "products.csv")
    products["price_inr"] = pd.to_numeric(products["price_inr"], errors="coerce")
    products = products.drop_duplicates(subset="product_id", keep="first")
    products.to_csv(clean_dir / "products.csv", index=False)

    # ── orders ─────────────────────────────────────────────────────────────
    orders = pd.read_csv(raw / "orders.csv", dtype=str)
    orders = orders.drop_duplicates()
    orders["order_ts"]         = pd.to_datetime(orders["order_ts"], errors="coerce")
    orders["quantity"]         = pd.to_numeric(orders["quantity"], errors="coerce")
    orders["discount_pct"]     = pd.to_numeric(orders["discount_pct"], errors="coerce")
    orders["order_amount"]     = pd.to_numeric(orders["order_amount"].str.strip(), errors="coerce")
    orders["delivery_minutes"] = pd.to_numeric(orders["delivery_minutes"], errors="coerce")
    orders.loc[orders["delivery_minutes"] < 0, "delivery_minutes"] = pd.NA
    orders = (
        orders
        .sort_values(["order_id", "order_ts"], kind="stable", na_position="first")
        .drop_duplicates(subset="order_id", keep="last")
    )
    orders.to_csv(clean_dir / "orders.csv", index=False)

    print("=== CLEANING SUMMARY ===")
    print(f"Customers : {len(customers)}")
    print(f"Products  : {len(products)}")
    print(f"Orders    : {len(orders)}  (unique order IDs)")
    print(f"Missing order amounts  : {orders['order_amount'].isna().sum()}")
    print(f"Missing delivery times : {orders['delivery_minutes'].isna().sum()}")

    return customers, products, orders


# ── Stage 2: validate ─────────────────────────────────────────────────────────
def validate(root: Path | None = None) -> None:
    """
    Run structural assertions on the cleaned data.

    Checks
    ------
    - Primary-key uniqueness (customer_id, product_id, order_id)
    - Referential integrity (orders → customers, orders → products)
    - No null order_id or order_ts
    - quantity in [1, 100], discount_pct in [0, 100]
    - No negative delivery_minutes remain
    """
    _, clean_dir, _ = _paths(root)

    customers = pd.read_csv(clean_dir / "customers.csv")
    products  = pd.read_csv(clean_dir / "products.csv")
    orders    = pd.read_csv(clean_dir / "orders.csv")

    assert customers["customer_id"].is_unique,  "Duplicate customer IDs"
    assert products["product_id"].is_unique,    "Duplicate product IDs"
    assert orders["order_id"].is_unique,        "Duplicate order IDs"
    assert orders["customer_id"].isin(customers["customer_id"]).all(), \
        "Orders reference customer IDs not in customers table"
    assert orders["product_id"].isin(products["product_id"]).all(), \
        "Orders reference product IDs not in products table"
    assert orders["order_id"].notna().all(),    "Null order IDs found"
    assert orders["order_ts"].notna().all(),    "Null order timestamps found"

    orders_num = orders.copy()
    orders_num["quantity"]     = pd.to_numeric(orders_num["quantity"], errors="coerce")
    orders_num["discount_pct"] = pd.to_numeric(orders_num["discount_pct"], errors="coerce")
    assert orders_num["quantity"].between(1, 100).all(),       "quantity out of range"
    assert orders_num["discount_pct"].between(0, 100).all(),   "discount_pct out of range"

    valid_del = pd.to_numeric(orders["delivery_minutes"], errors="coerce").dropna()
    assert (valid_del >= 0).all(), "Negative delivery_minutes found"

    print("PASS: all structural assertions cleared")


# ── Stage 3: load SQLite ──────────────────────────────────────────────────────
def load_db(root: Path | None = None) -> None:
    """
    Load the three cleaned DataFrames into data/instamart.db and verify
    that the row counts in the database match the CSV row counts.
    """
    import sqlite3

    _, clean_dir, db_path = _paths(root)

    customers = pd.read_csv(clean_dir / "customers.csv")
    products  = pd.read_csv(clean_dir / "products.csv")
    orders    = pd.read_csv(clean_dir / "orders.csv")

    with sqlite3.connect(db_path) as con:
        customers.to_sql("customers", con, if_exists="replace", index=False)
        products.to_sql("products",   con, if_exists="replace", index=False)
        orders.to_sql("orders",       con, if_exists="replace", index=False)

    with sqlite3.connect(db_path) as con:
        for tbl, df in [("customers", customers), ("products", products), ("orders", orders)]:
            db_n = con.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
            assert db_n == len(df), f"{tbl}: CSV has {len(df)} rows but DB has {db_n}"
            print(f"  {tbl:<12}: {db_n:>6} rows  ✓")

    print(f"\nSQLite database written to {db_path}")
    print("ALL CHECKPOINTS PASSED")


# ── Convenience: run all three stages ────────────────────────────────────────
def run(root: Path | None = None) -> None:
    """Run clean → validate → load_db in sequence."""
    clean(root)
    validate(root)
    load_db(root)


if __name__ == "__main__":
    run()
