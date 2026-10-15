import sqlite3
from pathlib import Path
import pandas as pd

customers = pd.read_csv("data/clean/customers.csv")
products = pd.read_csv("data/clean/products.csv")
orders = pd.read_csv("data/clean/orders.csv")

# Check primary-key uniqueness.
assert customers["customer_id"].is_unique
assert products["product_id"].is_unique
assert orders["order_id"].is_unique

# Check referential integrity.
assert orders["customer_id"].isin(customers["customer_id"]).all()
assert orders["product_id"].isin(products["product_id"]).all()

# Check essential fields.
assert orders["order_id"].notna().all()
assert orders["order_ts"].notna().all()
assert orders["quantity"].between(1, 100).all()
assert orders["discount_pct"].between(0, 100).all()

# Invalid negative delivery times should not remain.
valid_delivery = orders["delivery_minutes"].dropna()
assert (valid_delivery >= 0).all()

# Validate database row counts and primary-key uniqueness.
with sqlite3.connect("data/instamart.db") as conn:
    db_counts = {
        table: conn.execute(
            f"SELECT COUNT(*) FROM {table}"
        ).fetchone()[0]
        for table in ("customers", "products", "orders")
    }

    assert db_counts["customers"] == len(customers)
    assert db_counts["products"] == len(products)
    assert db_counts["orders"] == len(orders)

    assert conn.execute(
        "SELECT COUNT(*) - COUNT(DISTINCT customer_id) FROM customers"
    ).fetchone()[0] == 0

    assert conn.execute(
        "SELECT COUNT(*) - COUNT(DISTINCT product_id) FROM products"
    ).fetchone()[0] == 0

    assert conn.execute(
        "SELECT COUNT(*) - COUNT(DISTINCT order_id) FROM orders"
    ).fetchone()[0] == 0

print("ALL CHECKPOINTS PASSED")
