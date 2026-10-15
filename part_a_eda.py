import sqlite3
from pathlib import Path
import pandas as pd

orders = pd.read_csv("data/clean/orders.csv")
customers = pd.read_csv("data/clean/customers.csv")
products = pd.read_csv("data/clean/products.csv")

orders["order_ts"] = pd.to_datetime(orders["order_ts"], errors="coerce")

# Cross-check 1: status counts
pandas_status = orders["status"].value_counts().sort_index()

with sqlite3.connect("data/instamart.db") as conn:
    sql_status = pd.read_sql_query(
        "SELECT status, COUNT(*) AS count FROM orders GROUP BY status",
        conn
    ).set_index("status")["count"].sort_index()

assert pandas_status.equals(sql_status)
print("PASS: SQL and pandas order status counts match")

# Cross-check 2: monthly completed-order revenue
completed = orders[orders["status"] == "completed"].copy()
completed["order_month"] = completed["order_ts"].dt.strftime("%Y-%m")

pandas_monthly = (
    completed.groupby("order_month")["order_amount"]
    .sum(min_count=1).round(2)
)

with sqlite3.connect("data/instamart.db") as conn:
    sql_monthly = pd.read_sql_query("""
        SELECT substr(order_ts, 1, 7) AS order_month,
               ROUND(SUM(order_amount), 2) AS revenue_inr
        FROM orders
        WHERE status = 'completed'
        GROUP BY substr(order_ts, 1, 7)
        ORDER BY order_month
    """, conn).set_index("order_month")["revenue_inr"]

assert pandas_monthly.index.equals(sql_monthly.index)
assert (pandas_monthly - sql_monthly).abs().max() < 0.011
print("PASS: SQL and pandas monthly revenue match")

# EDA summary
report = []

report.append("INSTAMART CUSTOMER INTELLIGENCE — PART A EDA")
report.append("=" * 48)
report.append(f"Customers: {len(customers)}")
report.append(f"Products: {len(products)}")
report.append(f"Orders: {len(orders)}")
report.append(f"Completed orders: {(orders['status'] == 'completed').sum()}")
report.append(f"Cancelled orders: {(orders['status'] == 'cancelled').sum()}")

total_orders = len(orders)
cancelled = (orders["status"] == "cancelled").sum()
report.append(f"Cancellation rate: {cancelled / total_orders * 100:.2f}%")
report.append("")
report.append("Missing values in cleaned orders:")
report.append(orders.isna().sum().to_string())
report.append("")
report.append("Revenue by category (completed orders):")

category_revenue = (
    completed.merge(products, on="product_id", how="left")
    .groupby("category")["order_amount"]
    .sum()
    .sort_values(ascending=False)
)
report.append(category_revenue.round(2).to_string())

text = "\n".join(report)
Path("outputs/part_a_eda.txt").write_text(text, encoding="utf-8")

print("PASS: EDA report saved to outputs/part_a_eda.txt")
print("\n" + text)
