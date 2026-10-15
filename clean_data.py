from pathlib import Path
import pandas as pd

RAW = Path("data/raw")
CLEAN = Path("data/clean")
CLEAN.mkdir(parents=True, exist_ok=True)

# 1. CUSTOMERS
customers = pd.read_csv(RAW / "customers.csv")

customers["city"] = (
    customers["city"].astype("string").str.strip().str.title()
)
customers["age"] = pd.to_numeric(customers["age"], errors="coerce")
customers.loc[~customers["age"].between(0, 120), "age"] = pd.NA

customers["gender"] = (
    customers["gender"].astype("string").str.strip().str.lower()
)
gender_map = {
    "f": "female",
    "female": "female",
    "m": "male",
    "male": "male",
}
customers["gender"] = customers["gender"].map(gender_map).astype("string")

customers["signup_date"] = pd.to_datetime(
    customers["signup_date"], errors="coerce"
)

customers = customers.drop_duplicates(subset="customer_id", keep="first")
customers.to_csv(CLEAN / "customers.csv", index=False)

# 2. PRODUCTS
products = pd.read_csv(RAW / "products.csv")
products["price_inr"] = pd.to_numeric(
    products["price_inr"], errors="coerce"
)
products = products.drop_duplicates(subset="product_id", keep="first")
products.to_csv(CLEAN / "products.csv", index=False)

# 3. ORDERS
orders = pd.read_csv(RAW / "orders.csv", dtype=str)

# Remove only rows that are identical across every column.
orders = orders.drop_duplicates()

orders["order_ts"] = pd.to_datetime(orders["order_ts"], errors="coerce")
orders["quantity"] = pd.to_numeric(orders["quantity"], errors="coerce")
orders["discount_pct"] = pd.to_numeric(
    orders["discount_pct"], errors="coerce"
)
orders["order_amount"] = pd.to_numeric(
    orders["order_amount"].str.strip(), errors="coerce"
)
orders["delivery_minutes"] = pd.to_numeric(
    orders["delivery_minutes"], errors="coerce"
)

# Negative delivery durations are invalid; keep them missing, not zero.
orders.loc[orders["delivery_minutes"] < 0, "delivery_minutes"] = pd.NA

# Assumption: order_ts is the timestamp of each order-status record.
# Keep the latest timestamp as the final record for each order_id.
orders = orders.sort_values(
    ["order_id", "order_ts"], kind="stable", na_position="first"
)
orders = orders.drop_duplicates(subset="order_id", keep="last")

orders.to_csv(CLEAN / "orders.csv", index=False)

# Basic validation and summary
print("=== CLEANING SUMMARY ===")
print(f"Customers: {len(customers)}")
print(f"Products: {len(products)}")
print(f"Orders (unique order IDs): {len(orders)}")
print(f"Duplicate customer IDs: {customers['customer_id'].duplicated().sum()}")
print(f"Duplicate product IDs: {products['product_id'].duplicated().sum()}")
print(f"Duplicate order IDs: {orders['order_id'].duplicated().sum()}")
print(f"Missing order amounts: {orders['order_amount'].isna().sum()}")
print(f"Missing delivery times: {orders['delivery_minutes'].isna().sum()}")
print("Created data/clean/customers.csv, products.csv, orders.csv")
