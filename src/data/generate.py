"""
src.data.generate
=================
Deterministic synthetic data generator for the Instamart CIS project.

Generates:
  - data/raw/customers.csv
  - data/raw/products.csv
  - data/raw/orders.csv
  - data/raw/tickets.csv
  - data/raw/policy_eval.csv
  - data/raw/agent_scenarios.csv
  - data/policies/*.txt

Uses only the Python standard library — no third-party imports.
All data is synthetic and fictional. Fixed seed (42) ensures reproducibility.

Usage
-----
    from src.data.generate import run
    run()                          # generate into default data/ folders

    # or run as a script:
    python -m src.data.generate
"""

import csv
import math
import os
import random
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path


# ── Paths ─────────────────────────────────────────────────────────────────────
def _resolve_root() -> Path:
    """Return the project root (two levels above this file)."""
    return Path(__file__).resolve().parent.parent.parent


def _dirs(root: Path):
    raw = root / "data" / "raw"
    pol = root / "data" / "policies"
    raw.mkdir(parents=True, exist_ok=True)
    pol.mkdir(parents=True, exist_ok=True)
    return raw, pol


# ── Constants ─────────────────────────────────────────────────────────────────
CITIES = ["Chennai", "Mumbai", "Bengaluru", "Delhi", "Hyderabad", "Pune"]

PRODUCTS = [
    ("Basmati Rice 1kg",      "Grocery",       129),
    ("Toor Dal 1kg",          "Grocery",       165),
    ("Atta 5kg",              "Grocery",       245),
    ("Sunflower Oil 1L",      "Grocery",       155),
    ("Sugar 1kg",             "Grocery",        48),
    ("Salt 1kg",              "Grocery",        24),
    ("Tea Powder 250g",       "Grocery",       120),
    ("Poha 500g",             "Grocery",        42),
    ("Toned Milk 500ml",      "Dairy",          29),
    ("Curd 400g",             "Dairy",          45),
    ("Paneer 200g",           "Dairy",          95),
    ("Butter 100g",           "Dairy",          58),
    ("Cheese Slices 200g",    "Dairy",         130),
    ("Ghee 500ml",            "Dairy",         340),
    ("Eggs 12 Pack",          "Dairy",          84),
    ("Lassi 200ml",           "Dairy",          25),
    ("Potato Chips 52g",      "Snacks",         20),
    ("Namkeen 200g",          "Snacks",         60),
    ("Biscuits 150g",         "Snacks",         30),
    ("Dark Chocolate 80g",    "Snacks",        150),
    ("Instant Noodles 4 Pack","Snacks",         56),
    ("Popcorn 100g",          "Snacks",         45),
    ("Peanut Butter 340g",    "Snacks",        210),
    ("Cold Coffee Can 180ml", "Snacks",         90),
    ("Shampoo 340ml",         "Personal Care", 280),
    ("Face Wash 100ml",       "Personal Care", 190),
    ("Toothpaste 150g",       "Personal Care",  98),
    ("Soap 4 Pack",           "Personal Care", 150),
    ("Hand Wash 250ml",       "Personal Care",  99),
    ("Body Lotion 400ml",     "Personal Care", 350),
    ("Deodorant 150ml",       "Personal Care", 220),
    ("Sunscreen 50g",         "Personal Care", 399),
    ("Wireless Earbuds",      "Electronics",  1499),
    ("Power Bank 10000mAh",   "Electronics",  1299),
    ("USB-C Cable 1m",        "Electronics",   299),
    ("Phone Charger 20W",     "Electronics",   799),
    ("Smart Bulb",            "Electronics",   599),
    ("Bluetooth Speaker",     "Electronics",  1999),
    ("Phone Stand",           "Electronics",   349),
    ("Memory Card 64GB",      "Electronics",   549),
]

POLICY_DOCS = {
    "refund_policy.txt": (
        "Refund Policy. A refund can be requested only for orders whose status is completed. "
        "Orders with status cancelled were never charged, so no refund applies to them. "
        "If a completed order is delivered in more than 45 minutes, the customer receives a "
        "full refund of the order amount. If a completed order is delivered in 45 minutes or "
        "less, no late-delivery refund is paid. Refunds up to INR 1000 are approved "
        "automatically by the support system. A refund above INR 1000 must be approved by a "
        "human support lead before money is returned. Approved refunds are credited to the "
        "original payment method within 5 working days. A refund request must be raised "
        "within 7 days of the delivery date."
    ),
    "delivery_policy.txt": (
        "Delivery Policy. Standard delivery is promised within 30 minutes of order "
        "confirmation in all supported cities. The supported cities are Chennai, Mumbai, "
        "Bengaluru, Delhi, Hyderabad and Pune. A delivery fee of INR 25 is charged on orders "
        "below INR 199, and delivery is free on orders of INR 199 or more. During heavy rain "
        "the promised time may extend to 60 minutes and the app shows a rain notice. A rider "
        "will attempt to contact the customer twice before marking an order as undeliverable. "
        "Electronics items are delivered only after the customer shares a one time password "
        "with the rider."
    ),
    "payment_policy.txt": (
        "Payment Policy. Customers can pay using UPI, debit card, credit card or cash on "
        "delivery. Cash on delivery is available only for orders up to INR 2500. If a payment "
        "fails but money is deducted, the amount is returned automatically within 48 hours. A "
        "customer who sees two charges for one order should raise a billing ticket with the "
        "order identifier. Wallet balance can be used for any order and is never refunded as "
        "cash. Coupons cannot be combined with wallet balance on the same order."
    ),
    "privacy_policy.txt": (
        "Privacy Policy. Customer phone numbers and email addresses are used only for order "
        "updates and support. Support tickets are processed with personal details masked before "
        "any automated tool reads them. Delivery addresses are deleted from active systems "
        "180 days after the last order. A customer can ask for account deletion and the "
        "request is completed within 30 days. Payment card numbers are never stored by the "
        "support system."
    ),
    "escalation_policy.txt": (
        "Escalation Policy. A ticket about fraud or an unauthorized transaction is marked high "
        "priority and sent to the security team within 2 hours. Tickets from premium customers "
        "are answered within 1 hour, and all other tickets within 24 hours. If an automated "
        "assistant cannot resolve a ticket within 6 steps, it must hand the case to a human "
        "agent with a short summary. A human agent must call the customer back if such a "
        "high-value request is rejected."
    ),
}

POLICY_QA = [
    ("What delivery time makes a completed order eligible for a full refund?",
     "refund_policy.txt",   "more than 45 minutes"),
    ("Who must approve a refund above INR 1000?",
     "refund_policy.txt",   "human support lead"),
    ("How many days do I have to raise a refund request?",
     "refund_policy.txt",   "within 7 days"),
    ("What is the delivery fee on small orders?",
     "delivery_policy.txt", "INR 25"),
    ("On orders of what value is delivery free?",
     "delivery_policy.txt", "INR 199 or more"),
    ("Why does the electronics delivery need a one time password?",
     "delivery_policy.txt", "one time password"),
    ("Is cash on delivery available for a large order?",
     "payment_policy.txt",  "INR 2500"),
    ("How soon is money returned when a payment fails but is deducted?",
     "payment_policy.txt",  "48 hours"),
    ("Can wallet balance be refunded as cash?",
     "payment_policy.txt",  "never refunded as cash"),
    ("When are delivery addresses deleted from active systems?",
     "privacy_policy.txt",  "180 days"),
    ("How fast must a fraud ticket reach the security team?",
     "escalation_policy.txt", "within 2 hours"),
    ("After how many steps must an automated assistant hand over to a human?",
     "escalation_policy.txt", "6 steps"),
    ("What is the favourite colour of the company mascot?", "NONE", ""),
    ("Who won the cricket world cup final?",                "NONE", ""),
]

TICKET_TEMPLATES = {
    "billing": [
        "I was charged twice for order {o}. Please refund the extra payment. My email is {e}.",
        "The invoice for {o} shows a wrong amount. Call me on {p} to correct the bill.",
        "Extra money was deducted for order {o}, please check the bill. Contact {e}.",
    ],
    "delivery": [
        "My order {o} has not arrived and the rider is not picking up calls. Please call {p}.",
        "Delivery for {o} is very late, it was promised in 20 minutes. Write to {e}.",
        "The rider marked {o} as delivered but I never received the package. Call {p}.",
    ],
    "technical": [
        "The app crashes every time I open the cart. I tried to place {o} again and got an error. Email {e}.",
        "I cannot login, the OTP never arrives on {p}.",
        "The payment page shows an error and the app freezes when I apply a coupon. Order {o}.",
    ],
    "fraud": [
        "There is an unauthorized transaction on my card linked to order {o}. I did not place it. Call {p}.",
        "Someone hacked my account and used my wallet balance. This is fraud, contact {e}.",
        "I was charged for order {o} that I never made, it looks like fraud on my account. Call {p}.",
    ],
}


# ── Internal helpers ──────────────────────────────────────────────────────────
def _poisson(rng: random.Random, lam: float) -> int:
    L, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rng.random()
        if p <= L:
            return k
        k += 1


def _mangle_city(rng: random.Random, city: str) -> str:
    r = rng.random()
    if r < 0.20:  return city.lower()
    if r < 0.32:  return city.upper()
    if r < 0.42:  return f" {city} "
    return city


def _mangle_gender(rng: random.Random, gender: str) -> str:
    opts = {
        "male":   ["male", "Male", "M", "m", "MALE"],
        "female": ["female", "Female", "F", "f", "FEMALE"],
    }
    return rng.choice(opts[gender])


def _write_csv(path: Path, header: list, rows: list) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


# ── Public API ────────────────────────────────────────────────────────────────
def run(root: Path | None = None) -> None:
    """Generate all synthetic data and write to disk."""
    if root is None:
        root = _resolve_root()
    raw, pol = _dirs(root)

    rng = random.Random(42)
    cat_idx: dict[str, list[int]] = {}
    for i, (_, c, _) in enumerate(PRODUCTS):
        cat_idx.setdefault(c, []).append(i)

    # ── products ──────────────────────────────────────────────────────────────
    _write_csv(
        raw / "products.csv",
        ["product_id", "product_name", "category", "price_inr"],
        [[f"P{i+1:03d}", n, c, p] for i, (n, c, p) in enumerate(PRODUCTS)],
    )

    # ── customers ─────────────────────────────────────────────────────────────
    N_CUST = 1200
    IMPOSSIBLE_AGES = {10:280, 110:-4, 210:0, 310:150, 410:999, 510:200}
    cust_rows: list = []
    archetype: dict[str, str] = {}

    for i in range(N_CUST):
        cid = f"C{i+1:04d}"
        r   = rng.random()
        archetype[cid] = "premium" if r < 0.20 else "frequent" if r < 0.60 else "rare"
        signup = datetime(2024, 1, 1) + timedelta(days=rng.randint(0, 455))
        age    = IMPOSSIBLE_AGES.get(i, rng.randint(18, 60))
        age    = "" if (i not in IMPOSSIBLE_AGES and rng.random() < 0.05) else age
        gender = "" if rng.random() < 0.03 else _mangle_gender(rng, rng.choice(["male", "female"]))
        cust_rows.append([cid, _mangle_city(rng, rng.choice(CITIES)),
                          signup.strftime("%Y-%m-%d"), age, gender])
    _write_csv(raw / "customers.csv",
               ["customer_id", "city", "signup_date", "age", "gender"], cust_rows)

    # ── orders ────────────────────────────────────────────────────────────────
    WIN_START, WIN_DAYS = datetime(2025, 4, 1), 183
    RATE = {"premium": 1.0, "frequent": 3.5, "rare": 0.5}

    def pick_product(arch: str):
        if arch == "premium":
            pool = cat_idx["Electronics"] if rng.random() < 0.7 \
                   else cat_idx["Personal Care"] + cat_idx["Snacks"]
            return rng.choice(pool), rng.randint(1, 2)
        if arch == "frequent":
            return rng.choice(cat_idx["Grocery"] + cat_idx["Dairy"] + cat_idx["Snacks"]), rng.randint(1, 3)
        return rng.choice(cat_idx["Grocery"] + cat_idx["Dairy"] +
                          cat_idx["Snacks"] + cat_idx["Personal Care"]), rng.randint(1, 2)

    raw_orders: list = []
    oid = 100000
    for cid, arch in archetype.items():
        for _ in range(_poisson(rng, RATE[arch] * 6.0)):
            oid += 1
            pi, qty = pick_product(arch)
            ts   = WIN_START + timedelta(
                days=rng.randint(0, WIN_DAYS - 1),
                hours=rng.randint(7, 22),
                minutes=rng.randint(0, 59),
                seconds=rng.randint(0, 59),
            )
            disc   = rng.choice([0, 0, 0, 5, 10, 15, 20, 25, 40])
            status = "cancelled" if rng.random() < 0.10 else "completed"
            amount = round(PRODUCTS[pi][2] * qty * (1 - disc / 100), 2)
            deliv  = ""
            if status != "cancelled":
                deliv = (rng.randint(46, 75) if rng.random() < 0.10
                         else max(8, min(45, int(round(rng.gauss(24, 8))))))
                if rng.random() < 0.06:
                    deliv = ""
            raw_orders.append({
                "order_id":         f"ORD-{oid}",
                "customer_id":      cid,
                "product_id":       f"P{pi+1:03d}",
                "quantity":         qty,
                "order_ts":         ts,
                "status":           status,
                "discount_pct":     disc,
                "order_amount":     amount,
                "delivery_minutes": deliv,
            })

    out_rows: list = []
    for k, r in enumerate(raw_orders):
        if k % 100 == 7 and r["status"] == "completed" and r["delivery_minutes"] != "":
            r["delivery_minutes"] = -10
        if r["status"] == "completed" and rng.random() < 0.01:
            r["order_amount"] = "N/A" if rng.random() < 0.5 else "error"
        base = [r["order_id"], r["customer_id"], r["product_id"], r["quantity"],
                r["order_ts"].strftime("%Y-%m-%d %H:%M:%S"),
                r["status"], r["discount_pct"], r["order_amount"], r["delivery_minutes"]]
        out_rows.append(base)
        if rng.random() < 0.05:
            early = list(base)
            early[4] = (r["order_ts"] - timedelta(minutes=rng.randint(1, 10))).strftime("%Y-%m-%d %H:%M:%S")
            early[5] = "pending"; early[8] = ""
            out_rows.append(early)
        if rng.random() < 0.01:
            out_rows.append(list(base))

    rng.shuffle(out_rows)
    _write_csv(raw / "orders.csv",
               ["order_id","customer_id","product_id","quantity","order_ts",
                "status","discount_pct","order_amount","delivery_minutes"], out_rows)

    # ── support tickets ───────────────────────────────────────────────────────
    order_by_cust: dict[str, str] = {}
    for r in out_rows:
        order_by_cust.setdefault(r[1], r[0])
    cust_with_orders = sorted(order_by_cust)

    tickets: list = []
    cats = list(TICKET_TEMPLATES)
    for n in range(1, 157):
        cat = cats[(n - 1) % 4]
        cid = rng.choice(cust_with_orders)
        txt = rng.choice(TICKET_TEMPLATES[cat]).format(
            o=order_by_cust[cid],
            e=f"user{n}@example.com",
            p=str(rng.choice("6789")) + "".join(rng.choice("0123456789") for _ in range(9)),
        )
        tickets.append([f"T{n:04d}", cid, order_by_cust[cid], txt, cat])

    FILL = "The same problem keeps happening and nobody has given me a clear answer yet. "
    for n in range(157, 161):
        tickets.append([f"T{n:04d}", rng.choice(cust_with_orders), "", "", "none"])
    for n in range(161, 165):
        tickets.append([f"T{n:04d}", rng.choice(cust_with_orders), "",
                        ("My delivery was late. " + FILL * 40).strip(), "delivery"])
    rng.shuffle(tickets)
    _write_csv(raw / "tickets.csv",
               ["ticket_id","customer_id","order_id","text","true_category"], tickets)

    # ── policy documents ──────────────────────────────────────────────────────
    for fname, body in POLICY_DOCS.items():
        (pol / fname).write_text(body + "\n", encoding="utf-8")

    # ── policy evaluation questions ───────────────────────────────────────────
    _write_csv(raw / "policy_eval.csv",
               ["question","expected_doc","expected_fact"], POLICY_QA)

    # ── agent scenarios ───────────────────────────────────────────────────────
    cnt = Counter(r[0] for r in out_rows)

    def clean_row(r) -> bool:
        try:
            return (cnt[r[0]] == 1
                    and r[5] in ("completed", "cancelled")
                    and float(r[7]) > 0
                    and (r[5] == "cancelled" or (r[8] != "" and 8 <= int(r[8]) <= 75)))
        except (ValueError, TypeError):
            return False

    ok_rows = sorted((r for r in out_rows if clean_row(r)), key=lambda r: r[0])

    def first(pred):
        return next(r for r in ok_rows if pred(r))

    late_small = [r for r in ok_rows
                  if r[5] == "completed" and int(r[8]) > 45 and float(r[7]) <= 1000][:2]
    late_big   = first(lambda r: r[5] == "completed" and int(r[8]) > 45 and float(r[7]) > 1000)
    on_time    = first(lambda r: r[5] == "completed" and int(r[8]) <= 45)
    cancelled  = first(lambda r: r[5] == "cancelled")

    sc: list = []
    for r in late_small:
        sc.append((r[1], f"My order {r[0]} reached me very late. I would like a refund.", "REFUND", float(r[7])))
    sc.append((late_big[1], f"Order {late_big[0]} arrived far too late, please refund me.", "ESCALATE", float(late_big[7])))
    sc.append((on_time[1],  f"Please refund order {on_time[0]}, I did not like the delivery.", "DENIED", 0))
    sc.append((cancelled[1],f"I want my money back for order {cancelled[0]}.", "DENIED", 0))
    sc.append(("C0001",     "Refund for order ORD-999999 please.", "NOT_FOUND", 0))
    sc.append((late_small[0][1], f"[LOOP] Refund order {late_small[0][0]}.", "FALLBACK", 0))

    with (raw / "agent_scenarios.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scenario_id","customer_id","message","expected_outcome","expected_refund_inr"])
        for i, (cid, msg, oc, amt) in enumerate(sc, 1):
            w.writerow([f"S{i}", cid, msg, oc, f"{amt:.2f}"])

    print(f"generated: customers={len(cust_rows)} products={len(PRODUCTS)} "
          f"orders_raw={len(out_rows)} tickets={len(tickets)}")


if __name__ == "__main__":
    run()
