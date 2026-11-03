from pathlib import Path
import json
import sqlite3

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "data" / "instamart.db"
OUT = ROOT / "outputs"

st.set_page_config(
    page_title="Instamart Customer Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
[data-testid="stMetric"] {
    background: rgba(120, 120, 120, 0.07);
    border: 1px solid rgba(120, 120, 120, 0.18);
    padding: 16px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_database():
    with sqlite3.connect(DB_PATH) as con:
        customers = pd.read_sql_query("SELECT * FROM customers", con)
        orders = pd.read_sql_query("SELECT * FROM orders", con)
        products = pd.read_sql_query("SELECT * FROM products", con)
    orders["order_ts"] = pd.to_datetime(orders["order_ts"], errors="coerce")
    return customers, orders, products


@st.cache_data
def load_csv(filename):
    path = OUT / filename
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_jsonl(filename):
    path = OUT / filename
    if not path.exists():
        return pd.DataFrame()
    rows = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except (json.JSONDecodeError, ValueError):
                continue
    return pd.json_normalize(rows) if rows else pd.DataFrame()


def money(value):
    return f"₹{value:,.2f}"


def show_empty(message):
    st.info(message)


def section_title(title, description=None):
    st.subheader(title)
    if description:
        st.caption(description)


if not DB_PATH.exists():
    st.error(f"Database not found: {DB_PATH}")
    st.stop()

customers, orders, products = load_database()
segments = load_csv("customer_segments.csv")
profiles = load_csv("segment_profiles.csv")
metrics = load_csv("kmeans_metrics.csv")
retention = load_csv("retention_list.csv")
features = load_csv("rf_top_features.csv")
triage = load_csv("triage_results.csv")
budget = load_csv("triage_budget_demo.csv")
eval10 = load_csv("policy_eval_overlap10.csv")
eval0 = load_csv("policy_eval_overlap0.csv")
agent_runs = load_jsonl("agent_runs.jsonl")
ledger = load_jsonl("refund_ledger.jsonl")

st.sidebar.title("Instamart CIS")
st.sidebar.caption("Customer Intelligence & Support Copilot")
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Part A · Data Analysis", "Part B · Customer Intelligence",
     "Part C · Support Intelligence", "Part D · Policy & Agent"],
)
st.sidebar.divider()
st.sidebar.caption("Offline project dashboard")
st.sidebar.caption("Source: local SQLite database and project outputs")

# Common filtered order data
st.sidebar.subheader("Order filters")
statuses = sorted(orders["status"].dropna().astype(str).unique())
selected_statuses = st.sidebar.multiselect(
    "Order status", statuses, default=statuses
)
filtered_orders = orders[orders["status"].astype(str).isin(selected_statuses)]
joined_orders = filtered_orders.merge(
    products, on="product_id", how="left", suffixes=("", "_product")
)

if page == "Overview":
    st.title("Instamart Customer Intelligence")
    st.markdown(
        "An interactive view of customer analytics, segmentation, support "
        "triage, policy retrieval and refund-agent evaluation."
    )

    order_count = len(filtered_orders)
    customer_count = customers["customer_id"].nunique()
    product_count = products["product_id"].nunique()
    order_value = pd.to_numeric(
        filtered_orders["order_amount"], errors="coerce"
    ).sum()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Customers", f"{customer_count:,}")
    k2.metric("Filtered orders", f"{order_count:,}")
    k3.metric("Products", f"{product_count:,}")
    k4.metric("Filtered order value", money(order_value))

    st.divider()
    left, right = st.columns(2)

    with left:
        section_title("Order status distribution")
        status_df = (
            filtered_orders["status"].fillna("Unknown")
            .value_counts().rename_axis("Status")
            .reset_index(name="Orders")
        )
        if not status_df.empty:
            fig = px.pie(status_df, names="Status", values="Orders", hole=0.48)
            st.plotly_chart(fig, use_container_width=True)
        else:
            show_empty("No orders match the selected filters.")

    with right:
        section_title("Customer segments")
        if not segments.empty and "segment_name" in segments:
            seg_df = segments["segment_name"].fillna("Unknown").value_counts()
            fig = px.bar(
                x=seg_df.index, y=seg_df.values,
                labels={"x": "Segment", "y": "Customers"},
                text=seg_df.values,
            )
            fig.update_layout(xaxis_tickangle=-20)
            st.plotly_chart(fig, use_container_width=True)
        else:
            show_empty("Customer segmentation output is unavailable.")

    st.divider()
    left, right = st.columns(2)
    with left:
        section_title("Order value over time")
        trend = filtered_orders.dropna(subset=["order_ts"]).copy()
        if not trend.empty:
            trend["date"] = trend["order_ts"].dt.date
            daily = trend.groupby("date", as_index=False)["order_amount"].sum()
            fig = px.line(
                daily, x="date", y="order_amount", markers=True,
                labels={"date": "Date", "order_amount": "Order value (INR)"},
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            show_empty("No valid order timestamps are available.")

    with right:
        section_title("Support ticket outcomes")
        if not triage.empty and "status" in triage:
            counts = triage["status"].fillna("Unknown").value_counts()
            fig = px.bar(
                x=counts.index, y=counts.values,
                labels={"x": "Outcome", "y": "Tickets"},
                text=counts.values,
            )
            fig.update_layout(xaxis_tickangle=-20)
            st.plotly_chart(fig, use_container_width=True)
        else:
            show_empty("Ticket triage output is unavailable.")

    st.caption(
        "Order KPIs use the selected status filter. Customer and product totals "
        "represent the full source tables."
    )

elif page == "Part A · Data Analysis":
    st.title("Part A — Data Analysis")
    st.caption("Database-backed exploration of customers, orders and products.")

    a, b, c, d = st.columns(4)
    a.metric("Customers", f"{customers['customer_id'].nunique():,}")
    b.metric("Orders", f"{len(filtered_orders):,}")
    c.metric("Products", f"{products['product_id'].nunique():,}")
    d.metric(
        "Average filtered order",
        money(pd.to_numeric(filtered_orders["order_amount"], errors="coerce").mean())
        if not filtered_orders.empty else "₹0.00",
    )

    left, right = st.columns(2)
    with left:
        section_title("Products by category")
        cat = products["category"].fillna("Unknown").value_counts()
        fig = px.bar(
            x=cat.index, y=cat.values,
            labels={"x": "Category", "y": "Products"}, text=cat.values,
        )
        fig.update_layout(xaxis_tickangle=-25)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        section_title("Order status")
        counts = filtered_orders["status"].fillna("Unknown").value_counts()
        fig = px.bar(
            x=counts.index, y=counts.values,
            labels={"x": "Status", "y": "Orders"}, text=counts.values,
        )
        st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)
    with left:
        section_title("Order amount distribution")
        amounts = pd.to_numeric(filtered_orders["order_amount"], errors="coerce").dropna()
        if not amounts.empty:
            fig = px.histogram(
                x=amounts, nbins=30,
                labels={"x": "Order amount (INR)", "y": "Number of orders"},
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            show_empty("No order amounts to plot.")

    with right:
        section_title("Top product categories by order value")
        if not joined_orders.empty:
            category_value = (
                joined_orders.assign(category=joined_orders["category"].fillna("Unknown"))
                .groupby("category", as_index=False)["order_amount"].sum()
                .sort_values("order_amount", ascending=False)
            )
            fig = px.bar(
                category_value, x="category", y="order_amount",
                labels={"category": "Category", "order_amount": "Order value (INR)"},
            )
            fig.update_layout(xaxis_tickangle=-25)
            st.plotly_chart(fig, use_container_width=True)
        else:
            show_empty("No matching orders.")

    section_title("Recent orders", "Use the sidebar to filter by order status.")
    st.dataframe(
        filtered_orders.sort_values("order_ts", ascending=False).head(100),
        use_container_width=True, hide_index=True,
    )
    st.download_button(
        "Download filtered orders CSV",
        filtered_orders.to_csv(index=False).encode("utf-8"),
        file_name="instamart_filtered_orders.csv",
        mime="text/csv",
    )

elif page == "Part B · Customer Intelligence":
    st.title("Part B — Customer Intelligence")
    st.caption("Customer clusters, segment characteristics and lapse-model outputs.")

    if segments.empty:
        show_empty("customer_segments.csv was not found.")
    else:
        n_customers = segments["customer_id"].nunique()
        a, b, c = st.columns(3)
        a.metric("Segmented customers", f"{n_customers:,}")
        b.metric("Segments", f"{segments['segment_id'].nunique()}")
        c.metric(
            "Premium customers",
            f"{int(segments['is_premium'].astype(str).str.lower().eq('true').sum()):,}"
            if "is_premium" in segments else "N/A",
        )

        left, right = st.columns(2)
        with left:
            section_title("Customer count by segment")
            counts = segments["segment_name"].fillna("Unknown").value_counts()
            fig = px.bar(
                x=counts.index, y=counts.values,
                labels={"x": "Segment", "y": "Customers"},
                text=counts.values,
            )
            fig.update_layout(xaxis_tickangle=-25)
            st.plotly_chart(fig, use_container_width=True)

        with right:
            section_title("K-Means model metrics")
            if not metrics.empty and {"k", "silhouette"}.issubset(metrics.columns):
                fig = px.line(
                    metrics, x="k", y="silhouette", markers=True,
                    labels={"k": "Number of clusters (k)", "silhouette": "Silhouette score"},
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                show_empty("K-Means metrics output is unavailable.")

        section_title(
            "Customer cluster map (PCA)",
            "PCA reduces recency, frequency and monetary values to two visual dimensions. "
            "The points are projections of your actual customer records.",
        )
        required = {"recency", "frequency", "monetary", "segment_name"}
        if required.issubset(segments.columns) and len(segments) >= 3:
            numeric = segments[["recency", "frequency", "monetary"]].apply(
                pd.to_numeric, errors="coerce"
            )
            valid = numeric.notna().all(axis=1)
            if valid.sum() >= 3:
                scaled = StandardScaler().fit_transform(numeric.loc[valid])
                coords = PCA(n_components=2).fit_transform(scaled)
                scatter = segments.loc[valid].copy()
                scatter["PCA 1"] = coords[:, 0]
                scatter["PCA 2"] = coords[:, 1]
                fig = px.scatter(
                    scatter, x="PCA 1", y="PCA 2", color="segment_name",
                    hover_data=["customer_id", "recency", "frequency", "monetary"],
                    opacity=0.75,
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                show_empty("Not enough complete RFM records for PCA.")
        else:
            show_empty("RFM columns are missing from the segment output.")

        section_title("Segment profiles")
        if not profiles.empty:
            st.dataframe(profiles, use_container_width=True, hide_index=True)
            if {"segment_name", "mean_recency", "mean_frequency", "mean_monetary"}.issubset(profiles.columns):
                profile_long = profiles.melt(
                    id_vars="segment_name",
                    value_vars=["mean_recency", "mean_frequency", "mean_monetary"],
                    var_name="Measure", value_name="Mean value",
                )
                fig = px.bar(
                    profile_long, x="segment_name", y="Mean value", color="Measure",
                    barmode="group",
                )
                fig.update_layout(xaxis_tickangle=-20)
                st.plotly_chart(fig, use_container_width=True)

        left, right = st.columns(2)
        with left:
            section_title("Top lapse-model features")
            if not features.empty:
                fig = px.bar(
                    features.sort_values("importance"),
                    x="importance", y="feature", orientation="h",
                    labels={"importance": "Importance", "feature": "Feature"},
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                show_empty("Feature-importance output is unavailable.")
        with right:
            section_title("Retention priority list")
            if not retention.empty:
                st.dataframe(retention, use_container_width=True, hide_index=True)
            else:
                show_empty("Retention list is unavailable.")

elif page == "Part C · Support Intelligence":
    st.title("Part C — Support Intelligence")
    st.caption("Ticket routing, classification outcomes and token-budget behaviour.")

    if triage.empty:
        show_empty("triage_results.csv was not found.")
    else:
        a, b, c, d = st.columns(4)
        a.metric("Tickets", f"{len(triage):,}")
        b.metric(
            "High priority",
            f"{triage['priority'].astype(str).str.upper().eq('HIGH').sum():,}"
            if "priority" in triage else "N/A",
        )
        c.metric(
            "Tokens used",
            f"{pd.to_numeric(triage['tokens_used'], errors='coerce').sum():,.0f}"
            if "tokens_used" in triage else "N/A",
        )
        d.metric(
            "Flagged / skipped",
            f"{triage['status'].astype(str).str.contains('flagged|skipped', case=False, regex=True).sum():,}"
            if "status" in triage else "N/A",
        )

        left, right = st.columns(2)
        with left:
            section_title("Ticket category")
            if "category" in triage:
                counts = triage["category"].fillna("Unknown").value_counts()
                fig = px.pie(
                    names=counts.index, values=counts.values, hole=0.4,
                )
                st.plotly_chart(fig, use_container_width=True)
        with right:
            section_title("Classification status")
            counts = triage["status"].fillna("Unknown").value_counts()
            fig = px.bar(
                x=counts.index, y=counts.values,
                labels={"x": "Status", "y": "Tickets"}, text=counts.values,
            )
            fig.update_layout(xaxis_tickangle=-25)
            st.plotly_chart(fig, use_container_width=True)

        left, right = st.columns(2)
        with left:
            section_title("Priority distribution")
            if "priority" in triage:
                counts = triage["priority"].fillna("Unknown").value_counts()
                fig = px.bar(
                    x=counts.index, y=counts.values,
                    labels={"x": "Priority", "y": "Tickets"}, text=counts.values,
                )
                st.plotly_chart(fig, use_container_width=True)
        with right:
            section_title("Tokens per ticket")
            if "tokens_used" in triage:
                token_df = triage.copy()
                token_df["tokens_used"] = pd.to_numeric(token_df["tokens_used"], errors="coerce")
                fig = px.histogram(token_df, x="tokens_used", nbins=25)
                st.plotly_chart(fig, use_container_width=True)

        section_title("Budget demo vs normal run")
        if not budget.empty and "tokens_used" in budget:
            comparison = pd.DataFrame({
                "Run": ["Normal run", "Budget demo"],
                "Tickets": [len(triage), len(budget)],
                "Tokens": [
                    pd.to_numeric(triage.get("tokens_used", pd.Series(dtype=float)), errors="coerce").sum(),
                    pd.to_numeric(budget.get("tokens_used", pd.Series(dtype=float)), errors="coerce").sum(),
                ],
            })
            st.dataframe(comparison, use_container_width=True, hide_index=True)
            fig = px.bar(comparison, x="Run", y="Tickets", text="Tickets")
            st.plotly_chart(fig, use_container_width=True)

        section_title("Ticket records")
        category_filter = st.multiselect(
            "Filter categories",
            sorted(triage["category"].dropna().astype(str).unique())
            if "category" in triage else [],
        )
        view = triage.copy()
        if category_filter and "category" in view:
            view = view[view["category"].astype(str).isin(category_filter)]
        st.dataframe(view, use_container_width=True, hide_index=True)
        st.download_button(
            "Download ticket results CSV",
            view.to_csv(index=False).encode("utf-8"),
            file_name="instamart_ticket_results.csv",
            mime="text/csv",
        )

elif page == "Part D · Policy & Agent":
    st.title("Part D — Policy RAG & Refund Agent")
    st.caption("Retrieval evaluation, agent scenario outcomes, tool usage and audit records.")

    if not eval10.empty:
        total = len(eval10)
        doc_hits = int(eval10["retrieval_hit"].astype(str).str.lower().eq("true").sum()) if "retrieval_hit" in eval10 else 0
        fact_hits = int(eval10["fact_hit"].astype(str).str.lower().eq("true").sum()) if "fact_hit" in eval10 else 0
        a, b, c = st.columns(3)
        a.metric("Evaluation questions", f"{total}")
        b.metric("Retrieval hits", f"{doc_hits}/{total}")
        c.metric("Expected facts found", f"{fact_hits}/{total}")

        section_title("Policy retrieval evaluation")
        compare_rows = []
        for label, df in [("Overlap 10", eval10), ("Overlap 0", eval0)]:
            if df.empty:
                continue
            n = len(df)
            compare_rows.append({
                "Configuration": label,
                "Questions": n,
                "Retrieval hits": int(df["retrieval_hit"].astype(str).str.lower().eq("true").sum()) if "retrieval_hit" in df else 0,
                "Expected fact hits": int(df["fact_hit"].astype(str).str.lower().eq("true").sum()) if "fact_hit" in df else 0,
                "Fallback hits": int(df["fallback_hit"].astype(str).str.lower().eq("true").sum()) if "fallback_hit" in df else 0,
            })
        if compare_rows:
            compare_df = pd.DataFrame(compare_rows)
            st.dataframe(compare_df, use_container_width=True, hide_index=True)
            fig = go.Figure()
            for col in ["Retrieval hits", "Expected fact hits", "Fallback hits"]:
                fig.add_trace(go.Bar(name=col, x=compare_df["Configuration"], y=compare_df[col]))
            fig.update_layout(barmode="group", yaxis_title="Questions")
            st.plotly_chart(fig, use_container_width=True)

        section_title("Inspect policy evaluation rows")
        st.dataframe(eval10, use_container_width=True, hide_index=True)
    else:
        show_empty("Policy evaluation outputs are unavailable.")

    st.divider()
    section_title("Refund agent scenarios")
    if not agent_runs.empty:
        st.metric("Logged agent runs", f"{len(agent_runs):,}")
        if "status" in agent_runs:
            counts = agent_runs["status"].fillna("Unknown").value_counts()
            fig = px.pie(names=counts.index, values=counts.values, hole=0.4)
            st.plotly_chart(fig, use_container_width=True)
        if "llm_calls" in agent_runs:
            agent_runs["llm_calls"] = pd.to_numeric(agent_runs["llm_calls"], errors="coerce")
            fig = px.bar(
                agent_runs, x="scenario_id", y="llm_calls", color="status",
                labels={"scenario_id": "Scenario", "llm_calls": "LLM calls"},
            )
            st.plotly_chart(fig, use_container_width=True)
        st.dataframe(agent_runs, use_container_width=True, hide_index=True)
    else:
        show_empty("agent_runs.jsonl is missing or contains no valid records.")

    section_title("Refund / escalation ledger")
    if not ledger.empty:
        st.dataframe(ledger, use_container_width=True, hide_index=True)
    else:
        show_empty("No refund ledger records were found.")

    section_title("Part D implementation evidence")
    st.markdown("""
    - Policy retrieval evaluated with overlapping and non-overlapping chunks.
    - Refund decisions depend on order status and delivery-time policy.
    - Large refunds require human review.
    - Agent runs record outcomes, tool activity and token usage.
    - Checkpoint and resume behaviour is implemented in the project agent.
    """)
