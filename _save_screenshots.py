"""
_save_screenshots.py
Generates all project visualisation charts as PNG files
and saves them into docs/screenshots/.
Uses kaleido (bundled with plotly) for static image export.
"""
import json, os, sqlite3
from pathlib import Path

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

os.chdir('d:/INSTAMART_CIS_PROJECT')

OUT   = Path('docs/screenshots')
OUT.mkdir(parents=True, exist_ok=True)

W, H = 1100, 620   # consistent dimensions for every chart

def save(fig, name):
    path = OUT / name
    fig.update_layout(
        font=dict(family='Arial', size=13),
        paper_bgcolor='white',
        plot_bgcolor='white',
        margin=dict(l=60, r=40, t=60, b=60),
    )
    fig.write_image(str(path), width=W, height=H, scale=2)
    print(f'  saved  {name}')

print('=== Generating screenshots ===\n')

# ─── load data ────────────────────────────────────────────────────────────────
orders    = pd.read_csv('data/clean/orders.csv')
customers = pd.read_csv('data/clean/customers.csv')
products  = pd.read_csv('data/clean/products.csv')
segments  = pd.read_csv('outputs/customer_segments.csv')
profiles  = pd.read_csv('outputs/segment_profiles.csv')
k_metrics = pd.read_csv('outputs/kmeans_metrics.csv')
triage    = pd.read_csv('outputs/triage_results.csv')
features  = pd.read_csv('outputs/rf_top_features.csv')
retention = pd.read_csv('outputs/retention_list.csv')
eval10    = pd.read_csv('outputs/policy_eval_overlap10.csv')

orders['order_ts'] = pd.to_datetime(orders['order_ts'], errors='coerce')
completed = orders[orders['status'] == 'completed'].copy()
completed['order_month'] = completed['order_ts'].dt.strftime('%Y-%m')

agent_runs = []
with open('outputs/agent_runs.jsonl', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            try:
                agent_runs.append(json.loads(line))
            except Exception:
                pass
runs_df = pd.DataFrame(agent_runs) if agent_runs else pd.DataFrame()

# ══════════════════════════════════════════════════════════════════════════════
print('── Part A: Data Analysis ──')

# 1. Order status distribution (donut)
sc = orders['status'].value_counts().reset_index()
sc.columns = ['Status', 'Orders']
fig = px.pie(sc, names='Status', values='Orders',
             title='Order Status Distribution', hole=0.45,
             color_discrete_sequence=px.colors.qualitative.Set2)
save(fig, '01_order_status_distribution.png')

# 2. Monthly completed-order revenue (bar)
mr = completed.groupby('order_month')['order_amount'].sum().reset_index()
mr.columns = ['Month', 'Revenue (INR)']
fig = px.bar(mr, x='Month', y='Revenue (INR)',
             title='Monthly Completed-Order Revenue (Apr–Sep 2025)',
             text_auto='.2s',
             color='Revenue (INR)',
             color_continuous_scale='Blues')
fig.update_coloraxes(showscale=False)
save(fig, '02_monthly_revenue.png')

# 3. Revenue by product category (bar)
cat_rev = (
    completed.merge(products, on='product_id', how='left')
    .groupby('category')['order_amount'].sum()
    .sort_values(ascending=False).reset_index()
)
cat_rev.columns = ['Category', 'Revenue (INR)']
fig = px.bar(cat_rev, x='Category', y='Revenue (INR)',
             title='Revenue by Product Category',
             text_auto='.2s',
             color='Revenue (INR)',
             color_continuous_scale='Teal')
fig.update_coloraxes(showscale=False)
save(fig, '03_revenue_by_category.png')

# 4. Order amount distribution (histogram)
fig = px.histogram(
    orders['order_amount'].dropna(), nbins=40,
    title='Order Amount Distribution (INR)',
    labels={'value': 'Order Amount (INR)', 'count': 'Number of Orders'},
    color_discrete_sequence=['#636EFA']
)
save(fig, '04_order_amount_distribution.png')

# 5. Delivery time distribution (histogram + SLA line)
fig = px.histogram(
    completed['delivery_minutes'].dropna(), nbins=30,
    title='Delivery Time Distribution — Completed Orders',
    labels={'value': 'Delivery Minutes', 'count': 'Number of Orders'},
    color_discrete_sequence=['#EF553B']
)
fig.add_vline(x=45, line_dash='dash', line_color='navy',
              annotation_text='45-min SLA', annotation_position='top right')
save(fig, '05_delivery_time_distribution.png')

# 6. Products by category (bar)
cat_cnt = products['category'].value_counts().reset_index()
cat_cnt.columns = ['Category', 'Products']
fig = px.bar(cat_cnt, x='Category', y='Products',
             title='Products by Category', text_auto=True,
             color='Products', color_continuous_scale='Purples')
fig.update_coloraxes(showscale=False)
save(fig, '06_products_by_category.png')

# ══════════════════════════════════════════════════════════════════════════════
print('\n── Part B: Customer Intelligence ──')

# 7. K-Means inertia & silhouette dual-axis
fig = go.Figure()
fig.add_trace(go.Scatter(x=k_metrics['k'], y=k_metrics['inertia'],
                         name='Inertia', mode='lines+markers',
                         marker=dict(size=8), line=dict(color='#636EFA')))
fig.add_trace(go.Scatter(x=k_metrics['k'], y=k_metrics['silhouette'],
                         name='Silhouette', mode='lines+markers',
                         marker=dict(size=8), line=dict(color='#EF553B'),
                         yaxis='y2'))
fig.update_layout(
    title='K-Means: Inertia & Silhouette Scores (K = 2 to 8)',
    xaxis_title='Number of Clusters (K)',
    yaxis=dict(title='Inertia', color='#636EFA'),
    yaxis2=dict(title='Silhouette Score', overlaying='y',
                side='right', color='#EF553B'),
    legend=dict(x=0.6, y=0.95),
)
save(fig, '07_kmeans_inertia_silhouette.png')

# 8. Customer count by segment (bar)
seg_cnt = segments['segment_name'].fillna('Unknown').value_counts().reset_index()
seg_cnt.columns = ['Segment', 'Customers']
fig = px.bar(seg_cnt, x='Segment', y='Customers',
             title='Customer Count by Segment', text_auto=True,
             color='Segment',
             color_discrete_sequence=px.colors.qualitative.Pastel)
save(fig, '08_customer_count_by_segment.png')

# 9. Mean RFM per segment (grouped bar)
if {'segment_name','mean_recency','mean_frequency','mean_monetary'}.issubset(profiles.columns):
    prof_long = profiles.melt(
        id_vars='segment_name',
        value_vars=['mean_recency','mean_frequency','mean_monetary'],
        var_name='Metric', value_name='Mean Value'
    )
    fig = px.bar(prof_long, x='segment_name', y='Mean Value',
                 color='Metric', barmode='group',
                 title='Mean RFM Values per Segment',
                 labels={'segment_name': 'Segment'},
                 color_discrete_sequence=px.colors.qualitative.Bold)
    save(fig, '09_segment_rfm_profiles.png')

# 10. PCA cluster scatter
if {'recency','frequency','monetary','segment_name'}.issubset(segments.columns):
    numeric = segments[['recency','frequency','monetary']].apply(
        pd.to_numeric, errors='coerce')
    valid = numeric.notna().all(axis=1)
    if valid.sum() >= 3:
        scaled = StandardScaler().fit_transform(numeric.loc[valid])
        coords = PCA(n_components=2).fit_transform(scaled)
        sc_df  = segments.loc[valid].copy()
        sc_df['PC1'] = coords[:, 0]
        sc_df['PC2'] = coords[:, 1]
        fig = px.scatter(sc_df, x='PC1', y='PC2', color='segment_name',
                         opacity=0.7,
                         title='Customer Segments — PCA Projection (RFM Space)',
                         labels={'segment_name': 'Segment'},
                         color_discrete_sequence=px.colors.qualitative.Set1)
        save(fig, '10_pca_cluster_scatter.png')

# 11. Top feature importances (horizontal bar)
if not features.empty and {'feature','importance'}.issubset(features.columns):
    fig = px.bar(features.sort_values('importance'),
                 x='importance', y='feature', orientation='h',
                 title='Top Random Forest Feature Importances (Lapse Model)',
                 labels={'importance': 'Importance Score', 'feature': 'Feature'},
                 color='importance', color_continuous_scale='Oranges')
    fig.update_coloraxes(showscale=False)
    save(fig, '11_rf_feature_importances.png')

# 12. Retention list — lapse probability (bar)
if not retention.empty and 'lapse_probability' in retention.columns:
    ret = retention.copy()
    ret['lapse_probability'] = pd.to_numeric(ret['lapse_probability'], errors='coerce')
    fig = px.bar(ret.sort_values('lapse_probability', ascending=False),
                 x='customer_id', y='lapse_probability',
                 color='actually_lapsed',
                 title='Premium Customers — Predicted Lapse Probability',
                 labels={'customer_id':'Customer ID',
                         'lapse_probability':'Lapse Probability',
                         'actually_lapsed':'Actually Lapsed'},
                 color_continuous_scale='RdYlGn_r')
    save(fig, '12_retention_lapse_probability.png')

# ══════════════════════════════════════════════════════════════════════════════
print('\n── Part C: Support Ticket Triage ──')

# 13. Ticket category distribution (donut)
if 'category' in triage.columns:
    cat_counts = triage['category'].replace('', 'unclassified').fillna('unclassified').value_counts().reset_index()
    cat_counts.columns = ['Category', 'Tickets']
    fig = px.pie(cat_counts, names='Category', values='Tickets',
                 title='Ticket Category Distribution (Few-shot)',
                 hole=0.42,
                 color_discrete_sequence=px.colors.qualitative.Safe)
    save(fig, '13_ticket_category_distribution.png')

# 14. Ticket classification status (bar)
if 'status' in triage.columns:
    st_counts = triage['status'].fillna('unknown').value_counts().reset_index()
    st_counts.columns = ['Status', 'Tickets']
    fig = px.bar(st_counts, x='Status', y='Tickets', text_auto=True,
                 title='Ticket Classification Status Breakdown',
                 color='Tickets', color_continuous_scale='Tealgrn')
    fig.update_coloraxes(showscale=False)
    fig.update_layout(xaxis_tickangle=-20)
    save(fig, '14_ticket_classification_status.png')

# 15. Priority distribution (bar)
if 'priority' in triage.columns:
    pr_counts = triage['priority'].fillna('unknown').value_counts().reset_index()
    pr_counts.columns = ['Priority', 'Tickets']
    colours = {'HIGH': '#EF553B', 'NORMAL': '#636EFA', 'unknown': '#AAAAAA'}
    fig = px.bar(pr_counts, x='Priority', y='Tickets', text_auto=True,
                 title='Ticket Priority Distribution',
                 color='Priority',
                 color_discrete_map=colours)
    save(fig, '15_ticket_priority_distribution.png')

# 16. Tokens per ticket (histogram)
if 'tokens_used' in triage.columns:
    tok = pd.to_numeric(triage['tokens_used'], errors='coerce').dropna()
    fig = px.histogram(tok, nbins=25,
                       title='Token Usage per Ticket (Few-shot Run)',
                       labels={'value': 'Tokens Used', 'count': 'Tickets'},
                       color_discrete_sequence=['#AB63FA'])
    save(fig, '16_tokens_per_ticket.png')

# 17. Route distribution (bar)
if 'route' in triage.columns:
    rt_counts = triage['route'].fillna('unknown').value_counts().reset_index()
    rt_counts.columns = ['Route', 'Tickets']
    fig = px.bar(rt_counts, x='Route', y='Tickets', text_auto=True,
                 title='Ticket Routing Distribution',
                 color='Tickets', color_continuous_scale='Teal')
    fig.update_coloraxes(showscale=False)
    save(fig, '17_ticket_routing_distribution.png')

# ══════════════════════════════════════════════════════════════════════════════
print('\n── Part D: Policy Retrieval & Refund Agent ──')

# 18. Policy retrieval evaluation comparison (grouped bar)
compare = []
for label, path in [('Overlap 10','outputs/policy_eval_overlap10.csv'),
                    ('Overlap 0', 'outputs/policy_eval_overlap0.csv')]:
    p = Path(path)
    if not p.exists():
        continue
    df = pd.read_csv(p)
    n_ans = int((df['expected_doc'] != 'NONE').sum())
    n_unans = int((df['expected_doc'] == 'NONE').sum())
    rh = int(df.get('retrieval_hit', pd.Series(dtype=str)).astype(str).str.lower().eq('true').sum()) if 'retrieval_hit' in df else 0
    fh = int(df.get('fact_hit', pd.Series(dtype=str)).astype(str).str.lower().eq('true').sum()) if 'fact_hit' in df else 0
    fb = int(df.get('fallback_hit', pd.Series(dtype=str)).astype(str).str.lower().eq('true').sum()) if 'fallback_hit' in df else 0
    compare.append({'Configuration': label,
                    'Retrieval hits': rh,
                    'Fact hits': fh,
                    'Fallback hits': fb})

if compare:
    cdf = pd.DataFrame(compare)
    cdf_long = cdf.melt(id_vars='Configuration', var_name='Metric', value_name='Count')
    fig = px.bar(cdf_long, x='Configuration', y='Count',
                 color='Metric', barmode='group',
                 title='Policy Retrieval Evaluation: Overlap 10 vs Overlap 0',
                 color_discrete_sequence=px.colors.qualitative.Vivid)
    save(fig, '18_policy_retrieval_evaluation.png')

# 19. Agent scenario outcomes (bar)
if not runs_df.empty and 'scenario_id' in runs_df.columns:
    main = runs_df[runs_df['scenario_id'].isin([f'S{i}' for i in range(1,8)])].copy()
    if not main.empty:
        fig = px.bar(main, x='scenario_id', y='llm_calls', color='status',
                     title='LLM Calls per Refund Agent Scenario',
                     labels={'scenario_id': 'Scenario', 'llm_calls': 'LLM Calls'},
                     color_discrete_sequence=px.colors.qualitative.Set2)
        save(fig, '19_agent_llm_calls_per_scenario.png')

# 20. Token usage per scenario
if not runs_df.empty and 'tokens_used' in runs_df.columns:
    main = runs_df[runs_df['scenario_id'].isin([f'S{i}' for i in range(1,8)])].copy()
    if not main.empty:
        fig = px.bar(main, x='scenario_id', y='tokens_used', color='status',
                     title='Token Usage per Refund Agent Scenario',
                     labels={'scenario_id': 'Scenario', 'tokens_used': 'Tokens Used'},
                     color_discrete_sequence=px.colors.qualitative.Pastel)
        save(fig, '20_agent_token_usage_per_scenario.png')

# 21. Agent status distribution (donut)
if not runs_df.empty and 'status' in runs_df.columns:
    main = runs_df[runs_df['scenario_id'].isin([f'S{i}' for i in range(1,8)])].copy()
    if not main.empty:
        st = main['status'].value_counts().reset_index()
        st.columns = ['Status', 'Count']
        fig = px.pie(st, names='Status', values='Count',
                     title='Refund Agent Scenario Outcome Distribution',
                     hole=0.42,
                     color_discrete_sequence=px.colors.qualitative.Bold)
        save(fig, '21_agent_outcome_distribution.png')

print('\n=== Done ===')
saved = sorted(OUT.glob('*.png'))
print(f'Total PNG files saved: {len(saved)}')
for p in saved:
    print(f'  {p.name}  ({p.stat().st_size//1024} KB)')
