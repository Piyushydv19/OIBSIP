"""
utils/visualizations.py
========================
Reusable Plotly chart factory functions.
Each function returns a go.Figure ready for st.plotly_chart().
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# ── Colour palette ────────────────────────────
PALETTE  = px.colors.qualitative.Vivid
BG_COLOR = 'rgba(0,0,0,0)'   # transparent background for dark/light theme compat


def _base_layout(fig: go.Figure, title: str = '', height: int = 420) -> go.Figure:
    fig.update_layout(
        title=title,
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        height=height,
        margin=dict(t=50, b=40, l=40, r=20),
        font=dict(family='Inter, sans-serif', size=12),
    )
    return fig


# ──────────────────────────────────────────────
# CATEGORY CHARTS
# ──────────────────────────────────────────────

def top_categories_bar(df: pd.DataFrame, n: int = 10) -> go.Figure:
    data = df['Category'].value_counts().head(n).reset_index()
    data.columns = ['Category', 'Count']
    fig = px.bar(data, x='Count', y='Category', orientation='h',
                 color='Count', color_continuous_scale='Blues',
                 text='Count', title=f'Top {n} Categories by App Count')
    fig.update_traces(textposition='outside')
    return _base_layout(fig)


def avg_rating_by_category(df: pd.DataFrame) -> go.Figure:
    data = (df.groupby('Category')['Rating']
              .mean().sort_values(ascending=False)
              .reset_index())
    data.columns = ['Category', 'Avg Rating']
    fig = px.bar(data, x='Category', y='Avg Rating',
                 color='Avg Rating', color_continuous_scale='RdYlGn',
                 title='Average Rating by Category')
    fig.update_xaxes(tickangle=45)
    return _base_layout(fig, height=480)


def installs_by_category(df: pd.DataFrame, n: int = 15) -> go.Figure:
    data = (df.groupby('Category')['Installs']
              .sum().sort_values(ascending=False)
              .head(n).reset_index())
    data['Installs_M'] = data['Installs'] / 1e6
    fig = px.bar(data, x='Installs_M', y='Category', orientation='h',
                 color='Installs_M', color_continuous_scale='Purples',
                 text=data['Installs_M'].round(1),
                 title=f'Top {n} Categories by Total Installs (M)')
    fig.update_traces(textposition='outside')
    return _base_layout(fig)


# ──────────────────────────────────────────────
# RATINGS
# ──────────────────────────────────────────────

def rating_histogram(df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(df, x='Rating', nbins=20, color_discrete_sequence=['#4A90D9'],
                       title='Rating Distribution', marginal='box')
    return _base_layout(fig)


def rating_vs_installs(df: pd.DataFrame) -> go.Figure:
    sample = df.sample(min(2000, len(df)), random_state=42)
    sample = sample[sample['Installs'] > 0].copy()
    sample['log_installs'] = sample['Installs'].apply(lambda x: max(x, 1)).pipe(
        lambda s: s.apply(__import__('math').log10)
    )
    fig = px.scatter(sample, x='log_installs', y='Rating',
                     color='Category', opacity=0.5,
                     labels={'log_installs': 'Log10(Installs)'},
                     title='Rating vs Installs (log scale)')
    fig.update_traces(marker_size=4)
    return _base_layout(fig, height=460)


def rating_box_by_type(df: pd.DataFrame) -> go.Figure:
    fig = px.box(df, x='Type', y='Rating', color='Type',
                 color_discrete_map={'Free':'#2ECC71','Paid':'#E74C3C'},
                 title='Rating Distribution: Free vs Paid')
    return _base_layout(fig)


# ──────────────────────────────────────────────
# INSTALLS & POPULARITY
# ──────────────────────────────────────────────

def top_apps_bar(df: pd.DataFrame, col: str, label: str, n: int = 10) -> go.Figure:
    data = df.nlargest(n, col)[['App', col]].copy()
    fig = px.bar(data, x=col, y='App', orientation='h',
                 color=col, color_continuous_scale='Oranges',
                 title=f'Top {n} Apps by {label}')
    fig.update_traces(texttemplate='%{x:,.0f}', textposition='outside')
    return _base_layout(fig)


def installs_vs_reviews(df: pd.DataFrame) -> go.Figure:
    sample = df.sample(min(2000, len(df)), random_state=7)
    fig = px.scatter(sample, x='Reviews', y='Installs', color='Type',
                     log_x=True, log_y=True, opacity=0.5,
                     color_discrete_map={'Free':'#3498DB','Paid':'#E67E22'},
                     title='Installs vs Reviews (log-log)')
    fig.update_traces(marker_size=4)
    return _base_layout(fig, height=440)


def category_heatmap(df: pd.DataFrame) -> go.Figure:
    pivot = (df.groupby(['Category', 'Type'])['Installs']
               .sum().unstack(fill_value=0)
               .div(1e6)
               .round(1))
    fig = px.imshow(pivot, text_auto=True, aspect='auto',
                    color_continuous_scale='YlOrRd',
                    title='Installs (M) by Category × Type')
    return _base_layout(fig, height=620)


# ──────────────────────────────────────────────
# PRICING
# ──────────────────────────────────────────────

def free_vs_paid_pie(df: pd.DataFrame) -> go.Figure:
    counts = df['Type'].value_counts()
    fig = px.pie(names=counts.index, values=counts.values,
                 color=counts.index,
                 color_discrete_map={'Free':'#2ECC71','Paid':'#E74C3C'},
                 title='Free vs Paid Apps', hole=0.4)
    return _base_layout(fig)


def price_distribution(df: pd.DataFrame) -> go.Figure:
    paid = df[df['Price'] > 0]
    fig = px.histogram(paid, x='Price', nbins=30, log_y=True,
                       color_discrete_sequence=['#E67E22'],
                       title='Paid App Price Distribution')
    return _base_layout(fig)


def avg_price_by_category(df: pd.DataFrame) -> go.Figure:
    paid = df[df['Price'] > 0]
    data = paid.groupby('Category')['Price'].mean().sort_values(ascending=False).reset_index()
    data.columns = ['Category', 'Avg Price']
    fig = px.bar(data.head(15), x='Category', y='Avg Price',
                 color='Avg Price', color_continuous_scale='Reds',
                 title='Average Price by Category (Paid Apps)')
    fig.update_xaxes(tickangle=45)
    return _base_layout(fig, height=460)


# ──────────────────────────────────────────────
# SIZE
# ──────────────────────────────────────────────

def size_histogram(df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(df.dropna(subset=['Size_MB']), x='Size_MB',
                       nbins=40, color_discrete_sequence=['#9B59B6'],
                       title='App Size Distribution (MB)')
    return _base_layout(fig)


def size_vs_rating(df: pd.DataFrame) -> go.Figure:
    sample = df.dropna(subset=['Size_MB']).sample(min(2000, len(df)), random_state=3)
    fig = px.scatter(sample, x='Size_MB', y='Rating', color='Category',
                     opacity=0.5, title='App Size vs Rating')
    fig.update_traces(marker_size=4)
    return _base_layout(fig, height=460)


def size_box_by_category(df: pd.DataFrame, n: int = 12) -> go.Figure:
    top_cats = df['Category'].value_counts().head(n).index
    sub = df[df['Category'].isin(top_cats)].dropna(subset=['Size_MB'])
    fig = px.box(sub, x='Category', y='Size_MB', color='Category',
                 title=f'App Size Distribution – Top {n} Categories')
    fig.update_xaxes(tickangle=45)
    return _base_layout(fig, height=480)


# ──────────────────────────────────────────────
# SENTIMENT
# ──────────────────────────────────────────────

def sentiment_donut(summary: dict) -> go.Figure:
    labels = list(summary.keys())
    values = [v['count'] for v in summary.values()]
    fig = px.pie(names=labels, values=values, hole=0.5,
                 color=labels,
                 color_discrete_map={'Positive':'#2ECC71',
                                     'Negative':'#E74C3C',
                                     'Neutral':'#95A5A6'},
                 title='Sentiment Distribution')
    return _base_layout(fig)


def top_words_bar(df_words: pd.DataFrame, title: str, color: str) -> go.Figure:
    fig = px.bar(df_words.sort_values('Count'), x='Count', y='Word',
                 orientation='h', color_discrete_sequence=[color],
                 text='Count', title=title)
    fig.update_traces(textposition='outside')
    return _base_layout(fig)
