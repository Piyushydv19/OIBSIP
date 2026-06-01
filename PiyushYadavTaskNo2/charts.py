"""
utils/charts.py
===============
All Plotly visualisation functions for the Streamlit dashboard.
Each function returns a go.Figure.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Consistent segment colour palette
SEG_COLORS = px.colors.qualitative.Bold

_BG = "rgba(0,0,0,0)"

def _layout(fig, height=420):
    fig.update_layout(
        paper_bgcolor=_BG, plot_bgcolor=_BG,
        height=height, margin=dict(t=50, b=40, l=40, r=20),
        font=dict(family="Inter, sans-serif", size=12),
    )
    return fig


# ──────────────────────────────────────────────
# EDA CHARTS
# ──────────────────────────────────────────────

def income_distribution(df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(df, x="Income", nbins=50, marginal="box",
                       color_discrete_sequence=["#4A90D9"],
                       title="Income Distribution")
    return _layout(fig)


def age_distribution(df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(df, x="Age", nbins=30, marginal="violin",
                       color_discrete_sequence=["#E67E22"],
                       title="Customer Age Distribution")
    return _layout(fig)


def spend_by_category(df: pd.DataFrame) -> go.Figure:
    spend_cols = ["MntWines","MntFruits","MntMeatProducts",
                  "MntFishProducts","MntSweetProducts","MntGoldProds"]
    labels = ["Wines","Fruits","Meat","Fish","Sweets","Gold"]
    totals = df[spend_cols].sum().values
    fig = px.bar(x=labels, y=totals, color=labels,
                 color_discrete_sequence=SEG_COLORS,
                 labels={"x":"Category","y":"Total Spend ($)"},
                 title="Total Spend by Product Category",
                 text=[f"${v:,.0f}" for v in totals])
    fig.update_traces(textposition="outside")
    return _layout(fig)


def correlation_heatmap(df: pd.DataFrame, cols: list) -> go.Figure:
    corr = df[cols].corr().round(2)
    fig  = px.imshow(corr, text_auto=True, aspect="auto",
                     color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                     title="Feature Correlation Heatmap")
    return _layout(fig, height=520)


def purchase_channel_pie(df: pd.DataFrame) -> go.Figure:
    channels = {
        "Web":     df["NumWebPurchases"].sum(),
        "Store":   df["NumStorePurchases"].sum(),
        "Catalog": df["NumCatalogPurchases"].sum(),
        "Deals":   df["NumDealsPurchases"].sum(),
    }
    fig = px.pie(names=list(channels.keys()), values=list(channels.values()),
                 hole=0.45, title="Purchases by Channel",
                 color_discrete_sequence=SEG_COLORS)
    return _layout(fig)


def recency_histogram(df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(df, x="Recency", nbins=30,
                       color_discrete_sequence=["#9B59B6"],
                       title="Customer Recency (days since last purchase)")
    return _layout(fig)


# ──────────────────────────────────────────────
# ELBOW / SILHOUETTE
# ──────────────────────────────────────────────

def elbow_silhouette(eval_df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=eval_df["k"], y=eval_df["inertia"],
                   mode="lines+markers", name="Inertia",
                   line=dict(color="#4A90D9", width=2),
                   marker=dict(size=8)),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=eval_df["k"], y=eval_df["silhouette"],
                   mode="lines+markers", name="Silhouette Score",
                   line=dict(color="#E74C3C", width=2, dash="dash"),
                   marker=dict(size=8)),
        secondary_y=True,
    )
    fig.update_layout(title="Elbow Curve + Silhouette Scores",
                      xaxis_title="Number of Clusters (k)",
                      paper_bgcolor=_BG, plot_bgcolor=_BG,
                      height=420, font=dict(size=12))
    fig.update_yaxes(title_text="Inertia",          secondary_y=False)
    fig.update_yaxes(title_text="Silhouette Score", secondary_y=True)
    return fig


# ──────────────────────────────────────────────
# CLUSTER SCATTER (PCA)
# ──────────────────────────────────────────────

def pca_scatter(df: pd.DataFrame, pc1: np.ndarray, pc2: np.ndarray,
                pca_var: list) -> go.Figure:
    plot_df = df.copy()
    plot_df["PC1"] = pc1
    plot_df["PC2"] = pc2
    plot_df["Segment_str"] = plot_df["Segment_Label"]
    fig = px.scatter(
        plot_df, x="PC1", y="PC2",
        color="Segment_str",
        hover_data=["Income","TotalSpend","Age","Recency"],
        color_discrete_sequence=SEG_COLORS,
        opacity=0.7,
        title=f"Customer Segments — PCA 2D (var explained: "
              f"{pca_var[0]:.1%} + {pca_var[1]:.1%})",
        labels={"Segment_str": "Segment"},
    )
    fig.update_traces(marker_size=5)
    return _layout(fig, height=500)


# ──────────────────────────────────────────────
# SEGMENT PROFILE CHARTS
# ──────────────────────────────────────────────

def segment_bar(profile: pd.DataFrame, col: str, title: str) -> go.Figure:
    data = profile[col].reset_index()
    data.columns = ["Segment","Value"]
    fig = px.bar(data, x="Segment", y="Value",
                 color="Segment", text=data["Value"].round(1),
                 color_discrete_sequence=SEG_COLORS,
                 title=title)
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False)
    return _layout(fig)


def radar_chart(profile: pd.DataFrame, cols: list, col_labels: list) -> go.Figure:
    """Normalised radar chart comparing segments across key features."""
    # Normalise 0-1 per column
    norm = (profile[cols] - profile[cols].min()) / (
        profile[cols].max() - profile[cols].min() + 1e-9)

    fig = go.Figure()
    for i, (seg, row) in enumerate(norm.iterrows()):
        vals = list(row.values) + [row.values[0]]   # close the polygon
        cats = col_labels + [col_labels[0]]
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=cats, fill="toself",
            name=str(seg),
            line_color=SEG_COLORS[i % len(SEG_COLORS)],
            opacity=0.75,
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0,1])),
        title="Segment Radar — Normalised Feature Comparison",
        paper_bgcolor=_BG, height=500,
    )
    return fig


def spend_stacked_bar(df: pd.DataFrame) -> go.Figure:
    """Stacked bar of mean spend per product category by segment."""
    spend_cols = ["MntWines","MntFruits","MntMeatProducts",
                  "MntFishProducts","MntSweetProducts","MntGoldProds"]
    labels     = ["Wines","Fruits","Meat","Fish","Sweets","Gold"]
    grp = df.groupby("Segment_Label")[spend_cols].mean().reset_index()
    fig = go.Figure()
    for col, lbl in zip(spend_cols, labels):
        fig.add_trace(go.Bar(
            x=grp["Segment_Label"], y=grp[col],
            name=lbl,
        ))
    fig.update_layout(barmode="stack",
                      title="Average Spend by Product Category per Segment",
                      xaxis_title="Segment", yaxis_title="Mean Spend ($)",
                      paper_bgcolor=_BG, plot_bgcolor=_BG, height=450)
    return fig


def income_box(df: pd.DataFrame) -> go.Figure:
    fig = px.box(df, x="Segment_Label", y="Income",
                 color="Segment_Label",
                 color_discrete_sequence=SEG_COLORS,
                 title="Income Distribution by Segment")
    fig.update_layout(showlegend=False)
    return _layout(fig)


def rfm_scatter(df: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        df.sample(min(1000, len(df)), random_state=1),
        x="Recency", y="TotalSpend",
        color="Segment_Label", size="TotalPurchases",
        hover_data=["Income","Age"],
        color_discrete_sequence=SEG_COLORS,
        opacity=0.7,
        title="RFM View: Recency vs Total Spend (size = Total Purchases)",
        size_max=20,
        labels={"Segment_Label": "Segment"},
    )
    return _layout(fig, height=480)


def campaign_response_bar(df: pd.DataFrame) -> go.Figure:
    camp_rate = df.groupby("Segment_Label")["CampaignAccepted"].mean().reset_index()
    camp_rate.columns = ["Segment","Avg Campaigns Accepted"]
    fig = px.bar(camp_rate, x="Segment", y="Avg Campaigns Accepted",
                 color="Segment", text=camp_rate["Avg Campaigns Accepted"].round(2),
                 color_discrete_sequence=SEG_COLORS,
                 title="Average Campaign Acceptance Rate by Segment")
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False)
    return _layout(fig)


def segment_size_pie(df: pd.DataFrame) -> go.Figure:
    counts = df["Segment_Label"].value_counts().reset_index()
    counts.columns = ["Segment","Count"]
    fig = px.pie(counts, names="Segment", values="Count",
                 hole=0.45, color="Segment",
                 color_discrete_sequence=SEG_COLORS,
                 title="Segment Size Distribution")
    return _layout(fig)


def channel_heatmap(df: pd.DataFrame) -> go.Figure:
    ch_cols = ["NumWebPurchases","NumStorePurchases",
               "NumCatalogPurchases","NumDealsPurchases"]
    ch_labels = ["Web","Store","Catalog","Deals"]
    grp = df.groupby("Segment_Label")[ch_cols].mean()
    grp.columns = ch_labels
    fig = px.imshow(grp, text_auto=".1f", aspect="auto",
                    color_continuous_scale="Blues",
                    title="Avg Purchases by Channel × Segment")
    return _layout(fig, height=400)
