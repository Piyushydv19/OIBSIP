"""
app.py — Customer Segmentation Analysis Dashboard
Run: streamlit run app.py
"""

import os
import sys
import warnings
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

from utils.data_prep import (
    load_and_clean,
    engineer_features,
    get_cluster_matrix,
    CLUSTER_FEATURES,
)
from utils.clustering import (
    SegmentationPipeline,
    evaluate_k,
    auto_label_segments,
)
from utils import charts


# -----------------------------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Segmentation Analysis",
    page_icon="👥",
    layout="wide",
)

# -----------------------------------------------------------------------------
# CUSTOM CSS
# -----------------------------------------------------------------------------
st.markdown("""
<style>
.kpi{
    background:linear-gradient(135deg,#1a1a2e,#16213e);
    border-radius:14px;
    padding:20px;
    text-align:center;
    color:white;
}
.kpi-val{
    font-size:2rem;
    font-weight:700;
    color:#00d4ff;
}
.kpi-lbl{
    font-size:0.8rem;
    text-transform:uppercase;
}
.section{
    font-size:1.4rem;
    font-weight:700;
    border-left:4px solid #00d4ff;
    padding-left:12px;
    margin:20px 0;
}
.seg-card{
    border-radius:12px;
    padding:15px;
    margin-bottom:10px;
    background:#f8f9fa;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:

    st.title("👥 Customer Segmentation")

    uploaded_file = st.file_uploader(
        "Upload Customer Dataset",
        type=["csv"]
    )

    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "🏠 Overview & EDA",
            "🔬 Optimal Clusters",
            "🗺️ Segmentation Results",
            "📊 Segment Profiles",
            "📋 Insights & Recommendations"
        ]
    )

    st.markdown("---")

    n_clusters = st.slider(
        "Number of Clusters",
        min_value=2,
        max_value=8,
        value=4
    )

# -----------------------------------------------------------------------------
# DATA SOURCE
# -----------------------------------------------------------------------------
DEFAULT_DATASET = os.path.join(
    os.path.dirname(__file__),
    "data",
    "marketing_campaign.csv"
)

DATA_SOURCE = uploaded_file if uploaded_file else DEFAULT_DATASET

if uploaded_file:
    st.sidebar.success(f"✅ {uploaded_file.name}")
else:
    st.sidebar.info("📁 Using Default Dataset")

# -----------------------------------------------------------------------------
# LOAD DATA
# -----------------------------------------------------------------------------
@st.cache_data
def get_data(source):

    df_clean, report = load_and_clean(source)
    df_feat = engineer_features(df_clean)

    return df_clean, df_feat, report


try:
    df_clean, df_feat, quality_report = get_data(DATA_SOURCE)

except Exception as e:
    st.error(f"Dataset Error: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# FILTERS
# -----------------------------------------------------------------------------
with st.sidebar:

    st.markdown("---")
    st.subheader("Filters")

    income_min = int(df_feat["Income"].min())
    income_max = int(df_feat["Income"].max())

    income_range = st.slider(
        "Income Range",
        income_min,
        income_max,
        (income_min, income_max)
    )

    age_min = int(df_feat["Age"].min())
    age_max = int(df_feat["Age"].max())

    age_range = st.slider(
        "Age Range",
        age_min,
        age_max,
        (age_min, age_max)
    )

filtered = df_feat[
    df_feat["Income"].between(*income_range)
    &
    df_feat["Age"].between(*age_range)
]

# -----------------------------------------------------------------------------
# CLUSTERING
# -----------------------------------------------------------------------------
@st.cache_resource
def run_clustering(data, k):

    X = get_cluster_matrix(data)

    pipe = SegmentationPipeline(
        n_clusters=k
    )

    pipe.fit(X)

    return pipe


pipe = run_clustering(df_feat, n_clusters)

X_full = get_cluster_matrix(df_feat)

df_feat["Segment"] = pipe.predict(X_full)

profile_raw = pipe.profile(
    df_feat,
    CLUSTER_FEATURES
)

segment_labels = auto_label_segments(profile_raw)

df_feat["Segment_Label"] = (
    df_feat["Segment"]
    .map(segment_labels)
)

metrics = pipe.metrics()

pca_coords = pipe.pca_coords()

# -----------------------------------------------------------------------------
# KPI FUNCTION
# -----------------------------------------------------------------------------
def kpi(label, value):

    st.markdown(
        f"""
        <div class="kpi">
            <div class="kpi-lbl">{label}</div>
            <div class="kpi-val">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# -----------------------------------------------------------------------------
# PAGE 1
# -----------------------------------------------------------------------------
if page == "🏠 Overview & EDA":

    st.title("👥 Customer Segmentation Analysis")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        kpi("Customers", len(df_feat))

    with c2:
        kpi(
            "Average Income",
            f"${df_feat['Income'].mean():,.0f}"
        )

    with c3:
        kpi(
            "Average Spend",
            f"${df_feat['TotalSpend'].mean():,.0f}"
        )

    with c4:
        kpi(
            "Average Age",
            f"{df_feat['Age'].mean():.0f}"
        )

    st.markdown("## Data Quality")

    q1, q2, q3 = st.columns(3)

    q1.metric("Original Rows", quality_report["raw_rows"])
    q2.metric("Clean Rows", quality_report["clean_rows"])
    q3.metric("Removed", quality_report["removed"])

    st.plotly_chart(
        charts.income_distribution(filtered),
        use_container_width=True
    )

    st.plotly_chart(
        charts.age_distribution(filtered),
        use_container_width=True
    )

    st.plotly_chart(
        charts.spend_by_category(filtered),
        use_container_width=True
    )

    st.plotly_chart(
        charts.purchase_channel_pie(filtered),
        use_container_width=True
    )

    st.plotly_chart(
        charts.recency_histogram(filtered),
        use_container_width=True
    )

# -----------------------------------------------------------------------------
# PAGE 2
# -----------------------------------------------------------------------------
elif page == "🔬 Optimal Clusters":

    st.title("🔬 Optimal Clusters")

    X = get_cluster_matrix(df_feat).values

    eval_df = evaluate_k(
        X,
        k_range=range(2, 11)
    )

    st.plotly_chart(
        charts.elbow_silhouette(eval_df),
        use_container_width=True
    )

    st.dataframe(
        eval_df,
        use_container_width=True
    )

# -----------------------------------------------------------------------------
# PAGE 3
# -----------------------------------------------------------------------------
elif page == "🗺️ Segmentation Results":

    st.title("🗺️ Segmentation Results")

    st.plotly_chart(
        charts.pca_scatter(
            df_feat,
            pca_coords[:, 0],
            pca_coords[:, 1],
            metrics["pca_var_ratio"]
        ),
        use_container_width=True
    )

    c1, c2 = st.columns(2)

    with c1:
        st.plotly_chart(
            charts.segment_size_pie(df_feat),
            use_container_width=True
        )

    with c2:
        st.plotly_chart(
            charts.rfm_scatter(df_feat),
            use_container_width=True
        )

# -----------------------------------------------------------------------------
# PAGE 4
# -----------------------------------------------------------------------------
elif page == "📊 Segment Profiles":

    st.title("📊 Segment Profiles")

    profile = (
        df_feat
        .groupby("Segment_Label")[CLUSTER_FEATURES]
        .mean()
        .round(2)
    )

    radar_cols = [
        "Income",
        "TotalSpend",
        "TotalPurchases",
        "AvgOrderValue",
        "CampaignAccepted",
        "OnlineRatio"
    ]

    radar_labels = [
        "Income",
        "Spend",
        "Purchases",
        "Order Value",
        "Campaigns",
        "Online"
    ]

    st.plotly_chart(
        charts.radar_chart(
            profile,
            radar_cols,
            radar_labels
        ),
        use_container_width=True
    )

    st.dataframe(
        profile,
        use_container_width=True
    )

# -----------------------------------------------------------------------------
# PAGE 5
# -----------------------------------------------------------------------------
elif page == "📋 Insights & Recommendations":

    st.title("📋 Insights & Recommendations")

    segment_summary = (
        df_feat.groupby("Segment_Label")
        .agg(
            Customers=("ID", "count"),
            AvgIncome=("Income", "mean"),
            AvgSpend=("TotalSpend", "mean"),
            AvgPurchases=("TotalPurchases", "mean"),
        )
        .round(2)
    )

    st.dataframe(
        segment_summary,
        use_container_width=True
    )

    st.success(
        "🎯 High-value customers should receive loyalty rewards and premium offers."
    )

    st.info(
        "📧 Medium-value customers can be targeted with cross-selling campaigns."
    )

    st.warning(
        "⚠️ Low-engagement customers should receive win-back campaigns."
    )

# -----------------------------------------------------------------------------
# DOWNLOAD
# -----------------------------------------------------------------------------
st.sidebar.markdown("---")

st.sidebar.download_button(
    "⬇️ Download Segmented Data",
    df_feat.to_csv(index=False).encode(),
    file_name="segmented_customers.csv",
    mime="text/csv"
)