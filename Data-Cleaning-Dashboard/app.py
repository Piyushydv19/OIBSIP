import streamlit as st
import pandas as pd
import plotly.express as px
from io import BytesIO

from utils import (
    get_dataset_summary,
    handle_missing_values,
    remove_duplicates,
    standardize_data,
    detect_outliers,
    calculate_quality_score
)

st.set_page_config(
    page_title="Data Cleaning Dashboard",
    page_icon="🧹",
    layout="wide"
)

# -------------------------
# CUSTOM CSS
# -------------------------

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.metric-card {
    background-color: #1e1e1e;
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #333;
}

.reportview-container {
    background: #0f1117;
}

</style>
""", unsafe_allow_html=True)

# -------------------------
# HEADER
# -------------------------

st.title("🧹 Data Cleaning & Quality Analysis Dashboard")

st.markdown(
    "<p style='margin-top:-15px; color:gray; font-size:14px;'>By Piyush Yadav</p>",
    unsafe_allow_html=True
)

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV Dataset",
    type=["csv"]
)

st.sidebar.header("Cleaning Options")

apply_missing = st.sidebar.checkbox(
    "Handle Missing Values",
    value=True
)

apply_duplicates = st.sidebar.checkbox(
    "Remove Duplicates",
    value=True
)

apply_standardization = st.sidebar.checkbox(
    "Standardize Text",
    value=True
)

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    summary = get_dataset_summary(df)

    # -------------------------
    # CLEAN DATASET
    # -------------------------

    cleaned_df = df.copy()

    if apply_missing:
        cleaned_df = handle_missing_values(
            cleaned_df
        )

    if apply_duplicates:
        cleaned_df = remove_duplicates(
            cleaned_df
        )

    if apply_standardization:
        cleaned_df = standardize_data(
            cleaned_df
        )

    # -------------------------
    # KPI SECTION
    # -------------------------

    st.subheader("Dataset Summary")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Rows",
        summary["Rows"]
    )

    c2.metric(
        "Columns",
        summary["Columns"]
    )

    c3.metric(
        "Missing Values",
        summary["Missing Values"]
    )

    c4.metric(
        "Duplicates",
        summary["Duplicate Rows"]
    )

    # -------------------------
    # QUALITY SCORE
    # -------------------------

    st.divider()

    before_score = calculate_quality_score(df)
    after_score = calculate_quality_score(
        cleaned_df
    )

    q1, q2 = st.columns(2)

    q1.metric(
        "Quality Score Before",
        f"{before_score}%"
    )

    q2.metric(
        "Quality Score After",
        f"{after_score}%"
    )

    # -------------------------
    # BEFORE VS AFTER
    # -------------------------

    st.divider()

    st.subheader(
        "Before vs After Cleaning"
    )

    comparison = pd.DataFrame({

        "Metric": [
            "Rows",
            "Missing Values",
            "Duplicate Rows"
        ],

        "Before": [
            len(df),
            df.isnull().sum().sum(),
            df.duplicated().sum()
        ],

        "After": [
            len(cleaned_df),
            cleaned_df.isnull().sum().sum(),
            cleaned_df.duplicated().sum()
        ]
    })

    st.dataframe(
        comparison,
        use_container_width=True
    )

    # -------------------------
    # TABS
    # -------------------------

    tabs = st.tabs([
        "Overview",
        "Missing Values",
        "Outliers",
        "Cleaning Log",
        "Cleaned Dataset"
    ])

    # -------------------------
    # OVERVIEW
    # -------------------------

    with tabs[0]:

        st.subheader("Dataset Preview")

        st.dataframe(
            df.head(20),
            use_container_width=True
        )

        st.subheader("Data Types")

        st.dataframe(
            pd.DataFrame(
                df.dtypes,
                columns=["Data Type"]
            )
        )

    # -------------------------
    # MISSING VALUES
    # -------------------------

    with tabs[1]:

        missing_df = pd.DataFrame({

            "Column":
                df.columns,

            "Missing Values":
                df.isnull().sum().values
        })

        st.dataframe(
            missing_df,
            use_container_width=True
        )

        fig = px.bar(
            missing_df,
            x="Column",
            y="Missing Values",
            title="Missing Values by Column"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # -------------------------
    # OUTLIERS
    # -------------------------

    with tabs[2]:

        outlier_data = detect_outliers(df)

        outlier_df = pd.DataFrame(
            list(outlier_data.items()),
            columns=[
                "Column",
                "Outlier Count"
            ]
        )

        st.dataframe(
            outlier_df,
            use_container_width=True
        )

        numeric_cols = df.select_dtypes(
            include="number"
        ).columns

        if len(numeric_cols) > 0:

            selected = st.selectbox(
                "Select Numeric Column",
                numeric_cols
            )

            fig = px.box(
                df,
                y=selected,
                title=f"Outlier Analysis - {selected}"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    # -------------------------
    # CLEANING LOG
    # -------------------------

    with tabs[3]:

        st.subheader("Cleaning Report")

        report = f"""
DATA CLEANING REPORT

Rows Before : {df.shape[0]}
Rows After : {cleaned_df.shape[0]}

Columns : {df.shape[1]}

Missing Values Before :
{df.isnull().sum().sum()}

Missing Values After :
{cleaned_df.isnull().sum().sum()}

Duplicates Before :
{df.duplicated().sum()}

Duplicates After :
{cleaned_df.duplicated().sum()}

Quality Score Before :
{before_score}%

Quality Score After :
{after_score}%
"""

        st.text(report)

        st.download_button(
            "📄 Download Report",
            report,
            file_name="cleaning_report.txt"
        )

    # -------------------------
    # CLEANED DATASET
    # -------------------------

    with tabs[4]:

        st.subheader(
            "Cleaned Dataset"
        )

        st.dataframe(
            cleaned_df.head(25),
            use_container_width=True
        )

        csv = cleaned_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "📥 Download CSV",
            csv,
            file_name="cleaned_dataset.csv",
            mime="text/csv"
        )

        excel_buffer = BytesIO()

        with pd.ExcelWriter(
            excel_buffer,
            engine="xlsxwriter"
        ) as writer:

            cleaned_df.to_excel(
                writer,
                index=False
            )

        st.download_button(
            "📊 Download Excel",
            excel_buffer.getvalue(),
            file_name="cleaned_dataset.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

else:

    st.info(
        "👈 Upload a CSV file from the sidebar to begin analysis."
    )