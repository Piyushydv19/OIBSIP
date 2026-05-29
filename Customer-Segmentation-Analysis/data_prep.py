"""
utils/data_prep.py
==================
Data loading, cleaning, and feature engineering for customer segmentation.
"""

import pandas as pd
import numpy as np
from datetime import datetime


# ──────────────────────────────────────────────
# LOAD + CLEAN
# ──────────────────────────────────────────────

def load_and_clean(path: str) -> tuple[pd.DataFrame, dict]:
    """
    Load CSV, clean it, return (clean_df, quality_report_dict).
    Steps:
      1. Drop exact duplicates
      2. Fix Income: fill missing with median
      3. Remove obvious outliers (Income > 600k, Age > 100)
      4. Parse enrollment date → seniority in days
      5. Standardise Marital_Status categories
    """
    raw = pd.read_csv(path)
    report = {"raw_rows": len(raw), "raw_nulls": int(raw.isnull().sum().sum())}

    df = raw.copy()

    # 1. Duplicates
    df.drop_duplicates(inplace=True)

    # 2. Income: fill NaN with median
    df["Income"] = pd.to_numeric(df["Income"], errors="coerce")
    df["Income"].fillna(df["Income"].median(), inplace=True)

    # 3. Outlier removal
    df = df[df["Income"] < 600_000]
    df["Age"] = datetime.now().year - df["Year_Birth"]
    df = df[df["Age"].between(18, 100)]

    # 4. Enrollment seniority
    df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], errors="coerce")
    ref_date = df["Dt_Customer"].max()
    df["Seniority"] = (ref_date - df["Dt_Customer"]).dt.days
    df["Seniority"].fillna(df["Seniority"].median(), inplace=True)

    # 5. Simplify marital status
    df["Marital_Status"] = df["Marital_Status"].replace(
        {"Together": "Partner", "Married": "Partner",
         "Divorced": "Alone",   "Widow": "Alone", "Single": "Alone"}
    )

    # 6. Kids: total dependents
    df["Children"] = df["Kidhome"] + df["Teenhome"]

    df.reset_index(drop=True, inplace=True)
    report["clean_rows"]  = len(df)
    report["clean_nulls"] = int(df.isnull().sum().sum())
    report["removed"]     = report["raw_rows"] - report["clean_rows"]
    return df, report


# ──────────────────────────────────────────────
# FEATURE ENGINEERING
# ──────────────────────────────────────────────

SPEND_COLS = [
    "MntWines", "MntFruits", "MntMeatProducts",
    "MntFishProducts", "MntSweetProducts", "MntGoldProds",
]

PURCHASE_COLS = [
    "NumDealsPurchases", "NumWebPurchases",
    "NumCatalogPurchases", "NumStorePurchases",
]

CAMPAIGN_COLS = [
    "AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3",
    "AcceptedCmp4", "AcceptedCmp5",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create RFM + composite features used for clustering."""
    df = df.copy()

    # Total spend across all product categories
    df["TotalSpend"] = df[SPEND_COLS].sum(axis=1)

    # Total purchases across all channels
    df["TotalPurchases"] = df[PURCHASE_COLS].sum(axis=1)

    # Average order value (avoid /0)
    df["AvgOrderValue"] = df["TotalSpend"] / (df["TotalPurchases"] + 1)

    # Campaign engagement rate (0–5 campaigns accepted)
    df["CampaignAccepted"] = df[CAMPAIGN_COLS].sum(axis=1)

    # Preferred channel (fraction of purchases online vs store)
    total_ch = df["NumWebPurchases"] + df["NumStorePurchases"] + df["NumCatalogPurchases"] + 1
    df["OnlineRatio"]  = df["NumWebPurchases"]     / total_ch
    df["CatalogRatio"] = df["NumCatalogPurchases"] / total_ch

    # Spend per product share
    for col in SPEND_COLS:
        df[f"{col}_share"] = df[col] / (df["TotalSpend"] + 1)

    # RFM proxy
    # R = Recency (lower = better → invert for score)
    df["R_score"] = df["Recency"].max() - df["Recency"]
    # F = TotalPurchases
    df["F_score"] = df["TotalPurchases"]
    # M = TotalSpend
    df["M_score"] = df["TotalSpend"]

    return df


# ──────────────────────────────────────────────
# FEATURES FOR CLUSTERING
# ──────────────────────────────────────────────

CLUSTER_FEATURES = [
    "Income", "Age", "Recency", "TotalSpend", "TotalPurchases",
    "AvgOrderValue", "Children", "CampaignAccepted",
    "OnlineRatio", "CatalogRatio", "Seniority",
]


def get_cluster_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame of cluster features, no NaNs."""
    return df[CLUSTER_FEATURES].fillna(0)
