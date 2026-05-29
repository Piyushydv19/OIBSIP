"""
utils/data_cleaning.py
======================
All data cleaning functions for Google Play Store dataset.
"""

import pandas as pd
import numpy as np
import re


# ──────────────────────────────────────────────
# COLUMN CONVERTERS
# ──────────────────────────────────────────────

def clean_installs(val):
    """'1,000,000+' → 1000000"""
    if pd.isna(val):
        return np.nan
    return int(re.sub(r'[^0-9]', '', str(val)) or 0)


def clean_price(val):
    """'$4.99' or '0' → float"""
    if pd.isna(val):
        return 0.0
    return float(re.sub(r'[^\d.]', '', str(val)) or 0)


def clean_size(val):
    """'19M' → 19.0, '512k' → 0.5, 'Varies with device' → NaN"""
    if pd.isna(val):
        return np.nan
    val = str(val).strip()
    if 'varies' in val.lower():
        return np.nan
    if val.endswith('M'):
        return float(val[:-1])
    if val.endswith('k') or val.endswith('K'):
        return round(float(val[:-1]) / 1024, 3)
    try:
        return float(val)
    except ValueError:
        return np.nan


def clean_reviews(val):
    """'4,967' → 4967"""
    if pd.isna(val):
        return np.nan
    try:
        return int(re.sub(r'[^0-9]', '', str(val)) or 0)
    except Exception:
        return np.nan


# ──────────────────────────────────────────────
# MAIN CLEANING PIPELINE
# ──────────────────────────────────────────────

def clean_play_store(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full cleaning pipeline for googleplaystore.csv.
    Returns a clean copy.
    """
    df = df.copy()

    # 1. Drop duplicate rows
    df.drop_duplicates(subset='App', keep='first', inplace=True)

    # 2. Drop rows missing critical columns
    df.dropna(subset=['App', 'Category'], inplace=True)

    # 3. Convert numeric columns
    df['Reviews']  = df['Reviews'].apply(clean_reviews).astype('Int64')
    df['Installs'] = df['Installs'].apply(clean_installs).astype('Int64')
    df['Price']    = df['Price'].apply(clean_price)
    df['Size_MB']  = df['Size'].apply(clean_size)      # new numeric column
    df['Rating']   = pd.to_numeric(df['Rating'], errors='coerce')

    # 4. Fix Rating outliers (valid range 1–5)
    df.loc[~df['Rating'].between(1, 5), 'Rating'] = np.nan

    # 5. Clean category names
    df['Category'] = df['Category'].str.strip().str.upper().str.replace(' ', '_')

    # 6. Standardise Type column
    df['Type'] = df['Type'].str.strip().str.capitalize()
    df.loc[~df['Type'].isin(['Free', 'Paid']), 'Type'] = 'Free'

    # 7. Fill missing Rating with median per category
    df['Rating'] = df.groupby('Category')['Rating'].transform(
        lambda x: x.fillna(x.median())
    )
    # Global median fallback
    global_med = df['Rating'].median()
    df['Rating'] = df['Rating'].fillna(global_med)

    # 8. Fill missing Size_MB with category median
    df['Size_MB'] = df.groupby('Category')['Size_MB'].transform(
        lambda x: x.fillna(x.median())
    )
    df['Size_MB'] = df['Size_MB'].fillna(df['Size_MB'].median())

    # 9. Reset index
    df.reset_index(drop=True, inplace=True)

    return df


def data_quality_report(raw: pd.DataFrame, clean: pd.DataFrame) -> dict:
    """Return before/after quality stats dict."""
    return {
        "before": {
            "rows":        len(raw),
            "duplicates":  raw.duplicated(subset='App').sum(),
            "missing_rating": raw['Rating'].isna().sum(),
            "nulls_total": raw.isna().sum().sum(),
        },
        "after": {
            "rows":        len(clean),
            "duplicates":  0,
            "missing_rating": clean['Rating'].isna().sum(),
            "nulls_total": clean.isna().sum().sum(),
        },
    }
