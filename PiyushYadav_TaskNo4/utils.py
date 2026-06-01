import pandas as pd
import numpy as np


def get_dataset_summary(df):
    return {
        "Rows": df.shape[0],
        "Columns": df.shape[1],
        "Missing Values": df.isnull().sum().sum(),
        "Duplicate Rows": df.duplicated().sum(),
    }


def calculate_quality_score(df):

    missing_ratio = (
        df.isnull().sum().sum()
        / (df.shape[0] * df.shape[1])
    )

    duplicate_ratio = (
        df.duplicated().sum()
        / max(len(df), 1)
    )

    score = 100 - (
        missing_ratio * 50
        + duplicate_ratio * 50
    ) * 100

    return round(max(score, 0), 2)


def handle_missing_values(df):

    cleaned_df = df.copy()

    for col in cleaned_df.columns:

        if pd.api.types.is_numeric_dtype(
            cleaned_df[col]
        ):

            cleaned_df[col] = cleaned_df[col].fillna(
                cleaned_df[col].mean()
            )

        else:

            mode = (
                cleaned_df[col].mode()[0]
                if not cleaned_df[col].mode().empty
                else "Unknown"
            )

            cleaned_df[col] = cleaned_df[col].fillna(
                mode
            )

    return cleaned_df


def remove_duplicates(df):
    return df.drop_duplicates()


def standardize_data(df):

    df = df.copy()

    for col in df.select_dtypes(
        include="object"
    ).columns:

        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.title()
        )

    return df


def detect_outliers(df):

    outlier_info = {}

    numeric_cols = df.select_dtypes(
        include=np.number
    ).columns

    for col in numeric_cols:

        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)

        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        count = len(
            df[
                (df[col] < lower)
                | (df[col] > upper)
            ]
        )

        outlier_info[col] = count

    return outlier_info