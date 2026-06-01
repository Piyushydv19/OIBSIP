"""
utils/clustering.py
===================
K-Means clustering pipeline with optimal-k selection,
PCA reduction for visualisation, and segment profiling.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
import joblib
import os


# ──────────────────────────────────────────────
# ELBOW + SILHOUETTE SEARCH
# ──────────────────────────────────────────────

def evaluate_k(X_scaled: np.ndarray, k_range=range(2, 11)) -> pd.DataFrame:
    """
    For each k in k_range compute:
      - inertia  (elbow method)
      - silhouette score
      - davies-bouldin index (lower = better)
    Returns a DataFrame with one row per k.
    """
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        rows.append({
            "k":         k,
            "inertia":   km.inertia_,
            "silhouette": silhouette_score(X_scaled, labels, sample_size=500, random_state=42),
            "davies_bouldin": davies_bouldin_score(X_scaled, labels),
        })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────

class SegmentationPipeline:
    """
    Wraps: StandardScaler → KMeans → PCA(2D).
    Fit on feature matrix, attach cluster labels to df.
    """

    def __init__(self, n_clusters: int = 4):
        self.n_clusters = n_clusters
        self.scaler     = StandardScaler()
        self.kmeans     = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.pca        = PCA(n_components=2, random_state=42)
        self._fitted    = False

    # ── Fit ───────────────────────────────────────

    def fit(self, X: pd.DataFrame) -> "SegmentationPipeline":
        X_scaled   = self.scaler.fit_transform(X)
        self.kmeans.fit(X_scaled)
        self.pca.fit(X_scaled)
        self._X_scaled = X_scaled
        self._fitted   = True
        return self

    # ── Transform / predict ───────────────────────

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.kmeans.predict(X_scaled)

    def pca_coords(self, X: pd.DataFrame | None = None) -> np.ndarray:
        """Return 2-D PCA coords for scatter plot."""
        X_scaled = self._X_scaled if X is None else self.scaler.transform(X)
        return self.pca.transform(X_scaled)

    # ── Metrics ───────────────────────────────────

    def metrics(self) -> dict:
        labels = self.kmeans.labels_
        return {
            "inertia":        round(self.kmeans.inertia_, 2),
            "silhouette":     round(silhouette_score(self._X_scaled, labels,
                                                     sample_size=500, random_state=42), 4),
            "davies_bouldin": round(davies_bouldin_score(self._X_scaled, labels), 4),
            "pca_var_ratio":  [round(v, 4) for v in self.pca.explained_variance_ratio_],
        }

    # ── Segment profile ───────────────────────────

    def profile(self, df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
        """
        Compute per-segment mean of all feature_cols.
        Returns DataFrame indexed by Segment.
        """
        return (
            df.groupby("Segment")[feature_cols]
            .mean()
            .round(2)
        )

    # ── Save / load ───────────────────────────────

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "SegmentationPipeline":
        return joblib.load(path)


# ──────────────────────────────────────────────
# SEGMENT LABELLING
# ──────────────────────────────────────────────

def auto_label_segments(profile_df: pd.DataFrame) -> dict:
    """
    Heuristically assign human-readable labels to clusters
    based on Income and TotalSpend ranks.
    Returns dict {cluster_id: label}.
    """
    # Rank by TotalSpend descending
    ranked = profile_df["TotalSpend"].sort_values(ascending=False)
    labels_map = {}
    label_names = [
        "🏆 High-Value Champions",
        "💼 Loyal Mid-Tier",
        "🌱 Emerging Customers",
        "💤 At-Risk / Low Engagement",
    ]
    for rank, idx in enumerate(ranked.index):
        labels_map[idx] = label_names[rank] if rank < len(label_names) else f"Segment {idx}"
    return labels_map
