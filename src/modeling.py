# src/modeling.py

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

def run_modeling(feature_df: pd.DataFrame) -> pd.DataFrame:
    features = feature_df.drop(columns=["sample_id"], errors='ignore')

    if len(feature_df) < 2:
        # Handle single sample - no clustering or anomaly detection
        feature_df["cluster"] = 0  # default cluster
        feature_df["anomaly"] = 0  # not an anomaly
        feature_df["risk_score"] = compute_risk_score(features.iloc[0])
        return feature_df

    # --- Clustering ---
    kmeans = KMeans(n_clusters=2, random_state=42)
    feature_df["cluster"] = kmeans.fit_predict(features)

    # --- Anomaly Detection ---
    iso = IsolationForest(contamination=0.1, random_state=42)
    feature_df["anomaly"] = iso.fit_predict(features)
    feature_df["anomaly"] = feature_df["anomaly"].map({1: 0, -1: 1})  # 1 = not anomaly, -1 = anomaly

    # --- Risk Scoring ---
    feature_df["risk_score"] = feature_df.apply(lambda row: compute_risk_score(row), axis=1)

    return feature_df

def compute_risk_score(row) -> float:
    # You can fine-tune this function based on weights
    score = (
        0.3 * row["pause_avg_duration"] +
        0.2 * row["hesitation_count"] +
        0.2 * row["incomplete_sentences"] +
        0.2 * (1 - row["semantic_similarity"]) +  # lower similarity = more risk
        0.1 * (1 - row["lexical_diversity"])     # lower diversity = more risk
    )
    return round(score, 3)
