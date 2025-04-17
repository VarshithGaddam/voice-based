# src/visualization.py

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_heatmap(df: pd.DataFrame, output_path: str):
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.drop(columns=['sample_id', 'cluster', 'anomaly']).corr(), annot=True, cmap="coolwarm")
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_distributions(df: pd.DataFrame, output_dir: str):
    sns.countplot(x="cluster", data=df)
    plt.title("Cluster Distribution")
    plt.savefig(os.path.join(output_dir, "cluster_distribution.png"))
    plt.close()

    sns.countplot(x="anomaly", data=df)
    plt.title("Anomaly Distribution")
    plt.savefig(os.path.join(output_dir, "anomaly_distribution.png"))
    plt.close()

def plot_pairwise(df: pd.DataFrame, output_path: str):
    selected = ["pause_avg_duration", "speech_rate", "semantic_similarity", "lexical_diversity", "cluster"]
    sns.pairplot(df[selected], hue="cluster")
    plt.suptitle("Pairwise Feature Comparison", y=1.02)
    plt.savefig(output_path)
    plt.close()

def plot_box(df: pd.DataFrame, output_dir: str):
    for col in ["pause_count", "semantic_similarity", "lexical_diversity"]:
        sns.boxplot(x="risk_score", y=col, data=df)
        plt.title(f"{col} vs Risk Score")
        plt.savefig(os.path.join(output_dir, f"{col}_boxplot.png"))
        plt.close()

def save_all_plots(df: pd.DataFrame, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    plot_heatmap(df, os.path.join(output_dir, "heatmap.png"))
    plot_distributions(df, output_dir)
    plot_pairwise(df, os.path.join(output_dir, "pairplot.png"))
    plot_box(df, output_dir)
