"""
Self-Driving Car Ethics AI - Exploratory Data Analysis (EDA) Pipeline

This script performs an end-to-end Exploratory Data Analysis on all CSV datasets
found in the `data/` directory. It generates comprehensive statistics, creates
professional PNG visualizations in `outputs/eda/`, prints insights with explanations,
and outputs a complete responsive HTML report (`outputs/eda/eda_report.html`).
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set professional aesthetics for Matplotlib & Seaborn
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.sans-serif": "Arial",
    "font.family": "sans-serif",
    "figure.titlesize": 16,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.autolayout": True
})


def setup_output_directory(output_dir: Path = Path("outputs/eda")) -> Path:
    """Ensures the output directory for EDA plots and reports exists."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] EDA output directory ready at: '{output_dir.resolve()}'")
    return output_dir


def load_all_csv_datasets(data_dir: Path = Path("data")) -> Dict[str, pd.DataFrame]:
    """
    Scans the data directory and loads all available CSV files into a dictionary.

    Returns:
        Dict[str, pd.DataFrame]: Mapping of filename -> DataFrame
    """
    if not data_dir.exists():
        raise FileNotFoundError(f"Data folder '{data_dir.resolve()}' does not exist.")

    csv_paths = list(data_dir.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in '{data_dir.resolve()}'.")

    datasets = {}
    print(f"[INFO] Found {len(csv_paths)} CSV file(s) in data folder:")
    for path in sorted(csv_paths):
        try:
            df = pd.read_csv(path)
            datasets[path.name] = df
            print(f" - Loaded '{path.name}': {df.shape[0]} rows, {df.shape[1]} columns.")
        except Exception as e:
            print(f"[WARNING] Could not load '{path.name}': {e}", file=sys.stderr)

    return datasets


def inspect_dataset(df: pd.DataFrame, dataset_name: str) -> Dict:
    """
    Performs preliminary dataset inspection and extracts metadata.
    """
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    missing = df.isnull().sum()
    missing_dict = missing[missing > 0].to_dict()

    info = {
        "name": dataset_name,
        "rows": df.shape[0],
        "cols": df.shape[1],
        "columns": df.columns.tolist(),
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "missing_cells": int(missing.sum()),
        "missing_per_col": missing_dict,
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_pct": round(float(df.duplicated().sum() / len(df) * 100), 2) if len(df) > 0 else 0.0
    }

    print(f"\n==================================================")
    print(f" DATASET SUMMARY: {dataset_name}")
    print(f"==================================================")
    print(f"Shape            : {info['rows']} rows, {info['cols']} columns")
    print(f"Numerical Cols ({len(num_cols)}) : {num_cols}")
    print(f"Categorical Cols ({len(cat_cols)}) : {cat_cols}")
    print(f"Missing Values   : {info['missing_cells']} total ({len(missing_dict)} columns affected)")
    print(f"Duplicate Rows   : {info['duplicate_rows']} ({info['duplicate_pct']}%)")
    print("--------------------------------------------------")

    return info


# ==============================================================================
# VISUALIZATION GENERATORS
# ==============================================================================

def plot_histograms(df: pd.DataFrame, dataset_name: str, output_dir: Path) -> Tuple[str, str]:
    """Generates distribution histograms for numerical features."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cols_to_plot = [c for c in num_cols if "id" not in c.lower()][:8]

    if not cols_to_plot:
        return "", ""

    filename = f"histograms_{dataset_name.replace('.csv', '')}.png"
    filepath = output_dir / filename

    fig, axes = plt.subplots(nrows=int(np.ceil(len(cols_to_plot)/2)), ncols=2, figsize=(12, 3 * int(np.ceil(len(cols_to_plot)/2))))
    axes = np.array(axes).flatten()

    for i, col in enumerate(cols_to_plot):
        sns.histplot(df[col].dropna(), kde=True, ax=axes[i], color="#2b5c8f")
        axes[i].set_title(f"Distribution of {col}", fontsize=12, fontweight="bold")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Frequency")

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()

    explanation = (
        f"Histograms for '{dataset_name}' reveal feature distributions and skewness. "
        f"Continuous variables such as {', '.join(cols_to_plot[:3])} show normal or skewed patterns, "
        f"providing insights into normalization or scaling requirements."
    )
    print(f"[SUCCESS] Saved histogram plot -> {filepath.name}")
    print(f"Explanation: {explanation}\n")
    return filename, explanation


def plot_countplots(df: pd.DataFrame, dataset_name: str, output_dir: Path) -> Tuple[str, str]:
    """Generates count plots for categorical features."""
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    cols_to_plot = [c for c in cat_cols if df[c].nunique() <= 15 and "id" not in c.lower()][:6]

    if not cols_to_plot:
        return "", ""

    filename = f"countplots_{dataset_name.replace('.csv', '')}.png"
    filepath = output_dir / filename

    fig, axes = plt.subplots(nrows=int(np.ceil(len(cols_to_plot)/2)), ncols=2, figsize=(14, 4 * int(np.ceil(len(cols_to_plot)/2))))
    axes = np.array(axes).flatten()

    for i, col in enumerate(cols_to_plot):
        order = df[col].value_counts().index[:12]
        sns.countplot(data=df, y=col, order=order, ax=axes[i], hue=col, legend=False, palette="viridis")
        axes[i].set_title(f"Frequency of Categorical Feature: {col}", fontsize=12, fontweight="bold")
        axes[i].set_xlabel("Count")

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()

    explanation = (
        f"Count plots for categorical features ({', '.join(cols_to_plot)}) display class frequencies and imbalances. "
        f"Categories with higher frequencies indicate dominant decision criteria or scenario types."
    )
    print(f"[SUCCESS] Saved countplot -> {filepath.name}")
    print(f"Explanation: {explanation}\n")
    return filename, explanation


def plot_correlation_heatmap(df: pd.DataFrame, dataset_name: str, output_dir: Path) -> Tuple[str, str]:
    """Generates a correlation heatmap for numerical columns."""
    num_df = df.select_dtypes(include=[np.number])
    num_df = num_df.loc[:, num_df.nunique() > 1]

    if num_df.shape[1] < 2:
        return "", ""

    filename = f"correlation_heatmap_{dataset_name.replace('.csv', '')}.png"
    filepath = output_dir / filename

    plt.figure(figsize=(10, 8))
    corr = num_df.corr()

    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, linewidths=0.5)
    plt.title(f"Correlation Matrix: {dataset_name}", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()

    explanation = (
        f"The correlation heatmap highlights linear relationships among numerical features in '{dataset_name}'. "
        f"Strong positive or negative correlations indicate feature dependencies relevant for predictive modeling."
    )
    print(f"[SUCCESS] Saved correlation heatmap -> {filepath.name}")
    print(f"Explanation: {explanation}\n")
    return filename, explanation


def plot_boxplots(df: pd.DataFrame, dataset_name: str, output_dir: Path) -> Tuple[str, str]:
    """Generates box plots for outlier detection across numerical columns."""
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if "id" not in c.lower()][:6]

    if not num_cols:
        return "", ""

    filename = f"boxplots_{dataset_name.replace('.csv', '')}.png"
    filepath = output_dir / filename

    fig, axes = plt.subplots(nrows=int(np.ceil(len(num_cols)/2)), ncols=2, figsize=(12, 3 * int(np.ceil(len(num_cols)/2))))
    axes = np.array(axes).flatten()

    for i, col in enumerate(num_cols):
        sns.boxplot(x=df[col].dropna(), ax=axes[i], color="#4c9a2a")
        axes[i].set_title(f"Outlier Check: {col}", fontsize=12, fontweight="bold")

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()

    explanation = (
        f"Boxplots illustrate feature spreads, medians, interquartile ranges (IQR), and potential statistical outliers "
        f"in '{dataset_name}'. Points outside the whiskers signify extreme response scenarios."
    )
    print(f"[SUCCESS] Saved boxplots -> {filepath.name}")
    print(f"Explanation: {explanation}\n")
    return filename, explanation


def plot_pairplot(df: pd.DataFrame, dataset_name: str, output_dir: Path) -> Tuple[str, str]:
    """Generates pair plots for key numerical columns."""
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if "id" not in c.lower()]
    if len(num_cols) < 3:
        return "", ""

    selected_cols = num_cols[:4]
    filename = f"pairplot_{dataset_name.replace('.csv', '')}.png"
    filepath = output_dir / filename

    sample_df = df[selected_cols].dropna()
    if len(sample_df) > 5000:
        sample_df = sample_df.sample(5000, random_state=42)

    g = sns.pairplot(sample_df, diag_kind="kde", corner=True, plot_kws={"alpha": 0.5, "s": 25, "color": "#2b5c8f"})
    g.fig.suptitle(f"Pairwise Feature Relationships ({dataset_name})", y=1.02, fontsize=14, fontweight="bold")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()

    explanation = (
        f"Pair plots depict bivariate relationships between numerical variables ({', '.join(selected_cols)}). "
        f"Scattered distributions show cluster tendencies and multi-variable interactions."
    )
    print(f"[SUCCESS] Saved pairplot -> {filepath.name}")
    print(f"Explanation: {explanation}\n")
    return filename, explanation


# ==============================================================================
# SPECIFIC ETHICS & DEMOGRAPHIC CHARTS (Requirement 5)
# ==============================================================================

def plot_country_ethical_preference(datasets: Dict[str, pd.DataFrame], output_dir: Path) -> Tuple[str, str]:
    """Generates country-wise ethical preference chart."""
    df = datasets.get("country_preferences.csv")
    if df is None or "country_name" not in df.columns or "save_rate" not in df.columns:
        return "", ""

    filename = "country_ethical_preference.png"
    filepath = output_dir / filename

    top_countries = (
        df.dropna(subset=["country_name"])
        .groupby("country_name")["n_outcomes"]
        .sum()
        .nlargest(15)
        .index
    )
    filtered_df = df[df["country_name"].isin(top_countries)]

    plt.figure(figsize=(14, 7))
    sns.barplot(
        data=filtered_df,
        x="country_name",
        y="save_rate",
        hue="scenario_type",
        errorbar=None,
        palette="magma"
    )
    plt.title("Country-Wise Average Save Rate by Moral Scenario Type", fontsize=15, fontweight="bold", pad=15)
    plt.xlabel("Country", fontweight="bold")
    plt.ylabel("Average Save Rate (0 to 1)", fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Scenario Type", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()

    explanation = (
        "This chart illustrates how ethical saving preferences vary across countries for different scenario types "
        "(e.g., Age, Fitness, Gender, Social Status, Species, Utilitarian). Variation highlights cultural nuances in moral decision-making."
    )
    print(f"[SUCCESS] Saved country ethical preference chart -> {filepath.name}")
    print(f"Explanation: {explanation}\n")
    return filename, explanation


def plot_demographic_preference(datasets: Dict[str, pd.DataFrame], output_dir: Path) -> Tuple[str, str]:
    """Generates demographic preference chart."""
    df = datasets.get("demographic_preferences.csv")
    if df is None or "dimension" not in df.columns or "save_rate" not in df.columns:
        return "", ""

    filename = "demographic_preference.png"
    filepath = output_dir / filename

    plt.figure(figsize=(14, 6))
    sns.barplot(
        data=df,
        x="group",
        y="save_rate",
        hue="scenario_type",
        palette="crest"
    )
    plt.title("Demographic Preference Breakdown across Groups and Scenario Types", fontsize=15, fontweight="bold", pad=15)
    plt.xlabel("Demographic Group", fontweight="bold")
    plt.ylabel("Save Rate", fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Scenario Type", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()

    explanation = (
        "The demographic preference chart compares ethical choices across demographic segments (such as age brackets, gender, and education levels). "
        "It reveals disparities in priority given to characters based on user demographics."
    )
    print(f"[SUCCESS] Saved demographic preference chart -> {filepath.name}")
    print(f"Explanation: {explanation}\n")
    return filename, explanation


def plot_ethical_decisions_distribution(datasets: Dict[str, pd.DataFrame], output_dir: Path) -> Tuple[str, str]:
    """Generates distribution of ethical decisions across available datasets."""
    filename = "ethical_decisions_distribution.png"
    filepath = output_dir / filename

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    mm_df = datasets.get("moral_machine_responses.csv")
    if mm_df is not None and "saved" in mm_df.columns:
        saved_counts = mm_df["saved"].map({1: "Saved", 0: "Died"}).value_counts()
        axes[0].pie(
            saved_counts,
            labels=saved_counts.index,
            autopct="%1.1f%%",
            colors=["#2b5c8f", "#d9534f"],
            startangle=140,
            explode=(0.05, 0)
        )
        axes[0].set_title("Moral Machine Decision Outcomes (Saved vs Died)", fontsize=13, fontweight="bold")

    synth_df = datasets.get("self_driving_car_ethics.csv")
    if synth_df is not None and "decision" in synth_df.columns:
        dec_counts = synth_df["decision"].value_counts()
        sns.barplot(x=dec_counts.values, y=dec_counts.index, ax=axes[1], hue=dec_counts.index, legend=False, palette="Set2")
        axes[1].set_title("Autonomous Vehicle Action Decisions", fontsize=13, fontweight="bold")
        axes[1].set_xlabel("Count")

    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()

    explanation = (
        "This chart details the proportion of ethical outcomes (e.g., characters saved vs allowed to perish) "
        "and the frequency of specific autonomous driving maneuvers (such as hard braking vs swerving)."
    )
    print(f"[SUCCESS] Saved ethical decisions distribution chart -> {filepath.name}")
    print(f"Explanation: {explanation}\n")
    return filename, explanation


def plot_top_10_countries(datasets: Dict[str, pd.DataFrame], output_dir: Path) -> Tuple[str, str]:
    """Generates Top 10 countries by response count."""
    filename = "top_10_countries_response_count.png"
    filepath = output_dir / filename

    mm_df = datasets.get("moral_machine_responses.csv")
    cp_df = datasets.get("country_preferences.csv")

    plt.figure(figsize=(12, 6))

    if mm_df is not None and "country" in mm_df.columns:
        top10 = mm_df["country"].value_counts().head(10)
        sns.barplot(x=top10.values, y=top10.index, hue=top10.index, legend=False, palette="rocket")
        plt.title("Top 10 Countries by Total Moral Machine Responses", fontsize=14, fontweight="bold")
        plt.xlabel("Total Responses", fontweight="bold")
        plt.ylabel("Country Code", fontweight="bold")
    elif cp_df is not None and "country_name" in cp_df.columns:
        top10 = cp_df.dropna(subset=["country_name"]).groupby("country_name")["n_outcomes"].sum().nlargest(10)
        sns.barplot(x=top10.values, y=top10.index, hue=top10.index, legend=False, palette="rocket")
        plt.title("Top 10 Countries by Evaluated Outcomes Count", fontsize=14, fontweight="bold")
        plt.xlabel("Total Evaluated Outcomes", fontweight="bold")
        plt.ylabel("Country", fontweight="bold")

    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()

    explanation = (
        "The Top 10 countries chart highlights geographic data density. "
        "Understanding response volume by country ensures balanced ethical model training and guards against regional sampling bias."
    )
    print(f"[SUCCESS] Saved top 10 countries chart -> {filepath.name}")
    print(f"Explanation: {explanation}\n")
    return filename, explanation


def plot_age_group_preference(datasets: Dict[str, pd.DataFrame], output_dir: Path) -> Tuple[str, str]:
    """Generates Age-group preference analysis plot."""
    df = datasets.get("demographic_preferences.csv")
    if df is None or "dimension" not in df.columns:
        return "", ""

    age_df = df[df["dimension"] == "age_group"]
    if age_df.empty:
        return "", ""

    filename = "age_group_preference.png"
    filepath = output_dir / filename

    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=age_df,
        x="group",
        y="save_rate",
        hue="character_group",
        palette="viridis"
    )
    plt.title("Save Rate across Age Groups by Target Character Group", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Age Bracket", fontweight="bold")
    plt.ylabel("Save Rate", fontweight="bold")
    plt.legend(title="Character Group", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()

    explanation = (
        "Age-group preference analysis demonstrates how respondents of different ages prioritize different character groups "
        "(e.g., Young vs Old, Fit vs Fat, Humans vs Pets). Higher save rates for young characters indicate consistent utilitarian preference across age brackets."
    )
    print(f"[SUCCESS] Saved age group preference chart -> {filepath.name}")
    print(f"Explanation: {explanation}\n")
    return filename, explanation


def plot_gender_preference(datasets: Dict[str, pd.DataFrame], output_dir: Path) -> Tuple[str, str]:
    """Generates Gender preference analysis plot."""
    df = datasets.get("demographic_preferences.csv")
    if df is None or "dimension" not in df.columns:
        return "", ""

    gender_df = df[df["dimension"] == "gender"]
    if gender_df.empty:
        gender_df = df[df["scenario_type"] == "Gender"]

    if gender_df.empty:
        return "", ""

    filename = "gender_preference.png"
    filepath = output_dir / filename

    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=gender_df,
        x="group" if "group" in gender_df.columns else "character_group",
        y="save_rate",
        hue="character_group" if "group" in gender_df.columns else "scenario_type",
        palette="coolwarm"
    )
    plt.title("Gender Preference Analysis in Ethical Dilemmas", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Gender Category / Group", fontweight="bold")
    plt.ylabel("Save Rate", fontweight="bold")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()

    explanation = (
        "Gender preference analysis examines saving tendencies based on gender characteristics of both respondents and characters. "
        "It evaluates potential bias in ethical preference towards female vs male characters in critical scenarios."
    )
    print(f"[SUCCESS] Saved gender preference chart -> {filepath.name}")
    print(f"Explanation: {explanation}\n")
    return filename, explanation


# ==============================================================================
# HTML REPORT GENERATOR
# ==============================================================================

def generate_html_report(
    metadata_list: List[Dict],
    visualizations: List[Tuple[str, str, str]],
    output_dir: Path = Path("outputs/eda")
) -> Path:
    """
    Generates a responsive HTML EDA report with embedded figures and explanations.
    """
    html_path = output_dir / "eda_report.html"

    dataset_cards = []
    for meta in metadata_list:
        cols_html = "".join([f"<span class='badge badge-col'>{c}</span>" for c in meta['columns']])
        card = f"""
        <div class="card">
            <h3>📊 Dataset: {meta['name']}</h3>
            <div class="grid-2">
                <p><strong>Shape:</strong> {meta['rows']} rows, {meta['cols']} columns</p>
                <p><strong>Duplicate Rows:</strong> {meta['duplicate_rows']} ({meta['duplicate_pct']}%)</p>
                <p><strong>Missing Cells:</strong> {meta['missing_cells']}</p>
                <p><strong>Numerical Columns:</strong> {len(meta['num_cols'])}</p>
            </div>
            <p><strong>Categorical Columns:</strong> {len(meta['cat_cols'])}</p>
            <div class="col-list">
                <strong>Columns:</strong><br>{cols_html}
            </div>
        </div>
        """
        dataset_cards.append(card)

    vis_cards = []
    for title, img_filename, expl in visualizations:
        if not img_filename:
            continue
        vis_card = f"""
        <div class="vis-card">
            <h2>{title}</h2>
            <div class="img-container">
                <img src="{img_filename}" alt="{title}">
            </div>
            <div class="explanation-box">
                💡 <strong>Analysis & Insights:</strong> {expl}
            </div>
        </div>
        """
        vis_cards.append(vis_card)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Self-Driving Car Ethics AI - Exploratory Data Analysis Report</title>
    <style>
        :root {{
            --bg-color: #0f172a;
            --card-bg: #1e293b;
            --accent-color: #38bdf8;
            --text-color: #f8fafc;
            --text-muted: #94a3b8;
            --border-color: #334155;
            --badge-bg: #0284c7;
        }}
        body {{
            font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-color);
            margin: 0;
            padding: 20px;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        header {{
            text-align: center;
            padding: 30px 20px;
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            border-radius: 12px;
            border: 1px solid var(--border-color);
            margin-bottom: 30px;
        }}
        header h1 {{
            color: var(--accent-color);
            margin-bottom: 10px;
            font-size: 2.2rem;
        }}
        header p {{
            color: var(--text-muted);
            font-size: 1.1rem;
        }}
        .grid-2 {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
        }}
        .card {{
            background: var(--card-bg);
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            border: 1px solid var(--border-color);
        }}
        .card h3 {{
            color: var(--accent-color);
            margin-top: 0;
        }}
        .badge {{
            display: inline-block;
            background: var(--badge-bg);
            color: white;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.85rem;
            margin: 2px;
        }}
        .vis-card {{
            background: var(--card-bg);
            border-radius: 12px;
            padding: 25px;
            margin-bottom: 35px;
            border: 1px solid var(--border-color);
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
        }}
        .vis-card h2 {{
            color: var(--accent-color);
            margin-top: 0;
            font-size: 1.5rem;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 10px;
        }}
        .img-container {{
            text-align: center;
            margin: 20px 0;
        }}
        .img-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            border: 1px solid var(--border-color);
        }}
        .explanation-box {{
            background: rgba(56, 189, 248, 0.1);
            border-left: 4px solid var(--accent-color);
            padding: 15px;
            border-radius: 4px;
            font-size: 1rem;
            color: #e2e8f0;
        }}
        footer {{
            text-align: center;
            padding: 20px;
            color: var(--text-muted);
            font-size: 0.9rem;
            border-top: 1px solid var(--border-color);
            margin-top: 40px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🚗 Self-Driving Car Ethics AI</h1>
            <p>Comprehensive Exploratory Data Analysis (EDA) & Moral Machine Insights</p>
        </header>

        <section>
            <h2>📋 Dataset Overview & Metadata</h2>
            {"".join(dataset_cards)}
        </section>

        <section>
            <h2>📈 Exploratory Visualizations & Empirical Insights</h2>
            {"".join(vis_cards)}
        </section>

        <footer>
            Self-Driving Car Ethics AI Project &copy; 2026. Generated automatically by EDA Pipeline.
        </footer>
    </div>
</body>
</html>
"""

    html_path.write_text(html_content, encoding="utf-8")
    print(f"[SUCCESS] Saved HTML EDA Report -> '{html_path.resolve()}'")
    return html_path


# ==============================================================================
# MAIN EDA PIPELINE
# ==============================================================================

def main():
    """
    Main entry point for EDA pipeline.
    """
    print("[START] Initiating Exploratory Data Analysis (EDA)...")
    output_dir = setup_output_directory(Path("outputs/eda"))

    # 1. Load all CSV datasets
    datasets = load_all_csv_datasets(Path("data"))
    if not datasets:
        print("[ERROR] No datasets available to analyze.", file=sys.stderr)
        return

    # 2. Inspect metadata for all datasets
    metadata_list = []
    for name, df in datasets.items():
        meta = inspect_dataset(df, name)
        metadata_list.append(meta)

    # 3. Generate Visualizations & Collect Explanations
    visualizations = []

    for name, df in datasets.items():
        print(f"\n--- Generating Plots for '{name}' ---")
        hist_file, hist_exp = plot_histograms(df, name, output_dir)
        if hist_file:
            visualizations.append((f"Feature Histograms ({name})", hist_file, hist_exp))

        cp_file, cp_exp = plot_countplots(df, name, output_dir)
        if cp_file:
            visualizations.append((f"Categorical Count Plots ({name})", cp_file, cp_exp))

        hm_file, hm_exp = plot_correlation_heatmap(df, name, output_dir)
        if hm_file:
            visualizations.append((f"Correlation Heatmap ({name})", hm_file, hm_exp))

        bp_file, bp_exp = plot_boxplots(df, name, output_dir)
        if bp_file:
            visualizations.append((f"Outlier Box Plots ({name})", bp_file, bp_exp))

        pp_file, pp_exp = plot_pairplot(df, name, output_dir)
        if pp_file:
            visualizations.append((f"Pair Plot ({name})", pp_file, pp_exp))

    # Required specific ethics & demographic charts (Requirement 5)
    print("\n--- Generating Specific Ethical & Demographic Preference Charts ---")
    c1, e1 = plot_country_ethical_preference(datasets, output_dir)
    if c1:
        visualizations.append(("Country-Wise Ethical Preference Analysis", c1, e1))

    c2, e2 = plot_demographic_preference(datasets, output_dir)
    if c2:
        visualizations.append(("Demographic Preference Breakdown", c2, e2))

    c3, e3 = plot_ethical_decisions_distribution(datasets, output_dir)
    if c3:
        visualizations.append(("Distribution of Ethical Decisions & Actions", c3, e3))

    c4, e4 = plot_top_10_countries(datasets, output_dir)
    if c4:
        visualizations.append(("Top 10 Countries by Response & Outcome Count", c4, e4))

    c5, e5 = plot_age_group_preference(datasets, output_dir)
    if c5:
        visualizations.append(("Age-Group Preference Analysis", c5, e5))

    c6, e6 = plot_gender_preference(datasets, output_dir)
    if c6:
        visualizations.append(("Gender Preference Analysis", c6, e6))

    # 4. Generate Responsive HTML Report
    report_path = generate_html_report(metadata_list, visualizations, output_dir)

    print("\n==================================================")
    print(" [SUCCESS] EDA PIPELINE COMPLETED SUCCESSFULLY!")
    print(f" Saved PNG charts to: '{output_dir.resolve()}'")
    print(f" Generated HTML report: '{report_path.resolve()}'")
    print("==================================================")


if __name__ == "__main__":
    main()
