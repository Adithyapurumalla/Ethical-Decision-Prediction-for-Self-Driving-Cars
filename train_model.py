"""
Self-Driving Car Ethics AI - Production Machine Learning Pipeline

This module implements a production-grade machine learning training and evaluation
pipeline for the Self-Driving Car Ethics AI project.

Key Features:
- Dataset discovery and target column auto-detection
- Sklearn ColumnTransformer & Pipeline for median imputation, scaling, and one-hot encoding
- 80/20 Stratified train-test split with random_state=42
- Model training & comparison: Logistic Regression, Decision Tree, Random Forest, XGBoost
- 5-Fold Cross Validation & performance evaluation (Accuracy, Precision, Recall, F1, ROC-AUC)
- Automatic selection and serialization of the best model (models/best_model.pkl, models/preprocessor.pkl)
- Generation of visualization artifacts (Confusion Matrix, Feature Importance, Model Comparison, ROC Curves)
- Export of model_metrics.csv and detailed model_report.txt
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve, auc
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# Set visualization style
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.sans-serif": "Arial",
    "font.family": "sans-serif",
    "figure.autolayout": True
})


def setup_directories() -> Tuple[Path, Path]:
    """Ensures models/ and outputs/models/ directories exist."""
    models_dir = Path("models")
    outputs_dir = Path("outputs/models")

    models_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Models directory ready: '{models_dir.resolve()}'")
    print(f"[INFO] Outputs directory ready: '{outputs_dir.resolve()}'")

    return models_dir, outputs_dir


def load_dataset(data_dir: Path = Path("data"), preferred_file: str = "self_driving_car_ethics.csv") -> Tuple[pd.DataFrame, str]:
    """
    Loads the primary dataset or detects an available CSV file in data_dir.
    """
    target_path = data_dir / preferred_file
    if target_path.exists():
        df = pd.read_csv(target_path)
        print(f"[INFO] Loaded main dataset '{preferred_file}' ({df.shape[0]} rows, {df.shape[1]} columns).")
        return df, preferred_file

    csv_files = list(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in data directory '{data_dir.resolve()}'.")

    selected_file = csv_files[0]
    df = pd.read_csv(selected_file)
    print(f"[INFO] Preferred dataset not found. Selected '{selected_file.name}' ({df.shape[0]} rows, {df.shape[1]} columns).")
    return df, selected_file.name


def detect_target_column(df: pd.DataFrame, target_col: Optional[str] = None) -> str:
    """
    Automatically detects the target column or allows configuration.
    """
    if target_col and target_col in df.columns:
        print(f"[INFO] Target column manually specified: '{target_col}'")
        return target_col

    candidates = ["decision", "saved", "ethical_score", "target", "label", "choice", "action"]
    for candidate in candidates:
        if candidate in df.columns:
            print(f"[INFO] Auto-detected target column: '{candidate}'")
            return candidate

    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        detected = cat_cols[-1]
        print(f"[INFO] Target auto-detected from categorical columns: '{detected}'")
        return detected

    print(f"[WARNING] Available columns: {df.columns.tolist()}")
    fallback = df.columns[-1]
    print(f"[INFO] Fallback target column selected: '{fallback}'")
    return fallback


def build_preprocessor(X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    """
    Builds an sklearn ColumnTransformer pipeline for numerical and categorical features.

    - Numerical: SimpleImputer(median) -> StandardScaler
    - Categorical: SimpleImputer(most_frequent) -> OneHotEncoder
    """
    # Exclude ID / metadata columns
    ignore_cols = [col for col in X.columns if "id" in col.lower() or "name" in col.lower()]
    feature_cols = [col for col in X.columns if col not in ignore_cols]

    num_cols = X[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X[feature_cols].select_dtypes(include=["object", "category"]).columns.tolist()

    print(f"[INFO] Feature Processing Strategy:")
    print(f" - Numerical Features ({len(num_cols)}): {num_cols}")
    print(f" - Categorical Features ({len(cat_cols)}): {cat_cols}")
    if ignore_cols:
        print(f" - Ignored Identifier Columns: {ignore_cols}")

    num_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_cols),
            ("cat", cat_transformer, cat_cols)
        ],
        remainder="drop"
    )

    return preprocessor, num_cols, cat_cols


def get_model_dictionary() -> Dict[str, Any]:
    """Returns candidate classifiers for training & benchmark comparison."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=10),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": xgb.XGBClassifier(random_state=42, eval_metric="logloss")
    }


def evaluate_model(
    model_name: str,
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    label_encoder: LabelEncoder
) -> Dict[str, Any]:
    """
    Fits model pipeline, evaluates on test set, performs 5-fold CV, and computes metrics.
    """
    print(f"\n[TRAINING] Fitting {model_name}...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test) if hasattr(pipeline, "predict_proba") else None

    # Perform 5-fold Cross Validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1_weighted")

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    # ROC-AUC calculation
    roc_auc = np.nan
    try:
        if len(label_encoder.classes_) == 2:
            roc_auc = roc_auc_score(y_test, y_proba[:, 1])
        else:
            roc_auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted")
    except Exception:
        pass

    cm = confusion_matrix(y_test, y_pred)
    cls_report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0)

    print(f"[{model_name}] Test Acc: {acc:.4f} | F1: {f1:.4f} | 5-Fold CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    return {
        "name": model_name,
        "pipeline": pipeline,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "cv_mean_f1": cv_scores.mean(),
        "cv_std_f1": cv_scores.std(),
        "confusion_matrix": cm,
        "classification_report": cls_report,
        "y_pred": y_pred,
        "y_proba": y_proba
    }


# ==============================================================================
# VISUALIZATION PLOT EXPORTS
# ==============================================================================

def plot_confusion_matrix(cm: np.ndarray, labels: np.ndarray, model_name: str, output_dir: Path) -> Path:
    """Plots and saves confusion matrix heatmap."""
    filepath = output_dir / "confusion_matrix_best.png"
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, cbar=False)
    plt.title(f"Confusion Matrix ({model_name})", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Predicted Class", fontweight="bold")
    plt.ylabel("Actual Class", fontweight="bold")
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SUCCESS] Saved confusion matrix -> '{filepath.name}'")
    return filepath


def plot_model_comparison(metrics_df: pd.DataFrame, output_dir: Path) -> Path:
    """Plots comparative bar chart across models and metrics."""
    filepath = output_dir / "model_comparison.png"
    melted_df = metrics_df.melt(
        id_vars=["Model"],
        value_vars=["Accuracy", "Precision", "Recall", "F1 Score", "5-Fold CV F1"],
        var_name="Metric",
        value_name="Score"
    )

    plt.figure(figsize=(12, 6))
    sns.barplot(data=melted_df, x="Model", y="Score", hue="Metric", palette="Blues_d")
    plt.title("Model Benchmark Comparison", fontsize=15, fontweight="bold", pad=15)
    plt.ylim(0, 1.05)
    plt.xlabel("Model Architecture", fontweight="bold")
    plt.ylabel("Score (0.0 to 1.0)", fontweight="bold")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SUCCESS] Saved model comparison chart -> '{filepath.name}'")
    return filepath


def plot_feature_importance(best_model_info: Dict, preprocessor: ColumnTransformer, output_dir: Path) -> Optional[Path]:
    """Extracts and plots feature importances or linear coefficients for best model."""
    pipeline = best_model_info["pipeline"]
    model = pipeline.named_steps["classifier"]
    filepath = output_dir / "feature_importance_best.png"

    try:
        feature_names = preprocessor.get_feature_names_out()
        # Clean feature names (remove transformer prefixes like num__ or cat__)
        feature_names = [f.split("__")[-1] for f in feature_names]

        importances = None
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_).mean(axis=0)

        if importances is None or len(importances) != len(feature_names):
            return None

        fi_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
        fi_df = fi_df.sort_values("Importance", ascending=False).head(15)

        plt.figure(figsize=(10, 6))
        sns.barplot(data=fi_df, x="Importance", y="Feature", palette="viridis")
        plt.title(f"Top Feature Importances ({best_model_info['name']})", fontsize=14, fontweight="bold", pad=12)
        plt.xlabel("Relative Importance Score", fontweight="bold")
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[SUCCESS] Saved feature importance plot -> '{filepath.name}'")
        return filepath
    except Exception as e:
        print(f"[WARNING] Feature importance plotting skipped: {e}")
        return None


def plot_roc_curve(best_model_info: Dict, y_test: np.ndarray, labels: np.ndarray, output_dir: Path) -> Optional[Path]:
    """Plots binary or multiclass ROC curve for the best model."""
    y_proba = best_model_info["y_proba"]
    if y_proba is None:
        return None

    filepath = output_dir / "roc_curve_best.png"
    plt.figure(figsize=(8, 6))

    n_classes = len(labels)
    if n_classes == 2:
        fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
        roc_auc_val = auc(fpr, tpr)
        plt.plot(fpr, tpr, color="#2b5c8f", lw=2, label=f"ROC curve (AUC = {roc_auc_val:.3f})")
    else:
        for i in range(n_classes):
            binary_y_test = (y_test == i).astype(int)
            fpr, tpr, _ = roc_curve(binary_y_test, y_proba[:, i])
            roc_auc_val = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=1.5, label=f"Class '{labels[i]}' (AUC = {roc_auc_val:.2f})")

    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontweight="bold")
    plt.ylabel("True Positive Rate", fontweight="bold")
    plt.title(f"ROC Curve ({best_model_info['name']})", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SUCCESS] Saved ROC curve -> '{filepath.name}'")
    return filepath


# ==============================================================================
# REPORT GENERATION & ARTIFACT EXPORT
# ==============================================================================

def save_model_report(
    dataset_name: str,
    X_shape: Tuple[int, int],
    target_col: str,
    metrics_df: pd.DataFrame,
    best_info: Dict,
    models_dir: Path,
    outputs_dir: Path
) -> Path:
    """Exports model_report.txt detailing evaluation metrics and pipeline specs."""
    report_path = outputs_dir / "model_report.txt"

    lines = [
        "=" * 80,
        " SELF-DRIVING CAR ETHICS AI - MACHINE LEARNING TRAINING REPORT",
        "=" * 80,
        "",
        "1. DATASET & PREPROCESSING SUMMARY",
        "-" * 80,
        f"Primary Dataset Name   : {dataset_name}",
        f"Feature Matrix Shape   : {X_shape[0]} samples x {X_shape[1]} features",
        f"Target Column          : '{target_col}'",
        f"Split Ratio            : 80% Training / 20% Testing (random_state=42)",
        f"Numerical Scaling      : StandardScaler",
        f"Categorical Encoding   : OneHotEncoder (handle_unknown='ignore')",
        "",
        "2. MODEL BENCHMARK COMPARISON",
        "-" * 80,
        metrics_df.to_string(index=False),
        "",
        "3. BEST MODEL PERFORMANCE DETAILS",
        "-" * 80,
        f"Selected Best Model    : {best_info['name']}",
        f"Test Accuracy          : {best_info['accuracy']:.4f}",
        f"Weighted F1 Score      : {best_info['f1_score']:.4f}",
        f"5-Fold CV Mean F1      : {best_info['cv_mean_f1']:.4f} (+/- {best_info['cv_std_f1']:.4f})",
        "",
        "Classification Report:",
        best_info["classification_report"],
        "",
        "4. SAVED ARTIFACT LOCATIONS",
        "-" * 80,
        f"Best Model Binary      : {models_dir / 'best_model.pkl'}",
        f"Preprocessor Binary    : {models_dir / 'preprocessor.pkl'}",
        f"Metrics CSV            : {models_dir / 'model_metrics.csv'}",
        f"Confusion Matrix PNG   : {outputs_dir / 'confusion_matrix_best.png'}",
        f"Model Comparison PNG   : {outputs_dir / 'model_comparison.png'}",
        f"Feature Importance PNG : {outputs_dir / 'feature_importance_best.png'}",
        f"ROC Curve PNG          : {outputs_dir / 'roc_curve_best.png'}",
        "",
        "=" * 80,
        " END OF TRAINING REPORT",
        "=" * 80
    ]

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[SUCCESS] Saved detailed training report to '{report_path.resolve()}'")
    return report_path


# ==============================================================================
# MAIN PIPELINE EXECUTION
# ==============================================================================

def main():
    """
    Main execution workflow for production ML pipeline.
    """
    print("==================================================")
    print(" [START] Self-Driving Car Ethics AI - ML Pipeline")
    print("==================================================")

    # 1. Setup Directories
    models_dir, outputs_dir = setup_directories()

    # 2. Load Dataset
    df, dataset_name = load_dataset(Path("data"), preferred_file="self_driving_car_ethics.csv")

    # 3. Detect Target Column
    target_col = detect_target_column(df)

    # Separate Features and Target
    X = df.drop(columns=[target_col])
    y_raw = df[target_col]

    # Encode Target Labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    class_names = label_encoder.classes_
    print(f"[INFO] Target Classes ({len(class_names)}): {list(class_names)}")

    # 4. Build Preprocessor
    preprocessor, num_cols, cat_cols = build_preprocessor(X)

    # 5. Train / Test Split (80/20)
    stratify_option = y if len(np.unique(y)) > 1 and min(pd.Series(y).value_counts()) >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify_option
    )
    print(f"[INFO] Dataset split completed: Train={X_train.shape[0]} samples, Test={X_test.shape[0]} samples.")

    # Fit preprocessor on training data and transform both
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)

    # 6. Model Benchmarking
    candidate_models = get_model_dictionary()
    results = []

    for name, clf in candidate_models.items():
        # Construct full Pipeline with preprocessor + classifier
        model_pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", clf)
        ])

        res = evaluate_model(name, model_pipeline, X_train, y_train, X_test, y_test, label_encoder)
        results.append(res)

    # 7. Model Selection
    results.sort(key=lambda r: r["f1_score"], reverse=True)
    best_res = results[0]

    print("\n==================================================")
    print(f" [BEST MODEL SELECTED]: {best_res['name']}")
    print(f" F1-Score: {best_res['f1_score']:.4f} | Accuracy: {best_res['accuracy']:.4f} | 5-Fold CV F1: {best_res['cv_mean_f1']:.4f}")
    print("==================================================")

    # 8. Save Metrics CSV & Serialized Models
    metrics_summary = []
    for r in results:
        metrics_summary.append({
            "Model": r["name"],
            "Accuracy": round(r["accuracy"], 4),
            "Precision": round(r["precision"], 4),
            "Recall": round(r["recall"], 4),
            "F1 Score": round(r["f1_score"], 4),
            "ROC-AUC": round(r["roc_auc"], 4) if not np.isnan(r["roc_auc"]) else "N/A",
            "5-Fold CV F1": round(r["cv_mean_f1"], 4),
            "5-Fold CV Std": round(r["cv_std_f1"], 4)
        })
    metrics_df = pd.DataFrame(metrics_summary)
    metrics_csv_path = models_dir / "model_metrics.csv"
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"[SUCCESS] Exported model metrics to '{metrics_csv_path.resolve()}'")

    # Serialize Best Model and Preprocessor
    joblib.dump(best_res["pipeline"], models_dir / "best_model.pkl")
    joblib.dump(preprocessor, models_dir / "preprocessor.pkl")
    print(f"[SUCCESS] Saved best model binary to '{models_dir / 'best_model.pkl'}'")
    print(f"[SUCCESS] Saved preprocessor binary to '{models_dir / 'preprocessor.pkl'}'")

    # 9. Generate Visualizations inside outputs/models/
    plot_confusion_matrix(best_res["confusion_matrix"], class_names, best_res["name"], outputs_dir)
    plot_model_comparison(metrics_df, outputs_dir)
    plot_feature_importance(best_res, preprocessor, outputs_dir)
    plot_roc_curve(best_res, y_test, class_names, outputs_dir)

    # 10. Generate Model Report
    save_model_report(
        dataset_name,
        X.shape,
        target_col,
        metrics_df,
        best_res,
        models_dir,
        outputs_dir
    )

    print("\n==================================================")
    print(" [SUCCESS] ML PIPELINE TRAINED & EVALUATED SUCCESSFULLY!")
    print(" All artifacts saved to models/ and outputs/models/")
    print("==================================================")


if __name__ == "__main__":
    main()
