"""
Self-Driving Car Ethics AI - Data Preprocessing Module

This module automatically detects a CSV dataset in the `data/` folder,
loads it with pandas, computes key dataset statistics and data quality metrics,
prints them to console, and exports a report to `outputs/data_report.txt`.
"""

import sys
from pathlib import Path
from typing import Optional
import pandas as pd


def find_csv_file(data_dir: Path = Path("data")) -> Path:
    """
    Detects the first CSV file present in the data folder.

    Args:
        data_dir (Path): Path to the directory containing dataset files.

    Returns:
        Path: Path to the detected CSV file.

    Raises:
        FileNotFoundError: If the directory does not exist or no CSV file is found.
    """
    if not data_dir.exists() or not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory '{data_dir.resolve()}' does not exist.")

    csv_files = sorted(list(data_dir.glob("*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in '{data_dir.resolve()}'. Please add a dataset CSV file.")

    if len(csv_files) > 1:
        print(f"[INFO] Multiple CSV files found. Automatically selecting: {csv_files[0].name}")

    return csv_files[0]


def load_dataset(file_path: Path) -> pd.DataFrame:
    """
    Loads dataset from a CSV file into a pandas DataFrame.

    Args:
        file_path (Path): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.

    Raises:
        Exception: If reading the CSV fails.
    """
    try:
        print(f"[INFO] Loading dataset from '{file_path.name}'...")
        df = pd.read_csv(file_path)
        print(f"[SUCCESS] Dataset successfully loaded with {df.shape[0]} rows and {df.shape[1]} columns.")
        return df
    except pd.errors.EmptyDataError:
        raise ValueError(f"Dataset file '{file_path}' is empty.")
    except pd.errors.ParserError as e:
        raise ValueError(f"Error parsing CSV file '{file_path}': {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to read CSV file '{file_path}': {e}")


def generate_report(df: pd.DataFrame, file_name: str) -> str:
    """
    Generates a structured text report containing data quality and summary statistics.

    Args:
        df (pd.DataFrame): Dataset DataFrame.
        file_name (str): Name of the source file.

    Returns:
        str: Multi-line formatted text report.
    """
    sep = "=" * 80
    sub_sep = "-" * 80

    lines = [
        sep,
        f" DATA QUALITY & EXPLORATORY REPORT: {file_name}",
        sep,
        "",
        "1. DATASET SHAPE",
        sub_sep,
        f"Rows: {df.shape[0]}",
        f"Columns: {df.shape[1]}",
        "",
        "2. COLUMN NAMES & DATA TYPES",
        sub_sep,
        df.dtypes.to_string(),
        "",
        "3. FIRST 10 ROWS",
        sub_sep,
        df.head(10).to_string(),
        "",
        "4. LAST 10 ROWS",
        sub_sep,
        df.tail(10).to_string(),
        "",
        "5. MISSING VALUES",
        sub_sep,
        df.isnull().sum().to_string(),
        f"\nTotal Missing Cells: {df.isnull().sum().sum()}",
        "",
        "6. DUPLICATE ROWS",
        sub_sep,
        f"Duplicate Rows Count: {df.duplicated().sum()}",
        f"Duplicate Rows Percentage: {(df.duplicated().sum() / len(df) * 100):.2f}%" if len(df) > 0 else "0.00%",
        "",
        "7. SUMMARY STATISTICS (NUMERICAL & CATEGORICAL)",
        sub_sep,
        df.describe(include="all").to_string(),
        "",
        sep,
        " END OF REPORT",
        sep
    ]

    return "\n".join(lines)


def save_report(report_content: str, output_dir: Path = Path("outputs"), filename: str = "data_report.txt") -> Path:
    """
    Saves the data report string to an output file.

    Args:
        report_content (str): The report content to write.
        output_dir (Path): Target output directory.
        filename (str): Report filename.

    Returns:
        Path: Path to saved report file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / filename
    report_path.write_text(report_content, encoding="utf-8")
    print(f"[SUCCESS] Data quality report saved to '{report_path.resolve()}'")
    return report_path


def main() -> Optional[pd.DataFrame]:
    """
    Main function to execute the preprocessing pipeline:
    detects dataset, loads CSV, prints dataset inspection, and exports data report.
    """
    try:
        data_dir = Path("data")
        csv_path = find_csv_file(data_dir)
        df = load_dataset(csv_path)

        # Generate report content
        report_text = generate_report(df, csv_path.name)

        # Print report to console
        print("\n" + report_text + "\n")

        # Save report to outputs folder
        save_report(report_text, output_dir=Path("outputs"), filename="data_report.txt")

        return df

    except FileNotFoundError as fnf_err:
        print(f"[ERROR] {fnf_err}", file=sys.stderr)
        return None
    except Exception as err:
        print(f"[UNEXPECTED ERROR] {err}", file=sys.stderr)
        return None


if __name__ == "__main__":
    main()
