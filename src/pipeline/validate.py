# src/pipeline/validate.py
"""
Stage 4: Data Validation.
Schema and statistical checks that must pass before data can
flow downstream to training. Raises on failure to hard-stop the pipeline.
"""

import argparse
import logging
import sys

import pandas as pd


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("validate")


EXPECTED_COLUMNS = {
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
    "species",
    "sepal_area",
    "petal_area",
    "sepal_to_petal_length_ratio",
    "petal_length_bin",
}

VALID_SPECIES = {"setosa", "versicolor", "virginica"}

RANGE_CHECKS = {
    "sepal length (cm)": (3.0, 9.0),
    "sepal width (cm)": (1.5, 5.5),
    "petal length (cm)": (0.5, 8.0),
    "petal width (cm)": (0.05, 3.0),
}


class DataValidationError(Exception):
    """Raised when data fails validation."""


def validate(input_path: str) -> pd.DataFrame:
    """Read and validate a CSV file, then return its DataFrame."""
    df = pd.read_csv(input_path)
    errors = []

    # 1. Check required columns.
    missing_cols = EXPECTED_COLUMNS - set(df.columns)
    if missing_cols:
        errors.append(f"Missing expected columns: {missing_cols}")

    # 2. Check for nulls.
    if df.isnull().any().any():
        null_cols = df.columns[df.isnull().any()].tolist()
        errors.append(
            f"Unexpected null values in columns: {null_cols}"
        )

    # 3. Check species values only if the column exists.
    if "species" in df.columns:
        invalid_species = (
            set(df["species"].dropna().unique()) - VALID_SPECIES
        )
        if invalid_species:
            errors.append(
                f"Unexpected species values: {invalid_species}"
            )

    # 4. Check numeric ranges only for columns that exist.
    for col, (low, high) in RANGE_CHECKS.items():
        if col not in df.columns:
            continue

        values = pd.to_numeric(df[col], errors="coerce")
        out_of_range = values.notna() & (
            (values < low) | (values > high)
        )

        count = int(out_of_range.sum())
        if count:
            errors.append(
                f"{count} rows out of expected range for "
                f"'{col}' ({low}-{high})"
            )

    # 5. Raise an exception containing the detailed errors.
    if errors:
        message = (
            f"Validation failed with {len(errors)} error(s): "
            + "; ".join(errors)
        )
        logger.error(message)
        raise DataValidationError(message)

    logger.info(
        "Validation PASSED: %d rows, %d columns, all checks satisfied",
        len(df),
        df.shape[1],
    )
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/processed/iris_features.csv",
    )
    args = parser.parse_args()

    try:
        validate(args.input)
    except DataValidationError as e:
        logger.error("Pipeline halted: %s", e)
        sys.exit(1)
