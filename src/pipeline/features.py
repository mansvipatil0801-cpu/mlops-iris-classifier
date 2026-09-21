"""
Stage 3: Feature Engineering.
Derives new, model-useful features from the raw measurements.
"""

import argparse
import logging
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("features")


def engineer_features(input_path: str, output_path: str) -> pd.DataFrame:
    """Create new engineered features from the preprocessed Iris dataset."""

    # Load dataset
    df = pd.read_csv(input_path)

    # Original features:
    # sepal length, sepal width, petal length, petal width

    # 1. Sepal Area
    df["sepal_area"] = df["sepal length (cm)"] * df["sepal width (cm)"]

    # 2. Petal Area
    df["petal_area"] = df["petal length (cm)"] * df["petal width (cm)"]

    # 3. Sepal-to-Petal Length Ratio
    df["sepal_to_petal_length_ratio"] = (
        df["sepal length (cm)"] /
        df["petal length (cm)"].replace(0, pd.NA)
    )

    # 4. Petal Length Category
    df["petal_length_bin"] = pd.cut(
        df["petal length (cm)"],
        bins=[0, 2, 4.5, 7],
        labels=["short", "medium", "long"]
    )

    # Save engineered dataset
    df.to_csv(output_path, index=False)

    logger.info("Engineered %d features -> %s", df.shape[1], output_path)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/processed/iris_preprocessed.csv"
    )
    parser.add_argument(
        "--output",
        default="data/processed/iris_features.csv"
    )

    args = parser.parse_args()
    engineer_features(args.input, args.output)