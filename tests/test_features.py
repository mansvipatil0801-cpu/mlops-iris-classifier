
import pandas as pd

from src.pipeline.features import engineer_features


def test_sepal_and_petal_area_computed_correctly(tmp_path):
    df = pd.DataFrame({
        "sepal length (cm)": [5.0],
        "sepal width (cm)": [4.0],
        "petal length (cm)": [2.0],
        "petal width (cm)": [1.0],
        "species": ["setosa"],
    })

    input_path = tmp_path / "in.csv"
    output_path = tmp_path / "out.csv"
    df.to_csv(input_path, index=False)

    result = engineer_features(
        str(input_path), str(output_path)
    )

    assert result.loc[0, "sepal_area"] == 20.0
    assert result.loc[0, "petal_area"] == 2.0
    assert result.loc[
        0, "sepal_to_petal_length_ratio"
    ] == 2.5


def test_petal_length_bin_assigns_expected_category(tmp_path):
    df = pd.DataFrame({
        "sepal length (cm)": [5.0, 5.0, 5.0],
        "sepal width (cm)": [3.0, 3.0, 3.0],
        "petal length (cm)": [1.0, 3.0, 6.0],
        "petal width (cm)": [0.2, 1.0, 2.0],
        "species": ["setosa", "versicolor", "virginica"],
    })

    input_path = tmp_path / "in.csv"
    output_path = tmp_path / "out.csv"
    df.to_csv(input_path, index=False)

    result = engineer_features(
        str(input_path), str(output_path)
    )

    assert list(result["petal_length_bin"]) == [
        "short", "medium", "long"
    ]


def test_engineer_features_handles_zero_petal_length_without_crashing(
    tmp_path
):
    df = pd.DataFrame({
        "sepal length (cm)": [5.0],
        "sepal width (cm)": [3.0],
        "petal length (cm)": [0.0],
        "petal width (cm)": [0.2],
        "species": ["setosa"],
    })

    input_path = tmp_path / "in.csv"
    output_path = tmp_path / "out.csv"
    df.to_csv(input_path, index=False)

    result = engineer_features(
        str(input_path), str(output_path)
    )

    assert pd.isna(
        result.loc[0, "sepal_to_petal_length_ratio"]
    )


def test_engineer_features_preserves_row_count(tmp_path):
    import pandas as pd
    from src.pipeline.features import engineer_features

    input_df = pd.DataFrame({
        "sepal length (cm)": [5.1, 6.2],
        "sepal width (cm)": [3.5, 3.4],
        "petal length (cm)": [1.4, 4.5],
        "petal width (cm)": [0.2, 1.5],
        "species": ["setosa", "versicolor"],
    })

    input_file = tmp_path / "input.csv"
    output_file = tmp_path / "output.csv"
    input_df.to_csv(input_file, index=False)

    result = engineer_features(str(input_file), str(output_file))

    assert len(result) == len(input_df)


def test_engineer_features_returns_dataframe(tmp_path):
    import pandas as pd
    from src.pipeline.features import engineer_features

    input_df = pd.DataFrame({
        "sepal length (cm)": [5.1],
        "sepal width (cm)": [3.5],
        "petal length (cm)": [1.4],
        "petal width (cm)": [0.2],
        "species": ["setosa"],
    })

    input_file = tmp_path / "input.csv"
    output_file = tmp_path / "output.csv"
    input_df.to_csv(input_file, index=False)

    result = engineer_features(str(input_file), str(output_file))

    assert isinstance(result, pd.DataFrame)
