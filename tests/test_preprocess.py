
import pandas as pd
import pytest

from src.pipeline.preprocess import preprocess

NUMERIC_COLS = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]


@pytest.fixture
def valid_raw_df(tmp_path):
    df = pd.DataFrame({
        "sepal length (cm)": [5.1, 4.9, 4.7],
        "sepal width (cm)": [3.5, 3.0, 3.2],
        "petal length (cm)": [1.4, 1.4, 1.3],
        "petal width (cm)": [0.2, 0.2, 0.2],
        "species": ["setosa", "setosa", "setosa"],
        "collected_at": ["2026-01-01"] * 3,
    })

    path = tmp_path / "raw.csv"
    df.to_csv(path, index=False)
    return path


def test_preprocess_removes_duplicates(tmp_path, valid_raw_df):
    df = pd.read_csv(valid_raw_df)
    df_with_dupe = pd.concat(
        [df, df.iloc[[0]]], ignore_index=True
    )

    input_path = tmp_path / "raw_with_dupe.csv"
    output_path = tmp_path / "out.csv"
    df_with_dupe.to_csv(input_path, index=False)

    result = preprocess(str(input_path), str(output_path))

    assert len(result) == 3


def test_preprocess_imputes_missing_numeric_values(
    tmp_path, valid_raw_df
):
    df = pd.read_csv(valid_raw_df)
    df.loc[0, "sepal length (cm)"] = None

    input_path = tmp_path / "raw_with_missing.csv"
    output_path = tmp_path / "out.csv"
    df.to_csv(input_path, index=False)

    result = preprocess(str(input_path), str(output_path))

    assert result["sepal length (cm)"].isnull().sum() == 0

    expected_median = df["sepal length (cm)"].median()
    assert result.loc[0, "sepal length (cm)"] == pytest.approx(
        expected_median
    )


def test_preprocess_drops_rows_with_missing_target(
    tmp_path, valid_raw_df
):
    df = pd.read_csv(valid_raw_df)
    df.loc[0, "species"] = None

    input_path = tmp_path / "raw_missing_target.csv"
    output_path = tmp_path / "out.csv"
    df.to_csv(input_path, index=False)

    result = preprocess(str(input_path), str(output_path))

    assert len(result) == 2


def test_preprocess_handles_empty_dataframe(tmp_path):
    empty_df = pd.DataFrame(
        columns=NUMERIC_COLS + ["species", "collected_at"]
    )

    input_path = tmp_path / "empty.csv"
    output_path = tmp_path / "out.csv"
    empty_df.to_csv(input_path, index=False)

    result = preprocess(str(input_path), str(output_path))

    assert len(result) == 0
    assert list(result.columns[:4]) == NUMERIC_COLS

def test_preprocess_preserves_valid_rows():
    import pandas as pd
    from src.pipeline.preprocess import preprocess_data

    df = pd.DataFrame({
        "sepal_length": [5.1, 6.2],
        "sepal_width": [3.5, 3.4],
        "petal_length": [1.4, 4.5],
        "petal_width": [0.2, 1.5],
        "species": ["setosa", "versicolor"],
    })

    result = preprocess_data(df)
    assert len(result) == 2


def test_preprocess_returns_dataframe():
    import pandas as pd
    from src.pipeline.preprocess import preprocess_data

    df = pd.DataFrame({
        "sepal_length": [5.1],
        "sepal_width": [3.5],
        "petal_length": [1.4],
        "petal_width": [0.2],
        "species": ["setosa"],
    })

    result = preprocess_data(df)
    assert isinstance(result, pd.DataFrame)

def test_preprocess_preserves_valid_rows(tmp_path):
    import pandas as pd
    from src.pipeline.preprocess import preprocess

    input_file = tmp_path / "input.csv"
    output_file = tmp_path / "output.csv"

    df = pd.DataFrame({
        "sepal length (cm)": [5.1, 6.2],
        "sepal width (cm)": [3.5, 3.4],
        "petal length (cm)": [1.4, 4.5],
        "petal width (cm)": [0.2, 1.5],
        "species": ["setosa", "versicolor"],
    })
    df.to_csv(input_file, index=False)

    result = preprocess(str(input_file), str(output_file))

    assert len(result) == 2
    assert output_file.exists()


def test_preprocess_returns_dataframe(tmp_path):
    import pandas as pd
    from src.pipeline.preprocess import preprocess

    input_file = tmp_path / "input.csv"
    output_file = tmp_path / "output.csv"

    df = pd.DataFrame({
        "sepal length (cm)": [5.1],
        "sepal width (cm)": [3.5],
        "petal length (cm)": [1.4],
        "petal width (cm)": [0.2],
        "species": ["setosa"],
    })
    df.to_csv(input_file, index=False)

    result = preprocess(str(input_file), str(output_file))

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1

def test_preprocess_removes_collected_at_column(tmp_path):
    import pandas as pd
    from src.pipeline.preprocess import preprocess

    input_file = tmp_path / "input.csv"
    output_file = tmp_path / "output.csv"

    df = pd.DataFrame({
        "sepal length (cm)": [5.1],
        "sepal width (cm)": [3.5],
        "petal length (cm)": [1.4],
        "petal width (cm)": [0.2],
        "species": ["setosa"],
        "collected_at": ["2026-10-22"],
    })
    df.to_csv(input_file, index=False)

    result = preprocess(str(input_file), str(output_file))

    assert "collected_at" not in result.columns
    assert output_file.exists()
