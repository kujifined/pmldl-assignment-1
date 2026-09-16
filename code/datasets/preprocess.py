"""Stage 1: reproducible cleaning and splitting of Palmer Penguins data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

TARGET = "species"
NUMERIC_COLUMNS = [
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
]
CATEGORICAL_COLUMNS = ["island", "sex"]
REQUIRED_COLUMNS = [TARGET, *CATEGORICAL_COLUMNS, *NUMERIC_COLUMNS]


def _clean_values(
    frame: pd.DataFrame,
    numeric_medians: dict[str, float],
    categorical_modes: dict[str, str],
) -> pd.DataFrame:
    result = frame.copy()
    for column, median in numeric_medians.items():
        result[column] = pd.to_numeric(result[column], errors="coerce").fillna(median)
    for column, mode in categorical_modes.items():
        result[column] = result[column].fillna(mode).astype(str).str.strip()
    return result


def preprocess(
    raw_path: Path,
    output_dir: Path,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    raw = pd.read_csv(raw_path, na_values=["NA", "", "null"])
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(raw.columns))
    if missing_columns:
        raise ValueError(f"Raw data is missing columns: {missing_columns}")

    raw = raw.drop_duplicates().dropna(subset=[TARGET]).reset_index(names="row_id")
    train, test = train_test_split(
        raw,
        test_size=test_size,
        random_state=random_state,
        stratify=raw[TARGET],
    )

    numeric_medians = {column: float(train[column].median()) for column in NUMERIC_COLUMNS}
    categorical_modes = {
        column: str(train[column].mode(dropna=True).iloc[0]) for column in CATEGORICAL_COLUMNS
    }
    train = _clean_values(train, numeric_medians, categorical_modes)
    test = _clean_values(test, numeric_medians, categorical_modes)

    # Species have distinct body-size distributions. Compute IQR limits per
    # species to avoid treating a whole species as anomalous. The limits are
    # still learned exclusively from the training set.
    bounds: dict[str, dict[str, dict[str, float]]] = {}
    train_mask = pd.Series(True, index=train.index)
    for species, group in train.groupby(TARGET):
        bounds[str(species)] = {}
        for column in NUMERIC_COLUMNS:
            q1, q3 = group[column].quantile([0.25, 0.75])
            iqr = q3 - q1
            lower, upper = float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr)
            bounds[str(species)][column] = {"lower": lower, "upper": upper}
            train_rows = train[TARGET].eq(species)
            train_mask.loc[train_rows] &= train.loc[train_rows, column].between(lower, upper)

    train_before = len(train)
    train = train.loc[train_mask].sort_values("row_id").reset_index(drop=True)
    # Keep the holdout intact: choosing an outlier threshold using its true
    # species would make the reported evaluation overly optimistic.
    test = test.sort_values("row_id").reset_index(drop=True)
    if set(train[TARGET]) != set(raw[TARGET]) or set(test[TARGET]) != set(raw[TARGET]):
        raise ValueError("Cleaning removed an entire target class")
    if set(train.row_id) & set(test.row_id):
        raise AssertionError("Train and test sets overlap")

    output_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(output_dir / "train.csv", index=False)
    test.to_csv(output_dir / "test.csv", index=False)
    report = {
        "raw_rows": len(raw),
        "train_rows": len(train),
        "test_rows": len(test),
        "train_outliers_removed": train_before - len(train),
        "random_state": random_state,
        "test_size": test_size,
        "numeric_medians_from_train": numeric_medians,
        "categorical_modes_from_train": categorical_modes,
        "per_species_iqr_bounds_from_train": bounds,
    }
    (output_dir / "preprocessing_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=project_root / "data/raw/penguins.csv")
    parser.add_argument("--output-dir", type=Path, default=project_root / "data/processed")
    args = parser.parse_args()
    print(json.dumps(preprocess(args.input, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
