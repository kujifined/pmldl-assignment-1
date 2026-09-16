"""Stage 2: feature engineering, training, evaluation, and packaging."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "species"
NUMERIC_FEATURES = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
CATEGORICAL_FEATURES = ["island", "sex"]
FEATURES = [*CATEGORICAL_FEATURES, *NUMERIC_FEATURES]


def build_pipeline() -> Pipeline:
    numeric = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return Pipeline(
        [
            (
                "features",
                ColumnTransformer(
                    [("numeric", numeric, NUMERIC_FEATURES), ("categorical", categorical, CATEGORICAL_FEATURES)]
                ),
            ),
            ("classifier", LogisticRegression(max_iter=1_000, random_state=42)),
        ]
    )


def train(data_dir: Path, model_dir: Path, tracking_dir: Path) -> dict[str, float]:
    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    model = build_pipeline()
    model.fit(train_df[FEATURES], train_df[TARGET])
    predictions = model.predict(test_df[FEATURES])
    metrics = {
        "accuracy": float(accuracy_score(test_df[TARGET], predictions)),
        "f1_macro": float(f1_score(test_df[TARGET], predictions, average="macro")),
    }

    model_dir.mkdir(parents=True, exist_ok=True)
    tracking_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.joblib"
    metrics_path = model_dir / "metrics.json"
    confusion_matrix_path = model_dir / "confusion_matrix.json"
    joblib.dump(model, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    labels = sorted(test_df[TARGET].unique())
    confusion_matrix_path.write_text(
        json.dumps(
            {"labels": labels, "matrix": confusion_matrix(test_df[TARGET], predictions, labels=labels).tolist()},
            indent=2,
        ),
        encoding="utf-8",
    )

    mlflow.set_tracking_uri(tracking_dir.resolve().as_uri())
    mlflow.set_experiment("palmer-penguins-classification")
    with mlflow.start_run(run_name="logistic-regression"):
        mlflow.log_params({"model": "LogisticRegression", "random_state": 42, "features": len(FEATURES)})
        mlflow.log_metrics(metrics)
        mlflow.log_artifacts(str(model_dir), artifact_path="outputs")
        example = train_df[FEATURES].head(3)
        mlflow.sklearn.log_model(
            model,
            name="model",
            signature=infer_signature(example, model.predict(example)),
            input_example=example,
        )
    return metrics


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=project_root / "data/processed")
    parser.add_argument("--model-dir", type=Path, default=project_root / "models")
    parser.add_argument("--tracking-dir", type=Path, default=project_root / "mlruns")
    args = parser.parse_args()
    print(json.dumps(train(args.data_dir, args.model_dir, args.tracking_dir), indent=2))


if __name__ == "__main__":
    main()
