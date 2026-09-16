import importlib.util
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


preprocess_module = _load_module("preprocess_module", ROOT / "code/datasets/preprocess.py")
train_module = _load_module("train_module", ROOT / "code/models/train.py")
preprocess = preprocess_module.preprocess
train = train_module.train
FEATURES = train_module.FEATURES


def test_preprocess_is_reproducible_and_disjoint(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    preprocess(ROOT / "data/raw/penguins.csv", first)
    preprocess(ROOT / "data/raw/penguins.csv", second)
    train_a, test_a = pd.read_csv(first / "train.csv"), pd.read_csv(first / "test.csv")
    train_b, test_b = pd.read_csv(second / "train.csv"), pd.read_csv(second / "test.csv")
    pd.testing.assert_frame_equal(train_a, train_b)
    pd.testing.assert_frame_equal(test_a, test_b)
    assert not set(train_a.row_id) & set(test_a.row_id)
    assert not train_a.isna().any().any()
    assert not test_a.isna().any().any()
    assert set(train_a.species) == {"Adelie", "Chinstrap", "Gentoo"}
    assert "Biscoe" in set(train_a.island)
    assert len(test_a) == 69


def test_training_creates_usable_artifacts(tmp_path):
    data_dir, model_dir = tmp_path / "data", tmp_path / "models"
    preprocess(ROOT / "data/raw/penguins.csv", data_dir)
    metrics = train(data_dir, model_dir, tmp_path / "mlruns")
    model = joblib.load(model_dir / "model.joblib")
    sample = pd.read_csv(data_dir / "test.csv").head(1)
    assert model.predict(sample[FEATURES])[0] in {"Adelie", "Chinstrap", "Gentoo"}
    assert metrics["accuracy"] >= 0.8
    assert (model_dir / "metrics.json").exists()
    assert (model_dir / "confusion_matrix.json").exists()
