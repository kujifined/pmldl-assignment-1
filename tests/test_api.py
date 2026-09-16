import importlib.util
import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("api_app", ROOT / "code/deployment/api/app.py")
api_module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = api_module
spec.loader.exec_module(api_module)
app = api_module.app


def test_health_and_prediction(monkeypatch):
    model_path = Path(__file__).resolve().parents[1] / "models/model.joblib"
    monkeypatch.setenv("MODEL_PATH", str(model_path))
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        response = client.post(
            "/predict",
            json={
                "island": "Biscoe",
                "sex": "male",
                "bill_length_mm": 47.5,
                "bill_depth_mm": 15.2,
                "flipper_length_mm": 220,
                "body_mass_g": 5000,
            },
        )
        assert response.status_code == 200
        assert response.json()["prediction"] in {"Adelie", "Chinstrap", "Gentoo"}
        assert abs(sum(response.json()["probabilities"].values()) - 1) < 1e-5
