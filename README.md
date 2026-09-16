# PMLDL Assignment 1: Automated ML Deployment

This repository implements a reproducible three-stage MLOps pipeline for Palmer
Penguins species classification. It preprocesses raw data, trains and evaluates a
model with MLflow tracking, and deploys a FastAPI inference API and a Streamlit UI
in separate Docker containers. Airflow runs the complete pipeline every five
minutes.

## Architecture

```text
data/raw/penguins.csv
        |
        v
Stage 1: load -> validate -> split -> impute -> remove outliers
        |
        +-- data/processed/train.csv
        +-- data/processed/test.csv
        |
        v
Stage 2: sklearn feature pipeline -> train -> evaluate -> package -> MLflow
        |
        +-- models/model.joblib
        +-- models/metrics.json
        +-- models/confusion_matrix.json
        |
        v
Stage 3: Docker Compose
        +-- FastAPI API :8000
        +-- Streamlit UI :8501 --HTTP--> API
```

The model is a logistic regression classifier. Its numeric imputation/scaling and
categorical imputation/one-hot encoding are packaged inside one sklearn Pipeline,
so training and serving use identical transformations.

## Requirements

- Python 3.12
- Docker with Docker Compose v2
- macOS/Linux, or WSL 2 on Windows

The raw dataset is committed at `data/raw/penguins.csv`, so pipeline runs do not
depend on network access. The source and attribution are documented in
`data/raw/README.md`.

## Quick start

From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r requirements-api.txt

python code/datasets/preprocess.py
python code/models/train.py
docker compose -f code/deployment/docker-compose.yml up -d --build
```

Open:

- Streamlit application: <http://localhost:8501>
- FastAPI Swagger UI: <http://localhost:8000/docs>
- API health endpoint: <http://localhost:8000/health>

Example API request:

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "island": "Biscoe",
    "sex": "male",
    "bill_length_mm": 47.5,
    "bill_depth_mm": 15.2,
    "flipper_length_mm": 220,
    "body_mass_g": 5000
  }'
```

Stop the deployed services with:

```bash
docker compose -f code/deployment/docker-compose.yml down
```

## Run the full pipeline manually

After creating the environment and starting Docker:

```bash
./run_pipeline.sh
```

This command is idempotent: it recreates processed artifacts, trains and logs a
fresh model, and rebuilds/recreates both serving containers.

## Airflow automation

Airflow is best installed in a separate Python 3.12 environment with the official
constraints file, because Airflow has a larger dependency set:

```bash
python3.12 -m venv .airflow-venv
source .airflow-venv/bin/activate
AIRFLOW_VERSION=3.0.6
PYTHON_VERSION=3.12
python -m pip install \
  "apache-airflow==${AIRFLOW_VERSION}" \
  --constraint \
  "https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"
```

Set Airflow to use this repository's folders and start it:

```bash
export AIRFLOW_HOME="$PWD/services/airflow"
export AIRFLOW__CORE__DAGS_FOLDER="$PWD/services/airflow/dags"
export AIRFLOW__CORE__LOAD_EXAMPLES=False
airflow standalone
```

Open <http://localhost:8080>, sign in with the credentials printed by `airflow
standalone`, enable `pmldl_assignment_1_pipeline`, and optionally trigger it once
manually. The DAG uses `*/5 * * * *`, `catchup=False`, and `max_active_runs=1`.
Its tasks run in order:

```text
preprocess_data >> train_and_evaluate >> deploy
```

Airflow must run as a user that can access the Docker daemon. The project `.venv`
must exist because Stage 1 and Stage 2 use its Python executable.

## MLflow and tests

Every training run logs model parameters, accuracy, macro F1, the packaged model,
metrics JSON, and confusion matrix to local `mlruns/` storage.

```bash
source .venv/bin/activate
mlflow ui --backend-store-uri "file:$PWD/mlruns" --port 5000
pytest -q
```

Then open <http://localhost:5000> to inspect runs.

## Data engineering details

Stage 1 validates the schema and removes duplicates before a stratified 80/20 split
with seed 42. Numeric medians and categorical modes are learned from the training
split and applied to both splits. Per-species IQR bounds, learned from the training
split, remove outliers from training data only. The test split remains intact for
an unbiased evaluation. Categorical spelling is preserved, matching the values
sent by the UI to the API. `preprocessing_report.json` records statistics and row counts.

## Repository layout

```text
code/datasets/                 Stage 1 preprocessing
code/models/                   Stage 2 training and MLflow logging
code/deployment/api/           FastAPI service and Dockerfile
code/deployment/app/           Streamlit UI and Dockerfile
code/deployment/docker-compose.yml
data/raw/                      Versioned input data
data/processed/                Generated train/test artifacts
models/                        Generated packaged model and metrics
services/airflow/dags/         Scheduled end-to-end DAG
tests/                         Pipeline and API integration tests
```

## Demonstration checklist

1. Show a successful Airflow DAG run and the three tasks in sequence.
2. Open `models/metrics.json` and the MLflow run.
3. Show `docker compose ps` with healthy API and Streamlit containers.
4. Call `/predict` from Swagger UI.
5. Enter a second penguin in Streamlit and show the prediction and probabilities.
