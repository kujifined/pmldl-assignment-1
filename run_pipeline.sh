#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$PROJECT_ROOT/.venv/bin/python" "$PROJECT_ROOT/code/datasets/preprocess.py"
"$PROJECT_ROOT/.venv/bin/python" "$PROJECT_ROOT/code/models/train.py"
docker compose -f "$PROJECT_ROOT/code/deployment/docker-compose.yml" up -d --build --force-recreate --remove-orphans
