#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

uv run python main.py clear-test-data
uv run python main.py clear-training-data
uv run python main.py train
uv run python main.py test
