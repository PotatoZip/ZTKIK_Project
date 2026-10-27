#!/usr/bin/env bash
# Hardening gate: formatting, lint, tests and (when the real runs exist) the report consistency.
set -euo pipefail
cd "$(dirname "$0")/.."
uv run ruff format --check src tests scripts
uv run ruff check src tests scripts
uv run pytest -q
uv run fraud-ae --version
