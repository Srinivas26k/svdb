#!/bin/bash
# Helper script to run benchmark with the correct local environment

# Ensure we are using the local virtual environment which has the latest maturin build
VENV_PYTHON="./.venv/bin/python"

if [ ! -f "$VENV_PYTHON" ]; then
    echo "Error: Virtual environment python not found at $VENV_PYTHON"
    exit 1
fi

echo "Running Universal Benchmark using local env..."
$VENV_PYTHON debug_env.py "$@"
