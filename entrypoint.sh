#!/usr/bin/env bash
set -e

echo "================================================================="
echo "  Hiver SDE Internship: Customer Support AI System"
echo "================================================================="

# Resolve the correct Python binary dynamically across OS environments
if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
elif command -v python.exe >/dev/null 2>&1; then
    PYTHON_BIN="python.exe"
else
    echo "Error: Python binary not found in PATH." >&2
    exit 1
fi

if [ "$#" -gt 0 ]; then
    # If custom command was passed (e.g. bash, pytest, python src/pipeline.py)
    exec "$@"
else
    # Default: Run full verification deliverable
    echo "[1/4] Running automated unit test suite..."
    $PYTHON_BIN -m pytest tests/ -v

    echo ""
    echo "[2/4] Running full evaluation scorecard (Phase 12)..."
    $PYTHON_BIN src/phase12_eval_harness.py --brand SpotifyCares

    echo ""
    echo "[3/4] Running failure mode analysis (Phase 13)..."
    $PYTHON_BIN src/phase13_failure_analysis.py --brand SpotifyCares

    echo ""
    echo "[4/4] Executing end-to-end inference verification..."
    $PYTHON_BIN src/pipeline.py --query "My downloaded songs keep disappearing when offline"

    echo ""
    echo "================================================================="
    echo "  ALL DELIVERABLES & REPRODUCIBILITY TARGETS PASSED SUCCESSFULLY!"
    echo "================================================================="
fi