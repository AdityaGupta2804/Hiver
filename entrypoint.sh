#!/usr/bin/env bash
set -e

echo "================================================================="
echo "  Hiver SDE Internship: Customer Support AI System"
echo "================================================================="

if [ "$#" -gt 0 ]; then
    # If custom command was passed (e.g. bash, pytest, python src/pipeline.py)
    exec "$@"
else
    # Default: Run full verification deliverable
    echo "[1/4] Running automated unit test suite..."
    pytest tests/ -v

    echo ""
    echo "[2/4] Running full evaluation scorecard (Phase 12)..."
    python src/phase12_eval_harness.py --brand SpotifyCares

    echo ""
    echo "[3/4] Running failure mode analysis (Phase 13)..."
    python src/phase13_failure_analysis.py --brand SpotifyCares

    echo ""
    echo "[4/4] Executing end-to-end inference verification..."
    python src/pipeline.py --query "My downloaded songs keep disappearing when offline"

    echo ""
    echo "================================================================="
    echo "  ALL DELIVERABLES & REPRODUCIBILITY TARGETS PASSED SUCCESSFULLY!"
    echo "================================================================="
fi
