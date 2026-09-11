# Makefile for Hiver SDE Internship Assignment
# Fast 15-minute reproduction + full slow path targets

ifeq ($(OS),Windows_NT)
    PYTHON ?= .venv/Scripts/python.exe
else
    PYTHON ?= python3
endif
BRAND ?= SpotifyCares

.PHONY: all profile select clean-brand split taxonomy golden baselines retrieval generate escalate finetune evaluate failure pipeline test reproduce

all: reproduce

# Phase 1: Full-corpus brand profiling
profile:
	$(PYTHON) src/phase1_profiling.py

# Phase 2: Brand selection
select:
	$(PYTHON) src/phase2_brand_select.py --brand $(BRAND)

# Phase 3: Single-brand cleaning & thread reconstruction
clean-brand:
	$(PYTHON) src/phase3_cleaning.py --brand $(BRAND)

# Phase 4: Temporal data split
split:
	$(PYTHON) src/phase4_split.py --brand $(BRAND)

# Phase 5: Intent discovery & taxonomy freeze
taxonomy:
	$(PYTHON) src/phase5_taxonomy.py --brand $(BRAND) --propose
	$(PYTHON) src/phase5_taxonomy.py --brand $(BRAND) --freeze

# Phase 6: Golden evaluation set construction
golden:
	$(PYTHON) src/phase6_golden_set.py --brand $(BRAND) --sample
	$(PYTHON) src/phase6_golden_set.py --brand $(BRAND) --check

# Phase 7: Baselines E1-E4
baselines:
	$(PYTHON) src/phase7_baselines.py --brand $(BRAND) --run all --skip-e4

# Phase 8: Retrieval system E5-E7
retrieval:
	$(PYTHON) src/phase8_retrieval.py --brand $(BRAND) --build --run-experiments

# Phase 9: Response generation E9
generate:
	$(PYTHON) src/phase9_generation.py --brand $(BRAND)

# Phase 10: Escalation layer E10
escalate:
	$(PYTHON) src/phase10_escalation.py --brand $(BRAND)

# Phase 11: Fine-tuning experiment E8
finetune:
	$(PYTHON) src/phase11_finetune.py --brand $(BRAND)

# Phase 12: Full evaluation harness & scorecard
evaluate:
	$(PYTHON) src/phase12_eval_harness.py --brand $(BRAND)

# Phase 13: Failure mode analysis
failure:
	$(PYTHON) src/phase13_failure_analysis.py --brand $(BRAND)

# Run interactive inference pipeline
pipeline:
	$(PYTHON) src/pipeline.py --interactive

# Run automated unit tests
test:
	$(PYTHON) -m pytest tests/ -v

# 15-minute fast reproduction path (uses cached artifacts)
reproduce: select test
	$(PYTHON) src/phase7_baselines.py --brand $(BRAND) --run all --skip-e4
	$(PYTHON) src/phase8_retrieval.py --brand $(BRAND) --build --run-experiments
	$(PYTHON) src/phase9_generation.py --brand $(BRAND)
	$(PYTHON) src/phase10_escalation.py --brand $(BRAND)
	$(PYTHON) src/phase11_finetune.py --brand $(BRAND)
	$(PYTHON) src/phase12_eval_harness.py --brand $(BRAND)
	$(PYTHON) src/phase13_failure_analysis.py --brand $(BRAND)
	$(PYTHON) src/pipeline.py --query "My downloaded songs keep disappearing when offline"
	@echo "\n=== ALL HIVER REPRODUCIBILITY TARGETS PASSED ==="

