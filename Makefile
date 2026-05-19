# DREAM Makefile — convenience wrappers.
PYTHON ?= python
PYTHONPATH := .
export PYTHONPATH

.PHONY: help test sanity acquire-lax acquire-sfo ols envelope risk corridor eval all clean

help:
	@echo "Targets:"
	@echo "  make test           - run unit + smoke tests"
	@echo "  make sanity         - run sanity smoke pipeline (synthetic)"
	@echo "  make acquire-lax    - download all 8 real data sources for LAX 2024-08"
	@echo "  make acquire-sfo    - same for SFO"
	@echo "  make ols            - build OLS+SDF for LAX and SFO"
	@echo "  make envelope DATE=2024-08-02   - build dynamic envelope for LAX on DATE"
	@echo "  make risk           - train risk field for LAX (xgb)"
	@echo "  make corridor       - plan corridors for LAX × V1..V4 × B0..B4"
	@echo "  make eval           - compute joint KPIs for LAX and SFO"
	@echo "  make all            - full pipeline (scripts/run_all.sh)"

test:
	$(PYTHON) -m pytest -q

sanity:
	$(PYTHON) scripts/run_sanity.py

acquire-lax:
	$(PYTHON) scripts/acquire_all.py --airport KLAX --window 2024-08

acquire-sfo:
	$(PYTHON) scripts/acquire_all.py --airport KSFO --window 2024-08

ols:
	$(PYTHON) scripts/build_ols.py --airport KLAX
	$(PYTHON) scripts/build_ols.py --airport KSFO

DATE ?= 2024-08-02
envelope:
	$(PYTHON) scripts/build_envelope.py --airport KLAX --window $(DATE) --interval 15min

risk:
	$(PYTHON) scripts/sample_counterfactuals.py --airport KLAX --n 200000 --seed 42
	$(PYTHON) scripts/train_risk_field.py --model xgb --airport KLAX

corridor:
	@for V in V1 V2 V3 V4; do \
	  for B in B0 B1 B2 B3 B4; do \
	    $(PYTHON) scripts/plan_corridors.py --airport KLAX --vertiport $$V --baseline $$B --hours 8,11,17 || true; \
	  done; \
	done

eval:
	$(PYTHON) scripts/eval_safety_capacity_access.py --airport KLAX
	$(PYTHON) scripts/eval_safety_capacity_access.py --airport KSFO

all:
	bash scripts/run_all.sh

clean:
	rm -rf data/cache results/* figures/*.png figures/*.pdf
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name .pytest_cache -prune -exec rm -rf {} +
