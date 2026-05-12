PYTHON ?= python
PYTEST ?= pytest

EXPERIMENTS := exp_4_1_pvalue_approximation \
               exp_4_2_neufeldt_pruning \
               exp_4_2_1_cappelli_comparison \
               exp_4_2_2_cv_randomness \
               exp_4_3_real_data

.PHONY: all install test clean $(EXPERIMENTS) legacy

all: $(EXPERIMENTS)

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTEST)

$(EXPERIMENTS):
	$(PYTHON) experiments/$@.py

legacy:
	$(PYTHON) experiments/legacy/boosting_variants.py
	$(PYTHON) experiments/legacy/boosting_rmse_histogram.py

clean:
	rm -rf results/*/
	mkdir -p results
