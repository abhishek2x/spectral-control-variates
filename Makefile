PYTHON ?= python3
VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

.PHONY: venv install test bench plot smoke

venv:
	$(PYTHON) -m venv $(VENV)

install: venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

test: install
	$(PY) -m pytest tests/ -q

bench: install
	$(PY) benchmarks.py

plot: install
	$(PY) generate_plots.py

smoke: test
	$(PY) benchmarks.py
