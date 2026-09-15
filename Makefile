.PHONY: install dev test cover lint run serve janitor enrich index compose-build

VENV ?= .venv
PY := $(VENV)/bin/python
UV ?= uv

install:
	$(UV) venv $(VENV) --python 3.12 || $(UV) venv $(VENV) --python 3.13
	$(UV) pip install --python $(PY) -e ".[janitor,dev]"

test:
	$(PY) -m pytest -q

cover:
	$(PY) -m pytest --cov -q

lint:
	$(PY) -m pyflakes app tests

run serve:
	$(PY) -m app.cli serve --reload

init:
	$(PY) -m app.cli init

sync:
	$(PY) -m app.cli sync

janitor:
	$(PY) -m app.cli janitor -p

enrich:
	$(PY) -m app.cli enrich

index:
	$(PY) -m app.cli index

compose-build:
	docker compose build

compose-up:
	docker compose up -d
