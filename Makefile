.DEFAULT_GOAL := help
.PHONY: help install test cover lint format check run serve init sync janitor enrich index facets docker-build docker-up docker-down

VENV ?= .venv
PY := $(VENV)/bin/python
UV ?= uv

help: ## Muestra esta ayuda
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install: ## Crea el venv e instala todo (todos los extras + dev)
	$(UV) venv $(VENV) --python 3.12 || $(UV) venv $(VENV) --python 3.13
	$(UV) pip install --python $(PY) -e ".[janitor,lyrics,audio,dev]"

test: ## Corre los tests
	$(PY) -m pytest -q

cover: ## Tests con cobertura (exige 100% de líneas y ramas)
	$(PY) -m pytest --cov --cov-branch -q

lint: ## Revisa con ruff (check + format)
	$(PY) -m ruff check app tests
	$(PY) -m ruff format --check app tests

format: ## Formatea con ruff
	$(PY) -m ruff check app tests --fix
	$(PY) -m ruff format app tests

check: lint test ## Lint + tests (lo que corre el CI)

run serve: ## Levanta la UI web con recarga
	$(PY) -m app.cli serve --reload

init: ## Crea la base de datos y los directorios de datos
	$(PY) -m app.cli init

sync: ## Espeja la biblioteca de Navidrome a SQLite
	$(PY) -m app.cli sync

janitor: ## Corre el Janitor en modo simulación
	$(PY) -m app.cli janitor -p

enrich: ## Genera fichas de artista/álbum/canción
	$(PY) -m app.cli enrich

index: ## Construye embeddings + FTS
	$(PY) -m app.cli index

facets: ## Inspecciona el índice de facetas
	$(PY) -m app.cli facets

docker-build: ## Construye la imagen
	docker compose build

docker-up: ## Levanta el contenedor
	docker compose up -d

docker-down: ## Detiene el contenedor
	docker compose down
