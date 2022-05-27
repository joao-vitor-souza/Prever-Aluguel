install: 
	@echo "Instalando..."
	poetry install
	poetry run pre-commit install

activate:
	@echo "Ativando ambiente virtual"
	poetry shell

setup: install activate