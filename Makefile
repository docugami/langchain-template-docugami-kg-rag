format:
	poetry run ruff check . --fix
	poetry run black .

lint:
	poetry run ruff check .
	poetry run black --check .
	poetry run npx pyright .

test:
	poetry run pytest .