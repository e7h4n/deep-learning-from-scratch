test:
	coverage run -m pytest tests
	coverage report
	coverage xml

rebuild-env:
	git clean -dfx
	python -m venv .venv
	. .venv/bin/activate
	.venv/bin/pip install poetry
	.venv/bin/poetry install