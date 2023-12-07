test:
	coverage run -m pytest tests
	coverage report
	coverage xml
