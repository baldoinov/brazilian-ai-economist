.ONESHELL: clean lint
.PHONY: clean data lint requirements

clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} \;

lint:
	black src