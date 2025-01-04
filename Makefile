# Documentation commands
.PHONY: docs-serve docs-build docs-publish

# Serve documentation locally
docs-serve:
	PYTHONPATH=$(pwd)/bindings/python mkdocs serve

# Build documentation
docs-build:
	PYTHONPATH=$(pwd)/bindings/python mkdocs build

# Deploy documentation to GitHub Pages
docs-publish:
	PYTHONPATH=$(pwd)/bindings/python mkdocs gh-deploy
