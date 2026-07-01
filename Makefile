.PHONY: install train tune lint fmt test export app
install:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt && pre-commit install
train:
	python src/train.py
 tune:
	python src/tune.py
lint:
	ruff check . && mypy src || true
fmt:
	black .
test:
	pytest -q
export:
	./scripts/export_onnx.sh
app:
	python src/app.py --ckpt outputs/checkpoints/best.pt