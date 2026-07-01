.PHONY: install test train-smoke gradcam export-onnx app predict audit clean

install:
	python -m pip install --upgrade pip
	python -m pip install -r requirements.txt

test:
	pytest

train-smoke:
	python -m src.train --config configs/smoke_cpu.yaml

gradcam:
	python -m src.generate_gradcam_examples --config configs/smoke_cpu.yaml

export-onnx:
	python -m src.export --ckpt outputs/checkpoints/best.pt --out outputs/checkpoints/model.onnx --arch resnet18

app:
	python -m src.app --ckpt outputs/checkpoints/best.pt --arch resnet18

predict:
	python -m src.predict_image --image $(IMAGE) --ckpt outputs/checkpoints/best.pt --arch resnet18

audit:
	python -m src.agents.workflow

clean:
	rm -rf __pycache__
	rm -rf src/__pycache__
	rm -rf src/tests/__pycache__
	rm -rf .pytest_cache