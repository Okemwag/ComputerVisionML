# Medical Image Analysis - Makefile

.PHONY: install clean train evaluate predict setup-data help

# Default target
help:
	@echo "Available targets:"
	@echo "  install     - Install dependencies"
	@echo "  clean       - Clean temporary files and outputs"
	@echo "  train       - Run training with default configuration"
	@echo "  evaluate    - Run evaluation on trained model"
	@echo "  predict     - Run inference on new images"
	@echo "  setup-data  - Set up data directory structure"
	@echo "  help        - Show this help message"

# Install dependencies
install:
	pip install -r requirements.txt

# Clean temporary files and outputs
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf outputs/checkpoints/*
	rm -rf outputs/predictions/*
	rm -rf outputs/logs/*

# Run training with default configuration
train:
	python train.py --model unet --epochs 50 --batch-size 16

# Run evaluation (requires checkpoint path)
evaluate:
	@echo "Usage: make evaluate CHECKPOINT=path/to/checkpoint.pth"
	@echo "Example: make evaluate CHECKPOINT=outputs/checkpoints/unet_best.pth"

# Run inference (requires input and checkpoint)
predict:
	@echo "Usage: make predict INPUT=path/to/image CHECKPOINT=path/to/checkpoint.pth"
	@echo "Example: make predict INPUT=data/images/sample.tif CHECKPOINT=outputs/checkpoints/unet_best.pth"

# Set up data directory structure
setup-data:
	mkdir -p data/images data/masks
	mkdir -p outputs/checkpoints outputs/predictions outputs/logs
	@echo "Data directories created successfully"