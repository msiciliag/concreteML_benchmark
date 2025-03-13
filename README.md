# ConcreteML Benchmark

This repository contains benchmarks for ConcreteML, a library that enables privacy-preserving machine learning using fully homomorphic encryption (FHE).

## Overview

ConcreteML provides tools for creating machine learning models that can perform predictions on encrypted data without needing to decrypt it first. This benchmark suite evaluates performance, accuracy, and resource usage across different models and datasets.

## Features

- Benchmarks for various machine learning models (linear models, SGD classifiers)
- Performance comparisons between encrypted and unencrypted inference
- Accuracy measurements across different encryption parameters
- Resource utilization metrics (training time, compilation time, prediction time)
- Error logging on experiment tags

## Getting Started

```bash
# Clone the repository
git clone https://github.com/msiciliag/concreteML_benchmark.git
cd concreteML_benchmark

# Install dependencies using uv
uv sync

# Run benchmarks
uv run main.py --help
# OR
uv run main_random.py --help
```

## Configuration Files

The benchmark uses YAML configuration files to define experiments:

- `exp_logistic_regression.yaml`: Benchmarks for Logistic Regression models
- `exp_sdg_classifier.yaml`: Benchmarks for SGD Classifier models
- `example_test_config.yaml`: Minimal example for testing

## Requirements

- Python 3.12+
- ConcreteML
- Scikit-learn
- PyYAML
- Click
- MLflow

## TODO
- Complete README.
- KV store and hash implementation for progress.
- Fix system specs logging.
- Rewrite value list comprehensions over hyperparameters for readability.
