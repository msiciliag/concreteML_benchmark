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
- `example_tesst_config.yaml`: Minimal example for testing

## Requirements

- Python 3.12+
- ConcreteML
- Scikit-learn
- PyYAML
- Click
- MLflow
- msgpack

## TODO

### *Urgent*
- Add support for *neural networks*, ensuring differentiation between FHE and clear models during the fit process (quantization applied during fitting).
- Provide a method to vary hyperparameters when the fit() function is called for *neural networks*.

### Least urgent

#### Consider preprocessing function 
For fixing specific parameters such as: 
- *calculated n_informative* (and other make_classification() parameters)
- *class_weight* (LogisticReegression)
- *solver-penalty*  valid combinations (LogisticReegression)
#### Others
- Adjust min max step for floats, error:
```bash
return range(param['min'], param['max'], param['step'])
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: 'float' object cannot be interpreted as an integer
```
- Complete README.
- Fix system specs logging.
- Rewrite value list comprehensions over hyperparameters for readability.



## *_Caution_*

- ccp_alpha hyperparameter on dtclassifier fhe is not available