# ConcreteML Benchmark

This repository contains benchmarks for ConcreteML, a library that enables privacy-preserving machine learning using fully homomorphic encryption (FHE).

## Overview

This benchmark provides tools for creating classification experiments that can perform predictions on encrypted data without needing to decrypt it first. This benchmark suite evaluates performance, accuracy, and resource usage across different models and datasets. Giving you the ability to compare encrypted and unencrypted inference, measure accuracy across different encryption parameters, and log resource utilization metrics such as training time, compilation time, and prediction time.

## Features

- Benchmarks for various machine learning models (linear models, SGD classifiers, decision trees).
- Performance comparisons between encrypted and unencrypted inference.
- Accuracy measurements across different encryption parameters.
- Resource utilization metrics (training time, compilation time, prediction time).
- Error logging on experiment tags.
- Support for multiple datasets, including UCI Machine Learning Repository datasets.
- Iterate through the parameters defined lists in random or grid modes.

## Getting Started

```bash
# Clone the repository
git clone https://github.com/msiciliag/concreteML_benchmark.git
cd concreteML_benchmark

# Install dependencies using uv
uv sync

# Run benchmarks
uv run python run_benchmark.py --help
```

## Usage

To run the benchmarks, use the `run_benchmark.py` script with a configuration file:

```bash
uv run run_benchmark.py experiments_classification/tree/exp_rfclassifier_uci17.yaml --mode random
```

To see the results of the experiments, you can use the `mlflow` UI:

```bash
uv run mlflow ui
```

### Arguments

- `config_file`: (Required) Path to the YAML experiment configuration file.
- `progress_file`: (Optional) Path to the file to save progress. By default, a filename is generated based on the model and mode (e.g., `DecisionTreeClassifier_progress_basic.bin`).

### Options

- `--mode`: The execution mode. Can be `basic` (grid) or `random`. Default is `basic`.
- `--exp-number`: The number of configurations to test in `random` mode. Default is 20.
- `--random-seed`: The random seed for `random` mode. Default is 42.
- `--clear-progress`: If set, deletes the saved progress before starting a new run.

## Configuration Files

The benchmark uses YAML configuration files to define experiments. These files are located in the `experiments_classification/` directory, organized by model type. Each file provides a template for benchmarking a specific model, which you can adapt for your own datasets and hyperparameter tuning.

Below is a list of the primary example configurations available:

-   **Linear Models** (`experiments_classification/linear/`):
    -   `exp_logisticregression.yaml`: For Logistic Regression models.
    -   `exp_sdgclassifier.yaml`: For classifiers trained with Stochastic Gradient Descent.
-   **Tree-Based Models** (`experiments_classification/tree/`):
    -   `exp_dtclassifier.yaml`: For Decision Tree classifiers.
    -   `exp_rfclassifier.yaml`: For Random Forest classifiers.
    -   `exp_xgbclassifier.yaml`: For XGBoost classifiers.
-   **Other Models**:
    -   `experiments_classification/nearestneighbors/exp_kneighborsclassifier.yaml`: For K-Nearest Neighbors.
    -   TODO: `experiments_classification/neuralnet/exp_neuralnetclassifier.yaml`: For a simple Neural Network classifier.

A minimal test configuration is also available at `example_test_config.yaml`.

### UCI Dataset Variations

In addition to the base configurations, you will find specialized files for running benchmarks on datasets from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php). These files are named with a `_uci` suffix followed by a dataset ID number (e.g., `exp_dtclassifier_uci17.yaml`). The `ucimlrepo` library is used to fetch the data automatically.

### Create new experiment configurations
For more details on how to create new experiment configurations, see the [experiment examples README](./experiment_examples/README.md).

## Requirements

- Python 3.11+
- concrete-ml
- scikit-learn
- pyyaml
- click
- mlflow
- ucimlrepo
- msgpack
- concrete-python

## TODO

### Urgent
### Add neural network model benchmarks
- Implement neural network benchmarks using `NeuralNetClassifier` from `concrete.ml.sklearn` and basic sklearn in comparison.
- Ensure compatibility with FHE and standard scikit-learn interfaces.

### Less urgent

#### Consider preprocessing function
For fixing specific parameters such as:
- *calculated n_informative* (and other `make_classification()` parameters)
- *class_weight* (LogisticRegression)
- Valid *solver-penalty* combinations (LogisticRegression)

#### Others
- Adjust min max step for floats, error:
```bash
return range(param['min'], param['max'], param['step'])
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: 'float' object cannot be interpreted as an integer
```
- Complete `README` documentation.
- Fix system specs logging.
- Rewrite value list comprehensions over hyperparameters for readability.

### Add more task types
- Add benchmarks for additional machine learning tasks, such as regression or clustering, extending the current framework that only supports classification tasks.

## *_Caution_*

- The `ccp_alpha` hyperparameter in FHE `dtclassifier` is not available.
