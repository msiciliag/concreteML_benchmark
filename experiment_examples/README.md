# How to Build an Experiment Configuration

Experiment configurations are defined in YAML files, allowing you to specify the dataset, models, and parameters for benchmarking. This guide explains the structure of these files and how to create your own.

## File Structure

Each configuration file has three main sections: `task`, `concreteml`, and `models`.

Here is a simple example (`simple_experiment.yaml`):

```yaml
task:
  data:
    type: synthetic
    params:
      - param:
          name: n_samples
          values: [100, 200]
      - param:
          name: n_features
          values: [10, 20]
  type: classification

concreteml:
  model_params:
    - param:
        name: n_bits
        values: [6, 7]

models:
  - model:
      name: LogisticRegression
      module_name: sklearn.linear_model
      fhe_name: LogisticRegression
      fhe_module_name: concrete.ml.sklearn
      params:
        - param:
            name: C
            values: [1.0, 2.0]
```

---

### 1. The `task` Section

This section defines the machine learning task and the dataset to be used.

-   `data`: Specifies the data source.
    -   `type`: Can be `synthetic` for randomly generated data or `uci` to fetch a dataset from the UCI Machine Learning Repository.
    -   `params`: A list of parameters for the dataset.
        -   For `synthetic` data, you can specify parameters for `sklearn.datasets.make_classification`, such as `n_samples`, `n_features`, and `n_informative`.
        -   For `uci` data, you must provide the dataset `id`. For a random selection of datasets you can specify the id as `a`, this will select a random datasets from UCI, always checking if the task  defined in the next step is compatible with the dataset.
-   `type`: The type of machine learning task. Currently, only `classification` is supported.

**Example:**

```yaml
task:
  data:
    type: synthetic
    params:
      - param:
          name: n_samples
          values: [100, 200]
      - param:
          name: n_features
          values: [10, 20]
  type: classification
```

---

### 2. The `concreteml` Section

This section configures the parameters specific to Concrete ML's FHE models.

-   `model_params`: A list of FHE-related parameters to be benchmarked.
    -   `n_bits`: The number of bits for quantization is a common parameter to vary.

**Example:**

```yaml
concreteml:
  model_params:
    - param:
        name: n_bits
        values: [6, 7]
```

---

### 3. The `models` Section

This section defines a list of one or more models to be benchmarked. Each model is an item in the list.

-   `model`: Contains the configuration for a single model.
    -   `name`: The class name of the standard scikit-learn model (e.g., `LogisticRegression`).
    -   `module_name`: The module where the scikit-learn model is located (only supported `sklearn.linear_model`).
    -   `fhe_name`: The class name of the corresponding FHE-enabled model in Concrete ML (e.g., `LogisticRegression`).
    -   `fhe_module_name`: The module where the Concrete ML model is located (only supported `concrete.ml.sklearn`).
    -   `params`: A list of hyperparameters to test for the model. The benchmark will iterate through all combinations of these values (in `basic` mode) or a random selection (in `random` mode). For consulting the available parameters for each model, refer to the [Concrete ML documentation](https://docs.concrete.ml/).

**Example:**

```yaml
models:
  - model:
      name: LogisticRegression
      module_name: sklearn.linear_model
      fhe_name: LogisticRegression
      fhe_module_name: concrete.ml.sklearn
      params:
        - param:
            name: C
            values: [1.0, 2.0]
```
