# ConcreteML Benchmark

This repository contains benchmarks for ConcreteML, a library that enables privacy-preserving machine learning using fully homomorphic encryption (FHE).

## Overview

ConcreteML provides tools for creating machine learning models that can perform predictions on encrypted data without needing to decrypt it first. This benchmark suite evaluates performance, accuracy, and resource usage across different models and datasets.

## Features

- Benchmarks for various machine learning models (linear models, TODO: tree-based models, etc)
- Performance comparisons between encrypted and unencrypted inference
- Accuracy measurements across different encryption parameters
- TODO: Resource utilization metrics (memory, CPU, compilation time)

## Getting Started

```bash
# Clone the repository
git clone https://github.com/msiciliag/concreteML_benchmark.git
cd concreteML_benchmark

# Create a virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run benchmarks
```

## Requirements

- Python 3.7+
- ConcreteML
- Scikit-learn
- MLFlow

## License

[License information]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.