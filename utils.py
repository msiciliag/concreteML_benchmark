
import os
import msgpack
import hashlib
import platform
import mlflow
import importlib.metadata
import random
from itertools import product
import json

# --- Progress Management ---

def load_progress(progress_file):
    """Loads the set of completed configuration hashes from a file."""
    if not os.path.exists(progress_file):
        return set()
    try:
        with open(progress_file, "rb") as f:
            packed_data = f.read()
            if not packed_data:
                return set()
            unpacked_data = msgpack.unpackb(packed_data, raw=False)
            return set(unpacked_data)
    except Exception as e:
        print(f"Error reading progress file {progress_file}: {e}")
        return set()

def save_progress(progress, progress_file):
    """Saves the set of completed configuration hashes to a file."""
    with open(progress_file, "wb") as f:
        packed_data = msgpack.packb(list(progress), use_bin_type=True)
        f.write(packed_data)

def hash_config(config):
    """Computes a hash for a given configuration dictionary."""
    config_str = str(sorted(config.items()))
    return hashlib.md5(config_str.encode()).hexdigest()

# --- Configuration Generation ---

def _expand_param(param):
    """Expand a single parameter configuration into an iterable list of values."""
    if 'values' in param:
        return param['values']
    if 'min' in param and 'max' in param and 'step' in param:
        # Handle both float and int ranges correctly
        if isinstance(param['min'], float) or isinstance(param['max'], float) or isinstance(param['step'], float):
            values = []
            current = float(param['min'])
            while current <= float(param['max']):
                values.append(current)
                current += float(param['step'])
            return values
        return list(range(param['min'], param['max'] + 1, param['step']))
    raise ValueError(f"Unsupported parameter configuration: {param}")

def generate_experiment_configs(task_config: dict, concreteml_config: dict, model_config: dict, mode: str, exp_number: int):
    """Yields all possible experiment configurations from the config files."""
    task_params = {p["param"]["name"]: _expand_param(p["param"]) for p in task_config["data"]["params"]}
    fhe_params = {p["param"]["name"]: _expand_param(p["param"]) for p in concreteml_config["model_params"]}
    model_params = {p["param"]["name"]: _expand_param(p["param"]) for p in model_config["params"]}

    all_param_names = list(task_params.keys()) + list(fhe_params.keys()) + list(model_params.keys())
    all_param_values = list(task_params.values()) + list(fhe_params.values()) + list(model_params.values())

    if mode == 'basic':
        for values in product(*all_param_values):
            yield dict(zip(all_param_names, values))
    elif mode == 'random':
        for _ in range(exp_number):
            random_values = [random.choice(v) for v in all_param_values]
            yield dict(zip(all_param_names, random_values))

# --- MLflow Logging ---

def log_system_info():
    """Logs hardware and OS information to MLflow."""
    system_info = {
        "platform": platform.system(),
        "platform_release": platform.release(),
        "platform_version": platform.release(),  # Corrected: Use platform.release() instead of platform.version() for consistency
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "ram_gb": f"{round(os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / (1024. ** 3), 2)}"
    }
    mlflow.log_params(system_info)

def log_library_versions():
    """Logs versions of key libraries to MLflow."""
    libraries = ["scikit-learn", "concrete-ml"]
    versions = {}
    for lib in libraries:
        try:
            versions[f"{lib}_version"] = importlib.metadata.version(lib)
        except importlib.metadata.PackageNotFoundError:
            versions[f"{lib}_version"] = "not_installed"
    mlflow.log_params(versions)

def log_results(results: dict):
    """Logs a dictionary of metrics to MLflow."""
    mlflow.log_metrics(results)

def get_random_id(seed: int = None):
    """Gets a random id from the list of the avaliable datasets in UCI"""
    with open('ds/dataset_ids.json', 'r') as f:
        data = json.load(f)
    if not data:
        raise ValueError("Dataset IDs file is empty or malformed. Run `get_dataset_ids.py` to populate it.")
    id_list = [item for item in data]
    if seed is not None:
        random.seed(seed)
    random_index = random.randrange(len(id_list))
    random_id = id_list[random_index]
    return random_id
