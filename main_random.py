import yaml
import random
import time
import click
import platform
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from multiprocessing import Pool
from itertools import repeat
import importlib
import os
import json
import mlflow
import importlib.metadata

DEFAULT_PROGRESS_FILE = "progress_random.json"
DEFAULT_RANDOM_SEED = 42
DEFAULT_EXP_NUMBER = 20

def load_progress(progress_file):
    if os.path.exists(progress_file):
        try:
            with open(progress_file, "r") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON progress file: {e}")
            return {}
    return {}

def save_progress(progress, progress_file):
    with open(progress_file, "w") as f:
        json.dump(progress, f)

def instantiate_models(model_config: dict, param_config: dict, fhe_config: dict):
    """Creates instances of the models with the given configuration"""
    model_name, model_module = model_config["name"], model_config["module_name"]
    module = importlib.import_module(model_module)
    class_ = getattr(module, model_name)
    instance = class_(**param_config)
    
    fhe_model_name, fhe_model_module = model_config["fhe_name"], model_config["fhe_module_name"]
    module = importlib.import_module(fhe_model_module)
    class_ = getattr(module, fhe_model_name)
    instance_fhe = class_(**param_config, **fhe_config)
    
    return instance, instance_fhe

def expand_config_param(param):
    """Expand a configuration parameter into an iterable"""
    if 'values' in param:
        return param['values']
    if 'min' in param and 'max' in param and 'step' in param:
        return range(param['min'], param['max'], param['step'])
    raise ValueError(f"Unsupported parameter configuration: {param}")

def log_system_info():
    """Log the main specs of the machine"""
    system_info = {
        "platform": platform.system(),
        "platform-release": platform.release(),
        "platform-version": platform.version(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "ram": f"{round(os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / (1024. ** 3), 2)} GB"
    }
    for key, value in system_info.items():
        mlflow.log_param(key, value)

def log_library_versions():
    """Log the versions of the main libraries used"""
    libraries = ["scikit-learn", "concrete-ml"]
    for library in libraries:
        try:
            version = importlib.metadata.version(library)
            mlflow.log_param(f"{library}_version", version)
        except importlib.metadata.PackageNotFoundError:
            mlflow.log_param(f"{library}_version", "not installed")
    
def experiment(task_config: dict, concreteml_config: dict, model_config: dict, progress: dict, progress_file: str, exp_number: int):
    """Run an experiment with the given configurations"""
    task_configs = [(elem["param"]["name"], expand_config_param(elem["param"]))
                    for elem in task_config["data"]["params"]]
    concreteml_model_configs = [(elem["param"]["name"], expand_config_param(elem["param"]))
                                for elem in concreteml_config["model_params"]]
    model_configs = [(elem["param"]["name"], expand_config_param(elem["param"]))
                     for elem in model_config["params"]]
    
    task_config_names = [elem[0] for elem in task_configs]
    concreteml_model_config_names = [elem[0] for elem in concreteml_model_configs]
    model_config_names = [elem[0] for elem in model_configs]
    
    names = task_config_names + concreteml_model_config_names + model_config_names
    values = [elem[1] for elem in task_configs + concreteml_model_configs + model_configs]
        
    for i in range(exp_number):

        print(f"\nExperiment number {i+1} of the run\n")
        vals = [random.choice(val) for val in values]
        named_values = dict(zip(names, vals))
        config_key = str(tuple(named_values.items()))
        
        if config_key in progress:
            print(f"Skipping already tested configuration: {named_values}\n")
            continue
        
        results = {}
        with mlflow.start_run():

            try:
                model, fhe_model = instantiate_models(model_config=model_config,
                                                    param_config={k: v for k, v in named_values.items() if k in model_config_names},
                                                    fhe_config={k: v for k, v in named_values.items() if k in concreteml_model_config_names})
                
                dataset_config = {k: v for k, v in named_values.items() if k in task_config_names}
                X, y = make_classification(**dataset_config)
                X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
                
                
                experiment_name = f"{model_config['name']} Random Benchmark"
                mlflow.set_experiment(experiment_name)
                
                for param_name, param_value in named_values.items():
                    mlflow.log_param(param_name, param_value)

                log_system_info()
                log_library_versions()
                
                # Train clear model
                tic = time.perf_counter()
                model.fit(X_train, y_train)
                toc = time.perf_counter()
                results["train_time_clear"] = toc - tic
                mlflow.log_metric("train_time_clear", results["train_time_clear"])
                
                # Predictions with clear model
                tic = time.perf_counter()
                y_pred_clear = model.predict(X_test)
                toc = time.perf_counter()
                results["prediction_time_clear"] = toc - tic
                mlflow.log_metric("prediction_time_clear", results["prediction_time_clear"])
                
                results["accuracy_clear"] = accuracy_score(y_test, y_pred_clear)
                mlflow.log_metric("accuracy_clear", results["accuracy_clear"])
                results["f1_clear"] = f1_score(y_test, y_pred_clear)
                mlflow.log_metric("f1_clear", results["f1_clear"])
                results["auc_clear"] = roc_auc_score(y_test, y_pred_clear)
                mlflow.log_metric("auc_clear", results["auc_clear"])
                
                # Train FHE model
                tic = time.perf_counter()
                fhe_model.fit(X_train, y_train)
                toc = time.perf_counter()
                results["train_time_fhe"] = toc - tic
                mlflow.log_metric("train_time_fhe", results["train_time_fhe"])
                
                # Compile FHE model
                tic = time.perf_counter()
                fhe_model.compile(X_train)
                toc = time.perf_counter()
                results["compilation_time"] = toc - tic
                mlflow.log_metric("compilation_time", results["compilation_time"])
                
                # Predictions with FHE model
                tic = time.perf_counter()
                y_pred_fhe = fhe_model.predict(X_test, fhe="execute")
                toc = time.perf_counter()
                results["prediction_time_fhe"] = toc - tic
                mlflow.log_metric("prediction_time_fhe", results["prediction_time_fhe"])
                
                results["accuracy_fhe"] = accuracy_score(y_test, y_pred_fhe)
                mlflow.log_metric("accuracy_fhe", results["accuracy_fhe"])
                results["f1_fhe"] = f1_score(y_test, y_pred_fhe)
                mlflow.log_metric("f1_fhe", results["f1_fhe"])
                results["auc_fhe"] = roc_auc_score(y_test, y_pred_fhe)
                mlflow.log_metric("auc_fhe", results["auc_fhe"])
                
                # Log differences
                results["accuracy_diff"] = results["accuracy_fhe"] - results["accuracy_clear"]
                mlflow.log_metric("accuracy_diff", results["accuracy_diff"])
                results["f1_diff"] = results["f1_fhe"] - results["f1_clear"]
                mlflow.log_metric("f1_diff", results["f1_diff"])
                results["auc_diff"] = results["auc_fhe"] - results["auc_clear"]
                mlflow.log_metric("auc_diff", results["auc_diff"])
                results["prediction_time_diff"] = results["prediction_time_fhe"] - results["prediction_time_clear"]
                mlflow.log_metric("prediction_time_diff", results["prediction_time_diff"])

                print(results)
                mlflow.end_run(status="FINISHED")

            except Exception as e:
                error_msg = str(e)
                print(f"Error with configuration {named_values}: {e}")
                
                mlflow.set_tag("error_message", error_msg)
                mlflow.set_tag("error_type", type(e).__name__)
                
                mlflow.end_run(status="FAILED")
                continue

            finally:

                progress[config_key] = results
                save_progress(progress, progress_file)


@click.command()
@click.argument('config_file', type=click.Path(exists=True))
@click.argument('progress_file', type=click.Path(), required=False, default=DEFAULT_PROGRESS_FILE)
@click.option('--exp-number', type=int, required=False, default=DEFAULT_EXP_NUMBER)
@click.option('--random-seed', type=int, required=False, default=DEFAULT_RANDOM_SEED)
@click.option('--clear-progress', is_flag=True, help='Clear progress and start from scratch.')
def main(config_file, progress_file, exp_number, random_seed, clear_progress):
    """Main function to run the experiments"""
    if clear_progress and os.path.exists(progress_file):
        os.remove(progress_file)
    
    config = yaml.safe_load(open(config_file))
    n_models = len(config["models"])
    model_configs = [m["model"] for m in config["models"]]
    configs = list(zip(repeat(config["task"]), repeat(config["concreteml"]), model_configs))
    
    progress = load_progress(progress_file)

    random.seed(random_seed)
    
    with Pool(n_models) as p:
        p.starmap(experiment, [(task, concreteml, model, progress, progress_file, exp_number) for task, concreteml, model in configs])

if __name__ == '__main__':
    main()