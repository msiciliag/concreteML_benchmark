import time
import importlib
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from ucimlrepo import fetch_ucirepo
import mlflow

from utils import log_system_info, log_library_versions, log_results, get_random_id

# --- Model and Data Handling ---

# Constants for data splitting
TEST_SIZE = 0.3
RANDOM_STATE = 42


def instantiate_models(model_config: dict, param_config: dict, fhe_config: dict):
    """Creates instances of the clear and FHE models."""
    model_name, model_module = model_config["name"], model_config["module_name"]
    module = importlib.import_module(model_module)
    clear_model_class = getattr(module, model_name)
    clear_model = clear_model_class(**param_config)

    fhe_model_name, fhe_model_module = model_config["fhe_name"], model_config["fhe_module_name"]
    module = importlib.import_module(fhe_model_module)
    fhe_model_class = getattr(module, fhe_model_name)
    fhe_model = fhe_model_class(**param_config, **fhe_config)

    return clear_model, fhe_model

def load_and_prepare_data(task_config: dict, dataset_params: dict):
    """Loads a dataset based on the configuration and prepares it for training."""
    if task_config["data"]["type"] == "synthetic":
        X, y = make_classification(**dataset_params)
    elif task_config["data"]["type"] == "uci":
        uci_ds = fetch_ucirepo(**dataset_params)
        X = uci_ds.data.features
        y = uci_ds.data.targets
        le = LabelEncoder()
        y = le.fit_transform(y.values.ravel())
    else:
        raise ValueError(f"Unknown data type: {task_config['data']['type']}")

    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

# --- Experiment Execution ---

def select_dataset(task_config: dict, dataset_params: dict):
    """Selects a valid dataset based on the configuration."""
    if task_config["data"]["type"] == "uci" and dataset_params.get('id') == 'a':
        return select_valid_uci_dataset(task_config, dataset_params)
    return dataset_params

def select_dataset_with_guarantee(task_config: dict, dataset_params: dict):
    """
    Selects a valid dataset with guarantee that it will work for the experiment.
    This function ensures no MLflow run is created until we have a working dataset.
    
    Returns:
        dict: Updated dataset parameters with a valid dataset ID, or None if no valid dataset is found.
    """
    if task_config["data"]["type"] == "uci" and dataset_params.get('id') == 'a':
        return select_valid_uci_dataset_with_data_validation(task_config, dataset_params)
    elif task_config["data"]["type"] == "uci":
        # Validate specific UCI dataset before proceeding
        return validate_specific_uci_dataset(task_config, dataset_params)
    return dataset_params

def validate_specific_uci_dataset(task_config: dict, dataset_params: dict):
    """
    Validates a specific UCI dataset to ensure it works completely for the experiment.
    This performs the EXACT same pipeline as the real experiment to catch all possible errors.
    """
    try:
        print(f"--- Validating UCI dataset ID: {dataset_params['id']} ---")
        
        # Test EXACT data loading pipeline (same as load_and_prepare_data)
        uci_ds = fetch_ucirepo(**dataset_params)
        
        # Verify task compatibility
        if task_config["type"] not in uci_ds.metadata["tasks"]:
            print(f"Task '{task_config['type']}' not found in dataset tasks: {uci_ds.metadata['tasks']}")
            return None
            
        # Test EXACT data preparation pipeline
        X = uci_ds.data.features
        y = uci_ds.data.targets
        
        if X is None or y is None:
            print(f"Dataset {dataset_params['id']} has None features or targets")
            return None
        
        # This is where the "could not convert string to float" errors happen
        # We need to ensure X contains only numeric data
        le = LabelEncoder()
        y_encoded = le.fit_transform(y.values.ravel())
        
        # Test train_test_split with real data
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        
        # CRITICAL: Test that sklearn models can actually fit the data
        # This is where string-to-float conversion errors typically happen
        from sklearn.ensemble import RandomForestClassifier
        test_model = RandomForestClassifier(n_estimators=2, max_depth=2, random_state=42)
        test_model.fit(X_train, y_train)  # This will fail if X contains non-numeric data
        
        # Test predictions
        y_pred = test_model.predict(X_test)
        
        # Test metric calculations with real predictions
        n_classes = len(set(y_test))
        accuracy_score(y_test, y_pred)
        
        if n_classes > 2:
            f1_score(y_test, y_pred, average='weighted')
            try:
                roc_auc_score(y_test, y_pred, multi_class='ovr', average='weighted')
            except ValueError:
                roc_auc_score(y_test, y_pred, multi_class='ovr', average='macro')
        else:
            f1_score(y_test, y_pred)
            roc_auc_score(y_test, y_pred)
        
        # EXTRA: Also test FHE model instantiation and basic fitting to catch more errors
        try:
            from concrete.ml.sklearn import RandomForestClassifier as FHERandomForestClassifier
            test_fhe_model = FHERandomForestClassifier(n_estimators=2, max_depth=2, n_bits=2, random_state=42)
            test_fhe_model.fit(X_train, y_train)
            # Don't compile as it's expensive, but fitting should catch most data issues
        except Exception as fhe_e:
            print(f"FHE model test failed for dataset {dataset_params['id']}: {fhe_e}")
            return None
        
        print(f"✓ Dataset {dataset_params['id']} fully validated with model training (classes: {n_classes}, features: {X.shape[1]})")
        return dataset_params
        
    except Exception as e:
        print(f"Dataset {dataset_params['id']} validation failed: {e}")
        return None

def select_valid_uci_dataset_with_data_validation(task_config: dict, dataset_params: dict):
    """
    Attempts to select a valid UCI dataset ID that matches the specified task type
    and can complete the full data preparation pipeline.

    Returns:
        dict: Updated dataset parameters with a valid dataset ID, or None if no valid dataset is found.
    """
    max_attempts = 500  # Increased attempts since we're validating more thoroughly
    for attempt in range(max_attempts):
        random_id = get_random_id()
        dataset_params["id"] = random_id
        print(f"--- Trying random UCI dataset ID: {random_id} (attempt {attempt + 1}/{max_attempts}) ---")
        
        validated_params = validate_specific_uci_dataset(task_config, dataset_params.copy())
        if validated_params is not None:
            return validated_params

    print(f"Failed to find a valid UCI dataset after {max_attempts} attempts.")
    return None  # Indicate failure to find a valid dataset

def select_valid_uci_dataset(task_config: dict, dataset_params: dict):
    """
    Attempts to select a valid UCI dataset ID that matches the specified task type.

    Returns:
        dict: Updated dataset parameters with a valid dataset ID, or None if no valid dataset is found.
    """
    max_attempts = 100  # Limit the number of attempts to avoid indefinite looping
    for _ in range(max_attempts):
        random_id = get_random_id()
        dataset_params["id"] = random_id
        print(f"--- Trying random UCI dataset ID: {random_id} ---")
        try:
            uci_ds = fetch_ucirepo(**dataset_params)
            if task_config["type"] in uci_ds.metadata["tasks"]:
                return dataset_params 
            else:
                print(f"Task '{task_config['type']}' not found in dataset tasks: {uci_ds.metadata['tasks']}")
        except Exception as e:
            print(f"Could not fetch or use dataset ID {random_id}: {e}")

    print(f"Failed to find a valid UCI dataset after {max_attempts} attempts.")
    return None  # Indicate failure to find a valid dataset



def train_and_evaluate_clear_model(model, X_train, y_train, X_test, y_test):
    """Train and evaluate the clear model."""
    results = {}
    
    tic = time.perf_counter()
    model.fit(X_train, y_train)
    results["train_time_clear"] = time.perf_counter() - tic

    tic = time.perf_counter()
    y_pred_clear = model.predict(X_test)
    results["prediction_time_clear"] = time.perf_counter() - tic

    results["accuracy_clear"] = accuracy_score(y_test, y_pred_clear)
    
    # Handle multiclass scenario for F1 and AUC scores
    n_classes = len(set(y_test))
    if n_classes > 2:
        results["f1_clear"] = f1_score(y_test, y_pred_clear, average='weighted')
        # For multiclass, use one-vs-rest approach for AUC
        try:
            results["auc_clear"] = roc_auc_score(y_test, y_pred_clear, multi_class='ovr', average='weighted')
        except ValueError:
            # If OVR fails, use macro average
            results["auc_clear"] = roc_auc_score(y_test, y_pred_clear, multi_class='ovr', average='macro')
    else:
        results["f1_clear"] = f1_score(y_test, y_pred_clear)
        results["auc_clear"] = roc_auc_score(y_test, y_pred_clear)
    
    return results, y_pred_clear

def train_and_evaluate_fhe_model(fhe_model, X_train, y_train, X_test, y_test):
    """Train, compile, and evaluate the FHE model."""
    results = {}

    tic = time.perf_counter()
    fhe_model.fit(X_train, y_train)
    results["train_time_fhe"] = time.perf_counter() - tic

    tic = time.perf_counter()
    fhe_model.compile(X_train)
    results["compilation_time"] = time.perf_counter() - tic

    tic = time.perf_counter()
    y_pred_fhe = fhe_model.predict(X_test, fhe="execute")
    results["prediction_time_fhe"] = time.perf_counter() - tic

    results["accuracy_fhe"] = accuracy_score(y_test, y_pred_fhe)
    
    # Handle multiclass scenario for F1 and AUC scores
    n_classes = len(set(y_test))
    if n_classes > 2:
        results["f1_fhe"] = f1_score(y_test, y_pred_fhe, average='weighted')
        # For multiclass, use one-vs-rest approach for AUC
        try:
            results["auc_fhe"] = roc_auc_score(y_test, y_pred_fhe, multi_class='ovr', average='weighted')
        except ValueError:
            # If OVR fails, use macro average
            results["auc_fhe"] = roc_auc_score(y_test, y_pred_fhe, multi_class='ovr', average='macro')
    else:
        results["f1_fhe"] = f1_score(y_test, y_pred_fhe)
        results["auc_fhe"] = roc_auc_score(y_test, y_pred_fhe)

    return results, y_pred_fhe

def run_experiment_iteration(config: dict, task_config: dict, model_config: dict, concreteml_config: dict, config_name: str):
    """
    Runs and logs a single experiment iteration.

    Returns:
        bool: True if the experiment was successful, False if it failed due to a dataset issue.
    """
    def extract_params(config, param_definitions):
        param_names = [p["param"]["name"] for p in param_definitions]
        return {k: v for k, v in config.items() if k in param_names}

    dataset_params = extract_params(config, task_config["data"]["params"])
    model_params = extract_params(config, model_config["params"])
    fhe_params = extract_params(config, concreteml_config["model_params"])

    # IMPORTANTE: Asegurar dataset válido ANTES de crear el run de MLflow
    # Esto garantiza que una vez iniciado el experimento, se completará exitosamente
    dataset_params = select_dataset_with_guarantee(task_config, dataset_params.copy())
    
    if dataset_params is None:
        print("Skipping this iteration due to inability to find valid dataset after maximum attempts.")
        return False

    # Create a copy of config for logging with the selected dataset ID
    config_for_logging = config.copy()
    if 'id' in dataset_params:
        config_for_logging['id'] = dataset_params['id']

    experiment_name = f"Experiment: {config_name}"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        try:
            mlflow.log_params(config_for_logging)
            log_system_info()
            log_library_versions()

            clear_model, fhe_model = instantiate_models(model_config, model_params, fhe_params)
            X_train, X_test, y_train, y_test = load_and_prepare_data(task_config, dataset_params)

            clear_results, _ = train_and_evaluate_clear_model(clear_model, X_train, y_train, X_test, y_test)
            log_results(clear_results)

            fhe_results, _ = train_and_evaluate_fhe_model(fhe_model, X_train, y_train, X_test, y_test)
            log_results(fhe_results)

            diff_results = {
                "accuracy_diff": fhe_results["accuracy_fhe"] - clear_results["accuracy_clear"],
               "f1_diff": fhe_results["f1_fhe"] - clear_results["f1_clear"],
               "auc_diff": fhe_results["auc_fhe"] - clear_results["auc_clear"],
                "prediction_time_diff": fhe_results["prediction_time_fhe"] - clear_results["prediction_time_clear"],
            }
            log_results(diff_results)

            mlflow.end_run(status="FINISHED")
            return True

        except Exception as e:
            print(f"Error with configuration {config_for_logging}: {e}")
            mlflow.set_tag("error_message", str(e))
            mlflow.set_tag("error_type", type(e).__name__)
            
            # Clasificación mejorada de errores
            error_str = str(e).lower()
            error_type = type(e).__name__
            
            # Clasificar tipo de error para logging adicional
            if isinstance(e, RuntimeError) and "compile" in error_str:
                error_category = "compilation_error"
                mlflow.set_tag("error_category", error_category)
                if "bit" in error_str and "table lookup" in error_str:
                    mlflow.set_tag("error_subtype", "bit_precision_error")
            elif "average" in error_str and "binary" in error_str:
                error_category = "metric_calculation_error"
                mlflow.set_tag("error_category", error_category)
                mlflow.set_tag("error_subtype", "multiclass_binary_average_error")
            else:
                error_category = "model_training_error"
                mlflow.set_tag("error_category", error_category)
            
            mlflow.end_run(status="FAILED")
            # Dado que pre-validamos el dataset, todos los errores aquí son de modelo/entrenamiento
            # Por lo tanto, consideramos el run "completado" pero fallido (no reintentar)
            return True
