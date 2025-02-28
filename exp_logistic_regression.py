'''This script trains a logistic regression model with different configurations and logs the results to MLflow.
The configurations are optimized for each solver and include different values for the hyperparameters'''

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
from concrete.ml.sklearn.linear_model import LogisticRegression
import mlflow
from itertools import product
import datetime


def exp_logistic_regression(X_train, y_train, X_test, y_test):
    
    # configurations optimized for each solver
    solver_configs = [
        {"solver": "lbfgs", "penalty": "l2", "multi_class": "multinomial", "n_jobs": None},
        {"solver": "lbfgs", "penalty": "l2", "multi_class": "multinomial", "n_jobs": 1},
        {"solver": "lbfgs", "penalty": None, "multi_class": "multinomial", "n_jobs": None},
        {"solver": "lbfgs", "penalty": None, "multi_class": "multinomial", "n_jobs": 1},
        
        # n_jobs is ignored for liblinear
        {"solver": "liblinear", "penalty": "l1", "multi_class": "ovr"},
        {"solver": "liblinear", "penalty": "l2", "multi_class": "ovr"},
        
        {"solver": "newton-cg", "penalty": "l2", "multi_class": "multinomial", "n_jobs": None},
        {"solver": "newton-cg", "penalty": "l2", "multi_class": "multinomial", "n_jobs": 1},
        {"solver": "newton-cg", "penalty": None, "multi_class": "multinomial", "n_jobs": None},
        {"solver": "newton-cg", "penalty": None, "multi_class": "multinomial", "n_jobs": 1},
        
        {"solver": "newton-cholesky", "penalty": "l2", "multi_class": "ovr", "n_jobs": None},
        {"solver": "newton-cholesky", "penalty": "l2", "multi_class": "ovr", "n_jobs": 1},
        {"solver": "newton-cholesky", "penalty": None, "multi_class": "ovr", "n_jobs": None},
        {"solver": "newton-cholesky", "penalty": None, "multi_class": "ovr", "n_jobs": 1},
        
        {"solver": "sag", "penalty": "l2", "multi_class": "multinomial", "n_jobs": None},
        {"solver": "sag", "penalty": "l2", "multi_class": "multinomial", "n_jobs": 1},
        {"solver": "sag", "penalty": None, "multi_class": "multinomial", "n_jobs": None},
        {"solver": "sag", "penalty": None, "multi_class": "multinomial", "n_jobs": 1},
        
        # l1_ratio is only used when penalty is elasticnet (1.0 is equivalent to l1 penalty and 0.0 to l2)
        {"solver": "saga", "penalty": "elasticnet", "multi_class": "multinomial", "l1_ratio": 0.5, "n_jobs": None},
        {"solver": "saga", "penalty": "elasticnet", "multi_class": "multinomial", "l1_ratio": 0.5, "n_jobs": 1},
        {"solver": "saga", "penalty": "elasticnet", "multi_class": "multinomial", "l1_ratio": 0.75, "n_jobs": None},
        {"solver": "saga", "penalty": "elasticnet", "multi_class": "multinomial", "l1_ratio": 0.75, "n_jobs": 1},
        {"solver": "saga", "penalty": "l1", "multi_class": "multinomial", "n_jobs": None},
        {"solver": "saga", "penalty": "l1", "multi_class": "multinomial", "n_jobs": 1},
        {"solver": "saga", "penalty": "l2", "multi_class": "multinomial", "n_jobs": None},
        {"solver": "saga", "penalty": "l2", "multi_class": "multinomial", "n_jobs": 1},
        {"solver": "saga", "penalty": None, "multi_class": "multinomial", "n_jobs": None},
        {"solver": "saga", "penalty": None, "multi_class": "multinomial", "n_jobs": 1},
    ]
    
    C_values = [0.1, 1.0, 10.0]
    fit_intercept_values = [True, False]
    max_iter_values = [100, 1000]
    tol_values = [1e-4, 1e-3]
    random_state_values = [42, None]
    n_bits_values = [2, 4, 8, 16]
    
    for solver_config in solver_configs:
        for C, fit_intercept, max_iter, tol, random_state, n_bits in product(
            C_values, fit_intercept_values, max_iter_values, tol_values, random_state_values, n_bits_values
        ):
            try:
                current_params = {
                    "C": C,
                    "fit_intercept": fit_intercept,
                    "max_iter": max_iter,
                    "tol": tol,
                    "random_state": random_state,
                    "verbose": 0,
                    "n_bits": n_bits,
                    **solver_config
                }
                
                with mlflow.start_run(nested=True): 

                    for param_name, param_value in current_params.items():
                        mlflow.log_param(param_name, param_value)

                    model = LogisticRegression(**current_params)

                    start = datetime.datetime.now()
                    model.fit(X_train, y_train)
                    end = datetime.datetime.now()
                    training_time = (end-start).total_seconds()
                    mlflow.log_metric("training_time", training_time)

                    # clear prediction
                    
                    start = datetime.datetime.now()
                    y_pred = model.predict(X_test)
                    end = datetime.datetime.now()
                    prediction_time_clear = (end-start).total_seconds()
                    mlflow.log_metric("prediction_time_clear", prediction_time_clear)
                    
                    accuracy_clear = (y_pred == y_test).mean()
                    mlflow.log_metric("accuracy_clear", accuracy_clear)
                    f1_clear = f1_score(y_test, y_pred)
                    mlflow.log_metric("f1_clear", f1_clear)
                    auc_clear = roc_auc_score(y_test, y_pred)
                    mlflow.log_metric("auc_clear", auc_clear)

                    
                    # compilation

                    start = datetime.datetime.now()
                    model.compile(X_train)
                    end = datetime.datetime.now()
                    mlflow.log_metric("compilation_time", (end-start).total_seconds())

                    # encrypted prediction

                    start = datetime.datetime.now()
                    y_pred = model.predict(X_test, fhe="execute")
                    end = datetime.datetime.now()
                    prediction_time_fhe = (end-start).total_seconds()
                    mlflow.log_metric("prediction_time_fhe", prediction_time_fhe)
                    
                    accuracy_fhe = (y_pred == y_test).mean()
                    mlflow.log_metric("accuracy_fhe", accuracy_fhe)
                    f1_fhe = f1_score(y_test, y_pred)
                    mlflow.log_metric("f1_fhe", f1_fhe)
                    auc_fhe = roc_auc_score(y_test, y_pred)
                    mlflow.log_metric("auc_fhe", auc_fhe)

                    # log differences

                    mlflow.log_metric("accuracy_diff", accuracy_fhe - accuracy_clear)
                    mlflow.log_metric("f1_diff", f1_fhe - f1_clear)
                    mlflow.log_metric("auc_diff", auc_fhe - auc_clear)
                    mlflow.log_metric("prediction_time_diff", prediction_time_fhe - prediction_time_clear)

            except Exception as e:
                print(f"Error with parameters {current_params}: {str(e)}")
                continue



if __name__ == "__main__":
    
    mlflow.set_experiment("Logistic Regression")

    # dataset configurations mantain n_samples at 1000, but change n_features, n_informative and n_redundant, also random_state is fixed at 42
    dataset_configs = [
        {"n_samples" : 1000, "n_features" : 10, "n_informative" : 2, "n_redundant" : 8, "random_state": 42},
        {"n_samples" : 1000, "n_features" : 100, "n_informative" : 20, "n_redundant" : 80, "random_state": 42},
        {"n_samples" : 1000, "n_features" : 1000, "n_informative" : 200, "n_redundant" : 800, "random_state": 42},
        {"n_samples" : 1000, "n_features" : 10000, "n_informative" : 2000, "n_redundant" : 8000, "random_state": 42},
    ]
    
    for dataset_config in dataset_configs:

        with mlflow.start_run():
            
            for param_name, param_value in dataset_config.items():
                mlflow.log_param(param_name, param_value)
                
            X, y = make_classification(**dataset_config)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            exp_logistic_regression(X_train, y_train, X_test, y_test)
