'''
This script trains an SGDClassifier model with different configurations and logs the results to MLflow.
The configurations include various hyperparameter combinations to evaluate model performance,
focusing on parameters relevant to SGDClassifier and Concrete ML.
'''

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from concrete.ml.sklearn.linear_model import SGDClassifier
import mlflow
from itertools import product
import datetime

def exp_sgd_classifier(X_train, y_train, X_test, y_test):
    # Configurations - Hyperparameter grid for SGDClassifier
    loss_values = ['log_loss', 'modified_huber'] # Only these loss functions are supported by Concrete ML's SGDClassifier
    penalty_values = ['l2', 'l1', 'elasticnet', None]
    alpha_values = [0.0001, 0.001, 0.01]
    l1_ratio_values = [0.15, 0.5, 0.8] # only relevant for elasticnet
    fit_intercept_values = [True, False]
    max_iter_values = [1000, 2000]
    tol_values = [1e-3, 1e-4, None]
    learning_rate_values = ['optimal', 'constant', 'invscaling'] # 'adaptive' - might be complex for FHE
    eta0_values = [0.01, 0.1] # relevant for constant and invscaling
    power_t_values = [0.5, 0.25] # relevant for invscaling
    epsilon_values = [0.1, 0.01] # relevant for huber, epsilon_insensitive, squared_epsilon_insensitive - not used in this exp, but included for completeness
    average_values = [False, True, 10] # True and int values for averaged weights
    n_bits_values = [2, 4, 8, 16]


    for loss, penalty, alpha, l1_ratio, fit_intercept, max_iter, tol, learning_rate, eta0, power_t, epsilon, average, n_bits in product(
        loss_values, penalty_values, alpha_values, l1_ratio_values, fit_intercept_values, max_iter_values, tol_values, learning_rate_values, eta0_values, power_t_values, epsilon_values, average_values, n_bits_values
    ):
        try:
            current_params = {
                "loss": loss,
                "penalty": penalty,
                "alpha": alpha,
                "l1_ratio": l1_ratio,
                "fit_intercept": fit_intercept,
                "max_iter": max_iter,
                "tol": tol,
                "learning_rate": learning_rate,
                "eta0": eta0,
                "power_t": power_t,
                "epsilon": epsilon,
                "average": average,
                "shuffle": True, # keep shuffle True as default from sklearn
                "random_state": 42, # fixed random state for reproducibility
                "n_jobs": None, # default n_jobs
                "warm_start": False, # default warm_start
                "verbose": 0, # no verbose output
                "early_stopping": False, # no early stopping for simplicity in this experiment
                "validation_fraction": 0.1, # default validation fraction, not used as early_stopping is False
                "n_iter_no_change": 5, # default n_iter_no_change, not used as early_stopping is False
                "class_weight": None, # no class weight
                "n_bits": n_bits,
            }

            with mlflow.start_run(nested=True):

                for param_name, param_value in current_params.items():
                    mlflow.log_param(param_name, param_value)

                model = SGDClassifier(**current_params)

                start = datetime.datetime.now()
                model.fit(X_train, y_train)
                end = datetime.datetime.now()
                training_time = (end - start).total_seconds()
                mlflow.log_metric("training_time", training_time)

                # clear prediction

                start = datetime.datetime.now()
                y_pred = model.predict(X_test)
                end = datetime.datetime.now()
                prediction_time_clear = (end - start).total_seconds()
                mlflow.log_metric("prediction_time_clear", prediction_time_clear)

                accuracy_clear = accuracy_score(y_test, y_pred)
                mlflow.log_metric("accuracy_clear", accuracy_clear)
                f1_clear = f1_score(y_test, y_pred)
                mlflow.log_metric("f1_clear", f1_clear)
                auc_clear = roc_auc_score(y_test, y_pred)
                mlflow.log_metric("auc_clear", auc_clear)

                # compilation

                start = datetime.datetime.now()
                model.compile(X_train)
                end = datetime.datetime.now()
                mlflow.log_metric("compilation_time", (end - start).total_seconds())

                # encrypted prediction

                start = datetime.datetime.now()
                y_pred_fhe = model.predict(X_test, fhe="execute")
                end = datetime.datetime.now()
                prediction_time_fhe = (end - start).total_seconds()
                mlflow.log_metric("prediction_time_fhe", prediction_time_fhe)

                accuracy_fhe = accuracy_score(y_test, y_pred_fhe)
                mlflow.log_metric("accuracy_fhe", accuracy_fhe)
                f1_fhe = f1_score(y_test, y_pred_fhe)
                mlflow.log_metric("f1_fhe", f1_fhe)
                auc_fhe = roc_auc_score(y_test, y_pred_fhe)
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

    mlflow.set_experiment("SGDClassifier")

    dataset_configs = [
        {"n_samples" : 1000, "n_features" : 10, "n_informative" : 2, "n_redundant" : 3, "random_state": 42},
        {"n_samples" : 1000, "n_features" : 100, "n_informative" : 20, "n_redundant" : 30, "random_state": 42},
        {"n_samples" : 1000, "n_features" : 1000, "n_informative" : 200, "n_redundant" : 300, "random_state": 42},
        {"n_samples" : 1000, "n_features" : 10000, "n_informative" : 2000, "n_redundant" : 3000, "random_state": 42},
    ]
    for dataset_config in dataset_configs:

        with mlflow.start_run():

            for param_name, param_value in dataset_config.items():
                mlflow.log_param(param_name, param_value)

            X, y = make_classification(**dataset_config)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            exp_sgd_classifier(X_train, y_train, X_test, y_test)