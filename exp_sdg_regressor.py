'''
This script trains a SGDRegressor model with different configurations and logs the results to MLflow.
The configurations include various hyperparameter combinations to evaluate model performance.
'''

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from concrete.ml.sklearn.linear_model import SGDRegressor
import mlflow
from itertools import product
import datetime
import numpy as np

def exp_sgd_regressor(X_train, y_train, X_test, y_test):
    # Configurations - Hyperparameter grid for SGDRegressor
    loss_values = ['squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive']
    penalty_values = ['l2', 'l1', 'elasticnet', None]
    alpha_values = [0.0001, 0.001, 0.01]
    l1_ratio_values = [0.15, 0.5, 0.8] # only relevant for elasticnet
    fit_intercept_values = [True, False]
    max_iter_values = [1000, 2000]
    tol_values = [1e-3, 1e-4, None]
    learning_rate_values = ['invscaling', 'constant', 'optimal', 'adaptive']
    eta0_values = [0.01, 0.1] # relevant for constant and invscaling
    power_t_values = [0.25, 0.5] # relevant for invscaling
    epsilon_values = [0.1, 0.01] # relevant for huber, epsilon_insensitive, squared_epsilon_insensitive
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
                "warm_start": False, # default warm_start
                "verbose": 0, # no verbose output
                "early_stopping": False, # no early stopping for simplicity in this experiment
                "validation_fraction": 0.1, # default validation fraction, not used as early_stopping is False
                "n_iter_no_change": 5, # default n_iter_no_change, not used as early_stopping is False
                "n_bits": n_bits,
            }

            with mlflow.start_run(nested=True):

                for param_name, param_value in current_params.items():
                    mlflow.log_param(param_name, param_value)

                model = SGDRegressor(**current_params)

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

                r2_clear = r2_score(y_test, y_pred)
                mlflow.log_metric("r2_clear", r2_clear)
                mse_clear = mean_squared_error(y_test, y_pred)
                mlflow.log_metric("mse_clear", mse_clear)
                mae_clear = mean_absolute_error(y_test, y_pred)
                mlflow.log_metric("mae_clear", mae_clear)

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

                r2_fhe = r2_score(y_test, y_pred_fhe)
                mlflow.log_metric("r2_fhe", r2_fhe)
                mse_fhe = mean_squared_error(y_test, y_pred_fhe)
                mlflow.log_metric("mse_fhe", mse_fhe)
                mae_fhe = mean_absolute_error(y_test, y_pred_fhe)
                mlflow.log_metric("mae_fhe", mae_fhe)

                # log differences

                mlflow.log_metric("r2_diff", r2_fhe - r2_clear)
                mlflow.log_metric("mse_diff", mse_fhe - mse_clear)
                mlflow.log_metric("mae_diff", mae_fhe - mae_clear)
                mlflow.log_metric("prediction_time_diff", prediction_time_fhe - prediction_time_clear)


        except Exception as e:
            print(f"Error with parameters {current_params}: {str(e)}")
            continue


if __name__ == "__main__":

    mlflow.set_experiment("SGDRegressor")

    dataset_configs = [
        {"n_samples" : 1000, "n_features" : 10, "n_informative" : 5, "random_state": 42},
        {"n_samples" : 1000, "n_features" : 100, "n_informative" : 50, "random_state": 42},
        {"n_samples" : 1000, "n_features" : 1000, "n_informative" : 500, "random_state": 42},
        {"n_samples" : 1000, "n_features" : 10000, "n_informative" : 5000, "random_state": 42},
    ]
    for dataset_config in dataset_configs:

        with mlflow.start_run():

            for param_name, param_value in dataset_config.items():
                mlflow.log_param(param_name, param_value)

            X, y = make_regression(**dataset_config)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            exp_sgd_regressor(X_train, y_train, X_test, y_test)