'''
This script trains a Ridge Regression model with different configurations and logs the results to MLflow.
The configurations include various hyperparameter combinations to evaluate model performance.
'''

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from concrete.ml.sklearn.linear_model import Ridge
import mlflow
from itertools import product
import datetime

def exp_ridge(X_train, y_train, X_test, y_test):
    # Configurations - Hyperparameter grid for Ridge
    alpha_values = [0.1, 1.0, 10.0]
    fit_intercept_values = [True, False]
    copy_X_values = [True, False]
    max_iter_values = [None, 1000] # Include None for default
    tol_values = [1e-4, 1e-3, 1e-2]
    solver_values = ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga', 'lbfgs']
    positive_values = [False, True]
    n_bits_values = [2, 4, 8, 16]


    for alpha, fit_intercept, copy_X, max_iter, tol, solver, positive, n_bits in product(
        alpha_values, fit_intercept_values, copy_X_values, max_iter_values, tol_values, solver_values, positive_values, n_bits_values
    ):
        try:
            current_params = {
                "alpha": alpha,
                "fit_intercept": fit_intercept,
                "copy_X": copy_X,
                "max_iter": max_iter,
                "tol": tol,
                "solver": solver,
                "positive": positive,
                "random_state": 42, # fixed random state for reproducibility
                "n_bits": n_bits,
            }

            with mlflow.start_run(nested=True):

                for param_name, param_value in current_params.items():
                    mlflow.log_param(param_name, param_value)

                model = Ridge(**current_params)

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

    mlflow.set_experiment("Ridge Regression")

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

            exp_ridge(X_train, y_train, X_test, y_test)