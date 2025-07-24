import unittest
from unittest.mock import patch, MagicMock, call

import numpy as np
import pandas as pd

# It's good practice to place the module to be tested under a try-except block
# in case of import errors, which can happen in complex projects.
try:
    from experiment import (
        instantiate_models,
        load_and_prepare_data,
        train_and_evaluate_clear_model,
        train_and_evaluate_fhe_model,
        run_experiment_iteration
    )
except ImportError as e:
    print(f"Failed to import module for testing: {e}")
    # Exit if we can't import the code we need to test
    exit(1)


class TestExperiment(unittest.TestCase):

    def setUp(self):
        """Set up common test data and configurations."""
        self.model_config = {
            "name": "LogisticRegression",
            "module_name": "sklearn.linear_model",
            "fhe_name": "LogisticRegression",
            "fhe_module_name": "concrete.ml.sklearn"
        }
        self.param_config = {"C": 1.0}
        self.fhe_config = {"n_bits": 8}

        self.X_train = np.random.rand(10, 2)
        self.y_train = np.random.randint(0, 2, 10)
        self.X_test = np.random.rand(5, 2)
        self.y_test = np.random.randint(0, 2, 5)

    @patch('importlib.import_module')
    def test_instantiate_models(self, mock_import_module):
        """Test that models are instantiated with the correct parameters."""
        # Mock the entire chain of module and class loading
        mock_sklearn_module = MagicMock()
        mock_concreteml_module = MagicMock()

        # Configure the mock to return different modules for different calls
        mock_import_module.side_effect = [mock_sklearn_module, mock_concreteml_module]

        clear_model, fhe_model = instantiate_models(self.model_config, self.param_config, self.fhe_config)

        # Check that the correct modules were imported
        mock_import_module.assert_has_calls([
            call("sklearn.linear_model"),
            call("concrete.ml.sklearn")
        ])

        # Check that the clear model was instantiated correctly
        mock_sklearn_module.LogisticRegression.assert_called_once_with(**self.param_config)
        self.assertEqual(clear_model, mock_sklearn_module.LogisticRegression.return_value)

        # Check that the FHE model was instantiated correctly
        expected_fhe_params = {**self.param_config, **self.fhe_config}
        mock_concreteml_module.LogisticRegression.assert_called_once_with(**expected_fhe_params)
        self.assertEqual(fhe_model, mock_concreteml_module.LogisticRegression.return_value)

    @patch('experiment.make_classification')
    @patch('experiment.train_test_split')
    def test_load_and_prepare_data_synthetic(self, mock_train_test_split, mock_make_classification):
        """Test loading of synthetic data."""
        task_config = {"data": {"type": "synthetic"}}
        dataset_params = {"n_samples": 100}
        
        # Set return values for the mocked functions
        mock_make_classification.return_value = (MagicMock(), MagicMock())
        mock_train_test_split.return_value = (1, 2, 3, 4) # Dummy split data

        result = load_and_prepare_data(task_config, dataset_params)

        mock_make_classification.assert_called_once_with(**dataset_params)
        mock_train_test_split.assert_called_once()
        self.assertEqual(result, (1, 2, 3, 4))

    @patch('experiment.fetch_ucirepo')
    @patch('experiment.LabelEncoder')
    @patch('experiment.train_test_split')
    def test_load_and_prepare_data_uci(self, mock_train_test_split, mock_label_encoder, mock_fetch_ucirepo):
        """Test loading and preparation of UCI data."""
        task_config = {"data": {"type": "uci"}}
        dataset_params = {"id": 1}

        # Mock the UCI repo fetch result
        mock_repo = MagicMock()
        mock_repo.data.features = pd.DataFrame({'a': [1,2,3]})
        mock_repo.data.targets = pd.DataFrame({'b': ['x','y','z']})
        mock_fetch_ucirepo.return_value = mock_repo

        # Mock the LabelEncoder
        mock_encoder_instance = MagicMock()
        mock_label_encoder.return_value = mock_encoder_instance

        load_and_prepare_data(task_config, dataset_params)

        mock_fetch_ucirepo.assert_called_once_with(**dataset_params)
        mock_label_encoder.assert_called_once()
        mock_encoder_instance.fit_transform.assert_called_once()
        mock_train_test_split.assert_called_once()

    def test_load_and_prepare_data_unknown_type(self):
        """Test that an unknown data type raises a ValueError."""
        task_config = {"data": {"type": "unknown"}}
        with self.assertRaises(ValueError):
            load_and_prepare_data(task_config, {})

    @patch('experiment.time.perf_counter', side_effect=[0, 1, 1, 2]) # Mocking time
    @patch('experiment.accuracy_score', return_value=0.9)
    @patch('experiment.f1_score', return_value=0.8)
    @patch('experiment.roc_auc_score', return_value=0.7)
    def test_train_and_evaluate_clear_model(self, mock_auc, mock_f1, mock_acc, mock_time):
        """Test the training and evaluation pipeline for a clear model."""
        mock_model = MagicMock()
        results, _ = train_and_evaluate_clear_model(
            mock_model, self.X_train, self.y_train, self.X_test, self.y_test
        )

        mock_model.fit.assert_called_once_with(self.X_train, self.y_train)
        mock_model.predict.assert_called_once_with(self.X_test)

        self.assertEqual(results["train_time_clear"], 1)
        self.assertEqual(results["prediction_time_clear"], 1)
        self.assertEqual(results["accuracy_clear"], 0.9)
        self.assertEqual(results["f1_clear"], 0.8)
        self.assertEqual(results["auc_clear"], 0.7)

    @patch('experiment.time.perf_counter', side_effect=[0, 1, 1, 2, 2, 3]) # Mocking time
    @patch('experiment.accuracy_score', return_value=0.85)
    @patch('experiment.f1_score', return_value=0.75)
    @patch('experiment.roc_auc_score', return_value=0.65)
    def test_train_and_evaluate_fhe_model(self, mock_auc, mock_f1, mock_acc, mock_time):
        """Test the training, compilation, and evaluation pipeline for an FHE model."""
        mock_fhe_model = MagicMock()
        results, _ = train_and_evaluate_fhe_model(
            mock_fhe_model, self.X_train, self.y_train, self.X_test, self.y_test
        )

        mock_fhe_model.fit.assert_called_once_with(self.X_train, self.y_train)
        mock_fhe_model.compile.assert_called_once_with(self.X_train)
        mock_fhe_model.predict.assert_called_once_with(self.X_test, fhe="execute")

        self.assertEqual(results["train_time_fhe"], 1)
        self.assertEqual(results["compilation_time"], 1)
        self.assertEqual(results["prediction_time_fhe"], 1)
        self.assertEqual(results["accuracy_fhe"], 0.85)
        self.assertEqual(results["f1_fhe"], 0.75)
        self.assertEqual(results["auc_fhe"], 0.65)

    @patch('experiment.mlflow')
    @patch('experiment.log_system_info')
    @patch('experiment.log_library_versions')
    @patch('experiment.instantiate_models')
    @patch('experiment.load_and_prepare_data')
    @patch('experiment.train_and_evaluate_clear_model')
    @patch('experiment.train_and_evaluate_fhe_model')
    @patch('experiment.log_results')
    def test_run_experiment_iteration_success(self, mock_mlflow, mock_log_sys, mock_log_libs, mock_instantiate, mock_load_data, mock_eval_clear, mock_eval_fhe, mock_log_results):
        """Test a successful run of a full experiment iteration."""
        #TODO
        pass

    @patch('experiment.mlflow')
    @patch('experiment.instantiate_models', side_effect=Exception("Test Error"))
    def test_run_experiment_iteration_failure(self, mock_instantiate, mock_mlflow):
        """Test a failed run of an experiment iteration, ensuring error logging."""
        run_experiment_iteration({}, {"data": {"params": []}}, {"params": []}, {"model_params": []}, "test_config")

        mock_mlflow.start_run.assert_called_once()
        mock_mlflow.set_tag.assert_has_calls([
            call("error_message", "Test Error"),
            call("error_type", "Exception")
        ])
        mock_mlflow.end_run.assert_called_once_with(status="FAILED")

if __name__ == '__main__':
    unittest.main()