# tests/test_pipeline.py
# Tests for pipelines/pipeline_builder.py — MLflow logging, task-type detection,
# leaderboard output, and experiment history.
#
# Run with:  pytest tests/test_pipeline.py -v

import sys
import os
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def binary_classification_df():
    """Small synthetic dataset for binary classification."""
    np.random.seed(42)
    n = 60
    return pd.DataFrame({
        "age":         np.random.randint(18, 70, n),
        "tenure":      np.random.randint(0, 120, n),
        "monthly_charges": np.random.uniform(20, 100, n),
        "churn":       np.random.randint(0, 2, n),
    })


@pytest.fixture
def regression_df():
    """Small synthetic dataset for regression."""
    np.random.seed(7)
    n = 60
    return pd.DataFrame({
        "age":    np.random.randint(20, 60, n),
        "bmi":    np.random.uniform(18, 40, n),
        "smoker": np.random.choice(["yes", "no"], n),
        "charges": np.random.uniform(1000, 50000, n),
    })


# ── Unit tests: task-type detection ──────────────────────────────────────────

class TestTaskTypeDetection:
    """Verify automatic task-type inference from target cardinality."""

    def test_binary_target_detected_as_classification(self, binary_classification_df):
        """A binary (0/1) target should map to 'binary' problem_type."""
        from pipelines.pipeline_builder import run_pipeline

        with patch("pipelines.pipeline_builder.TabularPredictor") as mock_pred:
            instance = MagicMock()
            mock_pred.return_value = instance
            instance.fit.return_value = instance
            instance.leaderboard.return_value = pd.DataFrame(
                {"model": ["GBM"], "score_val": [0.85]}
            )
            instance.evaluate.return_value = {"accuracy": 0.85}

            with patch("pipelines.pipeline_builder.mlflow"):
                run_pipeline(binary_classification_df, target_col="churn")

        _, call_kwargs = mock_pred.call_args
        assert mock_pred.call_args is not None
        # problem_type should be binary or multiclass (not regression)
        args, kwargs = mock_pred.call_args
        assert kwargs.get("problem_type") in ("binary", "multiclass")

    def test_continuous_target_detected_as_regression(self, regression_df):
        """A continuous target (>10 unique values) should be detected as regression."""
        from pipelines.pipeline_builder import run_pipeline

        with patch("pipelines.pipeline_builder.TabularPredictor") as mock_pred:
            instance = MagicMock()
            mock_pred.return_value = instance
            instance.fit.return_value = instance
            instance.leaderboard.return_value = pd.DataFrame(
                {"model": ["RF"], "score_val": [-1500.0]}
            )
            instance.evaluate.return_value = {"root_mean_squared_error": 1500.0}

            with patch("pipelines.pipeline_builder.mlflow"):
                run_pipeline(regression_df, target_col="charges")

        args, kwargs = mock_pred.call_args
        assert kwargs.get("problem_type") == "regression"

    def test_explicit_task_type_overrides_detection(self, regression_df):
        """Explicit task_type='regression' should not be overridden by auto-detection."""
        from pipelines.pipeline_builder import run_pipeline

        with patch("pipelines.pipeline_builder.TabularPredictor") as mock_pred:
            instance = MagicMock()
            mock_pred.return_value = instance
            instance.fit.return_value = instance
            instance.leaderboard.return_value = pd.DataFrame(
                {"model": ["GBM"], "score_val": [-1200.0]}
            )
            instance.evaluate.return_value = {"root_mean_squared_error": 1200.0}

            with patch("pipelines.pipeline_builder.mlflow"):
                run_pipeline(regression_df, target_col="charges", task_type="regression")

        args, kwargs = mock_pred.call_args
        assert kwargs.get("problem_type") == "regression"

    def test_rows_with_missing_target_are_dropped_before_training(self, binary_classification_df):
        """
        AutoGluon hard-fails with 'train dataset label column cannot contain
        non-finite values' if any row has a NaN label — this broke a real upload
        (a Netflix titles CSV with some missing 'rating' values). Rows with a
        missing target must be dropped before predictor.fit() ever sees them.
        """
        from pipelines.pipeline_builder import run_pipeline

        df_with_gaps = binary_classification_df.copy()
        df_with_gaps.loc[[0, 5, 10], "churn"] = np.nan

        with patch("pipelines.pipeline_builder.TabularPredictor") as mock_pred:
            instance = MagicMock()
            mock_pred.return_value = instance
            instance.fit.return_value = instance
            instance.leaderboard.return_value = pd.DataFrame(
                {"model": ["GBM"], "score_val": [0.85]}
            )
            instance.evaluate.return_value = {"accuracy": 0.85}

            with patch("pipelines.pipeline_builder.mlflow"):
                run_pipeline(df_with_gaps, target_col="churn")

        fit_call_df = instance.fit.call_args.args[0]
        assert fit_call_df["churn"].isna().sum() == 0
        assert len(fit_call_df) == len(df_with_gaps) - 3


# ── Unit tests: MLflow logging ────────────────────────────────────────────────

class TestMLflowLogging:
    """Verify that pipeline runs log the correct params and metrics to MLflow."""

    def test_logs_target_and_task_type_params(self, binary_classification_df):
        """MLflow must log target column and task type as params."""
        from pipelines.pipeline_builder import run_pipeline

        with patch("pipelines.pipeline_builder.TabularPredictor") as mock_pred, \
             patch("pipelines.pipeline_builder.mlflow") as mock_mlflow:

            run_mock = MagicMock()
            mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=run_mock)
            mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

            instance = MagicMock()
            mock_pred.return_value = instance
            instance.fit.return_value = instance
            instance.leaderboard.return_value = pd.DataFrame(
                {"model": ["GBM"], "score_val": [0.88]}
            )
            instance.evaluate.return_value = {"accuracy": 0.88}

            run_pipeline(binary_classification_df, target_col="churn",
                         task_type="classification")

        # Verify log_param was called with target and task_type
        param_calls = {
            call.args[0]: call.args[1]
            for call in mock_mlflow.log_param.call_args_list
        }
        assert "target" in param_calls
        assert param_calls["target"] == "churn"
        assert "task_type" in param_calls

    def test_logs_accuracy_metric_for_classification(self, binary_classification_df):
        """MLflow must log 'accuracy' metric for classification tasks."""
        from pipelines.pipeline_builder import run_pipeline

        with patch("pipelines.pipeline_builder.TabularPredictor") as mock_pred, \
             patch("pipelines.pipeline_builder.mlflow") as mock_mlflow:

            mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=MagicMock())
            mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

            instance = MagicMock()
            mock_pred.return_value = instance
            instance.fit.return_value = instance
            instance.leaderboard.return_value = pd.DataFrame(
                {"model": ["GBM"], "score_val": [0.91]}
            )
            instance.evaluate.return_value = {"accuracy": 0.91}

            run_pipeline(binary_classification_df, target_col="churn",
                         task_type="classification")

        metric_calls = {
            call.args[0]: call.args[1]
            for call in mock_mlflow.log_metric.call_args_list
        }
        assert "accuracy" in metric_calls
        assert abs(metric_calls["accuracy"] - 0.91) < 1e-6


# ── Unit tests: pipeline output contract ─────────────────────────────────────

class TestPipelineOutput:
    """Verify run_pipeline returns the expected 4-tuple contract."""

    def test_returns_four_tuple(self, binary_classification_df):
        """run_pipeline must return (result_str, model_info_path, leaderboard, model_dir)."""
        from pipelines.pipeline_builder import run_pipeline

        with patch("pipelines.pipeline_builder.TabularPredictor") as mock_pred, \
             patch("pipelines.pipeline_builder.mlflow"):

            instance = MagicMock()
            mock_pred.return_value = instance
            instance.fit.return_value = instance
            instance.leaderboard.return_value = pd.DataFrame(
                {"model": ["GBM"], "score_val": [0.87]}
            )
            instance.evaluate.return_value = {"accuracy": 0.87}

            output = run_pipeline(binary_classification_df, target_col="churn",
                                  task_type="classification")

        assert len(output) == 4, "run_pipeline must return exactly 4 values"
        result_str, model_info_path, leaderboard, model_dir = output
        assert isinstance(result_str, str)
        assert "Pipeline complete" in result_str or "complete" in result_str.lower()

    def test_raises_on_missing_target_column(self, binary_classification_df):
        """run_pipeline must raise ValueError when target column is not in the dataframe."""
        from pipelines.pipeline_builder import run_pipeline

        with pytest.raises((ValueError, KeyError)):
            with patch("pipelines.pipeline_builder.mlflow"):
                run_pipeline(binary_classification_df, target_col="nonexistent_col")


# ── Unit tests: MLflow run history ───────────────────────────────────────────

class TestLoadRecentMLflowRuns:
    """Verify load_recent_mlflow_runs returns a DataFrame (empty or populated)."""

    def test_returns_dataframe(self):
        """Should always return a DataFrame, even when no experiments exist."""
        from pipelines.pipeline_builder import load_recent_mlflow_runs

        with patch("pipelines.pipeline_builder.mlflow") as mock_mlflow:
            mock_mlflow.get_experiment_by_name.return_value = None
            result = load_recent_mlflow_runs()

        assert isinstance(result, pd.DataFrame)

    def test_returns_limited_rows(self):
        """Should respect the limit parameter."""
        from pipelines.pipeline_builder import load_recent_mlflow_runs

        fake_runs = pd.DataFrame({
            "run_id":      [f"run_{i}" for i in range(10)],
            "start_time":  pd.date_range("2024-01-01", periods=10),
            "params.target": ["churn"] * 10,
            "params.task_type": ["classification"] * 10,
            "metrics.accuracy": [0.8 + i * 0.01 for i in range(10)],
        })

        mock_exp = MagicMock()
        mock_exp.experiment_id = "0"

        with patch("pipelines.pipeline_builder.mlflow") as mock_mlflow:
            mock_mlflow.get_experiment_by_name.return_value = mock_exp
            mock_mlflow.search_runs.return_value = fake_runs
            result = load_recent_mlflow_runs(limit=3)

        assert len(result) <= 3
