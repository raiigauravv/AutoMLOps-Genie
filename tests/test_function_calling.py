# tests/test_function_calling.py
# Tests for the LLM function-calling layer in genie_agent/genie_main.py
#
# Run with:  pytest tests/test_function_calling.py -v

import sys
import os
import pytest
from unittest.mock import MagicMock, patch

# Make sure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _mock_function_call_response(target: str, task_type: str):
    """Build a mock Gemini response that mimics a real function-calling reply."""
    function_call = MagicMock()
    function_call.name = "configure_ml_pipeline"
    function_call.args = {"target": target, "type": task_type}

    part = MagicMock()
    part.function_call = function_call

    content = MagicMock()
    content.parts = [part]

    candidate = MagicMock()
    candidate.content = content

    response = MagicMock()
    response.candidates = [candidate]
    return response


# ── Unit tests: extract_info_from_prompt ─────────────────────────────────────

class TestExtractInfoFromPrompt:
    """Verify that extract_info_from_prompt correctly handles function-call responses."""

    def test_classification_prompt(self):
        """Classification task is parsed correctly."""
        from genie_agent.genie_main import extract_info_from_prompt

        with patch("genie_agent.genie_main.model") as mock_model:
            mock_model.generate_content.return_value = _mock_function_call_response(
                "churn", "classification"
            )
            result = extract_info_from_prompt("Predict customer churn using all features")

        assert result["target"] == "churn"
        assert result["type"] == "classification"

    def test_regression_prompt(self):
        """Regression task is parsed correctly."""
        from genie_agent.genie_main import extract_info_from_prompt

        with patch("genie_agent.genie_main.model") as mock_model:
            mock_model.generate_content.return_value = _mock_function_call_response(
                "price", "regression"
            )
            result = extract_info_from_prompt("Forecast insurance premium price")

        assert result["target"] == "price"
        assert result["type"] == "regression"

    def test_fraud_detection_prompt(self):
        """Fraud label (binary classification) is parsed correctly."""
        from genie_agent.genie_main import extract_info_from_prompt

        with patch("genie_agent.genie_main.model") as mock_model:
            mock_model.generate_content.return_value = _mock_function_call_response(
                "is_fraud", "classification"
            )
            result = extract_info_from_prompt("Build a fraud detection classifier on is_fraud")

        assert result["target"] == "is_fraud"
        assert result["type"] == "classification"

    def test_returns_error_when_no_tool_calls(self):
        """Returns an error dict when the model returns no function_call."""
        from genie_agent.genie_main import extract_info_from_prompt

        part = MagicMock()
        part.function_call = None

        content = MagicMock()
        content.parts = [part]

        candidate = MagicMock()
        candidate.content = content

        response = MagicMock()
        response.candidates = [candidate]

        with patch("genie_agent.genie_main.model") as mock_model:
            mock_model.generate_content.return_value = response
            result = extract_info_from_prompt("Some ambiguous prompt")

        assert "error" in result

    def test_returns_error_on_api_exception(self):
        """Returns an error dict when the Gemini API raises an exception."""
        from genie_agent.genie_main import extract_info_from_prompt

        with patch("genie_agent.genie_main.model") as mock_model:
            mock_model.generate_content.side_effect = Exception("API timeout")
            result = extract_info_from_prompt("Predict target column")

        assert "error" in result
        assert "API timeout" in result["error"]

    def test_uses_tools_parameter(self):
        """Verify the model call is made with tool_config (real function calling, not plain chat)."""
        from genie_agent.genie_main import extract_info_from_prompt

        with patch("genie_agent.genie_main.model") as mock_model:
            mock_model.generate_content.return_value = _mock_function_call_response(
                "label", "classification"
            )
            extract_info_from_prompt("Predict label")

        call_kwargs = mock_model.generate_content.call_args.kwargs
        assert "tool_config" in call_kwargs, "Must use tool_config to force function calling"
        assert call_kwargs["tool_config"]["function_calling_config"]["mode"] == "ANY"

    def test_tool_choice_forces_correct_function(self):
        """Verify tool_config forces the configure_ml_pipeline function."""
        from genie_agent.genie_main import extract_info_from_prompt

        with patch("genie_agent.genie_main.model") as mock_model:
            mock_model.generate_content.return_value = _mock_function_call_response(
                "target", "regression"
            )
            extract_info_from_prompt("Predict target")

        call_kwargs = mock_model.generate_content.call_args.kwargs
        allowed = call_kwargs["tool_config"]["function_calling_config"]["allowed_function_names"]
        assert "configure_ml_pipeline" in allowed


# ── Integration test: genie_respond ──────────────────────────────────────────

class TestGenieRespond:
    """End-to-end tests for genie_respond — mocks Gemini but runs real pipeline logic."""

    def test_missing_target_column_returns_error(self):
        """Returns an error string when the LLM picks a column not in the dataframe."""
        import pandas as pd
        from genie_agent.genie_main import genie_respond

        df = pd.DataFrame({"age": [25, 30], "salary": [50000, 60000]})

        with patch("genie_agent.genie_main.model") as mock_model:
            mock_model.generate_content.return_value = _mock_function_call_response(
                "nonexistent_column", "classification"
            )
            result_str, parsed, *_ = genie_respond("Predict nonexistent_column", df)

        assert "not found" in result_str.lower()

    def test_error_in_extraction_propagates(self):
        """An extraction error is surfaced in the result string."""
        import pandas as pd
        from genie_agent.genie_main import genie_respond

        df = pd.DataFrame({"x": [1, 2], "y": [0, 1]})

        with patch("genie_agent.genie_main.model") as mock_model:
            mock_model.generate_content.side_effect = Exception("Connection refused")
            result_str, parsed, *_ = genie_respond("Predict y", df)

        assert "error" in result_str.lower()
