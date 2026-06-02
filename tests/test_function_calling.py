# tests/test_function_calling.py
# Tests for the LLM function-calling layer in genie_agent/genie_main.py
#
# Run with:  pytest tests/test_function_calling.py -v

import json
import sys
import os
import pytest
from unittest.mock import MagicMock, patch

# Make sure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _mock_tool_response(target: str, task_type: str):
    """Build a mock OpenAI response that mimics a real function-calling reply."""
    tool_call = MagicMock()
    tool_call.function.arguments = json.dumps({"target": target, "type": task_type})
    tool_call.function.name = "configure_ml_pipeline"

    message = MagicMock()
    message.tool_calls = [tool_call]

    choice = MagicMock()
    choice.message = message

    response = MagicMock()
    response.choices = [choice]
    return response


# ── Unit tests: extract_info_from_prompt ─────────────────────────────────────

class TestExtractInfoFromPrompt:
    """Verify that extract_info_from_prompt correctly handles function-call responses."""

    def test_classification_prompt(self):
        """Classification task is parsed correctly."""
        from genie_agent.genie_main import extract_info_from_prompt

        with patch("genie_agent.genie_main.client") as mock_client:
            mock_client.chat.completions.create.return_value = _mock_tool_response(
                "churn", "classification"
            )
            result = extract_info_from_prompt("Predict customer churn using all features")

        assert result["target"] == "churn"
        assert result["type"] == "classification"

    def test_regression_prompt(self):
        """Regression task is parsed correctly."""
        from genie_agent.genie_main import extract_info_from_prompt

        with patch("genie_agent.genie_main.client") as mock_client:
            mock_client.chat.completions.create.return_value = _mock_tool_response(
                "price", "regression"
            )
            result = extract_info_from_prompt("Forecast insurance premium price")

        assert result["target"] == "price"
        assert result["type"] == "regression"

    def test_fraud_detection_prompt(self):
        """Fraud label (binary classification) is parsed correctly."""
        from genie_agent.genie_main import extract_info_from_prompt

        with patch("genie_agent.genie_main.client") as mock_client:
            mock_client.chat.completions.create.return_value = _mock_tool_response(
                "is_fraud", "classification"
            )
            result = extract_info_from_prompt("Build a fraud detection classifier on is_fraud")

        assert result["target"] == "is_fraud"
        assert result["type"] == "classification"

    def test_returns_error_when_no_tool_calls(self):
        """Returns an error dict when the model returns no tool_calls."""
        from genie_agent.genie_main import extract_info_from_prompt

        message = MagicMock()
        message.tool_calls = []          # empty — model didn't call the function

        choice = MagicMock()
        choice.message = message

        response = MagicMock()
        response.choices = [choice]

        with patch("genie_agent.genie_main.client") as mock_client:
            mock_client.chat.completions.create.return_value = response
            result = extract_info_from_prompt("Some ambiguous prompt")

        assert "error" in result

    def test_returns_error_on_api_exception(self):
        """Returns an error dict when the OpenAI API raises an exception."""
        from genie_agent.genie_main import extract_info_from_prompt

        with patch("genie_agent.genie_main.client") as mock_client:
            mock_client.chat.completions.create.side_effect = Exception("API timeout")
            result = extract_info_from_prompt("Predict target column")

        assert "error" in result
        assert "API timeout" in result["error"]

    def test_uses_tools_parameter(self):
        """Verify the API call is made with tools= (real function calling, not plain chat)."""
        from genie_agent.genie_main import extract_info_from_prompt

        with patch("genie_agent.genie_main.client") as mock_client:
            mock_client.chat.completions.create.return_value = _mock_tool_response(
                "label", "classification"
            )
            extract_info_from_prompt("Predict label")

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert "tools" in call_kwargs, "Must use the tools= parameter for function calling"
        assert "tool_choice" in call_kwargs, "Must force the model to call the function"

    def test_tool_choice_forces_correct_function(self):
        """Verify tool_choice forces the configure_ml_pipeline function."""
        from genie_agent.genie_main import extract_info_from_prompt

        with patch("genie_agent.genie_main.client") as mock_client:
            mock_client.chat.completions.create.return_value = _mock_tool_response(
                "target", "regression"
            )
            extract_info_from_prompt("Predict target")

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        tool_choice = call_kwargs["tool_choice"]
        assert tool_choice["function"]["name"] == "configure_ml_pipeline"


# ── Integration test: genie_respond ──────────────────────────────────────────

class TestGenieRespond:
    """End-to-end tests for genie_respond — mocks OpenAI but runs real pipeline logic."""

    def test_missing_target_column_returns_error(self):
        """Returns an error string when the LLM picks a column not in the dataframe."""
        import pandas as pd
        from genie_agent.genie_main import genie_respond

        df = pd.DataFrame({"age": [25, 30], "salary": [50000, 60000]})

        with patch("genie_agent.genie_main.client") as mock_client:
            mock_client.chat.completions.create.return_value = _mock_tool_response(
                "nonexistent_column", "classification"
            )
            result_str, parsed, *_ = genie_respond("Predict nonexistent_column", df)

        assert "not found" in result_str.lower()

    def test_error_in_extraction_propagates(self):
        """An extraction error is surfaced in the result string."""
        import pandas as pd
        from genie_agent.genie_main import genie_respond

        df = pd.DataFrame({"x": [1, 2], "y": [0, 1]})

        with patch("genie_agent.genie_main.client") as mock_client:
            mock_client.chat.completions.create.side_effect = Exception("Connection refused")
            result_str, parsed, *_ = genie_respond("Predict y", df)

        assert "error" in result_str.lower()
