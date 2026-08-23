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

    def test_falls_back_to_live_model_list_when_all_candidates_retired(self):
        """
        If every hardcoded candidate model 404s (Google retired it), extract_info_from_prompt
        should ask the API which models are actually live right now rather than giving up —
        this is the exact failure mode that broke the deployed demo three times in a row.
        """
        from genie_agent import genie_main
        from genie_agent.genie_main import extract_info_from_prompt

        retired = Exception("404 models/whatever is not found for API version v1beta")
        live_model = MagicMock()
        live_model.generate_content.return_value = _mock_function_call_response(
            "churn", "classification"
        )

        live_info = MagicMock()
        live_info.name = "models/gemini-9.0-flash"
        live_info.supported_generation_methods = ["generateContent"]

        def fake_generative_model(model_name, **kwargs):
            if model_name == "gemini-9.0-flash":
                return live_model
            m = MagicMock()
            m.generate_content.side_effect = retired
            return m

        with patch("genie_agent.genie_main.model") as mock_model, \
             patch.object(genie_main.genai, "GenerativeModel", side_effect=fake_generative_model), \
             patch.object(genie_main.genai, "list_models", return_value=[live_info]):
            mock_model.generate_content.side_effect = retired
            result = extract_info_from_prompt("Predict customer churn")

        assert result == {"target": "churn", "type": "classification"}

    def test_returns_error_when_live_fallback_also_fails(self):
        """If even the live model list can't produce a working model, surface the last error."""
        from genie_agent import genie_main
        from genie_agent.genie_main import extract_info_from_prompt

        retired = Exception("404 models/whatever is not found for API version v1beta")

        def fake_generative_model(model_name, **kwargs):
            m = MagicMock()
            m.generate_content.side_effect = retired
            return m

        with patch("genie_agent.genie_main.model") as mock_model, \
             patch.object(genie_main.genai, "GenerativeModel", side_effect=fake_generative_model), \
             patch.object(genie_main.genai, "list_models", return_value=[]):
            mock_model.generate_content.side_effect = retired
            result = extract_info_from_prompt("Predict customer churn")

        assert "error" in result
        assert "All candidate Gemini models failed" in result["error"]

    def test_live_fallback_excludes_non_text_models_by_name(self):
        """
        ListModels' supported_generation_methods can say "generateContent" for a
        TTS/image/embedding model too (the method covers the endpoint, not the
        response modality) — a live TTS model actually got picked this way and
        broke the deployed demo with a 400 'response modalities' error. The
        fallback name list must exclude those before they're ever tried.
        """
        from genie_agent import genie_main
        from genie_agent.genie_main import extract_info_from_prompt

        retired = Exception("404 models/whatever is not found for API version v1beta")

        tts_info = MagicMock()
        tts_info.name = "models/gemini-2.5-flash-preview-tts"
        tts_info.supported_generation_methods = ["generateContent"]

        text_info = MagicMock()
        text_info.name = "models/gemini-9.0-flash"
        text_info.supported_generation_methods = ["generateContent"]

        text_model = MagicMock()
        text_model.generate_content.return_value = _mock_function_call_response(
            "churn", "classification"
        )

        called_with = []

        def fake_generative_model(model_name, **kwargs):
            called_with.append(model_name)
            if model_name == "gemini-9.0-flash":
                return text_model
            m = MagicMock()
            m.generate_content.side_effect = retired
            return m

        with patch("genie_agent.genie_main.model") as mock_model, \
             patch.object(genie_main.genai, "GenerativeModel", side_effect=fake_generative_model), \
             patch.object(genie_main.genai, "list_models", return_value=[tts_info, text_info]):
            mock_model.generate_content.side_effect = retired
            result = extract_info_from_prompt("Predict customer churn")

        assert result == {"target": "churn", "type": "classification"}
        assert "gemini-2.5-flash-preview-tts" not in called_with

    def test_modality_mismatch_error_is_retried_not_fatal(self):
        """
        If a non-text model slips past the name filter anyway, Gemini rejects it with
        a 400 'response modalities' error (not a 404) — that must still be treated as
        retryable, not returned as a fatal error on the first bad candidate.
        """
        from genie_agent import genie_main
        from genie_agent.genie_main import extract_info_from_prompt

        retired = Exception("404 models/whatever is not found for API version v1beta")
        modality_error = Exception(
            "400 The requested combination of response modalities (TEXT) is not "
            "supported by the model. models/gemini-2.5-flash-preview-tts accepts "
            "the following combination of response modalities: * AUDIO"
        )

        good_info = MagicMock()
        good_info.name = "models/gemini-9.0-flash"
        good_info.supported_generation_methods = ["generateContent"]

        good_model = MagicMock()
        good_model.generate_content.return_value = _mock_function_call_response(
            "churn", "classification"
        )

        def fake_generative_model(model_name, **kwargs):
            if model_name == "gemini-9.0-flash":
                return good_model
            m = MagicMock()
            m.generate_content.side_effect = modality_error
            return m

        with patch("genie_agent.genie_main.model") as mock_model, \
             patch.object(genie_main.genai, "GenerativeModel", side_effect=fake_generative_model), \
             patch.object(genie_main.genai, "list_models", return_value=[good_info]):
            mock_model.generate_content.side_effect = retired
            result = extract_info_from_prompt("Predict customer churn")

        assert result == {"target": "churn", "type": "classification"}


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
