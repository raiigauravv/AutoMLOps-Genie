# 📄 genie_agent/genie_main.py
# LLM-powered Genie: Parse user prompt via Gemini Function Calling & launch AutoML pipeline

import os
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
from pipelines.pipeline_builder import run_pipeline

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# ── Gemini Function / Tool schema ─────────────────────────────────────────────
# This is the real Gemini function-calling API (tools parameter).
# The model is forced to return structured JSON by calling this function,
# which guarantees type safety and eliminates fragile regex/JSON parsing.

_TOOLS = [
    {
        "function_declarations": [
            {
                "name": "configure_ml_pipeline",
                "description": (
                    "Extract the ML pipeline configuration from the user's natural language "
                    "task description. Identify the exact target column to predict and the "
                    "type of ML task required."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target": {
                            "type": "string",
                            "description": (
                                "The exact column name from the dataset that should be "
                                "predicted (e.g. 'churn', 'price', 'fraud_label')."
                            ),
                        },
                        "type": {
                            "type": "string",
                            "enum": ["classification", "regression"],
                            "description": (
                                "ML task type: 'classification' for categorical/discrete "
                                "targets, 'regression' for continuous numerical targets."
                            ),
                        },
                    },
                    "required": ["target", "type"],
                },
            }
        ]
    }
]

_SYSTEM_PROMPT = (
    "You are an expert ML engineer. Given a user's task description, "
    "call the configure_ml_pipeline function with the correct target column "
    "and task type. If the task type is ambiguous, infer it from context "
    "(e.g. predicting price → regression, predicting churn → classification)."
)

# Ordered by preference; gemini-2.0-flash was retired by Google. The 404 for
# a retired model only surfaces when generate_content() is actually called
# (GenerativeModel() itself never hits the API), so extract_info_from_prompt
# falls back to the next candidate on a 404 instead of failing outright.
_CANDIDATE_MODELS = ["gemini-2.5-flash", "gemini-2.0-flash-001", "gemini-1.5-flash"]

model = genai.GenerativeModel(
    model_name=_CANDIDATE_MODELS[0],
    tools=_TOOLS,
    system_instruction=_SYSTEM_PROMPT,
)

_TOOL_CONFIG = {
    "function_calling_config": {
        "mode": "ANY",
        "allowed_function_names": ["configure_ml_pipeline"],
    }
}


def extract_info_from_prompt(prompt: str) -> dict:
    """
    Use Gemini Function Calling to extract pipeline config from a natural-language prompt.

    Uses the `tools` parameter (real function calling) rather than prompt-engineering
    a JSON response — the model is *forced* to populate the function schema, giving
    us guaranteed structure and type validation with no fragile JSON parsing.

    Returns dict with keys: target (str), type (str) — or {"error": ...} on failure.
    """
    last_error = None
    for name in [None] + _CANDIDATE_MODELS[1:]:
        active_model = model if name is None else genai.GenerativeModel(
            model_name=name, tools=_TOOLS, system_instruction=_SYSTEM_PROMPT
        )
        try:
            response = active_model.generate_content(prompt, tool_config=_TOOL_CONFIG)

            candidates = response.candidates
            if not candidates or not candidates[0].content.parts:
                return {"error": "Model did not invoke the function — no function_call returned."}

            part = candidates[0].content.parts[0]
            function_call = getattr(part, "function_call", None)
            if not function_call or not function_call.name:
                return {"error": "Model did not invoke the function — no function_call returned."}

            return dict(function_call.args)

        except Exception as e:
            last_error = e
            # 404 means this model name is retired/unavailable — try the next candidate.
            if "404" in str(e) or "not found" in str(e).lower():
                continue
            return {"error": str(e)}

    return {"error": f"All candidate Gemini models failed: {last_error}"}


def genie_respond(prompt: str, df: pd.DataFrame):
    """
    Main Genie logic: parse prompt via function calling, run pipeline, return results.
    """
    parsed = extract_info_from_prompt(prompt)
    if "error" in parsed:
        return f"Error parsing prompt: {parsed['error']}", parsed, None, None, None

    target = parsed.get("target")
    if target not in df.columns:
        return (
            f"Target column '{target}' not found in the uploaded dataset. "
            f"Available columns: {list(df.columns)}",
            parsed, None, None, None,
        )

    result, model_info_path, leaderboard, model_dir = run_pipeline(
        df, target_col=target, task_type=parsed.get("type")
    )
    return result, parsed, model_info_path, leaderboard, model_dir
