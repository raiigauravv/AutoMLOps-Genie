# 📄 genie_agent/genie_main.py
# LLM-powered Genie: Parse user prompt via OpenAI Function Calling & launch AutoML pipeline

import os
import json
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from pipelines.pipeline_builder import run_pipeline

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# ── OpenAI Function / Tool schema ─────────────────────────────────────────────
# This is the real OpenAI function-calling API (tools parameter).
# The model is forced to return structured JSON by calling this function,
# which guarantees type safety and eliminates fragile regex/JSON parsing.

_TOOLS = [
    {
        "type": "function",
        "function": {
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
        },
    }
]

_SYSTEM_PROMPT = (
    "You are an expert ML engineer. Given a user's task description, "
    "call the configure_ml_pipeline function with the correct target column "
    "and task type. If the task type is ambiguous, infer it from context "
    "(e.g. predicting price → regression, predicting churn → classification)."
)


def extract_info_from_prompt(prompt: str) -> dict:
    """
    Use OpenAI Function Calling to extract pipeline config from a natural-language prompt.

    Uses the `tools` parameter (real function calling) rather than prompt-engineering
    a JSON response — the model is *forced* to populate the function schema, giving
    us guaranteed structure and type validation with no fragile JSON parsing.

    Returns dict with keys: target (str), type (str) — or {"error": ...} on failure.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            tools=_TOOLS,
            tool_choice={"type": "function", "function": {"name": "configure_ml_pipeline"}},
        )

        # Extract the function call result from the tool_calls list
        message = response.choices[0].message
        if not message.tool_calls:
            return {"error": "Model did not invoke the function — no tool_calls returned."}

        tool_call = message.tool_calls[0]
        parsed = json.loads(tool_call.function.arguments)
        return parsed

    except json.JSONDecodeError as e:
        return {"error": f"Function arguments were not valid JSON: {e}"}
    except Exception as e:
        return {"error": str(e)}


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
