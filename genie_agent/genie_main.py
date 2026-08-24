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


# Model names containing any of these are non-text-output variants (text-to-speech,
# image/video generation, embeddings, etc). They can still list "generateContent" as a
# supported method while only accepting non-TEXT response modalities, so name-based
# exclusion is needed on top of the method check.
_NON_TEXT_NAME_MARKERS = (
    "tts", "audio", "image", "vision", "embedding", "aqa", "video", "imagen", "veo",
)


def _is_retired_model_error(e: Exception) -> bool:
    msg = str(e).lower()
    return (
        "404" in msg
        or "not found" in msg
        or "response modalities" in msg  # e.g. a TTS-only model rejecting TEXT output
    )


def _live_fallback_model_names() -> list:
    """
    Ask the Gemini API which models it actually supports right now, for use only
    after every hardcoded candidate above has failed. Hardcoded model names go
    stale as Google retires them (this has happened repeatedly), so this is the
    one source of truth that can't drift out of date.

    ListModels' supported_generation_methods can say "generateContent" for models
    that only produce audio/image output (e.g. TTS models) — that method name
    covers the endpoint, not the response modality — so name-based filtering
    excludes those before they're ever tried, and extract_info_from_prompt also
    treats a modality-mismatch error as retryable in case a non-text model slips
    through this filter anyway.
    """
    try:
        names = [
            m.name.split("/")[-1]
            for m in genai.list_models()
            if "generateContent" in m.supported_generation_methods
        ]
        names = [
            n for n in names
            if not any(marker in n.lower() for marker in _NON_TEXT_NAME_MARKERS)
        ]
        names.sort(key=lambda n: "flash" not in n)  # prefer flash (cheaper/faster)
        return [n for n in names if n not in _CANDIDATE_MODELS]
    except Exception:
        return []


def extract_info_from_prompt(prompt: str) -> dict:
    """
    Use Gemini Function Calling to extract pipeline config from a natural-language prompt.

    Uses the `tools` parameter (real function calling) rather than prompt-engineering
    a JSON response — the model is *forced* to populate the function schema, giving
    us guaranteed structure and type validation with no fragile JSON parsing.

    Returns dict with keys: target (str), type (str) — or {"error": ...} on failure.
    """
    last_error = None
    names_to_try = [None] + _CANDIDATE_MODELS[1:]
    tried_live_fallback = False
    i = 0
    while i < len(names_to_try):
        name = names_to_try[i]
        i += 1
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
            if not _is_retired_model_error(e):
                return {"error": str(e)}
            # Every hardcoded candidate is retired — ask the API what's live instead
            # of failing outright.
            if i == len(names_to_try) and not tried_live_fallback:
                tried_live_fallback = True
                names_to_try.extend(_live_fallback_model_names())
            continue

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
        # Gemini extracts the target name from natural language, which can drift in
        # case from the actual CSV header (e.g. "churn" vs "Churn") — match
        # case-insensitively before giving up, and use the real column name below.
        case_insensitive_match = next(
            (col for col in df.columns if col.lower() == str(target).lower()), None
        )
        if case_insensitive_match is None:
            return (
                f"Target column '{target}' not found in the uploaded dataset. "
                f"Available columns: {list(df.columns)}",
                parsed, None, None, None,
            )
        target = case_insensitive_match
        parsed["target"] = target

    result, model_info_path, leaderboard, model_dir = run_pipeline(
        df, target_col=target, task_type=parsed.get("type")
    )
    return result, parsed, model_info_path, leaderboard, model_dir
