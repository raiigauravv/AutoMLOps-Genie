# 📄 genie_agent/genie_main.py
# LLM-powered Genie: Parse user prompt & launch AutoML pipeline

import os
import json
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from pipelines.pipeline_builder import run_pipeline

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

def extract_info_from_prompt(prompt: str) -> dict:
    """
    Use OpenAI GPT to extract target column and task type from user's NL prompt.
    """
    system_prompt = """
You are an ML assistant. Given a user's task description, extract:
- "target": the column to predict
- "type": "classification" or "regression"
Respond with valid JSON only.
"""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        parsed = json.loads(response.choices[0].message.content.strip())
        return parsed
    except Exception as e:
        return {"error": str(e)}

def genie_respond(prompt: str, df: pd.DataFrame):
    """
    Main Genie logic: parse prompt, run pipeline, return results.
    """
    parsed = extract_info_from_prompt(prompt)
    if "error" in parsed:
        return f"Error parsing prompt: {parsed['error']}", parsed, None, None, None

    target = parsed.get("target")
    if target not in df.columns:
        return f"Target column '{target}' not in uploaded dataset.", parsed, None, None, None

    result, model_info_path, leaderboard, model_dir = run_pipeline(df, target_col=target, task_type=parsed.get("type"))
    return result, parsed, model_info_path, leaderboard, model_dir
