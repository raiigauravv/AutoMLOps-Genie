# tests/conftest.py
# Stubs for heavy optional dependencies so the test suite runs without
# installing the full ML stack or requiring live API keys.
# pytest loads conftest.py before any test module is imported.

import os
import sys
from unittest.mock import MagicMock

# ── Provide a dummy API key so OpenAI() client init doesn't raise ─────────────
os.environ.setdefault("OPENAI_API_KEY", "sk-test-placeholder-key-for-tests")

# ── Stub out autogluon before pipeline_builder is imported ───────────────────
_autogluon_stub = MagicMock()
sys.modules.setdefault("autogluon", _autogluon_stub)
sys.modules.setdefault("autogluon.tabular", _autogluon_stub)

# ── Stub out shap (large C extension) ────────────────────────────────────────
sys.modules.setdefault("shap", MagicMock())

# ── Stub out joblib ───────────────────────────────────────────────────────────
sys.modules.setdefault("joblib", MagicMock())
