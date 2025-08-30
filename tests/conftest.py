# tests/conftest.py
import os
import sys

# 1) Ensure repo root on sys.path (so `import src.main` works)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# 2) Set dummy env vars BEFORE any test imports modules that touch providers
os.environ.setdefault("GEMINI_API_KEY", "DUMMY_KEY")
os.environ.setdefault("GEMINI_MODEL", "gemini-1.5-flash")

# Optional (safe):
os.environ.setdefault("OPENAI_API_KEY", "DUMMY_KEY")
os.environ.setdefault("ANTHROPIC_API_KEY", "DUMMY_KEY")
