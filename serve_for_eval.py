# serve_for_eval.py

# 1) Set dummy env vars BEFORE importing src.*
import os
os.environ.setdefault("GEMINI_API_KEY", "DUMMY_KEY")         # <-- correct name
os.environ.setdefault("GEMINI_MODEL", "gemini-1.5-flash")    # any non-empty model name
# optional, if other providers are referenced:
os.environ.setdefault("OPENAI_API_KEY", "DUMMY_KEY")
os.environ.setdefault("ANTHROPIC_API_KEY", "DUMMY_KEY")

# 2) Now import the app
import uvicorn
import src.main as main

# 3) Import emulator AFTER main so FastAPI models are available
from tests.test_pipeline_new import _emulate_pipeline

# 4) Patch the endpoint function to use the emulator (no LLM calls)
def patched_process_question(q: main.Question):
    return _emulate_pipeline(q.content)

main.process_question = patched_process_question

if __name__ == "__main__":
    uvicorn.run(main.app, host="127.0.0.1", port=8000)
