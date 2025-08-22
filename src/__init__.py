import os
from dotenv import load_dotenv
# from pydantic_ai.models.anthropic import AnthropicModel
# from pydantic_ai.models.fallback import FallbackModel
# from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

load_dotenv(".env")

GEMINI_MODEL=os.getenv("GEMINI_MODEL","")
GEMINI_API_KEY=os.getenv("GEMINI_API_KEY","")
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY","")
OPENAI_MODEL=os.getenv("OPENAI_MODEL","")
ANTHROPIC_MODEL=os.getenv("ANTHROPIC_MODEL","")
ANTHROPIC_API_KEY=os.getenv("ANTHROPIC_API_KEY","")

GOOGLE_PROVIDER = GoogleProvider(api_key=GEMINI_API_KEY)
GOOGLE_MODEL = GoogleModel(GEMINI_MODEL, provider=GOOGLE_PROVIDER)

# INNE MODELE
# ANTHROPIC_MODEL_ = AnthropicModel(ANTHROPIC_MODEL)
# OPENAI_MODEL_ = OpenAIModel(OPENAI_MODEL)
# FALLBACK_MODEL = FallbackModel(OPENAI_MODEL_,ANTHROPIC_MODEL_)

CURRENT_MODEL=GOOGLE_MODEL
CURRENT_PROVIDER=GOOGLE_PROVIDER