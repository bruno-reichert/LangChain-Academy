"""Model Initialization File

Configures the LLM model used throughout the course.

Default: Anthropic claude-haiku-4-5 (fast, cheap, great for learning).

═══════════════════════════════════════════════════════════════════════════
  ⚠  IMPORTANT: install the matching extra BEFORE swapping providers
═══════════════════════════════════════════════════════════════════════════

  Provider              Install command              Already installed?
  --------------------  ---------------------------  ---------------------
  Anthropic (default)   -                            yes (default dep)
  OpenAI                -                            yes (default dep)
  Azure OpenAI          uv sync --extra azure        no - install first
  AWS Bedrock           uv sync --extra bedrock      no - install first
  Google Vertex/Gemini  uv sync --extra google       no - install first

═══════════════════════════════════════════════════════════════════════════

To swap providers:
  1. Run the install command above (if needed).
  2. Comment out the active model line(s) below.
  3. Uncomment the section for your desired provider.
  4. Set the provider's env vars in `.env` (see notes inline).
"""

import os  # noqa: F401  # used in commented-out model examples below
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=True)

from langchain.chat_models import init_chat_model

# ═══ Zero-Dollar Stack Active Configuration ═══════════════════════════════════
# Primary Engine: Google Gemini 3.5 Flash (Default - 1M TPM free, multimodal)
# model = init_chat_model("google_genai:gemini-3.5-flash", timeout=60, max_retries=2)
# strong_model = init_chat_model("google_genai:gemini-3.5-flash", timeout=120, max_retries=2)

# --- GROQ PRESETS (Uncomment as needed) ---
# Groq 120B (Primary / Strong model fallback):
# model = init_chat_model("groq:openai/gpt-oss-120b", timeout=60, max_retries=2)
# strong_model = init_chat_model("groq:openai/gpt-oss-120b", timeout=120, max_retries=2)

# Groq 20B (Fast subagent / Sharded quota fallback):
# model = init_chat_model("groq:openai/gpt-oss-20b", timeout=60, max_retries=2)

# Groq Qwen 27B
# model = init_chat_model("groq:qwen/qwen3.8-27b", timeout=60, max_retries=2)


# ═══ OpenRouter (Free Hosted Open-Source Models) ══════════════════════════════
# Free models available; sign up at openrouter.ai and get an API key
# Requires OPENROUTER_API_KEY in .env

from langchain_openai import ChatOpenAI

# Default free endpoint suggested by the course:
model = ChatOpenAI(
    model="nvidia/nemotron-3-ultra-550b-a55b:free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    temperature=0.1,
)

strong_model = ChatOpenAI(
    model="nvidia/nemotron-3-ultra-550b-a55b:free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    temperature=0.1,
)


# Or any active free open-source model on OpenRouter, for example:

# PLEASE WORK
# model = ChatOpenAI(
#     model="cohere/north-mini-code:free",
#     base_url="https://openrouter.ai/api/v1",
#     api_key=os.environ.get("OPENROUTER_API_KEY"),
#     temperature=0.0,
# )