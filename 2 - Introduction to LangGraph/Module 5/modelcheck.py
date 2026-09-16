import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

# Initialize client (uses GROQ_API_KEY from your environment)
client = Groq()

# Fetch all available models
models = client.models.list()

# Filter and sort active models
active_models = sorted([m.id for m in models.data if m.active])

print(f"Found {len(active_models)} active Groq models:\n")
for model_id in active_models:
    print(f" - {model_id}")