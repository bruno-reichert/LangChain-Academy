import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

print("=== CHECKING GROQ MODELS ===")
try:
    from groq import Groq
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    groq_models = [m.id for m in groq_client.models.list().data]
    for m in sorted(groq_models):
        if "whisper" not in m and "guard" not in m:  # filter out audio/guard models
            print(f"  👉 {m}")
except Exception as e:
    print(f"Groq error: {e}")

print("\n=== CHECKING GOOGLE GEMINI MODELS ===")
try:
    from google import genai
    google_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    for m in google_client.models.list():
        if "gemini" in m.name:
            print(f"  👉 {m.name}")
except Exception as e:
    print(f"Google error: {e}")