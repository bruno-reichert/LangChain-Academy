# %% [markdown]
# # 0. Setup & Imports (Run this first)

# %%
import os
from pprint import pprint
from dotenv import load_dotenv, find_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.messages import HumanMessage, AIMessage

load_dotenv(find_dotenv())

# %% [markdown]
# # 1. Initialising and invoking a model (Groq)

# %%
model = init_chat_model(
    model="openai/gpt-oss-120b", 
    model_provider="groq"
)

response = model.invoke("Hello! Hello! You there?")
print("=== Content ===")
print(response.content)

print("\n=== Metadata ===")
pprint(response.response_metadata)

# %% [markdown]
# # 2. Customizing response (Temperature)

# %%
model_creative = init_chat_model(
    model="openai/gpt-oss-120b", 
    model_provider="groq",
    temperature=1.0
)

response = model_creative.invoke("What's the moon made of?")
print(response.content)

# %% [markdown]
# # 3. Other providers (Google Gemini)

# %%
gemini_model = init_chat_model(
    model="models/gemini-3.5-flash", 
    model_provider="google_genai"
)

response = gemini_model.invoke("What model of Air Fryer is best for a family of 4?")
if isinstance(response.content, list):
    print(response.content[0]['text'])
else:
    print(response.content)

# %% [markdown]
# # 4. Initialising and invoking an agent (Multi-turn)

# %%
agent = create_agent(model=gemini_model)

messages = [
    HumanMessage(content="What's a good place to buy a stapler? Mine is broken!"),
    AIMessage(content="Try a Staples! They have lots of good stuff."),
    HumanMessage(content="But there isn't a Staples nearby! I live in Sheboygan! I'm doomed! What do I do?")
]

response = agent.invoke({"messages": messages})

# Print just the AI's final answer:
final_answer = response["messages"][-1].content
if isinstance(final_answer, list):
    print(final_answer[0]['text'])
else:
    print(final_answer)