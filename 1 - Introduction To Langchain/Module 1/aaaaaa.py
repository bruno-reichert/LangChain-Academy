import json

notebook = {
    "cells": [
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from dotenv import load_dotenv, find_dotenv\n",
                "from langchain.chat_models import init_chat_model\n",
                "from langchain.agents import create_agent\n",
                "from langchain.messages import HumanMessage, AIMessage\n",
                "from pprint import pprint\n",
                "\n",
                "load_dotenv(find_dotenv())"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 1. Groq Model\n",
                "model = init_chat_model(model='openai/gpt-oss-120b', model_provider='groq')\n",
                "response = model.invoke('Hello! Hello! You there?')\n",
                "print(response.content)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 2. Gemini Model & Agent\n",
                "gemini = init_chat_model(model='models/gemini-3.5-flash', model_provider='google_genai')\n",
                "agent = create_agent(model=gemini)\n",
                "\n",
                "response = agent.invoke({\n",
                "    'messages': [\n",
                "        HumanMessage(content='What\\'s a good place to buy a stapler? Mine is broken!'),\n",
                "        AIMessage(content='Try a Staples! They have lots of good stuff.'),\n",
                "        HumanMessage(content='But there isn\\'t a Staples nearby! I live in Sheboygan! I\\'m doomed! What do I do?')\n",
                "    ]\n",
                "})\n",
                "\n",
                "ans = response['messages'][-1].content\n",
                "print(ans[0]['text'] if isinstance(ans, list) else ans)"
            ]
        }
    ],
    "metadata": {
        "language_info": {"name": "python"}
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

with open("1foundationalmodels.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print("Created 1foundationalmodels.ipynb successfully!")