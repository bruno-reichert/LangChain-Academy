# Generated from: 3multiagentsystems.ipynb
# Converted at: 2026-08-26T17:27:40.132Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# # Lesson 2.3: Multi-Agent Systems


from dotenv import load_dotenv

load_dotenv()

from langchain.tools import tool

@tool
def square_root(x: float) -> float:
    """Calculate the square root of a number"""
    return x ** 0.5

@tool
def square(x: float) -> float:
    """Calculate the square of a number"""
    return x ** 2

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

gemini_lite = init_chat_model(
        model="models/gemini-3.5-flash-lite",
        model_provider="google_genai"
    )

subagent_1 = create_agent(
    model=gemini_lite,
    tools=[square_root]
)

subagent_2 = create_agent(
    model=gemini_lite,
    tools=[square]
)

from langchain.messages import HumanMessage

@tool
def call_subagent_1(x: float) -> float:
    """Call subagent 1 in order to calculate the square root of a number"""
    response = subagent_1.invoke({"messages": [HumanMessage(content=f"Calculate the square root of {x}")]})
    return response["messages"][-1].content

@tool
def call_subagent_2(x: float) -> float:
    """Call subagent 2 in order to calculate the square of a number"""
    response = subagent_2.invoke({"messages": [HumanMessage(content=f"Calculate the square of {x}")]})
    return response["messages"][-1].content

## Creating the main agent

main_agent = create_agent(
    model=gemini_lite,
    tools=[call_subagent_1, call_subagent_2],
    system_prompt="You are a helpful assistant who can call subagents to calculate the square root or square of a number.")

question = "What is the square root of 456?"

response = main_agent.invoke({"messages": [HumanMessage(content=question)]})

ans = response["messages"][-1].content
print(ans[0]['text'] if isinstance(ans, list) else ans)

question = "What is the the result of 21 squared?"

response = main_agent.invoke({"messages": [HumanMessage(content=question)]})

ans = response["messages"][-1].content
print(ans[0]['text'] if isinstance(ans, list) else ans)