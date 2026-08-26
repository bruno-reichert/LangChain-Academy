# Generated from: 2contextandstate.ipynb
# Converted at: 2026-08-26T17:26:55.909Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# # Lesson 2.2: Context and State


from dotenv import load_dotenv

load_dotenv()

from dataclasses import dataclass

@dataclass
class ColourContext:
    favourite_colour: str = "blue"
    least_favourite_colour: str = "red"

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

gemini_lite = init_chat_model(
        model="models/gemini-3.5-flash-lite",
        model_provider="google_genai"
    )

agent = create_agent(
    model=gemini_lite,
    context_schema=ColourContext,
)

from langchain.messages import HumanMessage

response = agent.invoke(
    {"messages": [HumanMessage(content="What is my favourite colour?")]},
    context=ColourContext()
)

ans = response["messages"][-1].content
print(ans[0]['text'] if isinstance(ans, list) else ans)

# # 2. Accessing context


from langchain.tools import tool, ToolRuntime

@tool
def get_favourite_colour(runtime: ToolRuntime[ColourContext]) -> str:
    """Get the favourite colour of the user"""
    return runtime.context.favourite_colour

@tool
def get_least_favourite_colour(runtime: ToolRuntime[ColourContext]) -> str:
    """Get the least favourite colour of the user"""
    return runtime.context.least_favourite_colour

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

gemini_lite = init_chat_model(
        model="models/gemini-3.5-flash-lite",
        model_provider="google_genai",
    )

agent = create_agent(
    model=gemini_lite,
    context_schema=ColourContext,
    tools=[get_favourite_colour, get_least_favourite_colour]
)

response = agent.invoke(
    {"messages": [HumanMessage(content="What is my favourite colour?")]},
    context=ColourContext()
)

ans = response["messages"][-1].content
print(ans[0]['text'] if isinstance(ans, list) else ans)

response = agent.invoke(
    {"messages": [HumanMessage(content="What is my favourite colour?")]},
    context=ColourContext(favourite_colour="green")
)

ans = response["messages"][-1].content
print(ans[0]['text'] if isinstance(ans, list) else ans)

# # 3. State


from langchain.agents import AgentState

class CustomState(AgentState):
    favourite_colour: str

from langchain.tools import tool, ToolRuntime
from langgraph.types import Command
from langchain.messages import ToolMessage

@tool
def update_favourite_colour(favourite_colour: str, runtime: ToolRuntime) -> Command:
    """Update the favourite colour of the user in the state once they've revealed it."""
    return Command(update={
        "favourite_colour": favourite_colour, 
        "messages": [ToolMessage("Successfully updated favourite colour", tool_call_id=runtime.tool_call_id)]}
        )

@tool
def read_favourite_colour(runtime: ToolRuntime) -> str:
    """Read the favourite colour of the user from the state."""
    try:
        return runtime.state["favourite_colour"]
    except KeyError:
        return "No favourite colour found in state"

from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

gemini_lite = init_chat_model(
        model="models/gemini-3.5-flash-lite",
        model_provider="google_genai",
    )

agent = create_agent(
    gemini_lite,
    tools=[update_favourite_colour],
    checkpointer=InMemorySaver(),
    state_schema=CustomState
)

response = agent.invoke(
    { "messages": [HumanMessage(content="Hi! My favourite colour is green!")]},
    {"configurable": {"thread_id": "1"}}
)

ans = response["messages"][-1].content
print(ans[0]['text'] if isinstance(ans, list) else ans)

response = agent.invoke(
    { "messages": [HumanMessage(content="What's my favourite colour?")]},
    {"configurable": {"thread_id": "1"}}
)

ans = response["messages"][-1].content
print(ans[0]['text'] if isinstance(ans, list) else ans)

# #