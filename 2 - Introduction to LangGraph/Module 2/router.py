from dotenv import load_dotenv, find_dotenv
from langchain.chat_models import init_chat_model
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv(find_dotenv())

# 1. Tool
def multiply(a: int, b: int) -> int:
    """Multiplies a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

# 2. Free LLM with bound tool (Gemini 3.5 Flash Lite)
llm = init_chat_model(
    model="models/gemini-3.5-flash-lite", 
    model_provider="google_genai"
)
llm_with_tools = llm.bind_tools([multiply])

# 3. Node
def tool_calling_llm(state: MessagesState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# 4. Build graph
builder = StateGraph(MessagesState)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("tools", ToolNode([multiply]))
builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges(
    "tool_calling_llm",
    tools_condition,
)
builder.add_edge("tools", END)

# Compile graph
graph = builder.compile()