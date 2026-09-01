# Generated from: 3router.ipynb
# Converted at: 2026-09-01T19:09:31.033Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# # Router
# 
# ## Review
# 
# We built a graph that uses `messages` as state and a chat model with bound tools.
# 
# We saw that the graph can:
# 
# * Return a tool call
# * Return a natural language response
# 
# ## Goals
# 
# We can think of this as a router, where the chat model routes between a direct response or a tool call based upon the user input.
# 
# This is a simple example of an agent, where the LLM is directing the control flow either by calling a tool or just responding directly. 
# 
# ![Screenshot 2024-08-21 at 9.24.09 AM.png](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66dbac6543c3d4df239a4ed1_router1.png)
# 
# Let's extend our graph to work with either output! 
# 
# For this, we can use two ideas:
# 
# (1) Add a node that will call our tool.
# 
# (2) Add a conditional edge that will look at the chat model output, and route to our tool calling node or simply end if no tool call is performed. 
# 
# 


from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

from langchain.chat_models import init_chat_model

def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

def square_root(a: int) -> int:
    """Calculate the square root of a.

    Args:
        a: first int
    """
    return a ** 0.5

# 1. Initialize Groq
llm = init_chat_model(
    model="openai/gpt-oss-120b", 
    model_provider="groq"
)

llm_with_tools = llm.bind_tools([multiply, square_root])

#  We use the  [built-in `ToolNode`](https://langchain-ai.github.io/langgraph/reference/agents/#langgraph.prebuilt.tool_node.ToolNode) and simply pass a list of our tools to initialize it. 
#  
#  We use the [built-in `tools_condition`](https://langchain-ai.github.io/langgraph/reference/agents/#langgraph.prebuilt.tool_node.tools_condition) as our conditional edge.


from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition

# Node
def tool_calling_llm(state: MessagesState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# Build graph
builder = StateGraph(MessagesState)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("tools", ToolNode([multiply, square_root]))
builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges(
    "tool_calling_llm",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition,
)
builder.add_edge("tools", END)
graph = builder.compile()

# View
display(Image(graph.get_graph().draw_mermaid_png()))

from langchain_core.messages import HumanMessage
messages = [HumanMessage(content="Hello, what is 2 multiplied by 2?")]
messages = graph.invoke({"messages": messages})
for m in messages['messages']:
    m.pretty_print()

from langchain_core.messages import HumanMessage
messages = [HumanMessage(content="Hello, what is the square root of 25600?")]
messages = graph.invoke({"messages": messages})
for m in messages['messages']:
    m.pretty_print()

# Now, we can see that the graph runs the tool!
# 
# It responds with a `ToolMessage`.