# Generated from: 5agentwithmemory.ipynb
# Converted at: 2026-09-03T18:39:47.780Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# # Agent memory
# 
# ## Review
# 
# Previously, we built an agent that can:
# 
# * `act` - let the model call specific tools 
# * `observe` - pass the tool output back to the model 
# * `reason` - let the model reason about the tool output to decide what to do next (e.g., call another tool or just respond directly)
# 
# ![Screenshot 2024-08-21 at 12.45.32 PM.png](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66dbab7453080e6802cd1703_agent-memory1.png)
# 
# ## Goals
# 
# Now, we're going extend our agent by introducing memory.


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

# This will be a tool
def add(a: int, b: int) -> int:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    return a + b

def subtract(a: int, b: int) -> int:
    """Subtract b from a.

    Args:
        a: first int
        b: second int
    """
    return a - b

def divide(a: int, b: int) -> float:
    """Divide a and b.

    Args:
        a: first int
        b: second int
    """
    return a / b

tools = [add, multiply, divide, subtract]
llm = llm = init_chat_model(
    model="openai/gpt-oss-120b", 
    model_provider="groq"
)

# For this ipynb we set parallel tool calling to false as math generally is done sequentially, and this time we have 3 tools that can do math
# the OpenAI model specifically defaults to parallel tool calling for efficiency, see https://python.langchain.com/docs/how_to/tool_calling_parallel/
# play around with it and see how the model behaves with math equations!
llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)

from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, SystemMessage

# System message
sys_msg = SystemMessage(content="You are a helpful assistant tasked with performing arithmetic on a set of inputs.")

# Node
def assistant(state: MessagesState):
   return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition, ToolNode
from IPython.display import Image, display

# Graph
builder = StateGraph(MessagesState)

# Define nodes: these do the work
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

# Define edges: these determine how the control flow moves
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition,
)
builder.add_edge("tools", "assistant")
react_graph = builder.compile()

# Show
display(Image(react_graph.get_graph(xray=True).draw_mermaid_png()))

# ## Memory
# 
# Let's run our agent, as before.


messages = [HumanMessage(content="Add 3 and 4.")]
messages = react_graph.invoke({"messages": messages})
for m in messages['messages']:
    m.pretty_print()

messages = [HumanMessage(content="Multiply that by 2.")]
messages = react_graph.invoke({"messages": messages})
for m in messages['messages']:
    m.pretty_print()

# We don't retain memory of 7 from our initial chat!
# 
# This is because [state is transient](https://github.com/langchain-ai/langgraph/discussions/352#discussioncomment-9291220) to a single graph execution.
# 
# Of course, this limits our ability to have multi-turn conversations with interruptions. 
# 
# We can use [persistence](https://docs.langchain.com/oss/python/langgraph/persistence) to address this! 
# 
# LangGraph can use a checkpointer to automatically save the graph state after each step.
# 
# This built-in persistence layer gives us memory, allowing LangGraph to pick up from the last state update. 
# 
# One of the easiest checkpointers to use is the `MemorySaver`, an in-memory key-value store for Graph state.
# 
# All we need to do is simply compile the graph with a checkpointer, and our graph has memory!


from langgraph.checkpoint.memory import MemorySaver
memory = MemorySaver()
react_graph_memory = builder.compile(checkpointer=memory)

# When we use memory, we need to specify a `thread_id`.
# 
# This `thread_id` will store our collection of graph states.
# 
# Here is a cartoon:
# 
# * The checkpointer write the state at every step of the graph
# * These checkpoints are saved in a thread 
# * We can access that thread in the future using the `thread_id`
# 
# ![state.jpg](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66e0e9f526b41a4ed9e2d28b_agent-memory2.png)
# 


# Specify a thread
config = {"configurable": {"thread_id": "1"}}

# Specify an input
messages = [HumanMessage(content="Add 3 and 4.")]

# Run
messages = react_graph_memory.invoke({"messages": messages},config)
for m in messages['messages']:
    m.pretty_print()

# If we pass the same `thread_id`, then we can proceed from from the previously logged state checkpoint! 
# 
# In this case, the above conversation is captured in the thread.
# 
# The `HumanMessage` we pass (`"Multiply that by 2."`) is appended to the above conversation.
# 
# So, the model now know that `that` refers to the `The sum of 3 and 4 is 7.`.


messages = [HumanMessage(content="Multiply that by 2.")]
messages = react_graph_memory.invoke({"messages": messages}, config)
for m in messages['messages']:
    m.pretty_print()