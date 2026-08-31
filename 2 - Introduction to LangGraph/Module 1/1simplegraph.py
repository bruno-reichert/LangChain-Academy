# Generated from: 1simplegraph.ipynb
# Converted at: 2026-08-31T18:45:00.069Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# # Lesson 1: Simple Graph


# ## State
# 
# First, define the [State](https://docs.langchain.com/oss/python/langgraph/graph-api#state) of the graph. 
# 
# The State schema serves as the input schema for all Nodes and Edges in the graph.
# 
# Let's use the `TypedDict` class from python's `typing` module as our schema, which provides type hints for the keys.


from typing_extensions import TypedDict

class State(TypedDict):
    graph_state: str

# ## Nodes
# 
# [Nodes](https://docs.langchain.com/oss/python/langgraph/graph-api/#nodes) are just python functions.
# 
# The first positional argument is the state, as defined above.
# 
# Because the state is a `TypedDict` with schema as defined above, each node can access the key, `graph_state`, with `state['graph_state']`.
# 
# Each node returns a new value of the state key `graph_state`.
#   
# By default, the new value returned by each node [will override](https://docs.langchain.com/oss/python/langgraph/graph-api/#reducers) the prior state value.


def node_1(state):
    print("---Node 1---")
    return {"graph_state": state['graph_state'] +" I'll be visiting"}

def node_2(state):
    print("---Node 2---")
    return {"graph_state": state['graph_state'] +" Paris next year!"}

def node_3(state):
    print("---Node 3---")
    return {"graph_state": state['graph_state'] +" London next year!"}

# ## Edges
# 
# [Edges](https://docs.langchain.com/oss/python/langgraph/graph-api/#edges) connect the nodes.
# 
# Normal Edges are used if you want to *always* go from, for example, `node_1` to `node_2`.
# 
# [Conditional Edges](https://docs.langchain.com/oss/python/langgraph/graph-api/#conditional-edges) are used if you want to *optionally* route between nodes.
#  
# Conditional edges are implemented as functions that return the next node to visit based on some logic.


import random
from typing import Literal

def decide_travel(state) -> Literal["node_2", "node_3"]:
    
    # Often, we will use state to decide on the next node to visit
    user_input = state['graph_state'] 
    
    # Here, let's just do a 50 / 50 split between nodes 2, 3
    if random.random() < 0.5:

        # 50% of the time, we return Node 2
        return "node_2"
    
    # 50% of the time, we return Node 3
    return "node_3"

# ## Graph Construction
# 
# Now, we build the graph from our components defined above.
# 
# The [StateGraph class](https://docs.langchain.com/oss/python/langgraph/graph-api/#stategraph) is the graph class that we can use.
#  
# First, we initialize a StateGraph with the `State` class we defined above.
#  
# Then, we add our nodes and edges.
# 
# We use the  [`START` Node, a special node](https://docs.langchain.com/oss/python/langgraph/graph-api/#start-node) that sends user input to the graph, to indicate where to start our graph.
#  
# The [`END` Node](https://docs.langchain.com/oss/python/langgraph/graph-api/#end-node) is a special node that represents a terminal node. 
# 
# Finally, we [compile our graph](https://docs.langchain.com/oss/python/langgraph/graph-api/#compiling-your-graph) to perform a few basic checks on the graph structure. 
# 
# We can visualize the graph as a [Mermaid diagram](https://github.com/mermaid-js/mermaid).


from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END

# Build graph
builder = StateGraph(State)
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_node("node_3", node_3)

# Logic
builder.add_edge(START, "node_1")
builder.add_conditional_edges("node_1", decide_travel)
builder.add_edge("node_2", END)
builder.add_edge("node_3", END)

# Add
graph = builder.compile()

# View
display(Image(graph.get_graph().draw_mermaid_png()))

# ## Graph Invocation
# 
# The compiled graph implements the [runnable](https://reference.langchain.com/python/langchain_core/runnables/?h=runnables) protocol.
# 
# This provides a standard way to execute LangChain components. 
#  
# `invoke` is one of the standard methods in this interface.
# 
# The input is a dictionary `{"graph_state": "Hi, this is lance."}`, which sets the initial value for our graph state dict.
# 
# When `invoke` is called, the graph starts execution from the `START` node.
# 
# It progresses through the defined nodes (`node_1`, `node_2`, `node_3`) in order.
# 
# The conditional edge will traverse from node `1` to node `2` or `3` using a 50/50 decision rule. 
# 
# Each node function receives the current state and returns a new value, which overrides the graph state.
# 
# The execution continues until it reaches the `END` node.


graph.invoke({"graph_state" : "Hi! I'm Oliver!"})