# Generated from: 4filterandtrimmingmsgs.ipynb
# Converted at: 2026-09-05T18:44:03.629Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# # Filtering and trimming messages
# 
# ## Review
# 
# Now, we have a deeper understanding of a few things: 
# 
# * How to customize the graph state schema
# * How to define custom state reducers
# * How to use multiple graph state schemas
# 
# ## Goals
# 
# Now, we can start using these concepts with models in LangGraph!
#  
# In the next few sessions, we'll build towards a chatbot that has long-term memory.
# 
# Because our chatbot will use messages, let's first talk a bit more about advanced ways to work with messages in graph state.


# ## Messages as state
# 
# First, let's define some messages.


from pprint import pprint
from langchain_core.messages import AIMessage, HumanMessage
messages = [AIMessage(f"So you said you were researching ocean mammals?", name="Bot")]
messages.append(HumanMessage(f"Yes, I know about whales. But what others should I learn about?", name="Lance"))

for m in messages:
    m.pretty_print()

# Recall we can pass them to a chat model.


from langchain.chat_models import init_chat_model

llm = llm = init_chat_model(
    model="openai/gpt-oss-120b", 
    model_provider="groq"
)
llm.invoke(messages)

# We can run our chat model in a simple graph with `MessagesState`.


from IPython.display import Image, display
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph, START, END

# Node
def chat_model_node(state: MessagesState):
    return {"messages": llm.invoke(state["messages"])}

# Build graph
builder = StateGraph(MessagesState)
builder.add_node("chat_model", chat_model_node)
builder.add_edge(START, "chat_model")
builder.add_edge("chat_model", END)
graph = builder.compile()

# View
display(Image(graph.get_graph().draw_mermaid_png()))

output = graph.invoke({'messages': messages})
for m in output['messages']:
    m.pretty_print()

# ## Reducer
# 
# A practical challenge when working with messages is managing long-running conversations. 
# 
# Long-running conversations result in high token usage and latency if we are not careful, because we pass a growing list of messages to the model.
# 
# We have a few ways to address this.
# 
# First, recall the trick we saw using `RemoveMessage` and the `add_messages` reducer.


from langchain_core.messages import RemoveMessage

# Nodes
def filter_messages(state: MessagesState):
    # Delete all but the 2 most recent messages
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    return {"messages": delete_messages}

def chat_model_node(state: MessagesState):    
    return {"messages": [llm.invoke(state["messages"])]}

# Build graph
builder = StateGraph(MessagesState)
builder.add_node("filter", filter_messages)
builder.add_node("chat_model", chat_model_node)
builder.add_edge(START, "filter")
builder.add_edge("filter", "chat_model")
builder.add_edge("chat_model", END)
graph = builder.compile()

# View
display(Image(graph.get_graph().draw_mermaid_png()))

# Message list with a preamble
messages = [AIMessage("Hi.", name="Bot", id="1")]
messages.append(HumanMessage("Hi.", name="Lance", id="2"))
messages.append(AIMessage("So you said you were researching ocean mammals?", name="Bot", id="3"))
messages.append(HumanMessage("Yes, I know about whales. But what others should I learn about?", name="Lance", id="4"))

# Invoke
output = graph.invoke({'messages': messages})
for m in output['messages']:
    m.pretty_print()

# ## Filtering messages
# 
# If you don't need or want to modify the graph state, you can just filter the messages you pass to the chat model.
# 
# For example, just pass in a filtered list: `llm.invoke(messages[-1:])` to the model.


# Node
def chat_model_node(state: MessagesState):
    return {"messages": [llm.invoke(state["messages"][-1:])]}

# Build graph
builder = StateGraph(MessagesState)
builder.add_node("chat_model", chat_model_node)
builder.add_edge(START, "chat_model")
builder.add_edge("chat_model", END)
graph = builder.compile()

# View
display(Image(graph.get_graph().draw_mermaid_png()))

# Let's take our existing list of messages, append the above LLM response, and append a follow-up question.


messages.append(output['messages'][-1])
messages.append(HumanMessage(f"Tell me more about Narwhals!", name="Lance"))

for m in messages:
    m.pretty_print()

# Invoke, using message filtering
output = graph.invoke({'messages': messages})
for m in output['messages']:
    m.pretty_print()

# The state has all of the mesages.
# 
# But, let's look at the LangSmith trace to see that the model invocation only uses the last message:
# 
# https://smith.langchain.com/public/75aca3ce-ef19-4b92-94be-0178c7a660d9/r


# ## Trim messages
# 
# Another approach is to [trim messages](https://docs.langchain.com/oss/python/langgraph/add-memory#trim-messages), based upon a set number of tokens. 
# 
# This restricts the message history to a specified number of tokens.
# 
# While filtering only returns a post-hoc subset of the messages between agents, trimming restricts the number of tokens that a chat model can use to respond.
# 
# See the `trim_messages` below.


from langchain_core.messages import trim_messages

# Node
def chat_model_node(state: MessagesState):
    messages = trim_messages(
            state["messages"],
            max_tokens=100,
            strategy="last",
            token_counter=init_chat_model(model="openai/gpt-oss-120b", 
                model_provider="groq"),
            allow_partial=False,
        )
    return {"messages": [llm.invoke(messages)]}

# Build graph
builder = StateGraph(MessagesState)
builder.add_node("chat_model", chat_model_node)
builder.add_edge(START, "chat_model")
builder.add_edge("chat_model", END)
graph = builder.compile()

# View
display(Image(graph.get_graph().draw_mermaid_png()))

messages.append(output['messages'][-1])
messages.append(HumanMessage(f"Tell me where Orcas live!", name="Lance"))

import transformers

# Example of trimming messages
trim_messages(
            messages,
            max_tokens=100,
            strategy="last",
            token_counter=llm,
            allow_partial=False
        )

# Invoke, using message trimming in the chat_model_node 
messages_out_trim = graph.invoke({'messages': messages})

messages_out_trim

# Let's look at the LangSmith trace to see the model invocation:
# 
# https://smith.langchain.com/public/b153f7e9-f1a5-4d60-8074-f0d7ab5b42ef/r