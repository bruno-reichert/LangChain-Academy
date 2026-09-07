import asyncio
from typing import Literal
from dotenv import load_dotenv, find_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END, MessagesState

load_dotenv(find_dotenv())

# 1. LLM
model = init_chat_model(
    model="openai/gpt-oss-120b", 
    model_provider="groq"
)

# 2. State 
class State(MessagesState):
    summary: str

# 3. Nodes
def call_model(state: State, config: RunnableConfig):
    summary = state.get("summary", "")
    if summary:
        system_message = f"Summary of conversation earlier: {summary}"
        messages = [SystemMessage(content=system_message)] + state["messages"]
    else:
        messages = state["messages"]
    
    response = model.invoke(messages, config)
    return {"messages": response}

def summarize_conversation(state: State):
    summary = state.get("summary", "")
    if summary:
        summary_message = (
            f"This is summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
    else:
        summary_message = "Create a summary of the conversation above:"

    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = model.invoke(messages)
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    return {"summary": response.content, "messages": delete_messages}

def should_continue(state: State) -> Literal["summarize_conversation", END]:
    messages = state["messages"]
    if len(messages) > 6:
        return "summarize_conversation"
    return END

# 4. Build and Compile Graph
workflow = StateGraph(State)
workflow.add_node("conversation", call_model)
workflow.add_node("summarize_conversation", summarize_conversation)
workflow.add_edge(START, "conversation")
workflow.add_conditional_edges("conversation", should_continue)
workflow.add_edge("summarize_conversation", END)

# Expose graph for LangGraph server
graph = workflow.compile()


# 5. Standalone Execution (Wrapped so it doesn't crash server import)
async def test_streaming():
    config = {"configurable": {"thread_id": "3"}}
    input_message = HumanMessage(content="Tell me about the 49ers NFL team")
    
    print("\n--- Testing astream_events ---")
    async for event in graph.astream_events({"messages": [input_message]}, config, version="v2"):
        if event["event"] == "on_chat_model_stream" and event['metadata'].get('langgraph_node', '') == 'conversation':
            print(event["data"]["chunk"].content, end="", flush=True)
    print()

if __name__ == "__main__":
    asyncio.run(test_streaming())