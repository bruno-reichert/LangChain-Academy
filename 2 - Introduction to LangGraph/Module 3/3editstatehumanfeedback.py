import asyncio
from dotenv import load_dotenv, find_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv(find_dotenv())

# --- 1. Tools ---
def multiply(a: int, b: int) -> int:
    """Multiply a and b."""
    return a * b

def add(a: int, b: int) -> int:
    """Adds a and b."""
    return a + b

def divide(a: int, b: int) -> float:
    """Divide a by b."""
    return a / b

tools = [add, multiply, divide]

# --- 2. Model with Bound Tools (Groq) ---
llm = init_chat_model(
    model="openai/gpt-oss-120b", 
    model_provider="groq"
)
llm_with_tools = llm.bind_tools(tools)

# --- 3. Build Graph with `human_feedback` node ---
sys_msg = SystemMessage(content="You are a helpful assistant tasked with performing arithmetic on a set of inputs.")

# 1. Inherit from MessagesState with total=False
class State(MessagesState, total=False):
    pass  # Inherits 'messages' automatically!

# 2. Use your new State in the graph builder
builder = StateGraph(State)

# Placeholder node for human feedback
def human_feedback(state: MessagesState):
    pass

def assistant(state: MessagesState):
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

builder = StateGraph(State)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))
builder.add_node("human_feedback", human_feedback)

builder.add_edge(START, "human_feedback")
builder.add_edge("human_feedback", "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "human_feedback")

graph = builder.compile(interrupt_before=["human_feedback"])



# --- 4. Interactive Execution Sandbox ---
async def main():
    print("=" * 60)
    print("PART 1: Local In-Memory State Editing & Human Feedback")
    print("=" * 60)

    thread = {"configurable": {"thread_id": "feedback_thread_1"}}
    initial_input = {"messages": [HumanMessage(content="Multiply 2 and 3")]}

    # Step 1: Run until the human_feedback breakpoint
    print("▶️ Running graph until human_feedback breakpoint...")
    for event in graph.stream(initial_input, thread, stream_mode="values"):
        event["messages"][-1].pretty_print()

    # Step 2: Human edits state!
    user_input = input("\nTell me how you want to update the state (e.g. 'No, multiply 3 and 3!'): ")
    if not user_input.strip():
        user_input = "No, actually multiply 3 and 3!"

    print(f"\n✍️ Updating graph state as_node='human_feedback' with: '{user_input}'")
    graph.update_state(thread, {"messages": [HumanMessage(content=user_input)]}, as_node="human_feedback")

    # Step 3: Continue graph execution with None!
    print("\n▶️ Resuming graph with None...")
    for event in graph.stream(None, thread, stream_mode="values"):
        event["messages"][-1].pretty_print()

    # --- PART 2: LangGraph Local Server SDK State Editing ---
    print("\n" + "=" * 60)
    print("PART 2: LangGraph Server API State Editing")
    print("=" * 60)
    try:
        from langgraph_sdk import get_client
        client = get_client(url="http://127.0.0.1:2024")

        remote_thread = await client.threads.create()
        print(f"Created remote thread: {remote_thread['thread_id']}")

        # Stream until assistant breakpoint
        async for chunk in client.runs.stream(
            remote_thread["thread_id"],
            assistant_id="agent",
            input={"messages": [HumanMessage(content="Multiply 2 and 3")]},
            stream_mode="values",
            interrupt_before=["assistant"],
        ):
            if chunk.event == "values" and "messages" in chunk.data:
                print(f"Event: {chunk.data['messages'][-1]}")

        # Fetch and edit state remotely
        current_state = await client.threads.get_state(remote_thread['thread_id'])
        last_message = current_state['values']['messages'][-1]
        last_message['content'] = "No, actually multiply 3 and 3!"

        print("\n✍️ Updating remote state with edited message...")
        await client.threads.update_state(remote_thread['thread_id'], {"messages": last_message})

        # Resume remote run
        print("▶️ Resuming remote run with input=None...")
        async for chunk in client.runs.stream(
            remote_thread["thread_id"],
            assistant_id="agent",
            input=None,
            stream_mode="values",
            interrupt_before=["assistant"],
        ):
            if chunk.event == "values" and "messages" in chunk.data:
                print(f"Event: {chunk.data['messages'][-1]}")

    except Exception as e:
        print(f"\n⚠️ SDK Server Note: Make sure 'uv run langgraph dev' is running if testing Part 2! (Error: {e})")

if __name__ == "__main__":
    asyncio.run(main())