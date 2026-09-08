import asyncio
from dotenv import load_dotenv, find_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
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

# --- 3. Graph Logic ---
sys_msg = SystemMessage(content="You are a helpful assistant tasked with performing arithmetic on a set of inputs.")

def assistant(state: MessagesState):
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")

graph = builder.compile(interrupt_before=["tools"])


# --- 4. Interactive & Async Execution Sandbox ---
async def main():
    print("=" * 60)
    print("PART 1: Local In-Memory Breakpoint & Approval")
    print("=" * 60)

    thread_config = {"configurable": {"thread_id": "test_thread_1"}}
    initial_input = {"messages": [HumanMessage(content="Multiply 2 and 3")]}

    # Step 1: Run until the breakpoint
    print("▶️ Running graph until breakpoint...")
    for event in graph.stream(initial_input, thread_config, stream_mode="values"):
        event['messages'][-1].pretty_print()

    # Step 2: Inspect next pending node
    state = graph.get_state(thread_config)
    print(f"\n⏸️ GRAPH PAUSED! Next pending node to execute: {state.next}")

    # Step 3: Resume by passing None!
    user_approval = input("\nDo you approve calling the tool? (yes/no): ")
    if user_approval.strip().lower() == "yes":
        print("✅ Resuming graph with None...")
        for event in graph.stream(None, thread_config, stream_mode="values"):
            event['messages'][-1].pretty_print()
    else:
        print("❌ Operation cancelled by user.")

    # --- PART 2: LangGraph Local Server SDK Streaming ---
    print("\n" + "=" * 60)
    print("PART 2: LangGraph Local Server API Breakpoint Stream")
    print("=" * 60)
    try:
        from langgraph_sdk import get_client
        client = get_client(url="http://127.0.0.1:2024")

        thread = await client.threads.create()
        print(f"Created remote thread: {thread['thread_id']}")

        # Stream until breakpoint
        print("▶️ Streaming remote run until breakpoint...")
        async for chunk in client.runs.stream(
            thread["thread_id"],
            assistant_id="breakpoints",  # Ensure 'breakpoints' is in your langgraph.json!
            input={"messages": [HumanMessage(content="Multiply 2 and 3")]},
            stream_mode="values",
            interrupt_before=["tools"],
        ):
            if chunk.event == "values" and "messages" in chunk.data:
                print(f"Event: {chunk.data['messages'][-1]}")

        # Resume remote run with input=None
        print("\n✅ Resuming remote run with input=None...")
        async for chunk in client.runs.stream(
            thread["thread_id"],
            assistant_id="breakpoints",
            input=None,
            stream_mode="values",
            interrupt_before=["tools"],
        ):
            if chunk.event == "values" and "messages" in chunk.data:
                print(f"Event: {chunk.data['messages'][-1]}")

    except Exception as e:
        print(f"\n⚠️ SDK Server Note: Make sure 'uv run langgraph dev' is active if testing Part 2! (Error: {e})")

if __name__ == "__main__":
    asyncio.run(main())