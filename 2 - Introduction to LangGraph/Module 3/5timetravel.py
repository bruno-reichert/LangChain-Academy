import asyncio
from dotenv import load_dotenv, find_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState, START, END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

# 1. Load keys
load_dotenv(find_dotenv())

# 2. Tools
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

# 3. Model with Bound Tools (Groq)
llm = init_chat_model(
    model="openai/gpt-oss-120b", 
    model_provider="groq"
)
llm_with_tools = llm.bind_tools(tools)

# 4. Graph Construction
sys_msg = SystemMessage(content="You are a helpful assistant tasked with performing arithmetic on a set of inputs.")

def assistant(state: MessagesState):
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")

# Top-level compiled graph (Clean for langgraph dev)
graph = builder.compile()


# --- 5. Interactive Execution Sandbox ---
async def main():
    print("=" * 60)
    print("PART 1: Local In-Memory Time Travel (History, Replay & Fork)")
    print("=" * 60)

    # Local graph with memory checkpointer for history tracking
    local_memory = MemorySaver()
    local_graph = builder.compile(checkpointer=local_memory)

    thread = {"configurable": {"thread_id": "time_travel_1"}}
    initial_input = {"messages": [HumanMessage(content="Multiply 2 and 3")]}

    # Step 1: Initial Run
    print("▶️ Step 1: Running initial graph (2 * 3)...")
    for event in local_graph.stream(initial_input, thread, stream_mode="values"):
        event['messages'][-1].pretty_print()

    # Step 2: Browse State History
    all_states = [s for s in local_graph.get_state_history(thread)]
    print(f"\n📜 Total checkpoints saved in history: {len(all_states)}")

    # Step 3: Replay from a past checkpoint
    to_replay = all_states[-2]  # The initial user prompt checkpoint
    print(f"\n⏮️ Step 3: Replaying from checkpoint: {to_replay.config['configurable']['checkpoint_id']}...")
    for event in local_graph.stream(None, to_replay.config, stream_mode="values"):
        event['messages'][-1].pretty_print()

    # Step 4: Forking the Timeline (Changing 2*3 to 5*3)
    print("\n🌿 Step 4: Forking the state at that past checkpoint to 'Multiply 5 and 3'...")
    to_fork = all_states[-2]
    fork_config = local_graph.update_state(
        to_fork.config,
        {"messages": [HumanMessage(content='Multiply 5 and 3', id=to_fork.values["messages"][0].id)]},
    )

    print("▶️ Executing forked timeline...")
    for event in local_graph.stream(None, fork_config, stream_mode="values"):
        event['messages'][-1].pretty_print()


    # --- PART 2: LangGraph Local Server SDK Time Travel ---
    print("\n" + "=" * 60)
    print("PART 2: LangGraph Server SDK Time Travel (Replay & Fork)")
    print("=" * 60)
    try:
        from langgraph_sdk import get_client
        client = get_client(url="http://127.0.0.1:2024")

        remote_thread = await client.threads.create()
        print(f"Created remote thread: {remote_thread['thread_id']}")

        # Initial remote run
        print("\n▶️ Running remote thread (Multiply 2 and 3)...")
        async for chunk in client.runs.stream(
            remote_thread["thread_id"],
            assistant_id="agent",  # or whichever graph name is in your langgraph.json
            input={"messages": [HumanMessage(content="Multiply 2 and 3")]},
            stream_mode="updates",
        ):
            if chunk.data:
                node_data = chunk.data.get('assistant') or chunk.data.get('tools')
                if node_data and 'messages' in node_data:
                    print(f"  [{list(chunk.data.keys())[0]}]: {node_data['messages'][-1]}")

        # Fetch history remotely
        states = await client.threads.get_history(remote_thread['thread_id'])
        to_fork_remote = states[-2]

        print(f"\n🌿 Forking remote thread from checkpoint: {to_fork_remote['checkpoint_id']}")
        forked_input = {"messages": [HumanMessage(content="Multiply 3 and 3", id=to_fork_remote['values']['messages'][0]['id'])]}

        forked_config = await client.threads.update_state(
            remote_thread["thread_id"],
            forked_input,
            checkpoint_id=to_fork_remote['checkpoint_id']
        )

        print("▶️ Resuming forked remote run...")
        async for chunk in client.runs.stream(
            remote_thread["thread_id"],
            assistant_id="agent",
            input=None,
            stream_mode="updates",
            checkpoint_id=forked_config['checkpoint_id']
        ):
            if chunk.data:
                node_data = chunk.data.get('assistant') or chunk.data.get('tools')
                if node_data and 'messages' in node_data:
                    print(f"  [{list(chunk.data.keys())[0]}]: {node_data['messages'][-1]}")

        print("\n🎉 Time travel and forking completed successfully on both local and server runtimes!")

    except Exception as e:
        print(f"\n⚠️ SDK Server Note: Make sure 'uv run langgraph dev' is active if testing Part 2! (Error: {e})")

if __name__ == "__main__":
    asyncio.run(main())