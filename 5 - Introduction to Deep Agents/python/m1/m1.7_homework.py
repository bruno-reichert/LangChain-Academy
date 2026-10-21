# python/m1/m1.7_homework.py
"""M1.7 Homework: Design Your Own Multi-Thread Scenario.

THE IDEA
The lab used one topic (favorite colour) across two threads to show that
thread_a remembers it and thread_b doesn't. This homework asks you to
design your own multi-turn scenario and prove three things:
  1. that state persists within a thread across separate invoke() calls,
  2. that a different thread_id starts with no memory of it, and
  3. that this persistence lives in the MemorySaver instance rather than
     the thread_id string itself, by asking a brand-new agent with its
     own fresh MemorySaver the same question on thread_a's thread_id.

WHAT YOU FILL IN
  TODO 1: pick your own topic/fact for the agent to remember, and set up
    two or more of your own thread configs (different thread_ids).
  TODO 2: run the turns that demonstrate persistence (same thread
    remembers), isolation (a different thread doesn't), and checkpointer
    scope (a fresh MemorySaver on the same thread_id doesn't either).

RUN
  cd python
  uv run ./m1/m1.7_homework.py
"""

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from deepagents import create_deep_agent
from langgraph.checkpoint.memory import MemorySaver

from models import model

agent = create_deep_agent(
    model=model,
    checkpointer=MemorySaver(),
)


# ════════════════════════════════════════════════════════════════════════
# TODO 1: Pick your own topic and set up two or more thread configs.
#
# Requirements:
#   - Use a topic/fact of your own choosing (not favorite colour, the
#     lab's example).
#   - Define at least two thread configs with different thread_ids, e.g.
#     thread_a = {"configurable": {"thread_id": "my-thread-a"}}
# ════════════════════════════════════════════════════════════════════════

thread_a = {"configurable": {"thread_id": "user-jdm-project-101"}}
thread_b = {"configurable": {"thread_id": "user-jdm-project-102"}}

# ════════════════════════════════════════════════════════════════════════
# TODO 2: Run the turns that demonstrate persistence, isolation, and
# checkpointer scope.
#
# Requirements:
#   - In thread_a, send at least two turns: one that gives the agent your
#     fact, and a later one that asks it back. Print both responses.
#   - In thread_b (a different thread_id), ask the same follow-up
#     question with no prior context, and print the response. It should
#     NOT know the fact from thread_a.
#   - Build a SECOND agent with its own fresh MemorySaver(), and invoke it
#     on thread_a's thread_id, asking the same follow-up question. Print
#     the response and a line explaining why it doesn't know the fact
#     even though the thread_id matches: memory lives in the MemorySaver
#     instance, not in the thread_id string alone.
#
# Example shape for the second agent (delete this and write your own):
#   fresh_agent = create_deep_agent(model=model, checkpointer=MemorySaver())
#   result = fresh_agent.invoke(
#       {"messages": [{"role": "user", "content": "..."}]},
#       config=thread_a,
#   )
# ════════════════════════════════════════════════════════════════════════

def run_scenario():
    # ── Proof 1: Persistence within the same thread ──────────────────
    print("=== 1. Testing Persistence (Thread A) ===")
    turn_1 = agent.invoke(
        {"messages": [{"role": "user", "content": "I am planning to build an NA Mazda Miata with an ITB setup."}]},
        config=thread_a,
    )
    print("Thread A, Turn 1:")
    print(turn_1["messages"][-1].text)

    turn_2 = agent.invoke(
        {"messages": [{"role": "user", "content": "What engine setup and car did I say I was planning to build?"}]},
        config=thread_a,
    )
    print("\nThread A, Turn 2 (Recall):")
    print(turn_2["messages"][-1].text)

    # ── Proof 2: Isolation across different thread_ids ───────────────
    print("\n=== 2. Testing Isolation (Thread B) ===")
    turn_b = agent.invoke(
        {"messages": [{"role": "user", "content": "What engine setup and car did I say I was planning to build?"}]},
        config=thread_b,
    )
    print("Thread B, Turn 1 (Fresh thread on same agent):")
    print(turn_b["messages"][-1].text)

    # ── Proof 3: Checkpointer Scoping (Fresh MemorySaver) ────────────
    print("\n=== 3. Testing Checkpointer Instance Scope ===")
    fresh_agent = create_deep_agent(
        model=model,
        checkpointer=MemorySaver(),  # Brand new in-memory storage instance
    )
    fresh_result = fresh_agent.invoke(
        {"messages": [{"role": "user", "content": "What engine setup and car did I say I was planning to build?"}]},
        config=thread_a,  # Same thread_id string, but separate checkpointer!
    )
    print("Fresh Agent on Thread A's ID:")
    print(fresh_result["messages"][-1].text)
    print("\nExplanation: Even though the thread_id string matches ('user-jdm-project-101'), "
          "memory is scoped to the specific MemorySaver Python object in RAM. "
          "A new checkpointer instance starts with completely empty tables.")


run_scenario()
