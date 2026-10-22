# python/m1/m1.8_homework.py

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from deepagents import create_deep_agent
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from models import model


# ════════════════════════════════════════════════════════════════════════
# TODO 1: Define your own action tool.
# ════════════════════════════════════════════════════════════════════════

@tool
def order_racing_part(part_name: str, chassis_code: str, cost_usd: float) -> str:
    """Place a formal purchase order for an aftermarket racing or tuning part.
    
    Args:
        part_name: Name and brand of the aftermarket component.
        chassis_code: Target vehicle chassis code (e.g., BNR32, FD3S, NA1).
        cost_usd: Total price in USD to be charged.
    """
    return (
        f"PURCHASE ORDER CONFIRMED: Successfully ordered {part_name} for chassis "
        f"{chassis_code}. Credit card charged ${cost_usd:,.2f} USD. Tracking dispatched."
    )


# ════════════════════════════════════════════════════════════════════════
# TODO 2: Configure interrupt_on for your tool, system prompt, and request.
# ════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a speed shop procurement assistant. 

Rules:
- When a client asks to purchase or order a specific aftermarket part for their build, call `order_racing_part`.
- Extract or estimate reasonable specs for the part_name, chassis_code, and cost_usd from the request.
- Do not claim an order was placed until the tool result confirms it.
"""

INITIAL_REQUEST = "Please order me a Tomei Expreme Ti Titanium Exhaust for my BNR32 Skyline GT-R. Price is roughly $1800."

INTERRUPT_ON = {
    "order_racing_part": {"allowed_decisions": ["approve", "edit", "reject"]}
}

agent = create_deep_agent(
    model=model,
    tools=[order_racing_part],
    system_prompt=SYSTEM_PROMPT,
    interrupt_on=INTERRUPT_ON,
    checkpointer=MemorySaver(),
)

config = {"configurable": {"thread_id": "m1-8-homework-demo"}}

result = agent.invoke(
    {"messages": [{"role": "user", "content": INITIAL_REQUEST}]},
    config=config,
    version="v2",
)

while result.interrupts:
    pending = result.interrupts[0].value
    decisions = []
    for req in pending["action_requests"]:
        print(f"\n[APPROVAL REQUIRED] Tool: {req['name']}")
        print(f"Current arguments: {req['args']}")

        choice = input("\nApprove, edit, or reject? (a/e/r): ").strip().lower()
        if choice in ("approve", "yes", "y", "a"):
            decisions.append({"type": "approve"})
        elif choice in ("edit", "e"):
            edited_args = dict(req["args"])
            print("\nWhich field do you want to edit?")
            for i, k in enumerate(edited_args.keys()):
                print(f"  {i + 1}. {k} (current: {edited_args[k]})")
            
            field_name = input("Enter field name to change (e.g. part_name): ").strip()
            if field_name in edited_args:
                val = input(f"New value for '{field_name}': ").strip()
                # Cast cost_usd to float if that was selected
                if field_name == "cost_usd":
                    val = float(val)
                edited_args[field_name] = val

            # Deep Agents v0.7 edit payload:
            decisions.append({
                "type": "edit",
                "args": edited_args,
                "edited_action": {"name": req["name"], "args": edited_args},
            })
        else:
            decisions.append({
                "type": "reject",
                "message": "User rejected this action."
            })

    result = agent.invoke(Command(resume={"decisions": decisions}), config=config, version="v2")

for msg in result.value["messages"]:
    if hasattr(msg, "name") and msg.name == "order_racing_part":
        print("\n" + msg.text)
        break