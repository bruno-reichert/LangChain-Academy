from pathlib import Path
from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend  # <-- Free local shell!

from models import model

# Create a safe, isolated local sandbox directory inside m2/
sandbox_dir = Path(__file__).parent / "local_sandbox"
sandbox_dir.mkdir(exist_ok=True)

# Mount the local shell backend to that directory
backend = LocalShellBackend(root_dir=str(sandbox_dir), virtual_mode=True)

agent = create_deep_agent(
    model=model,
    backend=backend,
    system_prompt=(
        "You are a coding assistant. When asked to run code, write the script "
        "to a file first, then execute it. Show the output in your final answer."
    ),
)

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": (
                    "Write a Python script that prints the first 15 Fibonacci numbers, "
                    "save it to fib.py, and run it."
                ),
            }
        ]
    }
)
print(result["messages"][-1].text)