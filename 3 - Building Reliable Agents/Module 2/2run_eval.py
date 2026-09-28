"""
Run the schema-before-query evaluator against the officeflow-dataset using native async evaluation.
"""
import asyncio
import sys
from pathlib import Path
from dotenv import load_dotenv

# Use aevaluate for async agents!
from langsmith import aevaluate, uuid7

load_dotenv()

# Anchor directly to current folder
current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir))

import agent_v5
from agent_v5 import chat, load_knowledge_base
from eval_schema_check import schema_before_query


async def setup():
    """Load knowledge base before running evals."""
    kb_dir = str(current_dir / "knowledge_base")
    await load_knowledge_base(kb_dir)


# Natively async - no asyncio.run() inside the loop!
async def run_agent(inputs: dict) -> dict:
    """Invoke the agent with a fresh thread_id each time."""
    agent_v5.thread_id = str(uuid7())
    return await chat(inputs["question"])


async def main():
    await setup()

    # aevaluate keeps a single event loop alive and controls concurrency
    results = await aevaluate(
        run_agent,
        data="officeflow-dataset",
        evaluators=[schema_before_query],
        experiment_prefix="schema-check-v5",
        max_concurrency=1,  # <--- Steady, rate-limit friendly queue
    )


if __name__ == "__main__":
    asyncio.run(main())