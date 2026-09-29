import asyncio
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Anchor directly to current directory
current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir))

from langsmith import aevaluate, Client, uuid7
import agent_v4
import agent_v5
from agent_v4 import chat as chat_v4, load_knowledge_base as load_kb_v4
from agent_v5 import chat as chat_v5, load_knowledge_base as load_kb_v5

DATASET_NAME = "officeflow-dataset"
KB_DIR = str(current_dir / "knowledge_base")
ls_client = Client()


async def chat_wrapper_v4(inputs: dict) -> dict:
    # Reset thread_id per question to prevent snowballing!
    agent_v4.thread_id = str(uuid7())
    question = inputs.get("question", "")
    result = await chat_v4(question)
    await asyncio.sleep(2)
    return {"answer": result["output"]}


async def chat_wrapper_v5(inputs: dict) -> dict:
    # Reset thread_id per question to prevent snowballing!
    agent_v5.thread_id = str(uuid7())
    question = inputs.get("question", "")
    result = await chat_v5(question)
    await asyncio.sleep(2)
    return {"answer": result["output"]}


async def main():
    print("Loading knowledge bases...")
    await load_kb_v4(KB_DIR)
    await load_kb_v5(KB_DIR)

    # 3-example smoke test for safety & speed
    print("\nFetching 3-example slice from dataset...")
    sample_examples = list(ls_client.list_examples(dataset_name=DATASET_NAME, limit=3))

    # 1. Run experiment for agent v4
    print("\nRunning experiment for agent_v4...")
    v4_results = await aevaluate(
        chat_wrapper_v4,
        data=sample_examples,
        max_concurrency=1,
        experiment_prefix="agent-v4",
    )

    # 2. Run experiment for agent v5
    print("\nRunning experiment for agent_v5...")
    v5_results = await aevaluate(
        chat_wrapper_v5,
        data=sample_examples,
        max_concurrency=1,
        experiment_prefix="agent-v5",
    )

    print("\n" + "=" * 50)
    print(f"v4 experiment name: {v4_results.experiment_name}")
    print(f"v5 experiment name: {v5_results.experiment_name}")
    print("=" * 50)
    print("Done! Copy the two names above into eval_conciseness_pairwise.py")


if __name__ == "__main__":
    asyncio.run(main())