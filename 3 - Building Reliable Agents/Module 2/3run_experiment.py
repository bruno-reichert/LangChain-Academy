import asyncio
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir))

from langsmith import aevaluate, Client, uuid7
import agent_v5
from agent_v5 import chat, load_knowledge_base

dataset_name = "officeflow-dataset"
ls_client = Client()


async def chat_wrapper(inputs: dict) -> dict:
    """Wrapper with fresh thread_id per question to prevent the snowball effect!"""
    # 🚨 CRUCIAL FIX: Give each question its own fresh thread!
    agent_v5.thread_id = str(uuid7())

    question = inputs.get("question", "")
    result = await chat(question)
    await asyncio.sleep(2)
    return {"answer": result["output"], "messages": result["messages"]}


async def main():
    kb_path = str(current_dir / "knowledge_base")
    print(f"Loading knowledge base from {kb_path}...")
    await load_knowledge_base(kb_dir=kb_path)
    print()

    # Pull 3-example smoke test
    print("Fetching 3-example slice for fast evaluation...")
    sample_examples = list(ls_client.list_examples(dataset_name=dataset_name, limit=3))

    results = await aevaluate(
        chat_wrapper,
        data=sample_examples,
        max_concurrency=1,
        experiment_prefix="smoke-test-3",
    )
    print(f"\nEvaluation complete! Results: {results}")
    return results


if __name__ == "__main__":
    asyncio.run(main())