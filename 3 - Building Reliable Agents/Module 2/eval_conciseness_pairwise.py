"""
Pairwise conciseness evaluator for comparing two experiments.
"""
import os
import re
import sys
from dotenv import load_dotenv
from openai import OpenAI
from langsmith import evaluate

load_dotenv()

# Drop-in Groq client for the pairwise judge
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY"),
)
JUDGE_MODEL = "qwen/qwen3.8-27b"

CONCISENESS_PROMPT = """You are evaluating two responses to the same customer question.
Determine which response is MORE CONCISE while still providing all crucial information.

**Conciseness** means getting straight to the point, avoiding filler, and not repeating information.
**Crucial information** includes direct answers, necessary context, and required next steps.

A shorter response is NOT automatically better if it omits crucial information.

**Question:** {question}

**Response A:**
{response_a}

**Response B:**
{response_b}

Output your verdict as a single number:
1 if Response A is more concise while preserving crucial information
2 if Response B is more concise while preserving crucial information
0 if they are roughly equal"""


def conciseness_evaluator(inputs: dict, outputs: list[dict]) -> list[int]:
    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a conciseness evaluator. Respond with only a single number: 0, 1, or 2.",
            },
            {
                "role": "user",
                "content": CONCISENESS_PROMPT.format(
                    question=inputs["question"],
                    response_a=outputs[0].get("answer", "N/A"),
                    response_b=outputs[1].get("answer", "N/A"),
                ),
            },
        ],
    )

    content = response.choices[0].message.content.strip()
    # Robust parsing: find 0, 1, or 2
    match = re.search(r"\b([012])\b", content)
    preference = int(match.group(1)) if match else 0

    if preference == 1:
        return [1, 0]  # A wins
    elif preference == 2:
        return [0, 1]  # B wins
    else:
        return [0, 0]  # Tie


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python eval_conciseness_pairwise.py <experiment-a> <experiment-b>")
        print("Example: python eval_conciseness_pairwise.py agent-v4-xxxx agent-v5-yyyy")
        sys.exit(1)

    # LangSmith fetches the two experiment results from the cloud and pairs them up!
    evaluate(
        (sys.argv[1], sys.argv[2]),
        evaluators=[conciseness_evaluator],
        randomize_order=True,  # Protects against position bias
    )