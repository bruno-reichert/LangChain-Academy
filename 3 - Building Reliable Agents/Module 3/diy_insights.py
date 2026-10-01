"""
DIY Insight Agent: Spite-Powered Trace Analyzer ($0.00)
Extracts customer intents, calculates category percentages, and surfaces trends.
"""
import os
import json
import re
from dotenv import load_dotenv
from langsmith import Client
from openai import OpenAI

load_dotenv()

# 1. Connect to LangSmith and Groq
ls_client = Client()
groq_client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY"),
)
MODEL_NAME = "qwen/qwen3.8-27b"
PROJECT_NAME = os.getenv("LANGSMITH_PROJECT", "lca-reliable-agents")


def main():
    print("=" * 60)
    print(f"🕵️‍♂️ DIY INSIGHT AGENT: Analyzing Traces in '{PROJECT_NAME}'")
    print("=" * 60)

    # 2. Fetch root runs from your project
    print("📥 Pulling production customer traces from LangSmith...")
    runs = list(ls_client.list_runs(
        project_name=PROJECT_NAME,
        is_root=True,
        limit=35,  # Analyze 35 customer conversations in one fast batch
    ))

    # 3. Extract customer inputs
    user_questions = []
    for r in runs:
        if not r.inputs:
            continue
        # Extract question string
        q = r.inputs.get("question") or r.inputs.get("input") or str(r.inputs)
        if isinstance(q, str) and len(q.strip()) > 5:
            user_questions.append(q.strip())

    if not user_questions:
        print("No user questions found in traces.")
        return

    print(f"Found {len(user_questions)} customer inquiries. Analyzing with Qwen 27B...\n")

    # 4. Formatted prompt for Qwen
    formatted_questions = "\n".join(f"- {q}" for q in user_questions)

    analysis_prompt = f"""You are an expert AI Product Manager analyzing customer support traces for OfficeFlow Supply Co.

Here is a sample of real customer questions from our production traces:
{formatted_questions}

Analyze these questions and categorize them into meaningful business buckets (e.g., Product Inventory/Availability, Returns & Refunds, Shipping & Delivery, Ordering & Payment, Company/Location Inquiries, Customer Escalations).

Return ONLY valid JSON matching this exact structure:
{{
  "categories": [
    {{"category": "Category Name", "count": 12, "percentage": "34%", "sample_query": "example..."}}
  ],
  "executive_insights": [
    "Key trend or friction point 1...",
    "Key trend or friction point 2...",
    "Recommendation for the support team..."
  ]
}}"""

    # 5. Run Qwen analysis in one fast call
    response = groq_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a customer experience analytics engine. Respond with raw JSON only."},                                                                                    
            {"role": "user", "content": analysis_prompt}
        ],
        temperature=0.1,
    )

    content = response.choices[0].message.content.strip()

    # Clean markdown formatting if present
    content = re.sub(r"^```json\s*", "", content)
    content = re.sub(r"\s*```$", "", content)

    try:
        insights_data = json.loads(content)
    except json.JSONDecodeError:
        print("Raw response from model:\n", content)
        return

    # 6. Render Executive Dashboard in Terminal
    print("📊 CONVERSATION CATEGORY BREAKDOWN:")
    print("-" * 60)
    print(f"{'Category':<32} | {'Count':<6} | {'Share'}")
    print("-" * 60)
    for cat in insights_data.get("categories", []):
        name = cat.get("category", "Other")
        count = cat.get("count", 0)
        pct = cat.get("percentage", "0%")
        print(f"{name:<32} | {str(count):<6} | {pct}")

    print("\n💡 TOP INSIGHTS & CUSTOMER TRENDS:")
    print("-" * 60)
    for i, insight in enumerate(insights_data.get("executive_insights", []), 1):
        print(f" {i}. {insight}")
    print("=" * 60)
    print("✅ Total Cost: $0.00 | Paywall Defeated! 🎉\n")


if __name__ == "__main__":
    main()