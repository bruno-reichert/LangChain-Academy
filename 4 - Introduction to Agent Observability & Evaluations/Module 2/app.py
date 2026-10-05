import os
import json
import time
from pathlib import Path
from typing import List
import nest_asyncio
import pandas as pd
from openai import OpenAI
from langsmith import traceable
from langchain_core.documents import Document
from langchain_community.retrievers import TFIDFRetriever
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Stack Configuration (Crusty Nails Zero-Dollar Setup) ---
MODEL_NAME = "openai/gpt-oss-120b"  # Quota fallback: "qwen/qwen3.8-27b"
MODEL_PROVIDER = "groq"
APP_VERSION = 1.0

RAG_SYSTEM_PROMPT = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the latest question in the conversation. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.
"""

# Route standard OpenAI client to Groq
openai_client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("GROQ_API_KEY")
)

nest_asyncio.apply()


def get_vector_db_retriever():
    """Builds a zero-cost local TF-IDF retriever directly from union.parquet in milliseconds."""
    t0 = time.time()

    candidates = [
        Path("resources/union.parquet"),
        Path("./union.parquet"),
        Path("../resources/union.parquet"),
        Path(__file__).resolve().parent / "resources" / "union.parquet",
        Path(__file__).resolve().parent.parent / "resources" / "union.parquet",
    ]

    target_path = None
    for c in candidates:
        if c.exists():
            target_path = c
            break

    if not target_path:
        found = list(Path.cwd().rglob("union.parquet"))
        if found:
            target_path = found[0]

    if not target_path:
        raise FileNotFoundError(
            f"❌ Could not find 'union.parquet'! CWD: {Path.cwd()}"
        )

    df = pd.read_parquet(target_path)

    # Plural column extraction from parquet
    text_col = "texts" if "texts" in df.columns else "text"
    meta_col = "metadatas" if "metadatas" in df.columns else "metadata"

    raw_texts = df[text_col].tolist()
    raw_metas = df[meta_col].tolist() if meta_col in df.columns else [{}] * len(raw_texts)

    docs = []
    for text, meta in zip(raw_texts, raw_metas):
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except Exception:
                meta = {}
        docs.append(Document(page_content=str(text), metadata=meta if isinstance(meta, dict) else {}))

    retriever = TFIDFRetriever.from_documents(docs, k=4)
    return retriever


# Instantiate retriever once on import
retriever = get_vector_db_retriever()


@traceable(run_type="chain")
def retrieve_documents(question: str):
    return retriever.invoke(question)


@traceable(run_type="chain")
def generate_response(question: str, documents):
    # Rule #5: Token Diet [:600]
    formatted_docs = "\n\n".join(doc.page_content[:600] for doc in documents)
    messages = [
        {
            "role": "system",
            "content": RAG_SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": f"Context: {formatted_docs} \n\n Question: {question}"
        }
    ]
    return call_openai(messages)


@traceable(
    run_type="llm",
    metadata={
        "ls_provider": MODEL_PROVIDER,
        "ls_model_name": MODEL_NAME
    }
)
def call_openai(messages: List[dict]) -> str:
    return openai_client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
    )


@traceable(run_type="chain")
def langsmith_rag(question: str):
    documents = retrieve_documents(question)
    response = generate_response(question, documents)
    return response.choices[0].message.content


# Quick test when executed directly
if __name__ == "__main__":
    print("Testing app.py standalone...")
    test_q = "What is LangSmith?"
    ans = langsmith_rag(test_q)
    print(f"\nQuestion: {test_q}\nAnswer:\n{ans}")