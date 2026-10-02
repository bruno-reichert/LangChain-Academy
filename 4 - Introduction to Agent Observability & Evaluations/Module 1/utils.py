import os
import json
import time
from pathlib import Path
import pandas as pd
from langchain_core.documents import Document
from langchain_community.retrievers import TFIDFRetriever

RAG_PROMPT = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the latest question in the conversation. 
If you don't know the answer, just say that you don't know. 
The pre-existing conversation may provide important context to the question.
Use three sentences maximum and keep the answer concise.

Conversation: {conversation}
Context: {context} 
Question: {question}
Answer:"""

def get_vector_db_retriever():
    t0 = time.time()
    print("[1/3] 🔍 Locating union.parquet...")

    candidates = [
        Path("resources/union.parquet"),
        Path("./union.parquet"),
        Path("../resources/union.parquet"),
        Path(__file__).resolve().parent / "resources" / "union.parquet",
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
            f"❌ Could not find 'union.parquet'! Current working directory: {Path.cwd()}"
        )

    print(f"[2/3] 📖 Found '{target_path}'. Reading parquet into DataFrame...")
    df = pd.read_parquet(target_path)
    print(f"      Loaded {len(df)} document chunks from parquet.")

    # Explicitly target the pluralized column names ('texts' and 'metadatas')
    text_col = "texts" if "texts" in df.columns else "text"
    meta_col = "metadatas" if "metadatas" in df.columns else "metadata"

    print(f"      Using text column: '{text_col}', metadata column: '{meta_col}'")

    # Fast column extraction (vectorized lists: 0.02s vs 60s iterrows)
    raw_texts = df[text_col].tolist()
    raw_metas = df[meta_col].tolist() if meta_col in df.columns else [{}] * len(raw_texts)

    docs = []
    for text, meta in zip(raw_texts, raw_metas):
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except Exception:
                meta = {}
        docs.append(
            Document(
                page_content=str(text),
                metadata=meta if isinstance(meta, dict) else {}
            )
        )

    print(f"[3/3] ⚡ Building local TF-IDF index across {len(docs)} documents...")
    retriever = TFIDFRetriever.from_documents(docs, k=4)
    print(f"✅ Retriever ready in {time.time() - t0:.2f} seconds! Zero API calls, zero web scraping.")
    return retriever