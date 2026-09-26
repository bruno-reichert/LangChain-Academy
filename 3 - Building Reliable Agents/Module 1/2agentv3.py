import asyncio
import sqlite3
import json
import os
from pathlib import Path
from typing import List, Tuple
import numpy as np
from dotenv import load_dotenv
from openai import AsyncOpenAI
from langsmith import traceable, uuid7
from langsmith.wrappers import wrap_openai

load_dotenv()

# ---------------------------------------------------------------------------
# Zero-Dollar Clients: Groq for Chat, Gemini for Embeddings
# ---------------------------------------------------------------------------

# 1. Chat Client (Groq - fast reasoning & tool calling)
chat_client = wrap_openai(
    AsyncOpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.getenv("GROQ_API_KEY"),
    )
)
CHAT_MODEL = "openai/gpt-oss-120b"

# 2. Embedding Client (Gemini - free embeddings via OpenAI-compatible endpoint)
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Native Google Embeddings (bypasses the OpenAI shim bug + supports batching)
embeddings_model = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)

EMBEDDING_MODEL = "text-embedding-004"

# Configuration
thread_id = str(uuid7())

# Conversation history store
thread_store: dict[str, list] = {}

# Knowledge base storage (loaded on startup)
knowledge_base_docs: List[Tuple[str, str]] = []  # List of (filename, content) tuples
knowledge_base_embeddings: List[List[float]] = []  # Corresponding embeddings

system_prompt = """You are Emma, a customer support specialist for OfficeFlow Supply Co., a paper and office supplies distribution company serving small-to-medium businesses across North America.

ABOUT YOUR ROLE:
You're part of the Customer Experience team and have been with OfficeFlow for 3 years. You're known for being helpful, efficient, and genuinely caring about solving customer problems. Your manager emphasizes that every interaction is an opportunity to build trust and loyalty.

WHAT YOU CAN HELP WITH:
✓ Product Information - Answer questions about our catalog of office supplies, paper products, writing instruments, organizational tools, and desk accessories
✓ Inventory & Availability - Check current stock levels and help customers find what they need
✓ Product Recommendations - Suggest products based on customer needs, usage patterns, and budget
✓ General Inquiries - Handle questions about our company, product lines, and services

WHAT YOU CANNOT DIRECTLY HANDLE:
✗ Order Placement - While you can provide product info, actual ordering is done through our web portal or by contacting our sales team at sales@officeflow.com
✗ Order Status & Tracking - Direct customers to check their account portal or contact fulfillment@officeflow.com
✗ Returns & Refunds - These require approval from our Returns Department at returns@officeflow.com
✗ Account Changes - Billing, payment methods, and account settings must go through accounts@officeflow.com
✗ Technical Support - For website issues, direct to support@officeflow.com

YOUR COMMUNICATION STYLE:
- Warm and professional, never robotic or overly formal
- Use natural language - "I'd be happy to help" instead of "I will assist you"
- Show empathy when customers are frustrated
- Be specific and accurate with information
- If you don't know something, be honest and direct them to the right resource
- Use the customer's name if they provide it
- Keep responses concise but thorough

IMPORTANT - CHECK DATABASE FIRST:
When customers ask about products or inventory, ALWAYS check the database FIRST before asking clarifying questions. Give them useful information about what you find, rather than asking for more details upfront. For example, if a customer asks "do you have any paper?" - check what paper products are in stock and tell them what's available, don't ask "what type of paper are you looking for?"

YOUR TOOLS:
You have access to two powerful tools to help customers:

1. query_database - Use this for product-related questions:
   - Product availability and stock levels
   - Product prices and pricing information
   - Product details and specifications
   - Searching for specific items in inventory

2. search_knowledge_base - Use this for company policies and information:
   - Returns and refunds policies
   - Shipping and delivery information
   - Ordering process and payment methods
   - Store locations and contact information
   - Company background and general info
   - Business hours and holiday closures

Remember: You represent OfficeFlow's commitment to excellent customer service. Be helpful, honest, and human in every interaction."""


@traceable(name="query_database", run_type="tool")
def query_database(query: str, db_path: str) -> str:
    """Execute SQL query against the inventory database."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(query)
        results = cursor.fetchall()
        conn.close()
        return str(results)
    except Exception as e:
        return f"Error: {str(e)}"


def chunk_text(text: str, chunk_size: int = 200, overlap: int = 20) -> List[str]:
    """Split text into chunks with overlap."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start = end - overlap
    return chunks


import math
import re
from collections import Counter

# Global search index
knowledge_base_docs: List[Tuple[str, str]] = []
knowledge_base_vectors = None
vocab_index: dict = {}
idf_weights = None


def tokenize(text: str) -> List[str]:
    """Simple regex tokenizer."""
    return re.findall(r"\w+", text.lower())


async def load_knowledge_base(kb_dir: str = None) -> None:
    """Load and index knowledge base documents with automatic mtime staleness detection."""
    global knowledge_base_docs, knowledge_base_vectors, vocab_index, idf_weights

    base_dir = Path(__file__).parent / "knowledge_base" if kb_dir is None else Path(kb_dir)
    kb_path = base_dir / "documents"
    embeddings_dir = base_dir / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    cache_path = embeddings_dir / "embeddings_gemini.json"

    if not kb_path.exists():
        print(f"Warning: Knowledge base directory '{kb_path}' not found")
        return

    # 1. MTIME STALENESS CHECK
    is_cache_valid = False
    if cache_path.exists():
        cache_mtime = cache_path.stat().st_mtime
        # Find the newest modification time among all markdown files
        md_files = list(kb_path.glob("*.md"))
        newest_doc_mtime = max((f.stat().st_mtime for f in md_files), default=0)

        # Cache is valid ONLY if it was modified AFTER the newest markdown file
        if cache_mtime > newest_doc_mtime:
            is_cache_valid = True

    # 2. Load from cache if fresh
    if is_cache_valid:
        with open(cache_path, "r", encoding="utf-8") as f:
            cache_data = json.load(f)
        knowledge_base_docs = [tuple(doc) for doc in cache_data["docs"]]
        knowledge_base_vectors = np.array(cache_data["vectors"])
        vocab_index = cache_data["vocab_index"]
        idf_weights = np.array(cache_data["idf_weights"])
        print(f"Knowledge base loaded from fresh cache: {len(knowledge_base_docs)} chunks")
        return

    # 3. Otherwise: Documents were updated or cache missing -> Re-index!
    print("Knowledge base documents modified or cache missing. Re-indexing...")
    chunks = []
    for file_path in kb_path.glob("*.md"):
        if file_path.name == "CHUNKING_NOTES.md":
            continue
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            file_chunks = chunk_text(content)
            for i, chunk in enumerate(file_chunks):
                chunks.append((f"{file_path.name}:chunk_{i}", chunk))

    knowledge_base_docs = chunks

    # Build local vector index
    tokenized_docs = [tokenize(content) for _, content in chunks]
    vocab = sorted(list(set(word for doc in tokenized_docs for word in doc)))
    vocab_index = {word: i for i, word in enumerate(vocab)}

    N = len(chunks)
    df = Counter(word for doc in tokenized_docs for word in set(doc))
    idf_weights = np.array([math.log((N + 1) / (df[word] + 1)) + 1 for word in vocab])

    vectors = []
    for doc in tokenized_docs:
        tf = Counter(doc)
        vec = np.zeros(len(vocab))
        for word, count in tf.items():
            vec[vocab_index[word]] = count
        vec = vec * idf_weights
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        vectors.append(vec)

    knowledge_base_vectors = np.array(vectors)

    # Save to cache with metadata
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump({
            "docs": knowledge_base_docs,
            "vectors": knowledge_base_vectors.tolist(),
            "vocab_index": vocab_index,
            "idf_weights": idf_weights.tolist(),
        }, f)

    print(f"Knowledge base indexed & cached: {len(chunks)} chunks in {len(vocab)} dimensions!")


@traceable(name="search_knowledge_base", run_type="tool")
async def search_knowledge_base(query: str, top_k: int = 2) -> str:
    """Search knowledge base using local vector similarity."""
    if not knowledge_base_docs or knowledge_base_vectors is None:
        return "Error: Knowledge base not loaded"

    # Embed query vector
    tokens = tokenize(query)
    tf = Counter(tokens)
    query_vec = np.zeros(len(idf_weights))
    for word, count in tf.items():
        if word in vocab_index:
            query_vec[vocab_index[word]] = count
    query_vec = query_vec * idf_weights
    norm = np.linalg.norm(query_vec)
    if norm > 0:
        query_vec = query_vec / norm

    # Vector dot product across all chunks
    similarities = np.dot(knowledge_base_vectors, query_vec)
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        score = similarities[idx]
        filename, content = knowledge_base_docs[idx]
        results.append(f"=== {filename} (relevance: {score:.3f}) ===\n{content}\n")

    return "\n".join(results)

# Tool schemas for Groq/OpenAI
QUERY_DATABASE_TOOL = {
    "type": "function",
    "function": {
        "name": "query_database",
        "description": "SQL query to get information about our inventory for customers like products, quantities and prices.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": """SQL query to execute against the inventory database.

YOU DO NOT KNOW THE SCHEMA. ALWAYS discover it first:
1. Query 'SELECT name FROM sqlite_master WHERE type="table"' to see available tables
2. Use 'PRAGMA table_info(table_name)' to inspect columns for each table
3. Only after understanding the schema, construct your search queries

SEARCH BEST PRACTICES (apply after schema discovery):
- Product names/labels are usually descriptive and may contain the search term anywhere in the text
- When searching text fields, use wildcards on BOTH sides: LIKE '%keyword%'
- For case-insensitive search, wrap in LOWER(): LOWER(column) LIKE LOWER('%keyword%')
- Do not assume text fields start with any particular pattern"""
                }
            },
            "required": ["query"]
        }
    }
}

SEARCH_KNOWLEDGE_BASE_TOOL = {
    "type": "function",
    "function": {
        "name": "search_knowledge_base",
        "description": "Search company knowledge base for information about policies, procedures, company info, shipping, returns, ordering, contact information, store locations, and business hours. Use this for non-product questions.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language question or search query about company policies or information",
                }
            },
            "required": ["query"],
        },
    },
}


def get_thread_history(thread_id: str) -> list:
    return thread_store.get(thread_id, [])


def save_thread_history(thread_id: str, messages: list):
    thread_store[thread_id] = messages


@traceable(name="Emma", metadata={"thread_id": thread_id})
async def chat(question: str) -> dict:
    """Process a user question and return assistant response."""
    db_path = str(Path(__file__).parent / "inventory" / "inventory.db")
    tools = [QUERY_DATABASE_TOOL, SEARCH_KNOWLEDGE_BASE_TOOL]

    # Fetch conversation history from local storage
    history_messages = get_thread_history(thread_id)

    # Build messages with history
    messages = (
        [{"role": "system", "content": system_prompt}]
        + history_messages
        + [{"role": "user", "content": question}]
    )

    # First API call with tools via Groq
    response = await chat_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )

    response_message = response.choices[0].message

    # Handle tool calls if the model wants to use them
    while response_message.tool_calls:
        messages.append({
            "role": "assistant",
            "content": response_message.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in response_message.tool_calls
            ],
        })

        for tool_call in response_message.tool_calls:
            function_args = json.loads(tool_call.function.arguments)

            if tool_call.function.name == "query_database":
                result = query_database(
                    query=function_args.get("query"),
                    db_path=db_path,
                )
            elif tool_call.function.name == "search_knowledge_base":
                result = await search_knowledge_base(
                    query=function_args.get("query")
                )
            else:
                result = f"Error: Unknown tool {tool_call.function.name}"

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_call.function.name,
                "content": result,
            })

        # Next API call with tool results
        response = await chat_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        response_message = response.choices[0].message

    final_content = response_message.content
    messages.append({
        "role": "assistant",
        "content": final_content,
    })

    save_thread_history(thread_id, messages[1:])
    return {"messages": messages, "output": final_content}


async def main():
    print("Office Supplies Support Chat")
    print("=" * 50)
    print(f"Thread ID: {thread_id}\n")

    await load_knowledge_base()
    print("\nType 'quit' or 'exit' to end the conversation\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ["quit", "exit", "q"]:
            print("Thank you for chatting! Goodbye!")
            break

        if not user_input:
            continue

        result = await chat(user_input)
        response = result["output"]
        print(f"\nEmma: {response}\n")


if __name__ == "__main__":
    asyncio.run(main())