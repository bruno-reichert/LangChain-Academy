import asyncio
import operator
from typing import Annotated, Any, List
from typing_extensions import TypedDict

from dotenv import find_dotenv, load_dotenv
from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import WikipediaLoader
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_tavily import TavilySearch  # or TavilySearchResults / TavilyClient
from langgraph.errors import InvalidUpdateError
from langgraph.graph import END, START, StateGraph

# 1. Load keys
load_dotenv(find_dotenv())

# 2. LLM Model (Groq)
llm = init_chat_model(
    model="openai/gpt-oss-120b",
    model_provider="groq",
)


# --- SECTION 1: FAN-OUT / FAN-IN REDUCER CONCEPTS ---
class ReturnNodeValue:

    def __init__(self, node_secret: str):
        self._value = node_secret

    def __call__(self, state: dict) -> Any:
        print(f"Adding {self._value} to {state['state']}")
        return {"state": [self._value]}


def sorting_reducer(left, right):
    """Combines and sorts the values in a list"""
    if not isinstance(left, list):
        left = [left]
    if not isinstance(right, list):
        right = [right]
    return sorted(left + right, reverse=False)


# --- SECTION 2: REAL PARALLEL RAG GRAPH (Wikipedia + Web Search) ---
class RAGState(TypedDict):
    question: str
    answer: str
    context: Annotated[list, operator.add]  # Appends parallel outputs!


def search_web(state: RAGState):
    """Retrieve docs from web search in parallel"""
    try:
        tavily_search = TavilySearch(max_results=3)
        data = tavily_search.invoke({"query": state["question"]})
        search_docs = data.get("results", data)
        formatted_search_docs = "\n\n---\n\n".join(
            [
                f'<Document href="{doc.get("url", "")}">\n{doc.get("content", "")}\n</Document>'
                for doc in search_docs
            ]
        )
        return {"context": [formatted_search_docs]}
    except Exception as e:
        return {"context": [f"Web search error: {e}"]}


def search_wikipedia(state: RAGState):
    """Retrieve docs from Wikipedia in parallel"""
    try:
        search_docs = WikipediaLoader(
            query=state["question"], load_max_docs=2
        ).load()
        formatted_search_docs = "\n\n---\n\n".join(
            [
                f'<Document source="{doc.metadata.get("source", "")}">\n{doc.page_content}\n</Document>'
                for doc in search_docs
            ]
        )
        return {"context": [formatted_search_docs]}
    except Exception as e:
        return {"context": [f"Wikipedia error: {e}"]}


def generate_answer(state: RAGState):
    """Node to synthesize parallel context and answer question"""
    context = state["context"]
    question = state["question"]

    answer_template = (
        """Answer the question {question} using this context:\n\n{context}"""
    )
    answer_instructions = answer_template.format(
        question=question, context="\n\n".join(context)
    )

    answer = llm.invoke(
        [
            SystemMessage(content=answer_instructions),
            HumanMessage(content="Answer the question accurately and concisely."),
        ]
    )
    return {"answer": answer.content}


# Build Parallel Graph
builder = StateGraph(RAGState)
builder.add_node("search_web", search_web)
builder.add_node("search_wikipedia", search_wikipedia)
builder.add_node("generate_answer", generate_answer)

# Parallel Fan-Out from START to both search nodes
builder.add_edge(START, "search_wikipedia")
builder.add_edge(START, "search_web")

# Fan-In from both search nodes to generate_answer
builder.add_edge("search_wikipedia", "generate_answer")
builder.add_edge("search_web", "generate_answer")
builder.add_edge("generate_answer", END)

# Compiled graph for server / local use
graph = builder.compile()


# --- SECTION 3: ASYNC & SDK TEST RUNNER ---
async def main():
    print("=" * 60)
    print("PART 1: Local Parallel Execution (Wikipedia + Web Search)")
    print("=" * 60)

    question = "How were Nvidia's Q2 2025 earnings?"
    print(f"❓ Question: {question}")
    print("⚡ Fetching Wikipedia and Tavily in PARALLEL...")

    result = graph.invoke({"question": question})

    print("\n=== SYNTHESIZED ANSWER ===")
    print(result["answer"])

    # --- PART 2: LangGraph Server SDK Streaming ---
    print("\n" + "=" * 60)
    print("PART 2: LangGraph Server SDK Streaming")
    print("=" * 60)
    try:
        from langgraph_sdk import get_client

        client = get_client(url="http://127.0.0.1:2024")

        thread = await client.threads.create()
        print(f"Created remote thread: {thread['thread_id']}")

        async for event in client.runs.stream(
            thread["thread_id"],
            assistant_id="parallelization",  # Ensure this matches langgraph.json
            input={"question": "How were Nvidia Q2 2025 earnings?"},
            stream_mode="values",
        ):
            if event.data and "answer" in event.data and event.data["answer"]:
                print(f"\nStreamed Answer:\n{event.data['answer']}")

    except Exception as e:
        print(f"\n⚠️ SDK Note: Start 'uv run langgraph dev' if testing Part 2! (Error: {e})")


if __name__ == "__main__":
    asyncio.run(main())