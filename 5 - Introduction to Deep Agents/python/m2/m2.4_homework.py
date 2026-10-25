# python/m2/m2.4_homework.py
"""M2.4 Homework: Ask Your Own Question, Then Grade It.

THE IDEA
Lab 2 asked one fixed, dependent-query question about the Chinook
database and let the interpreter chain four SQL queries together inside
a single eval() call. This homework asks you to pose your OWN question
about the database (anything query_chinook and a little JavaScript can
answer, simple or dependent, your call), and then, since this lesson is
also about evaluating what an agent's code produces, write your own quick
eval check that judges whether the agent's answer looks right. There's no
single right question or eval method here, that's the point.

WHAT YOU FILL IN
  TODO 1: write your own natural-language question about the Chinook
    database (see the schema hints in SYSTEM below) for the interpreter
    agent to answer using eval() and query_chinook.
  TODO 2: write a small eval_answer(...) function that independently
    checks whether the agent's final answer looks correct. However you
    want to do this is fine: run your own SQL query and compare, check
    for an expected keyword or number, or just print both side by side
    for your own judgment call. Pick whatever level of rigor makes sense
    for your question.

RUN
  cd python
  uv run ./m2/m2.4_homework.py
"""

import json
import sqlite3
import uuid
from pathlib import Path

from deepagents import create_deep_agent
from langchain.tools import tool
from langchain_quickjs import CodeInterpreterMiddleware

from models import model

DB_PATH = Path(__file__).resolve().parent / "chinook.db"

SYSTEM = (
    "You are a sales analyst for Chinook Digital Music Store. "
    "Use the query_chinook tool to query the database. "
    "Key tables: Artist(ArtistId, Name), Album(AlbumId, Title, ArtistId), "
    "Track(TrackId, Name, AlbumId, GenreId), Genre(GenreId, Name), "
    "Customer(CustomerId, FirstName, LastName, Country), "
    "Invoice(InvoiceId, CustomerId), "
    "InvoiceLine(InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity). "
    "Revenue is InvoiceLine.UnitPrice * InvoiceLine.Quantity. "
    "The eval tool supports Programmatic Tool Calling (PTC): JavaScript "
    "running inside eval() can call query_chinook via tools.queryChinook()."
)


@tool
def query_chinook(sql: str) -> str:
    """Execute a read-only SQL query against the Chinook database. Returns a JSON-encoded string."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.execute(sql)
        rows = [dict(row) for row in cursor.fetchall()]
        return json.dumps(rows)
    finally:
        conn.close()


# ════════════════════════════════════════════════════════════════════════
# TODO 1: Write your own question about the Chinook database.
# ════════════════════════════════════════════════════════════════════════

TASK = (
    "Which country generated the highest total sales revenue, "
    "and who is the top-spending customer from that specific country? "
    "State the country, the customer's full name, and how much they spent."
)


# ════════════════════════════════════════════════════════════════════════
# TODO 2: Write a simple eval check for the agent's answer.
# ════════════════════════════════════════════════════════════════════════

def eval_answer(answer_text: str) -> None:
    """Independently calculates ground truth via sqlite3 and validates the answer."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        # Ground truth step 1: Top country by revenue
        cursor.execute("""
            SELECT c.Country, SUM(il.UnitPrice * il.Quantity) as total_rev
            FROM Customer c
            JOIN Invoice i ON c.CustomerId = i.CustomerId
            JOIN InvoiceLine il ON i.InvoiceId = il.InvoiceId
            GROUP BY c.Country
            ORDER BY total_rev DESC
            LIMIT 1;
        """)
        top_country, country_rev = cursor.fetchone()

        # Ground truth step 2: Top customer in that country
        cursor.execute("""
            SELECT c.FirstName || ' ' || c.LastName as full_name, 
                   SUM(il.UnitPrice * il.Quantity) as customer_spend
            FROM Customer c
            JOIN Invoice i ON c.CustomerId = i.CustomerId
            JOIN InvoiceLine il ON i.InvoiceId = il.InvoiceId
            WHERE c.Country = ?
            GROUP BY c.CustomerId
            ORDER BY customer_spend DESC
            LIMIT 1;
        """, (top_country,))
        top_customer, customer_spend = cursor.fetchone()

    finally:
        conn.close()

    print("\n" + "=" * 50)
    print("INDEPENDENT EVALUATION RESULTS:")
    print("=" * 50)
    print(f"Ground Truth Country:  {top_country} (${country_rev:.2f})")
    print(f"Ground Truth Customer: {top_customer} (${customer_spend:.2f})")
    print("-" * 50)

    # Verification assertions
    country_match = top_country.lower() in answer_text.lower()
    customer_match = (
        top_customer.lower() in answer_text.lower() or 
        all(part.lower() in answer_text.lower() for part in top_customer.split())
    )

    if country_match and customer_match:
        print(" Verdict: [PASS] - The agent identified both the top country and top customer accurately!")
    else:
        print("❌ Verdict: [FAIL] - Agent output missed the expected ground truth.")
    print("=" * 50)

agent = create_deep_agent(
    model=model,
    tools=[query_chinook],
    middleware=[CodeInterpreterMiddleware(ptc=["query_chinook"])],
    system_prompt=SYSTEM,
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": TASK}]},
    config={"configurable": {"thread_id": str(uuid.uuid4())}},
)

answer = result["messages"][-1].text
print(answer)
eval_answer(answer)
