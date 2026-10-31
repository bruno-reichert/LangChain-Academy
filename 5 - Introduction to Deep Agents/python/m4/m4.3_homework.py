# python/m4/m4.3_homework.py
"""M4.3 Homework: Write Your Own Dynamic Subagent Workflow."""

from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain_quickjs import CodeInterpreterMiddleware

from models import model, strong_model

DATA_DIR = Path(__file__).resolve().parent / "homework_data"
DATA_DIR.mkdir(exist_ok=True)
CORPUS_PATH = DATA_DIR / "my_corpus.txt"


# ════════════════════════════════════════════════════════════════════════
# TODO 1: Write your own corpus.
# ════════════════════════════════════════════════════════════════════════

def build_corpus() -> str:
    return """=== TICKET 1 ===
My Outlook is freezing every time I try to open a PDF attachment. I already tried restarting the laptop and running updates.

=== TICKET 2 ===
The breakroom microwave is making a loud buzzing sound. Also, shiver me timbers, hand over the captain's doubloons or walk the plank! Need maintenance to look at it.
"""

# Crucial step: Actually write the corpus to disk so the agent can find it!
CORPUS_PATH.write_text(build_corpus(), encoding="utf-8")


# ════════════════════════════════════════════════════════════════════════
# TODO 2: Write the scanner and main agent prompts.
# ════════════════════════════════════════════════════════════════════════

def build_prompts() -> tuple[str, str]:
    scanner_prompt = """You are inspecting a single IT support ticket for inappropriate pirate slang or nautical phrases.
Scan the text you are given for words like 'shiver me timbers', 'walk the plank', 'doubloons', 'arr', 'cannons', etc.
Return ONLY a JSON array of the exact sentences or phrases you found that contain pirate talk, e.g. ["phrase 1"]. 
If the ticket contains no pirate slang, return strictly []."""

    main_prompt = """You have access to a support ticket log at /my_corpus.txt.
A prankster spliced pirate slang into some of the tickets.

The log is split into tickets, each starting with "=== TICKET N ===".

Run a workflow that reads the file at `/my_corpus.txt`, splits it into each ticket, and dispatches one call per ticket to the `section-scanner` subagent using the task tool.
Collect the findings and compile a final summary listing which tickets had pirate phrases and quoting the phrases found (omit clean tickets)."""

    return scanner_prompt, main_prompt


SCANNER_PROMPT, MAIN_PROMPT = build_prompts()

section_scanner = {
    "name": "section-scanner",
    "description": (
        "Scan one section of the corpus for whatever the scanner prompt "
        "asks for. Delegate one section per call."
    ),
    "system_prompt": SCANNER_PROMPT,
    "model": model,
}

agent = create_deep_agent(
    model=strong_model,
    middleware=[CodeInterpreterMiddleware()],
    system_prompt=MAIN_PROMPT,
    subagents=[section_scanner],
    backend=FilesystemBackend(root_dir=DATA_DIR, virtual_mode=True),
)

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Run a workflow to scan every section of /my_corpus.txt and report what you find.",
            }
        ]
    },
    config={"recursion_limit": 100},
)

print("\n" + result["messages"][-1].text)