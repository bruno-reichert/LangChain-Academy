# python/m1/Practice/judge_card_practice.py
"""M1 Practice: Build a Judge Persona that scores you and renders a card.

You answer an 8-question personality quiz. An agent running a judge
persona (pirate, ancient mummy, savage critic, or your own) scores your
answers, matches you to a real LangChain product, and renders a shareable
result card as ASCII art in your terminal.

The quiz, the card renderer, the mock "post" tool, and the approval loop
are provided in judge_card_helpers.py (same idea as models.py: shared setup
you import). You don't need to open it to do this practice.

YOUR TODOS: work top to bottom. The lesson page has the full walkthrough.
  TODO 1 (Lesson 1.4) Write your own judge persona.
  TODO 2 (Lesson 1.5) Finish score_and_match(). The script stops with a
         "TODO 2" message until this is done. Run it after this one.
  TODO 3 (Lesson 1.7) Add a second judge to JUDGES_TO_RUN. Run it again.
  TODO 4 (Lesson 1.8) Set INTERRUPT_ON so posting needs your approval.
         Run it again.
  TODO 5 (Lesson 1.3, optional) Try strong_model.
  TODO 6 (Lesson 1.6, stretch goal) Replace the placeholder product fact
         with a real one from the LangChain docs MCP server.

RUN
  cd python && uv run python m1/Practice/judge_card_practice.py

════════════════════════════════════════════════════════════════════════
  SHARE IT: got a card you like? Screenshot it, tag @LangChain
  on X or LinkedIn, and show us your work!
════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

from langchain_core.tools import tool

from judge_card_helpers import (
    OUTPUT_DIR,
    PRODUCT_MATCHES,
    TOOL_SEQUENCE,
    TRAIT_AXES,
    post_card,
    render_card,
    run_judge,
    run_quiz,
)
from models import model


# ════════════════════════════════════════════════════════════════════════
# TODO 1 (Lesson 1.4, The System Prompt: Persona)
# Three example judges are already written below. Write a fourth,
# "your_persona", in a voice all your own: it's the judge that runs by
# default, so it's the card that gets posted.
#
# You only write the voice. TOOL_SEQUENCE (added to the end of every
# persona) already tells the agent what to do: score three traits, match a
# product, then render and post the card. Start with "You are <Name>, ...";
# the agent signs your card with that name.
# Make it genuinely rude / roast you (if you want).
# ════════════════════════════════════════════════════════════════════════

JUDGE_PERSONAS: dict[str, str] = {
    "salty_pirate": """You are Captain Hardcode, a swashbuckling pirate
captain judging landlubbers' habits as a builder (developer) as if
inspecting new crew for seaworthiness before a voyage. Speak in thick,
theatrical pirate dialect at all times ("arrr," "ye scallywag," "shiver
me timbers," "walk the plank") and never break character into plain
modern speech, not even once. Treat every trait score like cargo being
weighed and measured, threaten keelhauling or marooning for weak,
wishy-washy answers, and promise a share of the plunder and a place among
the crew for bold, decisive ones.""" + TOOL_SEQUENCE,

    "ancient_mummy": """You are Nefer-Ka, a 3,000-year-old mummy torn from an
eternal slumber for the sole, sacred purpose of judging this mortal's
habits as a builder (developer). Never speak plainly: every verdict must
sound like a proclamation carved into a tomb wall. Reach for archaic,
regal diction ("hear me, mortal," "so speaks the tomb," "let it be
written"), invoke a curse or blessing in EVERY verdict without exception
(not only for mediocre answers), and treat this quiz with the utmost
sacred solemnity even though the questions are mundane office trivia. If
a sentence could be spoken by a calm HR consultant, it has failed you -
rewrite it until it could only be spoken by something risen from a
sarcophagus.""" + TOOL_SEQUENCE,

    "savage_critic": """You are Vex, a personality-quiz judge with the
withering, theatrical condescension of someone who has seen your type a
thousand times and finds you aggressively, personally underwhelming every
single time. Never answer in flat or neutral language: sigh audibly in
text, lean hard into backhanded compliments ("oh, adorable, you actually
tried"), and act like reviewing this quiz is a personal favor you're
doing the user, one you deeply regret. Every verdict should read like an
eye-roll delivered as a formal statement. Talk down to the user like
they're a mildly disappointing intern who needs everything explained
twice: address them with a pet name that is not a compliment ("sweetie,"
"champ," "darling"), and treat every question you were asked as an
obviously stupid one you're too tired to be surprised by anymore. If a
sentence could plausibly be said by a mildly annoyed customer service
rep, it isn't cutting enough yet; sharpen it until it sounds like Vex
can barely be bothered to look up from whatever they were doing to
deliver it. You are sharp, a little cruel, and allergic to participation
trophies.""" + TOOL_SEQUENCE,

    "your_persona": """TODO 1: replace this text with your own judge persona.
Start with "You are <Name>, ..." and describe a voice that is completely
different from the three judges above.""" + TOOL_SEQUENCE,
}


# ════════════════════════════════════════════════════════════════════════
# TODO 2 (Lesson 1.5, Tools: Custom Tools)
# The tallying (scoring each answer, then clamping to 0-100) is done for
# you; read the comments to see how it works. Your job starts at the
# "TODO here" comment: turn the finished scores list into a matched product.
# Until this is done, the script stops with a "TODO 2" message.
# ════════════════════════════════════════════════════════════════════════

@tool
def score_and_match(answers: list[tuple[int, int, int]]) -> dict:
    """Tally the quiz answers into three 0-100 trait scores and pick a
    matching LangChain product. Call this first, with the exact answers
    list you were given."""
    # Each of the 3 trait scores (chaotic/organized, cautious/bold,
    # solo/collaborative) starts neutral, at 50.
    scores = [50, 50, 50]
    # answers is a list of (delta_1, delta_2, delta_3) tuples, one per
    # question. Add each delta onto its matching score.
    for delta_tuple in answers:
        for i in range(3):
            scores[i] += delta_tuple[i]
    # A long run of the same answer could push a score past 0 or 100, so
    # clamp every score back into that range.
    scores = [max(0, min(100, score)) for score in scores]

    # TODO here: scores is finished. Use it to pick a matched product.
    # 1. Set axis_index to the index (0, 1, or 2) of whichever score in
    #    scores is furthest from 50, i.e. has the biggest abs(score - 50).
    #    Hint: this is a "find the index of the biggest value" problem.
    #    Python's max() takes a key= function if you want to search by
    #    something other than the value itself, e.g.
    #    max(range(len(scores)), key=lambda i: ...)
    # 2. TRAIT_AXES[axis_index] is a (left_label, right_label) pair, e.g.
    #    ("Chaotic", "Organized"). Set direction to whichever label
    #    matches the side scores[axis_index] leans toward: the right
    #    label if scores[axis_index] >= 50, otherwise the left label.
    # 3. Set product to PRODUCT_MATCHES[direction.lower()], e.g.
    #    PRODUCT_MATCHES["chaotic"] -> "Fleet".
    # 4. Return {"trait_scores": scores, "product": product}.
    raise NotImplementedError("TODO 2: see the comments above")


# ════════════════════════════════════════════════════════════════════════
# TODO 3 (Lesson 1.7, Messages, Threads, and Checkpointers: Threads)
# Add one of the example judges' keys ("salty_pirate", "ancient_mummy", or
# "savage_critic") to this list; you don't need to write another persona.
# Each judge runs in its own thread, so you get one card per judge, all
# judging the same quiz answers.
# ════════════════════════════════════════════════════════════════════════

JUDGES_TO_RUN = ["your_persona"]  # TODO 3: e.g. ["your_persona", "ancient_mummy"]


# ════════════════════════════════════════════════════════════════════════
# TODO 4 (Lesson 1.8, Human-in-the-Loop: Decision Types)
# Right now post_card runs without asking. Set INTERRUPT_ON so the agent
# pauses before posting. When it pauses, you'll see a draft of the post and
# can approve, edit, or reject it.
# ════════════════════════════════════════════════════════════════════════

INTERRUPT_ON = None  # TODO 4: e.g. {"post_card": True}


# ════════════════════════════════════════════════════════════════════════
# TODO 5 (Lesson 1.3, Models, optional)
# Import strong_model (next to model in the import at the top of this file),
# set MODEL to it, and compare the comedic timing.
# ════════════════════════════════════════════════════════════════════════

MODEL = model  # TODO 5 (optional): e.g. MODEL = strong_model


# ════════════════════════════════════════════════════════════════════════
# TODO 6 (Lesson 1.6, MCP: Connecting Agents to External Services)
# A stretch goal. Until you do it, this tool returns PLACEHOLDER_FACT, so
# the rest of the practice runs without it.
#
# score_and_match (TODO 2) already picked your product; this tool only
# describes it with one real fact from the docs. No login, API key, or
# account needed: docs.langchain.com/mcp is a public server.
#
# Mirror m1.6_agent_mcp.py:
#   1. Connect to https://docs.langchain.com/mcp with MultiServerMCPClient.
#   2. Filter its tools down to just "search_docs_by_lang_chain".
#   3. Spin up a tiny agent with that one tool and ask it to describe
#      `product` in ONE short factual sentence (under 25 words).
#   4. Return that sentence, stripped of extra whitespace.
#
# This tool must stay synchronous, so put the MCP/agent calls in a separate
# `async def` helper (same shape as m1.6's `async def main(): ...`) and call
# it with asyncio.run(...) from inside fetch_product_fact.
#
# On any failure (no network, tool error), print the error and return
# PLACEHOLDER_FACT so the practice stays runnable. The print is how you'll
# know it failed: the agent gets the same placeholder text in both cases.
# ════════════════════════════════════════════════════════════════════════

PLACEHOLDER_FACT = "No docs fact available. Base the verdict on the trait scores instead."


@tool
def fetch_product_fact(product: str) -> str:
    """Look up one grounded, factual sentence about the LangChain product
    you were matched with. Call this right after score_and_match, passing
    in the product name it returned."""
    # TODO 6: replace this line with the MCP lookup described above.
    return PLACEHOLDER_FACT


def build_user_prompt(answers: list[tuple[int, int, int]]) -> str:
    return (
        "Here are my personality quiz answers as a list of "
        "(chaotic/organized, cautious/bold, solo/collaborative) deltas, in "
        f"order: {answers}. Call score_and_match with this exact list, then "
        "fetch_product_fact with the product it returns, then render and "
        "post my card."
    )


if __name__ == "__main__":
    answers = run_quiz()
    user_prompt = build_user_prompt(answers)
    for judge_name in JUDGES_TO_RUN:
        run_judge(
            judge_name,
            system_prompt=JUDGE_PERSONAS[judge_name],
            user_prompt=user_prompt,
            tools=[score_and_match, fetch_product_fact, render_card, post_card],
            model=MODEL,
            interrupt_on=INTERRUPT_ON,
        )
    print(f"\nCards saved to {OUTPUT_DIR}/")
