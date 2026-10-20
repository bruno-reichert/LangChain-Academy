# python/m1/m1.5_homework.py
"""M1.5 Homework: Build Your Own Custom Tool.

THE IDEA
The lab wired up one custom tool (read_sql) for one fixed topic (the
Chinook music database). This homework asks you to do the same thing for
a topic YOU pick: something you actually know or care about (a game, a
sport, a show, your favorite band's discography, local trivia, whatever).
There's no single correct topic or persona here, that's the point. Two
students doing this homework could end up with two completely different
tools and agents.

WHAT YOU FILL IN
  TODO 1: write your own custom tool with the @tool decorator. Pick any
    topic, store a small lookup (a dict is fine, no API needed) of facts
    about it, and return one back based on the argument the model passes.
  TODO 2: write a system prompt that gives the agent a persona of your
    choosing and tells it to use your tool before answering.

RUN
  cd python
  uv run ./m1/m1.5_homework.py
"""

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_core.tools import tool

from deepagents import create_deep_agent
from models import model


# ════════════════════════════════════════════════════════════════════════
# TODO 1: Define your own custom tool.
#
# Requirements:
#   - Keep the @tool decorator.
#   - Give it a real docstring: one sentence the model will read to decide
#     when to call this tool.
#   - Have it take at least one argument and return a string.
#   - The lookup data can just live in this file (a dict, a list, whatever
#     fits your topic). No external API or key needed.
#
# Example shape (delete this and write your own):
#   @tool
#   def lookup_something(query: str) -> str:
#       """One sentence describing what this returns and when to call it."""
#       ...
# ════════════════════════════════════════════════════════════════════════

@tool
def lookup_jdm_specs(query: str) -> str:
    """Query specs, chassis codes, engine designations, and historical trivia for iconic 70s–90s JDM legends."""
    cars = {
        "toyota supra": (
            "Chassis: A80 / Mk IV (1993-2002). Engine: 3.0L Twin-Turbo 2JZ-GTE Inline-6. "
            "Famous for its bulletproof cast-iron block capable of 1000+ hp on stock internals, "
            "Getrag 6-speed manual, and iconic curved rear wing."
        ),
        "nissan skyline": (
            "Lineage: From the legendary 1969-1972 'Hakosuka' PGC10/KPGC10 and 1973 'Kenmeri' KPGC110 "
            "(powered by the rev-happy 2.0L S20 twin-cam 24-valve straight-6) to the modern era BNR32 'Godzilla' "
            "(1989-1994) with the 2.6L Twin-Turbo RB26DETT and ATTESA E-TS AWD that won 29 out of 29 JTCC races."
        ),
        "mazda rx-7": (
            "Chassis: FD3S (1992-2002). Engine: 1.3L Twin-Turbo 13B-REW Sequential Rotary. "
            "Famous for its lightweight balance, 50:50 weight distribution, high-revving rotary sound, "
            "and gorgeous timeless styling."
        ),
        "toyota ae86": (
            "Chassis: AE86 Sprinter Trueno / Corolla Levin (1983-1987). Engine: 1.6L 4A-GE DOHC 16-valve Inline-4. "
            "Front-engine, rear-wheel drive, ultra-lightweight chassis (~950 kg). The car that launched modern "
            "drifting culture under Keiichi Tsuchiya (Drift King) and immortalized in Initial D."
        ),
        "honda nsx": (
            "Chassis: NA1 (1990-1997). Engine: 3.0L Naturally Aspirated C30A V6 with VTEC. "
            "World's first mass-produced all-aluminum monocoque chassis, tuned and refined by Formula 1 "
            "legend Ayrton Senna at Suzuka Circuit. Redline: 8,000 RPM."
        ),
        "honda beat": (
            "Chassis: PP1 (1991-1996). Engine: 656cc Naturally Aspirated E07A Inline-3 with MTREC "
            "(Multi Throttle Responsive Engine Control). Mid-engine, rear-wheel drive Kei-car sports convertible. "
            "Screams to an 8,500 RPM redline without a turbo, styled by Pininfarina, and the final car approved by Soichiro Honda."
        ),
        "mazda mx-5": (
            "Chassis: NA (1989-1997, sold as Eunos Roadster in Japan / Miata in US). "
            "Engine: 1.6L B6-ZE / 1.8L BP-ZE DOHC Inline-4. Designed under the Jinba Ittai philosophy "
            "('rider and horse as one body') with pop-up headlights, manual soft-top, and 50:50 weight distribution. "
            "The benchmark for accessible lightweight sports cars."
        ),
        "mitsubishi lancer evolution": (
            "Chassis: Evo III / IV / V / VI (1995-1999). Engine: 2.0L Turbo 4G63T Inline-4. "
            "Rally weapon developed for Group A WRC, piloted by Tommi Mäkinen to 4 consecutive World Championships."
        ),
        "subaru impreza wrx": (
            "Chassis: GC8 (1992-2000). Engine: 2.0L Turbo EJ20 Boxer-4. "
            "Featuring Symmetrical AWD, blue and gold 555 WRC livery, driven by Colin McRae to win the 1995 WRC title."
        ),
        "nissan fairlady z": (
            "Chassis: S30 / 240Z (1969-1978) & Z32 300ZX (1989-2000). "
            "The 240Z brought Japanese sports cars to the world stage with an L24 inline-6; "
            "the Z32 featured the VG30DETT twin-turbo V6 with advanced 90s CAD styling."
        ),
    }

    q = query.lower()
    for name, specs in cars.items():
        if name in q or any(part in q for part in name.split()):
            return f"Found record for {name.upper()}: {specs}"

    available = ", ".join(k.title() for k in cars.keys())
    return f"No record found for '{query}'. Available legends: {available}"

# ════════════════════════════════════════════════════════════════════════
# TODO 2: Write a system prompt for your agent.
#
# Give it a persona (a name, a voice, a personality, anything you want)
# and tell it to call your_custom_tool (rename it if you like) before
# answering, the same way the lab's SYSTEM_PROMPT pointed the agent at
# read_sql.
# ════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are Kenji, a grizzled, veteran mechanic and engine builder from Kanagawa who has been wrenching on Japanese machinery since the 1970s. 

Persona and Comportment Rules:
- You are dead serious, no-nonsense, and speak with quiet, hardened authority. You have seen kids wrap fiberglass kits around guardrails on the Wangan and the Touge for thirty years.
- You are ALWAYS smoking a Seven Stars cigarette. Periodically include brief action cues in asterisks describing you tapping the ash, lighting a fresh one with a battered Zippo, or talking with it dangling from your bottom lip (e.g. *taps ash onto the concrete*, *takes a slow drag from his Seven Stars*).
- You despise modern computerized econoboxes and showroom hype. You respect engineering purity, forged internals, and proper balance.
- CRITICAL TOOL INSTRUCTION: Before answering ANY question about a car's specs, lineage, or history, you MUST consult your shop binder by calling `lookup_jdm_specs`. Never rely on hearsay or guess engine codes—real builders check the spec sheet first.
"""

agent = create_deep_agent(
    model=model,
    name="Homework_Agent",
    tools=[lookup_jdm_specs],
    system_prompt=SYSTEM_PROMPT,
)

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Hey Kenji, I'm thinking about getting an old Skyline or maybe a Beat for a project. What's the real story with them?",
            }
        ]
    }
)

print(result["messages"][-1].text)
