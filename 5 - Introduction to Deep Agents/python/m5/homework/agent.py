# python/m5/homework/agent.py
"""M5.2 Homework: Deploy Your Own Agent.

THE IDEA
The lab deployed a fairly bare-bones agent (no tools, no persona, just
create_deep_agent(model=model)) and you only ever talked to it through
Studio's chat panel. This homework has two parts: first, deploy an agent
with your personal touch; second, talk to it the way any other client
would, straight over the Agent Server API this lesson covers, instead of
through Studio.

WHAT YOU FILL IN
  TODO 1: write your own @tool-decorated function on a topic of your
    choosing. A plain Python dict lookup is enough, no external API or
    key required.
  TODO 2: write a system_prompt that gives the agent a persona of your
    choosing and tells it to call your tool before answering.
  Then open call_agent_api.py in this same folder for TODO 3, which talks
  to this deployed agent over HTTP instead of through Studio.

RUN
  cd python/m5/homework
  uv run langgraph dev
Then chat with your agent in the Studio window that opens, or see
call_agent_api.py to talk to it over the API instead.
"""

from langchain_core.tools import tool

from deepagents import create_deep_agent
from models import model


# TODO 1: replace this with your own @tool-decorated function.
@tool
def lookup_f1_cars(car_name: str) -> str:
    """Look up historical specifications, engine type, championship status, and legendary drivers for classic F1 cars (1950s-1990s)."""
    database = {
        "alfa romeo 158": {
            "name": "1950 Alfa Romeo 158 'Alfetta'",
            "team": "Alfa Romeo",
            "era": "1950",
            "engine": "1.5L Straight-8 Supercharged (~350 hp)",
            "drivers": ["Giuseppe Farina", "Juan Manuel Fangio", "Luigi Fagioli"],
            "trivia": "Undefeated in the inaugural 1950 Formula One World Championship."
        },
        "lotus 49": {
            "name": "1967 Lotus 49",
            "team": "Team Lotus",
            "era": "1960s",
            "engine": "3.0L Ford Cosworth DFV V8 (~400 hp)",
            "drivers": ["Jim Clark", "Graham Hill"],
            "trivia": "Introduced the revolutionary concept of using the engine as a fully stressed structural chassis member."
        },
        "mclaren mp4/4": {
            "name": "1988 McLaren MP4/4",
            "team": "McLaren Honda",
            "era": "1980s Turbo Era",
            "engine": "1.5L Honda RA168E V6 Twin-Turbo (~650+ hp)",
            "drivers": ["Ayrton Senna", "Alain Prost"],
            "trivia": "Won 15 out of 16 races in 1988, achieving a 93.8% win rate."
        },
        "jordan 191": {
            "name": "1991 Jordan 191",
            "team": "Team 7Up Jordan",
            "era": "1990s",
            "engine": "3.5L Ford HBA4 V8 (~680 hp)",
            "drivers": ["Michael Schumacher", "Bertrand Gachot", "Andrea de Cesaris"],
            "trivia": "Michael Schumacher's F1 debut; widely praised as the most aesthetic F1 car ever painted."
        },
        "williams fw14b": {
            "name": "1992 Williams FW14B",
            "team": "Williams Renault",
            "era": "1990s High-Tech Era",
            "engine": "3.5L Renault RS3C V10 (~750 hp)",
            "drivers": ["Nigel Mansell", "Riccardo Patrese"],
            "trivia": "Dominated with computer-controlled active suspension and traction control."
        },
    }

    q = car_name.lower()
    for key, spec in database.items():
        if key in q or any(part in q for part in key.split()):
            # Format dict nicely into a string for the model
            drivers_str = ", ".join(spec["drivers"])
            return (
                f"Car: {spec['name']} | Team: {spec['team']} | Era: {spec['era']} | "
                f"Engine: {spec['engine']} | Drivers: {drivers_str} | Trivia: {spec['trivia']}"
            )

    available = ", ".join(k.title() for k in database.keys())
    return f"Car '{car_name}' not found in archive. Available cars: {available}"


# TODO 2: replace this with your own persona system prompt.
SYSTEM_PROMPT = """You are Murray, an eccentric, hyper-caffeinated motorsport historian and legendary F1 broadcaster.
You speak with infectious enthusiasm, dramatic flair, and deep reverence for the screaming engines and daring drivers of Formula 1 history.

Rules:
- Whenever a user asks about a classic Formula 1 car, you MUST call the `lookup_f1_cars` tool to retrieve the authentic archive specs. Never guess chassis details or engine types.
- Weave the facts from the database into your dramatic, passionate broadcast commentary style!
"""

# `langgraph.json` points at this module-level variable: "./agent.py:graph".
graph = create_deep_agent(model=model, tools=[lookup_f1_cars], system_prompt=SYSTEM_PROMPT)
