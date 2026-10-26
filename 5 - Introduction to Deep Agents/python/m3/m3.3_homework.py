# python/m3/m3.2_homework.py
"""M3.2 Homework: Bundle a Reference File Into Your Skill."""

import tempfile
from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends.filesystem import FilesystemBackend

from models import model

SKILL_NAME = "craft-bespoke-cocktail"
REFERENCE_PATH = f"/skills/{SKILL_NAME}/reference.md"


# ════════════════════════════════════════════════════════════════════════
# TODO 1: Write your own SKILL.md content.
# ════════════════════════════════════════════════════════════════════════

def build_skill_md() -> str:
    return f"""---
name: {SKILL_NAME}
description: Use when the user asks to craft, mix, or formulate a bespoke cocktail recipe.
---

# Craft a Bespoke Cocktail

Follow these steps to construct a balanced cocktail recipe to our speakeasy's house standard.

**Step 1: Identify Profile**: Confirm the user's desired base spirit and preferred style (e.g. sour, spirit-forward, highball, non-alcoholic).

**Step 2: Consult House Reference Guide**: 
Before writing the spec, you MUST call `read_file` on `{REFERENCE_PATH}` to retrieve our bar's proprietary syrup brix levels, house acid-adjusted citrus formulas, zero-proof bases, and standardized dilution ratios. Do NOT invent your own measurements or guess our house specs.

**Step 3: Construct the Spec**: Build the recipe using the exact proportions, non-alcoholic alternatives, and house ingredients specified in reference.md.

**Step 4: Output the Drink Ticket**: Present the final recipe with:
- Drink Name
- Exact measurements in ounces (oz) using house syrups/acids/bases from reference.md
- Preparation technique (Shake vs. Stir)
- Glassware and Ice Specification
- Garnish
"""


# ════════════════════════════════════════════════════════════════════════
# TODO 2: Write reference.md's content.
# ════════════════════════════════════════════════════════════════════════

def build_reference_md() -> str:
    return """# The House Cocktail Reference & Specs Manual

## 1. Proprietary House Syrups & Modifiers
- **House 2:1 Demerara Gomme**: 2 parts raw Demerara sugar to 1 part filtered water, 2% gum arabic for velvety mouthfeel. (Rich, caramel, viscous).
- **Orgeat Royale**: 65 Brix blanched almond syrup infused with roasted almond extract, brandy essence, and 3 drops orange flower water per 500ml.
- **Zero-Proof Botanical Apertif Cordial**: Infusion of gentian root, bitter orange peel, hibiscus, and juniper berry sweetened with 1.5:1 wildflower honey syrup.

## 2. Acid-Adjusted Citrus Formulations
- **Acid-Adjusted Grapefruit**: Fresh white grapefruit juice fortified with 3.2% citric acid and 0.8% malic acid powder by weight to equal lime's 6% titratable acidity.
- **Clarified Super Lemon Juice**: Lemon peel oleo-citrate clarified through agar-agar; shelf-stable for 30 days at 4°C.

## 3. Zero-Proof / Non-Alcoholic Spirit Alternatives
- **House Distilled Botanical Base (NA)**: Hydrosol blend of juniper, crushed coriander, angelica root, and Persian dried lime (substitute 1:1 for dry gin).
- **Smoked Tea & Oak Redux (NA)**: Heavy Lapsang Souchong cold-brew reduced with toasted American oak chips and verjus (substitute 1:1 for aged whiskey or rum).

## 4. House Standard Ratios & Build Rules
- **Standard Sour Formula**: 2.0 oz Base (Alcoholic or House NA) : 0.75 oz Acid-Adjusted Citrus : 0.75 oz House Syrup.
- **Spirit-Forward / Old Fashioned Ratio**: 2.25 oz Base : 1 barspoon (0.125 oz) 2:1 Demerara Gomme : 2 dashes house bitters.
- **Low-ABV / Highball Build**: 1.5 oz Base : 0.75 oz Zero-Proof Cordial : 3.5 oz Top-tier soda water, built in Collins glass with spear ice.

## 5. Ice & Glassware Standards
- Shaken Drinks (contains citrus or cream): Hard shake with dense Kold-Draft cubes for 12 seconds. Double strain into chilled Nick & Nora or Coupe.
- Stirred Drinks: Stir with Japanese yarai mixing glass for 45 revolutions. Strain over single 2-inch crystal-clear ice block in Double Old Fashioned glass.
"""


# Write the skill to a scratch directory so it's discoverable through a
# FilesystemBackend, the same mechanism the lab uses for python/m3/skills/.
_tmp_root = Path(tempfile.mkdtemp(prefix="m3_2_homework_"))
_skill_dir = _tmp_root / "skills" / SKILL_NAME
_skill_dir.mkdir(parents=True, exist_ok=True)
(_skill_dir / "SKILL.md").write_text(build_skill_md(), encoding="utf-8")
(_skill_dir / "reference.md").write_text(build_reference_md(), encoding="utf-8")

backend = FilesystemBackend(root_dir=str(_tmp_root), virtual_mode=True)
print(f"Skill files written to: {_skill_dir}")


# ════════════════════════════════════════════════════════════════════════
# TODO 3: Write a system prompt and a triggering question.
# ════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are Julian, the fastidious Head Mixologist at The Velvet Quill speakeasy.
You speak with quiet passion about balance, hospitality, and cocktail architecture.
You treat non-alcoholic drinks with the exact same reverence and culinary rigor as high-proof spirits.
Always adhere strictly to our house manual and reference guides.
"""

USER_QUESTION = (
    "Hey Julian, could you craft a bespoke sour cocktail for me? I love bourbon or rich whiskey flavors, "
    "but my guest wants a non-alcoholic version with the same balance and complexity. "
    "What are your house specs for both?"
)

agent = create_deep_agent(
    model=model,
    name="Homework_Agent",
    backend=backend,
    skills=["/skills"],
    system_prompt=SYSTEM_PROMPT,
)

result = agent.invoke({"messages": [{"role": "user", "content": USER_QUESTION}]})
print("\n" + result["messages"][-1].text)

read_calls = [
    call
    for msg in result["messages"]
    for call in getattr(msg, "tool_calls", [])
    if call["name"] == "read_file"
]
reference_was_read = any(call["args"].get("file_path") == REFERENCE_PATH for call in read_calls)
print(f"\n--- Did the agent read {REFERENCE_PATH}? {reference_was_read} ---")
if not reference_was_read:
    print(
        "It didn't. Either SKILL.md isn't clearly telling it to, or the "
        "task is answerable without the details in reference.md."
    )