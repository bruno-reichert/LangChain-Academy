# python/m2/m2.2_homework.py

from pathlib import Path

from deepagents import FilesystemPermission, create_deep_agent
from deepagents.backends import FilesystemBackend

from models import model


# ════════════════════════════════════════════════════════════════════════
# TODO 1: Configure a backend for a topic of your choosing.
# ════════════════════════════════════════════════════════════════════════

garage_dir = Path(__file__).parent / "garage"
garage_dir.mkdir(exist_ok=True)

service_log = garage_dir / "r32_service_log.md"
service_log.write_text("""# 1991 Nissan Skyline GT-R (BNR32) - Service Log

## Vehicle Specs:
- Engine: RB26DETT Twin-Turbo
- Odometer: 114,200 km

## Completed Maintenance:
- 112,000 km: Motul 300V 15W-50 oil and OEM filter replacement.
- 113,500 km: Replaced timing belt and water pump (Nismo Heavy Duty).
- 114,000 km: Flushed ATTESA E-TS hydraulic reservoir with Nissan special fluid.

## Open Maintenance Items:
- Front brake pads nearing wear indicators (< 3mm remaining). Rotors in good shape.
""")

backend = FilesystemBackend(root_dir=str(garage_dir), virtual_mode=True)


# ════════════════════════════════════════════════════════════════════════
# TODO 2: Write the task, and optionally a permission rule.
# ════════════════════════════════════════════════════════════════════════

TASK = (
    "Read /r32_service_log.md, check the open maintenance items, and update the log. "
    "Mark the front brakes as completed at 114,200 km with 'Installed Endless MX72 brake pads "
    "and flushed system with Motul RBF 660 brake fluid'. Remove it from open items."
)

# Apply our ordered permission: deny any deletion operations!
permissions: list[FilesystemPermission] = [
    FilesystemPermission(
        operations=["delete"],
        paths=["/**"],
        mode="deny",
    ),
]

agent = create_deep_agent(
    model=model,
    backend=backend,
    permissions=permissions,
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": TASK}]},
    config={"configurable": {"thread_id": "homework-m2.2"}},
)

print(result["messages"][-1].text)