# python/m2/m2.3_homework.py

import os
import shutil
from pathlib import Path

# Prevent Tkinter and encoding crashes on Windows (Rule #16)
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["MPLBACKEND"] = "Agg"

from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend
from models import model

# ════════════════════════════════════════════════════════════════════════
# TODO 1: System prompt for data visualization assistant
# ════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = (
    "You are an automotive data telemetry analyst. When asked to run code, write the "
    "Python script to a file first using write_file (use parameter file_path), then execute it using python. "
    "Rules: "
    "- Only call exact tool names: `write_file`, `read_file`, `execute`. Never invent or append commentary to tool names. "
    "- When using matplotlib, ALWAYS include `import matplotlib; matplotlib.use('Agg')` at the very top. "
    "- Use relative paths without leading slashes (e.g. 'telemetry.json', 'homework_chart.png')."
)

# ════════════════════════════════════════════════════════════════════════
# TODO 2: Two tasks proving persistence
# ════════════════════════════════════════════════════════════════════════

# TASK_ONE = (
#     "Generate lap time data (in seconds) for 10 laps of a Nissan Skyline GT-R around Suzuka Circuit. "
#     "Save the lap numbers and lap times to 'telemetry.json' using write_file. Print the data summary."
# )

TASK_TWO = (
    "Read the saved 'telemetry.json' file (do NOT regenerate the data). "
    "Write a Python script that creates a clean line chart of the lap times across all 10 laps "
    "using matplotlib. Save the chart as 'homework_chart.png' and execute the script."
)

# ════════════════════════════════════════════════════════════════════════
# TODO 3: Point CHART_PATH at where TASK_TWO saves the chart
# ════════════════════════════════════════════════════════════════════════

CHART_PATH = "homework_chart.png"

# Setup local persistent sandbox directory
sandbox_dir = Path(__file__).resolve().parent / "local_sandbox"
sandbox_dir.mkdir(exist_ok=True)

backend = LocalShellBackend(root_dir=str(sandbox_dir), virtual_mode=True)

agent = create_deep_agent(
    model=model,
    backend=backend,
    system_prompt=SYSTEM_PROMPT,
)

print("Starting Homework 2.3 execution...")

# # Task 1: Generate data and save to file
# print("\n>>> Running Task 1...")
# result1 = agent.invoke({"messages": [{"role": "user", "content": TASK_ONE}]})
# print("\n--- Task 1 Summary ---")
# print(result1["messages"][-1].text)

# # Reset Groq's 60-second sliding token bucket completely to 0!
# print("\n>>> Resting the telemetry engine for 60 seconds to reset Groq's token bucket...")
# import time
# for remaining in range(60, 0, -10):
#     print(f"    Bucket cooldown: {remaining}s remaining...")
#     time.sleep(10)

# Task 2: Read file back and chart it with a 100% fresh 8,000 TPM bucket!
print("\n>>> Running Task 2 (Plotting)...")
result2 = agent.invoke({"messages": [{"role": "user", "content": TASK_TWO}]})
print("\n--- Task 2 Result ---")
print(result2["messages"][-1].text)


# Verify chart existence
chart_candidate = sandbox_dir / "home" / "user" / CHART_PATH
if not chart_candidate.exists():
    chart_candidate = sandbox_dir / CHART_PATH

if chart_candidate.exists():
    out_path = Path(__file__).parent / "homework_chart.png"
    shutil.copy(chart_candidate, out_path)
    print(f"\n[SUCCESS] Telemetry chart saved locally to: {out_path}")