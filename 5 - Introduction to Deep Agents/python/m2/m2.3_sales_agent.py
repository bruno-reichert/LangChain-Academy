import sys
import os
import shutil
from pathlib import Path

# Force UTF-8 and headless matplotlib
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["MPLBACKEND"] = "Agg"

from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend
from models import model

sandbox_dir = Path(__file__).resolve().parent / "local_sandbox"
sandbox_dir.mkdir(exist_ok=True)

# Ensure chinook.db is in the sandbox
source_db = Path(__file__).resolve().parent.parent / "m1" / "chinook.db"
target_db = sandbox_dir / "chinook.db"
if source_db.exists() and not target_db.exists():
    shutil.copy(source_db, target_db)

backend = LocalShellBackend(root_dir=str(sandbox_dir), virtual_mode=True)

agent = create_deep_agent(
    model=model,
    backend=backend,
    system_prompt=(
        "You are a sales data analyst with access to the Chinook music store database at /chinook.db. "
        "Rules: "
        "- When writing Python scripts for matplotlib, ALWAYS include `import matplotlib; matplotlib.use('Agg')` at the very top. "
        "- Always use parameter `file_path` when calling write_file. "
        "- Write the script to file_path='/generate_chart.py', execute it with python, and confirm /genre_revenue.png is saved."
    ),
)

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": (
                    "Query /chinook.db to get total revenue by genre. Create a clean donut chart "
                    "showing each genre's share of total sales revenue. Group any genres < 3% into 'Other'. "
                    "Save the chart to /genre_revenue.png."
                ),
            }
        ]
    }
)

print(result["messages"][-1].text)

chart_file = sandbox_dir / "genre_revenue.png"
if chart_file.exists():
    out_path = Path(__file__).resolve().parent / "genre_revenue.png"
    shutil.copy(chart_file, out_path)
    print(f"\n[DONE] Donut chart generated at: {out_path}")