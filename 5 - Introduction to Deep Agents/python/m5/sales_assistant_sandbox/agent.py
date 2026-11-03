# python/m5/sales_assistant_sandbox/agent.py
"""Chinook Sales Assistant.

Configured with a local isolated sandbox backend, pre-seeded with skills
and AGENTS.md, with a QuickJS code interpreter for arithmetic and
an execute tool for plotting charts locally.
"""

from __future__ import annotations

import contextlib
import logging
import os
import shutil
from pathlib import Path

from async_research import build_async_research_middleware
from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend, StateBackend
from langchain_core.runnables import RunnableConfig
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_quickjs import CodeInterpreterMiddleware
from langgraph_sdk.runtime import ServerRuntime
from subagents import build_subagents
from tools.html import markdown_to_html

from models import strong_model

# Force headless matplotlib and UTF-8 encoding across all agent child processes
os.environ["MPLBACKEND"] = "Agg"
os.environ["PYTHONIOENCODING"] = "utf-8"

logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent

SYSTEM_PROMPT = (
    "You are a sales assistant for Jane Peacock, a Sales Support Agent at "
    "Chinook, an online music distributor. Follow your operating manual (loaded "
    "from your memory) and use the matching playbook from /skills/ for each task.\n\n"
    "Your entire filesystem — skills, memory, and anything you write — lives "
    "inside an isolated sandbox. To produce a chart, write a Python script with "
    "write_file and run it with execute, saving the image under /outputs/. "
    "When using matplotlib, ALWAYS include `import matplotlib; matplotlib.use('Agg')` at the top."
)

MAIL_SERVER = {"transport": "streamable-http", "url": "http://127.0.0.1:5002/mcp"}

_enable_search = bool(os.environ.get("TAVILY_API_KEY"))
if not _enable_search:
    logger.info("TAVILY_API_KEY not set — newsletter research subagent disabled.")

# ── Local Sandbox Setup ($0 Zero-Dollar Cloud Alternative) ───────────────────
LOCAL_SANDBOX_DIR = HERE / "local_sandbox"
LOCAL_SANDBOX_DIR.mkdir(exist_ok=True)
(LOCAL_SANDBOX_DIR / "outputs").mkdir(exist_ok=True)


def _seed_local_sandbox():
    """Copy /skills and /AGENTS.md into the local sandbox if not present."""
    agents_md = LOCAL_SANDBOX_DIR / "AGENTS.md"
    if not agents_md.exists():
        agents_md.write_bytes((HERE / "AGENTS.md").read_bytes())

    dest_skills = LOCAL_SANDBOX_DIR / "skills"
    if not dest_skills.exists() and (HERE / "skills").exists():
        shutil.copytree(HERE / "skills", dest_skills)


_seed_local_sandbox()

_local_backend = LocalShellBackend(root_dir=str(LOCAL_SANDBOX_DIR), virtual_mode=True)


@contextlib.asynccontextmanager
async def make_graph(config: RunnableConfig, runtime: ServerRuntime):
    # Route real execution runs to our local isolated sandbox, introspection to StateBackend
    backend = _local_backend if runtime.execution_runtime else StateBackend()

    client = MultiServerMCPClient({"mock-mail": MAIL_SERVER})
    mail_tools = await client.get_tools()
    middleware = [CodeInterpreterMiddleware(ptc=["execute", "write_file"])]
    if _enable_search:
        middleware.append(build_async_research_middleware())

    yield create_deep_agent(
        model=strong_model,
        tools=[markdown_to_html] + mail_tools,
        system_prompt=SYSTEM_PROMPT,
        subagents=build_subagents(backend, mail_tools=mail_tools),
        skills=["/skills"],
        memory=["/AGENTS.md"],
        backend=backend,
        middleware=middleware,
        name="chinook-sales-assistant",
    )