import os
import sys
import asyncio
from dotenv import load_dotenv, find_dotenv # type: ignore
from langchain_mcp_adapters.client import MultiServerMCPClient # type: ignore
from langchain.chat_models import init_chat_model # type: ignore
from langchain.agents import create_agent # type: ignore

load_dotenv(find_dotenv())

async def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    server_path = os.path.join(current_dir, "mcp_server.py")

    # 1. Connect to BOTH MCP servers simultaneously!
    client = MultiServerMCPClient({
        "local_server": {
            "transport": "stdio",
            "command": sys.executable,
            "args": [server_path]
        },
        "time": {
            "transport": "stdio",
            "command": "uv",
            "args": [
                "run",
                "python",
                "-m",
                "mcp_server_time",
                "--local-timezone=America/New_York"
            ]
        }
    })

    print("🔌 Connecting to BOTH MCP servers (local_server + time)...")
    
    # 2. Fetch all tools from all servers
    tools = await client.get_tools()
    print(f"\n✅ Successfully loaded {len(tools)} total tools:")
    for t in tools:
        print(f"  👉 {t.name}: {t.description}")

    # 3. Initialize Gemini
    gemini_lite = init_chat_model(
        model="models/gemini-3.5-flash-lite",
        model_provider="google_genai"
    )

    # 4. Create Agent equipped with ALL tools
    agent = create_agent(
        model=gemini_lite,
        tools=tools,
        system_prompt="You are a helpful assistant with access to web search, time, and documentation tools."
    )

    # 5. Test the Time Tool!
    print("\n🤖 Asking agent for the time...")
    question = "What is the current time in New York, and what is the time in Tokyo right now?"
    
    response = await agent.ainvoke({
        "messages": [{"role": "user", "content": question}]
    })

    print("\n=== AGENT RESPONSE ===")
    ans = response["messages"][-1].content
    print(ans[0]['text'] if isinstance(ans, list) else ans)

if __name__ == "__main__":
    asyncio.run(main())