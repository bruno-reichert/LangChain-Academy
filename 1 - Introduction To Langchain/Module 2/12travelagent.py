import asyncio
from dotenv import load_dotenv, find_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from langchain.messages import HumanMessage

load_dotenv(find_dotenv())

async def main():
    # 1. Connect to Kiwi Travel MCP server over HTTP!
    client = MultiServerMCPClient(
        {
            "travel_server": {
                "transport": "streamable_http",
                "url": "https://mcp.kiwi.com"
            }
        }
    )

    print("✈️ Connecting to Kiwi Travel MCP server...")
    tools = await client.get_tools()
    print(f"✅ Loaded {len(tools)} travel tools!")

    # 2. Initialize Gemini
    gemini_lite = init_chat_model(
        model="models/gemini-3.5-flash-lite",
        model_provider="google_genai"
    )

    # 3. Create Travel Agent (with Memory checkpointer)
    agent = create_agent(
        model=gemini_lite,
        tools=tools,
        system_prompt="You are a travel agent. No follow up questions.",
        checkpointer=InMemorySaver()
    )

    config = {"configurable": {"thread_id": "travel_1"}}

    # 4. Search for flights!
    print("🔎 Searching for flights from SFO to Tokyo...")
    response = await agent.ainvoke(
        {"messages": [HumanMessage(content="Get me a direct flight from San Francisco to Tokyo on March 31st")]},
        config=config
    )

    print("\n=== FLIGHT DETAILS ===")
    ans = response["messages"][-1].content
    print(ans[0]['text'] if isinstance(ans, list) else ans)

if __name__ == "__main__":
    asyncio.run(main())