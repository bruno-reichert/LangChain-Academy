import os
from dataclasses import dataclass
from typing import Any, Dict, List
from dotenv import load_dotenv, find_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool, ToolRuntime
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, HumanInTheLoopMiddleware, ModelRequest
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv(find_dotenv())

# 1. Base Model
model = init_chat_model(
    model="models/gemini-3.5-flash-lite", 
    model_provider="google_genai"
)

# 2. Auth Schema
@dataclass
class AuthContext:
    user_name: str = "Seán"
    user_email: str = "sean@company.com"
    is_authenticated: bool = True

# 3. Database
MOCK_INBOX_DB: Dict[str, List[Dict[str, str]]] = {
    "sean@company.com": [
        {
            "id": "msg_001",
            "from": "manager_sarah@company.com",
            "subject": "Urgent: Q3 Budget Review",
            "body": "Hi Seán, we need your team's finalized Q3 budget numbers before 4 PM today. Can you confirm if they are ready?",
        }
    ]
}

# 4. Tools
@tool
def read_inbox(runtime: ToolRuntime[AuthContext]) -> str:
    """Reads unread emails for the user."""
    user_email = runtime.context.user_email if runtime.context else "sean@company.com"
    emails = MOCK_INBOX_DB.get(user_email, [])
    if not emails:
        return "Inbox is empty."
    return f"From: {emails[0]['from']}\nSubject: {emails[0]['subject']}\nBody: {emails[0]['body']}"

@tool
def send_email(to: str, subject: str, body: str, runtime: ToolRuntime[AuthContext]) -> str:
    """Sends an email reply. Requires Human Approval."""
    return f"✅ Email sent to {to} with subject '{subject}'."

# 5. Dynamic Prompt & HITL
@dynamic_prompt
def assistant_prompt(request: ModelRequest) -> str:
    user_name = request.runtime.context.user_name if request.runtime.context else "Seán"
    return f"""You are the Executive Email Assistant for {user_name}.
1. Use `read_inbox` to check messages.
2. Use `send_email` to reply, signing off with 'Best regards, {user_name}'.
3. If your `send_email` call is rejected with critique, revise it and call `send_email` again."""

hitl = HumanInTheLoopMiddleware(
    interrupt_on={"read_inbox": False, "send_email": True},
    description_prefix="⚠️ OUTGOING EMAIL APPROVAL REQUIRED"
)

# 6. Expose the agent graph variable
agent = create_agent(
    model=model,
    tools=[read_inbox, send_email],
    context_schema=AuthContext,
    middleware=[assistant_prompt, hitl],
    # checkpointer=InMemorySaver()
)