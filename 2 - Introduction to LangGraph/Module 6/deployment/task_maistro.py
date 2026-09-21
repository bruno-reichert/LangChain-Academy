"""
task_mAIstro: Production Memory Agent
Manages Profile, ToDo Collection, and Procedural Instructions via LangGraph Store.
"""

import uuid
from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    merge_message_runs,
)
from langchain_core.runnables import RunnableConfig
from langchain.chat_models import init_chat_model
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.store.base import BaseStore
from trustcall import create_extractor


# ---------------------------------------------------------------------------
# 1. Model & Helpers
# ---------------------------------------------------------------------------

model = init_chat_model(
    model="models/gemini-3.5-flash-lite",
    model_provider="google_genai",
    max_retries=5,
)


class Spy:
    """Collects tool calls made by Trustcall for agent introspection."""

    def __init__(self):
        self.called_tools = []

    def __call__(self, run):
        q = [run]
        while q:
            r = q.pop()
            if r.child_runs:
                q.extend(r.child_runs)
            if r.run_type == "chat_model":
                self.called_tools.append(
                    r.outputs["generations"][0][0]["message"]["kwargs"]["tool_calls"]
                )


def extract_tool_info(tool_calls, schema_name="Memory"):
    """Extract information from tool calls for both patches and new memories."""
    changes = []
    for call_group in tool_calls:
        for call in call_group:
            if call["name"] == "PatchDoc":
                changes.append({
                    "type": "update",
                    "doc_id": call["args"]["json_doc_id"],
                    "planned_edits": call["args"]["planned_edits"],
                    "value": call["args"]["patches"][0]["value"],
                })
            elif call["name"] == schema_name:
                changes.append({
                    "type": "new",
                    "value": call["args"],
                })

    result_parts = []
    for change in changes:
        if change["type"] == "update":
            result_parts.append(
                f"Document {change['doc_id']} updated:\n"
                f"Plan: {change['planned_edits']}\n"
                f"Added content: {change['value']}"
            )
        else:
            result_parts.append(
                f"New {schema_name} created:\nContent: {change['value']}"
            )
    return "\n\n".join(result_parts)


# ---------------------------------------------------------------------------
# 2. Schemas
# ---------------------------------------------------------------------------

class UpdateMemory(TypedDict):
    """Decision on what memory type to update."""
    update_type: Literal["user", "todo", "instructions"]


class Profile(BaseModel):
    """User profile schema."""
    name: Optional[str] = Field(description="The user's name", default=None)
    location: Optional[str] = Field(description="The user's location", default=None)
    job: Optional[str] = Field(description="The user's job", default=None)
    connections: list[str] = Field(
        description="Personal connections (family, friends, coworkers)",
        default_factory=list,
    )
    interests: list[str] = Field(
        description="User interests and hobbies",
        default_factory=list,
    )


class ToDo(BaseModel):
    """Schema for individual ToDo items."""
    task: str = Field(description="The task to be completed.")
    time_to_complete: Optional[int] = Field(
        description="Estimated time to complete the task (minutes)."
    )
    deadline: Optional[datetime] = Field(
        description="When the task needs to be completed by",
        default=None,
    )
    solutions: list[str] = Field(
        description="Specific, actionable solutions or vendors",
        min_items=1,
        default_factory=list,
    )
    status: Literal["not started", "in progress", "done", "archived"] = Field(
        description="Current status of the task",
        default="not started",
    )


profile_extractor = create_extractor(
    model,
    tools=[Profile],
    tool_choice="Profile",
)


# ---------------------------------------------------------------------------
# 3. Prompts
# ---------------------------------------------------------------------------

MODEL_SYSTEM_MESSAGE = """You are a helpful chatbot. 

You are designed to be a companion to a user, helping them keep track of their ToDo list.

You have a long term memory which keeps track of three things:
1. The user's profile (general information about them) 
2. The user's ToDo list
3. General instructions for updating the ToDo list

Here is the current User Profile (may be empty if no information has been collected yet):
<user_profile>
{user_profile}
</user_profile>

Here is the current ToDo List (may be empty if no tasks have been added yet):
<todo>
{todo}
</todo>

Here are the current user-specified preferences for updating the ToDo list (may be empty if no preferences have been specified yet):
<instructions>
{instructions}
</instructions>

Here are your instructions for reasoning about the user's messages:

1. Reason carefully about the user's messages as presented below. 

2. Decide whether any of your long-term memory should be updated:
- If personal information was provided about the user, update the user's profile by calling UpdateMemory tool with type `user`
- If tasks are mentioned, update the ToDo list by calling UpdateMemory tool with type `todo`
- If the user has specified preferences for how to update the ToDo list, update the instructions by calling UpdateMemory tool with type `instructions`

3. Tell the user that you have updated your memory, if appropriate:
- Do not tell the user you have updated the user's profile
- Tell the user when you update the todo list
- Do not tell the user that you have updated instructions

4. Err on the side of updating the todo list. No need to ask for explicit permission.

5. Respond naturally to the user after a tool call was made to save memories, or if no tool call was made."""

TRUSTCALL_INSTRUCTION = """Reflect on following interaction. 

Use the provided tools to retain any necessary memories about the user. 

Use parallel tool calling to handle updates and insertions simultaneously.

System Time: {time}"""

CREATE_INSTRUCTIONS = """Reflect on the following interaction.

Based on this interaction, update your instructions for how to update ToDo list items. 

Use any feedback from the user to update how they like to have items added, etc.

Your current instructions are:

<current_instructions>
{current_instructions}
</current_instructions>"""


# ---------------------------------------------------------------------------
# 4. Graph Nodes & Routing
# ---------------------------------------------------------------------------

def task_mAIstro(state: MessagesState, config: RunnableConfig, store: BaseStore):
    """Load memories from the store and generate personalized response / routing decision."""
    user_id = config["configurable"]["user_id"]

    # 1. Profile
    profile_memories = store.search(("profile", user_id))
    user_profile = profile_memories[0].value if profile_memories else None

    # 2. ToDos
    todo_memories = store.search(("todo", user_id))
    todo = "\n".join(f"{mem.value}" for mem in todo_memories)

    # 3. Instructions
    inst_memories = store.search(("instructions", user_id))
    instructions = inst_memories[0].value if inst_memories else ""

    system_msg = MODEL_SYSTEM_MESSAGE.format(
        user_profile=user_profile, todo=todo, instructions=instructions
    )

    # Invoke with tool binding (Gemini clean invocation without parallel_tool_calls=False)
    response = model.bind_tools([UpdateMemory]).invoke(
        [SystemMessage(content=system_msg)] + state["messages"]
    )
    return {"messages": [response]}


def update_profile(state: MessagesState, config: RunnableConfig, store: BaseStore):
    """Reflect on chat history and update the user profile."""
    user_id = config["configurable"]["user_id"]
    namespace = ("profile", user_id)

    existing_items = store.search(namespace)
    existing_memories = (
        [(item.key, "Profile", item.value) for item in existing_items]
        if existing_items
        else None
    )

    formatted_instruction = TRUSTCALL_INSTRUCTION.format(time=datetime.now().isoformat())
    updated_messages = list(
        merge_message_runs(
            messages=[SystemMessage(content=formatted_instruction)]
            + state["messages"][:-1]
        )
    )

    result = profile_extractor.invoke({
        "messages": updated_messages,
        "existing": existing_memories,
    })

    for r, rmeta in zip(result["responses"], result["response_metadata"]):
        store.put(
            namespace,
            rmeta.get("json_doc_id", str(uuid.uuid4())),
            r.model_dump(mode="json"),
        )

    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            {
                "role": "tool",
                "content": "updated profile",
                "tool_call_id": tool_calls[0]["id"],
            }
        ]
    }


def update_todos(state: MessagesState, config: RunnableConfig, store: BaseStore):
    """Reflect on chat history and update the ToDo collection."""
    user_id = config["configurable"]["user_id"]
    namespace = ("todo", user_id)

    existing_items = store.search(namespace)
    tool_name = "ToDo"
    existing_memories = (
        [(item.key, tool_name, item.value) for item in existing_items]
        if existing_items
        else None
    )

    formatted_instruction = TRUSTCALL_INSTRUCTION.format(time=datetime.now().isoformat())
    updated_messages = list(
        merge_message_runs(
            messages=[SystemMessage(content=formatted_instruction)]
            + state["messages"][:-1]
        )
    )

    spy = Spy()
    todo_extractor = create_extractor(
        model,
        tools=[ToDo],
        tool_choice=tool_name,
        enable_inserts=True,
    ).with_listeners(on_end=spy)

    result = todo_extractor.invoke({
        "messages": updated_messages,
        "existing": existing_memories,
    })

    for r, rmeta in zip(result["responses"], result["response_metadata"]):
        store.put(
            namespace,
            rmeta.get("json_doc_id", str(uuid.uuid4())),
            r.model_dump(mode="json"),
        )

    tool_calls = state["messages"][-1].tool_calls
    todo_update_msg = extract_tool_info(spy.called_tools, tool_name)
    return {
        "messages": [
            {
                "role": "tool",
                "content": todo_update_msg,
                "tool_call_id": tool_calls[0]["id"],
            }
        ]
    }


def update_instructions(state: MessagesState, config: RunnableConfig, store: BaseStore):
    """Reflect on chat history and update custom procedural instructions."""
    user_id = config["configurable"]["user_id"]
    namespace = ("instructions", user_id)

    existing_memory = store.get(namespace, "user_instructions")
    system_msg = CREATE_INSTRUCTIONS.format(
        current_instructions=existing_memory.value if existing_memory else None
    )

    new_memory = model.invoke(
        [SystemMessage(content=system_msg)]
        + state["messages"][:-1]
        + [HumanMessage(content="Please update the instructions based on the conversation")]
    )

    store.put(namespace, "user_instructions", {"memory": new_memory.content})
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            {
                "role": "tool",
                "content": "updated instructions",
                "tool_call_id": tool_calls[0]["id"],
            }
        ]
    }


def route_message(state: MessagesState, config: RunnableConfig, store: BaseStore) -> Literal[END, "update_todos", "update_instructions", "update_profile"]:
    """Route based on whether task_mAIstro called UpdateMemory."""
    message = state["messages"][-1]
    if len(message.tool_calls) == 0:
        return END

    tool_call = message.tool_calls[0]
    update_type = tool_call["args"]["update_type"]
    if update_type == "user":
        return "update_profile"
    elif update_type == "todo":
        return "update_todos"
    elif update_type == "instructions":
        return "update_instructions"
    else:
        raise ValueError(f"Unrecognized update_type: {update_type}")


# ---------------------------------------------------------------------------
# 5. Graph Compilation for Deployment
# ---------------------------------------------------------------------------

builder = StateGraph(MessagesState)

builder.add_node(task_mAIstro)
builder.add_node(update_todos)
builder.add_node(update_profile)
builder.add_node(update_instructions)

builder.add_edge(START, "task_mAIstro")
builder.add_conditional_edges("task_mAIstro", route_message)
builder.add_edge("update_todos", "task_mAIstro")
builder.add_edge("update_profile", "task_mAIstro")
builder.add_edge("update_instructions", "task_mAIstro")

# Production Rule: Compile cleanly without checkpointer or store.
# LangGraph Server / CLI injects SQLite/Postgres persistence natively at runtime.
graph = builder.compile()