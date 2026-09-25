import os
from openai import OpenAI
from langsmith.wrappers import wrap_openai
from langsmith import traceable
from dotenv import load_dotenv

load_dotenv()

# Drop-in Groq via the OpenAI SDK wrapped with LangSmith tracing
client = wrap_openai(
    OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.environ.get("GROQ_API_KEY"),
    )
)

MODEL_NAME = "openai/gpt-oss-120b"


@traceable(run_type="tool")
def weather_retriever():
    """Retrieve current weather information."""
    random_number = os.urandom(1)[0]  # Generate a random number between 0 and 255
    if random_number < 128:
        return "It is raining today"
    elif random_number < 192:
        return "It is cloudy today"
    return "It is sunny today"


# Tool schema (OpenAI standard format)
WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "weather_retriever",
        "description": "Get the current weather conditions",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}


@traceable
def agent(question: str) -> dict:
    messages = [{"role": "user", "content": question}]

    # First API call with tool available
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        tools=[WEATHER_TOOL],
        tool_choice="auto",
    )

    response_message = response.choices[0].message

    # Handle tool calls if the model wants to use them
    if response_message.tool_calls:
        # Add assistant's tool call to messages
        messages.append({
            "role": "assistant",
            "content": response_message.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in response_message.tool_calls
            ],
        })

        # Execute the tool call(s)
        for tool_call in response_message.tool_calls:
            if tool_call.function.name == "weather_retriever":
                result = weather_retriever()

                # Add tool result to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": "weather_retriever",
                    "content": result,
                })

        # Make second API call with tool results
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=[WEATHER_TOOL],
            tool_choice="auto",
        )
        response_message = response.choices[0].message

    messages.append({"role": "assistant", "content": response_message.content})
    return {"messages": messages, "output": response_message.content}


if __name__ == "__main__":
    result = agent("What is the weather today in Baltimore?")
    print("\nResult:", result["output"])