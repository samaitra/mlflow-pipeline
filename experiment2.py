import requests
import mlflow
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

from mlflow.entities import SpanType

# Decorated with @mlflow.trace to trace the function call.

mlflow.set_tracking_uri("http://localhost:5001")
mlflow.set_experiment("weather-api")

# Initialize OpenAI client
client = OpenAI()

@mlflow.trace(span_type=SpanType.TOOL)
def get_weather(latitude, longitude):
    # Input validation
    if not isinstance(latitude, (int, float)) or not isinstance(longitude, (int, float)):
        raise ValueError("Latitude and longitude must be numbers")
    if not (-90 <= latitude <= 90) or not (-180 <= longitude <= 180):
        raise ValueError("Invalid coordinate range")
    
    try:
        response = requests.get(
            f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,wind_speed_10m&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m",
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        return data["current"]["temperature_2m"]
    except (requests.RequestException, KeyError, ValueError) as e:
        raise RuntimeError(f"Weather API error: {e}")


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current temperature for provided coordinates in celsius.",
            "parameters": {
                "type": "object",
                "properties": {
                    "latitude": {"type": "number"},
                    "longitude": {"type": "number"},
                },
                "required": ["latitude", "longitude"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }
]

import json


# Define a simple tool calling agent
@mlflow.trace(span_type=SpanType.AGENT)
def run_tool_agent(question: str):
    messages = [{"role": "user", "content": question}]

    # Invoke the model with the given question and available tools
    response = client.chat.completions.create(
        model="o4-mini",
        messages=messages,
        tools=tools,
    )
    ai_msg = response.choices[0].message
    messages.append(ai_msg)

    # If the model requests tool call(s), invoke the function with the specified arguments
    if tool_calls := ai_msg.tool_calls:
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            if function_name == "get_weather":
                # Invoke the tool function with the provided arguments
                try:
                    args = json.loads(tool_call.function.arguments)
                    # Validate required arguments exist
                    if "latitude" not in args or "longitude" not in args:
                        raise ValueError("Missing required arguments: latitude and longitude")
                    tool_result = get_weather(**args)
                except (json.JSONDecodeError, ValueError, TypeError) as e:
                    tool_result = f"Error parsing arguments: {e}"
            else:
                raise RuntimeError("An invalid tool is returned from the assistant!")

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(tool_result),
                }
            )

        # Sent the tool results to the model and get a new response
        response = client.chat.completions.create(model="o4-mini", messages=messages)

    return response.choices[0].message.content

# Run the tool calling agent
question = "What's the weather like in Seattle?"
answer = run_tool_agent(question)    