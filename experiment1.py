import mlflow
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# This will create a new experiment called "Tracing Quickstart" and set it as active
mlflow.set_experiment("Tracing Quickstart")


# Enable automatic tracing for all OpenAI API calls
mlflow.openai.autolog()

# Instantiate the OpenAI client
client = OpenAI()

# Invoke chat completion API
response = client.chat.completions.create(
    model="o4-mini",
    messages=[
        {"role": "system", "content": "You are a helpful weather assistant."},
        {"role": "user", "content": "What's the weather like in Seattle?"},
    ],
)