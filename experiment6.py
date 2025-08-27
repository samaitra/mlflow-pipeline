import mlflow
import openai
from dotenv import load_dotenv
from openai import OpenAI
import time

load_dotenv()
client = OpenAI()

target_text = """
MLflow is an open source platform for managing the end-to-end machine learning lifecycle.
It tackles four primary functions in the ML lifecycle: Tracking experiments, packaging ML
code for reuse, managing and deploying models, and providing a central model registry.
MLflow currently offers these functions as four components: MLflow Tracking,
MLflow Projects, MLflow Models, and MLflow Registry.
"""

# Load the prompt
prompt = mlflow.genai.load_prompt("prompts:/summarization-prompt/2")

# Use the prompt with an LLM
client = openai.OpenAI()
response = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": prompt.format(num_sentences=1, sentences=target_text),
        }
    ],
    model="gpt-4o-mini",
)

print(response.choices[0].message.content)

