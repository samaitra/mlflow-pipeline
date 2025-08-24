import os
import openai
import mlflow
from mlflow.genai import scorer
from mlflow.genai.scorers import Correctness, Guidelines
from dotenv import load_dotenv
from fastapi import FastAPI
from openai import OpenAI

load_dotenv()
app = FastAPI()
client = OpenAI()

# Set up local MLflow tracking
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("GenAI Evaluation Quickstart")

# Define a simple Q&A dataset with questions and expected answers
eval_dataset = [
    {
        "inputs": {"question": "What is the capital of France?"},
        "expectations": {"expected_response": "Paris"},
    },
    {
        "inputs": {"question": "Who was the first person to build an airplane?"},
        "expectations": {"expected_response": "Wright Brothers"},
    },
    {
        "inputs": {"question": "Who wrote Romeo and Juliet?"},
        "expectations": {"expected_response": "William Shakespeare"},
    },
]


def qa_predict_fn(question: str) -> str:
    """Simple Q&A prediction function using OpenAI"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant. Answer questions concisely.",
            },
            {"role": "user", "content": question},
        ],
    )
    return response.choices[0].message.content


@scorer
def is_concise(outputs: str) -> bool:
    """Evaluate if the answer is concise (less than 5 words)"""
    return len(outputs.split()) <= 5


scorers = [
    Correctness(),
    Guidelines(name="is_english", guidelines="The answer must be in English"),
    is_concise,
]

results = mlflow.genai.evaluate(
    data=eval_dataset,
    predict_fn=qa_predict_fn,
    scorers=scorers,
)