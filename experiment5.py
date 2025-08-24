import mlflow
from mlflow.genai.scorers import Correctness, RelevanceToQuery, Guidelines

from dotenv import load_dotenv
from fastapi import FastAPI
from openai import OpenAI

load_dotenv()
app = FastAPI()
client = OpenAI()

# Set up local MLflow tracking
mlflow.set_tracking_uri("http://localhost:5001")
mlflow.set_experiment("Use Predefined LLM Scorers")

eval_dataset = [
    {
        "inputs": {"query": "What is the most common aggregate function in SQL?"},
        "outputs": "The most common aggregate function in SQL is SUM().",
        "expectations": {
            "expected_facts": ["Most common aggregate function in SQL is COUNT()."],
        },
    },
    {
        "inputs": {"query": "How do I use MLflow?"},
        # verbose answer
        "outputs": "Hi, I'm a chatbot that answers questions about MLflow. Thank you for asking a great question! I know MLflow well and I'm glad to help you with that. You will love it! MLflow is a Python-based platform that provides a comprehensive set of tools for logging, tracking, and visualizing machine learning models and experiments throughout their entire lifecycle. It consists of four main components: MLflow Tracking for experiment management, MLflow Projects for reproducible runs, MLflow Models for standardized model packaging, and MLflow Model Registry for centralized model lifecycle management. To get started, simply install it with 'pip install mlflow' and then use mlflow.start_run() to begin tracking your experiments with automatic logging of parameters, metrics, and artifacts. The platform creates a beautiful web UI where you can compare different runs, visualize metrics over time, and manage your entire ML workflow efficiently. MLflow integrates seamlessly with popular ML libraries like scikit-learn, TensorFlow, PyTorch, and many others, making it incredibly easy to incorporate into your existing projects!",
        "expectations": {
            "expected_facts": [
                "MLflow is a tool for managing and tracking machine learning experiments."
            ],
        },
    },
]

results = mlflow.genai.evaluate(
    data=eval_dataset,
    scorers=[
        Correctness(),
        RelevanceToQuery(),
        Guidelines(
            name="is_concise",
            guidelines="The answer must be concise and straight to the point.",
        ),
    ],
)