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
mlflow.set_experiment("GenAI Evaluation Local")

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


def evaluate_qa_model():
    """Local evaluation function that works without Databricks"""
    results = []
    
    with mlflow.start_run():
        mlflow.log_param("model", "gpt-4o-mini")
        mlflow.log_param("dataset_size", len(eval_dataset))
        
        for i, item in enumerate(eval_dataset):
            question = item["inputs"]["question"]
            expected = item["expectations"]["expected_response"]
            
            # Get model prediction
            prediction = qa_predict_fn(question)
            
            # Evaluate correctness (simple string matching)
            is_correct = expected.lower() in prediction.lower()
            
            # Evaluate conciseness
            is_concise_answer = is_concise(prediction)
            
            # Evaluate if answer is in English (simple check)
            is_english = all(ord(c) < 128 for c in prediction)
            
            # Log results
            result = {
                "question": question,
                "expected": expected,
                "prediction": prediction,
                "correctness": is_correct,
                "conciseness": is_concise_answer,
                "is_english": is_english
            }
            results.append(result)
            
            # Log individual metrics
            mlflow.log_metric(f"correctness_{i}", float(is_correct))
            mlflow.log_metric(f"conciseness_{i}", float(is_concise_answer))
            mlflow.log_metric(f"is_english_{i}", float(is_english))
        
        # Calculate and log aggregate metrics
        total_correct = sum(r["correctness"] for r in results)
        total_concise = sum(r["conciseness"] for r in results)
        total_english = sum(r["is_english"] for r in results)
        
        accuracy = total_correct / len(results)
        conciseness_rate = total_concise / len(results)
        english_rate = total_english / len(results)
        
        mlflow.log_metric("overall_accuracy", accuracy)
        mlflow.log_metric("overall_conciseness", conciseness_rate)
        mlflow.log_metric("overall_english_rate", english_rate)
        
        # Log detailed results as artifact
        import json
        with open("evaluation_results.json", "w") as f:
            json.dump(results, f, indent=2)
        mlflow.log_artifact("evaluation_results.json")
        
        print(f"Evaluation completed!")
        print(f"Overall Accuracy: {accuracy:.2%}")
        print(f"Overall Conciseness: {conciseness_rate:.2%}")
        print(f"Overall English Rate: {english_rate:.2%}")
        
        return results


if __name__ == "__main__":
    results = evaluate_qa_model()
