import mlflow
import os
from fastapi import FastAPI, Request
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
import time

load_dotenv()
app = FastAPI()
client = OpenAI()

# Initialize MLflow with retry logic
def setup_mlflow():
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            mlflow.set_tracking_uri("http://localhost:5001")
            mlflow.set_experiment("chat-api")
            mlflow.openai.autolog()
            print(f"Successfully connected to MLflow on attempt {attempt + 1}")
            return True
        except Exception as e:
            print(f"Attempt {attempt + 1} failed to connect to MLflow: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print("Failed to connect to MLflow after all retries. Continuing without MLflow...")
                return False

# Setup MLflow in background
print("Setting up MLflow...")
mlflow_available = setup_mlflow()


class ChatRequest(BaseModel):
    message: str


def process_chat(message: str, user_id: str, session_id: str):
    # Only use MLflow tracing if available
    if mlflow_available:
        try:
            with mlflow.trace() as trace:
                # Update trace with user and session context
                mlflow.update_current_trace(
                    metadata={
                        "mlflow.trace.session": session_id,
                        "mlflow.trace.user": user_id,
                    }
                )
                
                # Process chat message using OpenAI API
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": message},
                    ],
                )
                return response.choices[0].message.content
        except Exception as e:
            print(f"MLflow tracing failed: {e}. Continuing without tracing...")
    
    # Fallback without MLflow tracing
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": message},
        ],
    )
    return response.choices[0].message.content


@app.post("/chat")
def handle_chat(request: Request, chat_request: ChatRequest):
    session_id = request.headers.get("X-Session-ID", "default-session")
    user_id = request.headers.get("X-User-ID", "default-user")
    response_text = process_chat(chat_request.message, user_id, session_id)
    return {"response": response_text}


@app.get("/")
async def root():
    return {"message": "FastAPI MLflow Tracing Example"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)