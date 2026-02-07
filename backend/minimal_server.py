from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import os
from dotenv import load_dotenv
import cohere
from qdrant_client import QdrantClient
from qdrant_client.http import models
import logging

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="RAG Chatbot API",
    description="Retrieval-Augmented Generation chatbot API for AI book",
    version="1.0.0"
)

# Initialize clients with error handling
try:
    cohere_client = cohere.Client(os.getenv("COHERE_API_KEY"))
    print("✅ Cohere client initialized")
except Exception as e:
    print(f"❌ Failed to initialize Cohere client: {e}")
    cohere_client = None

try:
    qdrant_client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY")
    )
    print("✅ Qdrant client initialized")
except Exception as e:
    print(f"❌ Failed to initialize Qdrant client: {e}")
    qdrant_client = None

# Define request/response models
class MessageRequest(BaseModel):
    message: str
    selected_text: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    sources: List[str]
    tokens_used: int

@app.get("/")
def read_root():
    return {"message": "RAG Chatbot API is running!"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)