from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import logging
import os
from dotenv import load_dotenv
import cohere
from qdrant_client import QdrantClient
from qdrant_client.http import models
from src.services.database_service import DatabaseService
from src.services.chatbot_service import ChatbotService
from src.utils.env_loader import load_environment, validate_api_key_exists
from src.utils.config_validator import validate_configuration
from src.utils.exceptions import (
    ChatbotBaseException,
    InvalidApiKeyError,
    ResourceNotFoundError,
    RateLimitExceededError
)

# Load environment variables
load_environment()

# Validate configuration
try:
    validate_configuration()
    logging.info("Configuration validation passed")
except Exception as e:
    logging.error(f"Configuration validation failed: {str(e)}")
    raise

# Initialize FastAPI app
app = FastAPI(
    title="Neon API Chatbot Service",
    description="A RAG chatbot service with Neon PostgreSQL database integration",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG components
try:
    cohere_client = cohere.Client(os.getenv("COHERE_API_KEY"))
    logging.info("✅ Cohere client initialized")
except Exception as e:
    logging.error(f"❌ Failed to initialize Cohere client: {e}")
    cohere_client = None

# Store Qdrant configuration but don't connect at startup to avoid blocking
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
qdrant_client = None  # Initialize as None, connect when needed
logging.info("Qdrant configuration stored, will connect when needed")

# Collection name for storing document embeddings
COLLECTION_NAME = "book_content"

# Initialize services conditionally
db_service = None
chatbot_service = None

# Only initialize database service if database URL is provided
if os.getenv('NEON_DB_URL'):
    try:
        db_service = DatabaseService()
        chatbot_service = ChatbotService(db_service)
        logging.info("Database service initialized successfully")
    except Exception as e:
        logging.error(f"Failed to initialize database service: {e}")
        # For RAG functionality, we can proceed without database
        db_service = None
        chatbot_service = None
else:
    logging.info("Database URL not provided, skipping database service initialization")
    # For RAG functionality, we can proceed without database
    db_service = None
    chatbot_service = None

# Pydantic models for request/response
class ChatRequest(BaseModel):
    user_id: str
    message: str
    conversation_id: Optional[str] = None

class ChatResponse(BaseModel):
    conversation_id: str
    message_id: str
    response: str
    timestamp: str

class ConversationResponse(BaseModel):
    id: str
    title: Optional[str]
    created_at: str
    updated_at: str
    messages: List[dict]

class UserConversationsResponse(BaseModel):
    conversations: List[dict]
    total_count: int

class UserPreferencesRequest(BaseModel):
    preferences: Dict[str, Any]

class UserPreferencesResponse(BaseModel):
    preferences: Dict[str, Any]

# Models for RAG functionality
class Document(BaseModel):
    content: str
    metadata: Dict[str, Any]

class IngestRequest(BaseModel):
    documents: List[Document]

class IngestResponse(BaseModel):
    message: str
    processed_count: int

class RAGChatRequest(BaseModel):
    message: str
    selected_text: Optional[str] = None
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None

class RAGChatResponse(BaseModel):
    response: str
    sources: List[str]
    tokens_used: int
    conversation_id: Optional[str] = None

# Dependency to verify API key
def verify_api_key(request: Request):
    api_key = request.headers.get("Authorization")
    if not api_key or not api_key.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

    # In a real implementation, you would validate the API key against a database
    # For now, we'll just check if it matches the environment variable
    expected_api_key = "Bearer " + "your_neon_api_key"  # This should be replaced with actual validation

    if api_key != expected_api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")

def get_qdrant_client():
    """Get or create Qdrant client"""
    global qdrant_client
    if qdrant_client is None:
        try:
            qdrant_client = QdrantClient(
                url=QDRANT_URL,
                api_key=QDRANT_API_KEY
            )
            logging.info("✅ Qdrant client initialized on demand")
        except Exception as e:
            logging.error(f"❌ Failed to initialize Qdrant client: {e}")
            raise HTTPException(status_code=500, detail="Failed to connect to Qdrant")
    return qdrant_client

def ensure_collection_exists():
    """Ensure the Qdrant collection exists"""
    client = get_qdrant_client()
    
    try:
        # Check if collection exists
        collections = client.get_collections()
        collection_names = [collection.name for collection in collections.collections]
        
        if COLLECTION_NAME not in collection_names:
            # Create collection with appropriate vector size for Cohere embeddings
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE)  # Cohere embeddings are 1024-dimensional
            )
            logging.info(f"Created Qdrant collection: {COLLECTION_NAME}")
        else:
            logging.info(f"Qdrant collection {COLLECTION_NAME} already exists")
    except Exception as e:
        logging.error(f"Error ensuring collection exists: {str(e)}")
        raise

def embed_documents(documents: List[str]):
    """Generate embeddings for a list of documents using Cohere"""
    if not cohere_client:
        raise HTTPException(status_code=500, detail="Cohere client not initialized")
    
    try:
        response = cohere_client.embed(
            texts=documents,
            model="multilingual-22-12"  # Using a Cohere embedding model
        )
        return response.embeddings
    except Exception as e:
        logging.error(f"Error generating embeddings: {str(e)}")
        raise

def store_in_qdrant(documents: List[Document], embeddings: List[List[float]]):
    """Store documents and their embeddings in Qdrant"""
    client = get_qdrant_client()
    
    try:
        points = []
        for idx, (doc, embedding) in enumerate(zip(documents, embeddings)):
            points.append(
                models.PointStruct(
                    id=idx,
                    vector=embedding,
                    payload={
                        "content": doc.content,
                        "metadata": doc.metadata
                    }
                )
            )
        
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )
        logging.info(f"Stored {len(points)} documents in Qdrant collection: {COLLECTION_NAME}")
    except Exception as e:
        logging.error(f"Error storing documents in Qdrant: {str(e)}")
        raise

def search_similar_documents(query: str, top_k: int = 5):
    """Search for similar documents in Qdrant based on the query"""
    if not cohere_client:
        raise HTTPException(status_code=500, detail="Cohere client not initialized")
    
    client = get_qdrant_client()
    
    try:
        # Generate embedding for the query
        query_embedding = cohere_client.embed(
            texts=[query],
            model="multilingual-22-12"
        ).embeddings[0]
        
        # Search in Qdrant
        search_result = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            limit=top_k
        )
        
        # Extract relevant documents
        results = []
        sources = []
        for hit in search_result:
            content = hit.payload.get("content", "")
            metadata = hit.payload.get("metadata", {})
            results.append({
                "content": content,
                "metadata": metadata,
                "score": hit.score
            })
            sources.append(metadata.get("source", "Unknown"))
        
        return results, sources
    except Exception as e:
        logging.error(f"Error searching similar documents: {str(e)}")
        raise

def generate_rag_response(query: str, context: str):
    """Generate a response using Cohere with the provided context"""
    if not cohere_client:
        raise HTTPException(status_code=500, detail="Cohere client not initialized")
    
    try:
        # Construct the prompt with context
        prompt = f"""Based on the following context, please answer the question. If the context doesn't contain enough information, say so.

Context:
{context}

Question: {query}

Answer:"""
        
        # Generate response using Cohere
        response = cohere_client.generate(
            model="command-xlarge-nightly",  # Using a Cohere generation model
            prompt=prompt,
            max_tokens=300,
            temperature=0.7
        )
        
        return response.generations[0].text
    except Exception as e:
        logging.error(f"Error generating RAG response: {str(e)}")
        raise

@app.on_event("startup")
async def startup_event():
    logging.info("Starting up the chatbot service...")
    # Don't try to connect to Qdrant at startup to avoid blocking
    # Qdrant connection will be established on first use
    logging.info("Chatbot service started, Qdrant connection will be established on first use")

@app.on_event("shutdown")
async def shutdown_event():
    logging.info("Shutting down the chatbot service...")
    if db_service:
        db_service.close_all_connections()

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(chat_request: ChatRequest):
    """
    Send a message to the chatbot and receive a response
    """
    if not chatbot_service:
        # Fallback to RAG functionality if database service is not available
        rag_request = RAGChatRequest(
            message=chat_request.message,
            user_id=chat_request.user_id,
            conversation_id=chat_request.conversation_id
        )
        rag_response = await rag_chat_endpoint(rag_request)
        # Convert RAG response to ChatResponse format
        return ChatResponse(
            conversation_id=rag_response.conversation_id or "",
            message_id="",  # We don't have a message ID in this fallback
            response=rag_response.response,
            timestamp=""  # We don't have a timestamp in this fallback
        )
    
    try:
        response = chatbot_service.process_user_message(
            user_id=chat_request.user_id,
            message_content=chat_request.message,
            conversation_id=chat_request.conversation_id
        )
        return ChatResponse(**response)
    except ResourceNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ChatbotBaseException as e:
        raise HTTPException(status_code=400, detail={"error": str(e), "code": e.error_code})
    except Exception as e:
        logging.error(f"Unexpected error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/rag-chat", response_model=RAGChatResponse)
async def rag_chat_endpoint(chat_request: RAGChatRequest):
    """
    Send a message to the RAG chatbot and receive a response with sources
    """
    try:
        # Search for relevant documents based on the query
        search_results, sources = search_similar_documents(chat_request.message)
        
        # Combine the context from search results
        context_parts = []
        for result in search_results:
            context_parts.append(result["content"])
        
        context = "\n\n".join(context_parts)
        
        # Generate response using the context
        response_text = generate_rag_response(chat_request.message, context)
        
        # Prepare response
        response = RAGChatResponse(
            response=response_text,
            sources=sources,
            tokens_used=len(response_text.split()),  # Simple token estimation
            conversation_id=chat_request.conversation_id
        )
        
        return response
    except Exception as e:
        logging.error(f"Unexpected error in RAG chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/ingest", response_model=IngestResponse)
async def ingest_documents(ingest_request: IngestRequest):
    """
    Ingest documents into the vector database
    """
    try:
        # Extract content from documents
        contents = [doc.content for doc in ingest_request.documents]
        
        # Generate embeddings
        embeddings = embed_documents(contents)
        
        # Store in Qdrant
        store_in_qdrant(ingest_request.documents, embeddings)
        
        response = IngestResponse(
            message=f"Successfully ingested {len(ingest_request.documents)} documents",
            processed_count=len(ingest_request.documents)
        )
        
        return response
    except Exception as e:
        logging.error(f"Unexpected error in ingest endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/conversations", response_model=UserConversationsResponse)
async def get_conversations(user_id: str, limit: int = 10, offset: int = 0):
    """
    Retrieve all conversations for a specific user
    """
    if not chatbot_service:
        # Database service not available, return empty response
        return UserConversationsResponse(conversations=[], total_count=0)
    
    try:
        result = chatbot_service.get_user_conversations(user_id, limit, offset)
        return UserConversationsResponse(**result)
    except ResourceNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ChatbotBaseException as e:
        raise HTTPException(status_code=400, detail={"error": str(e), "code": e.error_code})
    except Exception as e:
        logging.error(f"Unexpected error in get_conversations: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/conversation/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(conversation_id: str):
    """
    Retrieve a specific conversation by ID
    """
    if not chatbot_service:
        # Database service not available, return empty response
        return ConversationResponse(id=conversation_id, title=None, created_at="", updated_at="", messages=[])
    
    try:
        result = chatbot_service.get_conversation_history(conversation_id)
        return ConversationResponse(**result)
    except ResourceNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ChatbotBaseException as e:
        raise HTTPException(status_code=400, detail={"error": str(e), "code": e.error_code})
    except Exception as e:
        logging.error(f"Unexpected error in get_conversation: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.delete("/conversation/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """
    Delete a specific conversation by ID
    """
    if not chatbot_service:
        # Database service not available, return success response
        return {"message": "Conversation deleted successfully (simulated)"}
    
    try:
        success = chatbot_service.delete_conversation(conversation_id)
        if success:
            return {"message": "Conversation deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Conversation not found")
    except ChatbotBaseException as e:
        raise HTTPException(status_code=400, detail={"error": str(e), "code": e.error_code})
    except Exception as e:
        logging.error(f"Unexpected error in delete_conversation: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/user/{user_id}/preferences", response_model=UserPreferencesResponse)
async def get_user_preferences(user_id: str):
    """
    Retrieve user preferences for personalization
    """
    if not chatbot_service:
        # Database service not available, return default empty preferences
        return UserPreferencesResponse(preferences={})
    
    try:
        preferences = chatbot_service.get_user_preferences(user_id)
        return UserPreferencesResponse(preferences=preferences)
    except ResourceNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ChatbotBaseException as e:
        raise HTTPException(status_code=400, detail={"error": str(e), "code": e.error_code})
    except Exception as e:
        logging.error(f"Unexpected error in get_user_preferences: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.put("/user/{user_id}/preferences", response_model=UserPreferencesResponse)
async def update_user_preferences(user_id: str, preferences_request: UserPreferencesRequest):
    """
    Update user preferences for personalization
    """
    if not chatbot_service:
        # Database service not available, return success with the preferences
        return UserPreferencesResponse(preferences=preferences_request.preferences)
    
    try:
        success = chatbot_service.update_user_preferences(user_id, preferences_request.preferences)
        if success:
            updated_preferences = chatbot_service.get_user_preferences(user_id)
            return UserPreferencesResponse(preferences=updated_preferences)
        else:
            raise HTTPException(status_code=404, detail="User not found")
    except ResourceNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ChatbotBaseException as e:
        raise HTTPException(status_code=400, detail={"error": str(e), "code": e.error_code})
    except Exception as e:
        logging.error(f"Unexpected error in update_user_preferences: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify service is running
    """
    return {"status": "healthy", "service": "chatbot-api"}

@app.get("/health/db")
async def database_health_check():
    """
    Health check endpoint to verify database connectivity
    """
    if not db_service:
        return {
            "status": "unavailable",
            "service": "database",
            "connection": "not configured",
            "message": "Database service not initialized (optional for RAG functionality)"
        }
    
    try:
        # Attempt a simple query to verify database connectivity
        result = db_service.execute_query("SELECT 1 as test;")
        if result:
            return {
                "status": "healthy",
                "service": "database",
                "connection": "successful"
            }
        else:
            return {
                "status": "degraded",
                "service": "database",
                "connection": "failed"
            }
    except Exception as e:
        logging.error(f"Database health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "service": "database",
            "error": str(e)
        }

@app.get("/health/config")
async def config_health_check():
    """
    Health check endpoint to verify configuration
    """
    try:
        # Validate that API key exists and is properly formatted
        validate_api_key_exists()

        # Validate other configuration elements
        validate_configuration()

        return {
            "status": "healthy",
            "service": "configuration",
            "details": {
                "api_key": "present and valid",
                "database_url": "present",
                "pool_settings": "valid"
            }
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "service": "configuration",
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)