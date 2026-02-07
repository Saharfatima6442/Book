from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
<<<<<<< HEAD
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
=======
import os
import logging
import asyncio
import uuid
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
import json
import jwt  # Requires: pip install pyjwt

# For this example, we'll use a mock AI service
# In a real implementation, you would integrate with OpenAI, Hugging Face, or similar
class MockAIService:
    @staticmethod
    def get_response(user_input: str, context: Optional[str] = None) -> str:
        """Mock AI service that simulates generating responses"""
        user_input_lower = user_input.lower()
        
        # Enhanced responses based on book content
        if any(term in user_input_lower for term in ["physical ai", "physical artificial intelligence"]):
            return (
                "Physical AI represents a paradigm shift from traditional artificial intelligence by integrating "
                "computational systems with the physical world. Unlike conventional AI that operates primarily on "
                "abstract data representations, Physical AI systems must perceive, reason about, and interact with "
                "tangible environments subject to the laws of physics, materials science, and mechanics. "
                "Key concepts include embodied cognition, sensory-motor integration, and physics-aware systems."
            )
        elif any(term in user_input_lower for term in ["ros", "robot operating system", "ros2"]):
            return (
                "ROS 2 (Robot Operating System 2) serves as the nervous system of your humanoid robot. "
                "It provides the middleware infrastructure connecting all components of your robot system. "
                "Key concepts include:\n\n"
                "- Nodes: Individual processes performing computation\n"
                "- Topics: Asynchronous publisher-subscriber communication\n"
                "- Services: Synchronous request-response communication\n"
                "- Actions: Long-running tasks with feedback"
            )
        elif any(term in user_input_lower for term in ["navigation", "path planning", "slam"]):
            return (
                "Advanced navigation for humanoid robots involves several key components:\n\n"
                "1. Simultaneous Localization and Mapping (SLAM) for understanding the environment\n"
                "2. Hierarchical path planning with global and local planners\n"
                "3. Dynamic obstacle avoidance for moving objects\n"
                "4. Social navigation considering human interactions\n\n"
                "These systems integrate multiple sensor modalities like cameras, LiDAR, and IMUs "
                "to create comprehensive environmental understanding."
            )
        elif any(term in user_input_lower for term in ["gazebo", "unity", "simulation", "digital twin"]):
            return (
                "Digital twin environments like Gazebo and Unity are essential for developing and testing "
                "Physical AI systems:\n\n"
                "- Gazebo: Robot-centric simulation with realistic physics\n"
                "- Unity: Game-engine approach with high-quality visuals\n"
                "- NVIDIA Isaac Sim: Advanced simulation for AI robotics\n\n"
                "Simulation allows testing of algorithms in safe, repeatable environments "
                "before deployment on physical robots."
            )
        elif any(term in user_input_lower for term in ["locomotion", "humanoid", "walking", "movement"]):
            return (
                "Humanoid locomotion is one of the most challenging aspects of Physical AI:\n\n"
                "1. Balance control using feedback from IMUs and force sensors\n"
                "2. Walking pattern generation using techniques like ZMP (Zero Moment Point)\n"
                "3. Terrain adaptation for different surfaces\n"
                "4. Recovery strategies for disturbances\n\n"
                "Controllers often use CPGs (Central Pattern Generators) or RL (Reinforcement Learning)."
            )
        elif any(term in user_input_lower for term in ["ai planning", "decision making", "planning"]):
            return (
                "AI planning and decision-making in Physical AI systems involve multiple levels of abstraction:\n\n"
                "1. Task planning: High-level goal decomposition\n"
                "2. Motion planning: Path finding and obstacle avoidance\n"
                "3. Control: Low-level actuator commands\n\n"
                "These systems must operate under uncertainty and adapt to environmental changes."
            )
        else:
            # Default response that guides user to book content
            return (
                f"I understand you're asking about '{user_input}'. "
                f"In the context of Physical AI and Humanoid Robotics, this topic would typically involve "
                f"considerations of perception, decision-making, and safe interaction with the physical world. "
                f"Based on the book content, relevant chapters might include:\n\n"
                f"- Chapter 1: Introduction to Physical AI\n"
                f"- Chapter 2: The Robotic Nervous System (ROS2)\n"
                f"- Chapter 3: The Digital Twin (Gazebo/Unity)\n"
                f"- Chapter 4: The AI Robot Brain (NVIDIA Isaac)\n"
                f"- Chapter 5: Vision-Language-Action (VLA)\n"
                f"- Chapter 6: Humanoid Locomotion and Interaction\n"
                f"- Chapter 7: Conversational Multimodal Robotics\n"
                f"- Chapter 8: Edge AI Deployment\n"
                f"- Chapter 9: Advanced Perception and Navigation\n"
                f"- Chapter 10: AI Planning and Decision Making\n"
                f"- Chapter 11: Safety and Ethics\n"
                f"- Chapter 12: Capstone Project\n\n"
                f"Could you be more specific about what aspect you'd like to know more about?"
            )

# Models
class ChatMessage(BaseModel):
    content: str
    role: str  # "user" or "assistant"
    timestamp: float

class ChatRequest(BaseModel):
    message: str
    context: Optional[str] = None
    user_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    timestamp: float

class UserSession(BaseModel):
    session_id: str
    user_id: str
    created_at: str
    messages: List[ChatMessage]

class HealthResponse(BaseModel):
    status: str
    version: str

class AuthRequest(BaseModel):
    username: str
    password: str

class AuthResponse(BaseModel):
    token: str
    user_id: str

# JWT configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Initialize FastAPI app
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logging.info("Physical AI Book Chatbot API starting up...")
    logging.info(f"Running on Hugging Face Spaces environment: {os.environ.get('SPACE_ID', 'Not in Space')}")
    yield
    # Shutdown
    logging.info("Physical AI Book Chatbot API shutting down...")

app = FastAPI(
    title="Physical AI Book Chatbot API",
    description="API for the Physical AI & Humanoid Robotics book chatbot, deployed on Hugging Face Spaces",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware (important for Hugging Face Spaces)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For Hugging Face Spaces, more restrictive in production
>>>>>>> 21ec9ac21bae15c0a6c4f5fd1b014840530cb68f
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

<<<<<<< HEAD
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
=======
# In-memory storage for session data (use a proper DB in production)
# For Hugging Face Spaces, this will reset when the Space hibernates
user_sessions: Dict[str, List[UserSession]] = {}

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Could not validate credentials")
        return user_id
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="Could not validate credentials")

def get_current_user(request: Request):
    token = request.headers.get("Authorization")
    if token and token.startswith("Bearer "):
        token = token[7:]
        return verify_token(token)
    else:
        # For demo purposes, create a temporary user ID if no token provided
        # In production, you would require authentication
        return str(uuid.uuid4())

@app.get("/", response_class=str)
async def root():
    """Root endpoint with API information"""
    return """
    <h1>Physical AI Book Chatbot API</h1>
    <p>This is the backend API for the Physical AI & Humanoid Robotics book chatbot.</p>
    <p>Use the /chat endpoint to interact with the chatbot.</p>
    <p>API documentation available at: <a href="/docs">/docs</a></p>
    """

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(status="healthy", version="1.0.0")

@app.post("/auth", response_model=AuthResponse)
async def authenticate_user(auth_request: AuthRequest):
    """Authenticate user and return a JWT token"""
    # In a real implementation, verify credentials against a database
    # For this demo, we'll just create a token for any user
    user_id = f"user_{auth_request.username}"
    token = create_access_token(data={"sub": user_id})
    return AuthResponse(token=token, user_id=user_id)

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Main chat endpoint that processes user messages and returns AI responses"""
    try:
        # Get or create user ID
        user_id = request.user_id or str(uuid.uuid4())
        
        # Create or get user session
        if user_id not in user_sessions:
            user_sessions[user_id] = []
        
        # Create a new session or get the active one
        if not user_sessions[user_id] or len(user_sessions[user_id][-1].messages) > 20:
            # Create a new session after 20 messages or if no active session
            new_session = UserSession(
                session_id=str(uuid.uuid4()),
                user_id=user_id,
                created_at=datetime.now().isoformat(),
                messages=[]
            )
            user_sessions[user_id].append(new_session)
        
        current_session = user_sessions[user_id][-1]
        
        # Add user message to session
        user_msg = ChatMessage(
            content=request.message,
            role="user",
            timestamp=datetime.now().timestamp()
        )
        current_session.messages.append(user_msg)
        
        # Get AI response
        ai_response = MockAIService.get_response(request.message, request.context)
        
        # Add AI response to session
        ai_msg = ChatMessage(
            content=ai_response,
            role="assistant",
            timestamp=datetime.now().timestamp()
        )
        current_session.messages.append(ai_msg)
        
        return ChatResponse(
            response=ai_response,
            session_id=current_session.session_id,
            timestamp=datetime.now().timestamp()
        )
    except Exception as e:
        logging.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/chat/history")
async def get_user_history(request: Request):
    """Retrieve conversation history for the current user"""
    try:
        user_id = get_current_user(request)
        
        if user_id not in user_sessions or not user_sessions[user_id]:
            return {"user_id": user_id, "sessions": []}
        
        # Return all sessions for the user
        user_sessions_data = []
        for session in user_sessions[user_id]:
            user_sessions_data.append({
                "session_id": session.session_id,
                "created_at": session.created_at,
                "message_count": len(session.messages),
                "messages": [
                    {
                        "role": msg.role,
                        "content": msg.content,
                        "timestamp": msg.timestamp
                    }
                    for msg in session.messages
                ]
            })
        
        return {"user_id": user_id, "sessions": user_sessions_data}
    except Exception as e:
        logging.error(f"Error retrieving history: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/chat/sessions")
async def get_user_sessions_list(request: Request):
    """Get a list of user's sessions (without full history)"""
    try:
        user_id = get_current_user(request)
        
        if user_id not in user_sessions or not user_sessions[user_id]:
            return {"user_id": user_id, "sessions": []}
        
        sessions_list = []
        for session in user_sessions[user_id]:
            sessions_list.append({
                "session_id": session.session_id,
                "created_at": session.created_at,
                "message_count": len(session.messages)
            })
        
        return {"user_id": user_id, "sessions": sessions_list}
    except Exception as e:
        logging.error(f"Error retrieving sessions list: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.delete("/chat/session/{session_id}")
async def delete_session(request: Request, session_id: str):
    """Delete a specific session for the current user"""
    try:
        user_id = get_current_user(request)
        
        if user_id not in user_sessions:
            raise HTTPException(status_code=404, detail="No sessions found for user")
        
        # Find and delete the session
        user_sessions[user_id] = [
            session for session in user_sessions[user_id] 
            if session.session_id != session_id
        ]
        
        return {"message": f"Session {session_id} deleted for user {user_id}"}
    except Exception as e:
        logging.error(f"Error deleting session: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

# For Hugging Face Spaces, we need an app variable
app_instance = app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
>>>>>>> 21ec9ac21bae15c0a6c4f5fd1b014840530cb68f
