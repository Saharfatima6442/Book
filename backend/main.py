from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
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
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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