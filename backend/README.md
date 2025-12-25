# Physical AI Book Chatbot API

This is the backend API for the Physical AI & Humanoid Robotics book chatbot. It provides natural language processing capabilities to answer questions about the book content.

## Features

- Natural language interface for book content
- Context-aware responses
- Conversation history management
- Session-based interactions
- RESTful API design

## Deployment on Hugging Face Spaces

This application is designed for deployment on Hugging Face Spaces using the Docker runtime.

### Steps to Deploy on Hugging Face Spaces:

1. **Create a new Space**:
   - Go to [huggingface.co/spaces](https://huggingface.co/spaces)
   - Click "Create new Space"
   - Choose:
     - **SDK**: Docker
     - **License**: Choose appropriate license
     - **Hardware**: CPU Basic or higher (depending on needs)

2. **Repository Structure**:
   ```
   ├── app.py          # Application entrypoint for HF Spaces
   ├── main.py         # FastAPI application
   ├── requirements.txt # Python dependencies
   ├── Dockerfile      # Container configuration
   └── README.md       # This file
   ```

3. **Configure Environment**:
   - Add the required environment variables in the Space settings if needed
   - For production, add `CHATBOT_API_KEY` for authentication

4. **Wait for Deployment**:
   - Hugging Face will build and deploy your Space automatically
   - The API will be available at `https://your-username-space-name.hf.space`

### API Endpoints

- `GET /` - Root endpoint with API information
- `GET /health` - Health check
- `POST /chat` - Chat endpoint
  - Request body: `{"message": "your question", "context": "optional page context", "session_id": "optional session id"}`
  - Response: `{"response": "AI response", "session_id": "session id", "timestamp": timestamp}`
- `GET /chat/history/{session_id}` - Get conversation history
- `DELETE /chat/session/{session_id}` - Clear session

### API Usage Example

```bash
curl -X POST "https://your-username-space-name.hf.space/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "What is Physical AI?", "context": "Introduction chapter"}'
```

## Development

To run the application locally:

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

## Architecture

- FastAPI backend with Python
- Docker containerization for deployment
- In-memory session storage (for stateless Spaces - persistent DB in production)
- Mock AI service (integrate with OpenAI, Hugging Face models, or other NLP services in production)

## Note on Hugging Face Spaces Limitations

- Session data is stored in-memory and will reset when the Space hibernates
- For production use, integrate with a persistent database
- For cost efficiency, Spaces may hibernate when not in use