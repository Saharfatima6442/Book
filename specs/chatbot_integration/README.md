# Chatbot Integration Feature

This feature integrates an AI-powered chatbot into the Physical AI & Humanoid Robotics book, providing readers with an interactive way to ask questions, find book topics, and maintain personalized session history.

## Components

### Backend API
- FastAPI-based server deployed on Hugging Face Spaces
- JWT-based authentication system
- Handles natural language processing and response generation
- Maintains user-specific conversation history
- Manages session creation and retrieval
- Can be integrated with various AI services (OpenAI, Hugging Face models, etc.)

### Frontend Component
- React-based chat interface integrated with Docusaurus
- Floating chat widget on all book pages
- User authentication and session management
- History view with session switching
- Context-aware responses based on current page content
- Quick-ask buttons for common book topics
- Responsive design for all devices

## Deployment

### Backend (Hugging Face Spaces)
1. Create a new Space with Docker SDK
2. Upload the backend files (`main.py`, `requirements.txt`, `Dockerfile`, etc.)
3. Set the `JWT_SECRET_KEY` environment variable in Space settings
4. The Space will automatically build and deploy the API
5. API will be available at `https://your-username-space-name.hf.space`

### Frontend (Docusaurus Site)
1. Add the chatbot component files to your Docusaurus `src/components/Chatbot` directory
2. Update the layout wrapper at `src/theme/Layout.tsx` to include the chatbot
3. Build and deploy your Docusaurus site as usual

## Configuration

To configure the frontend to connect to your backend:

1. Update the `CHATBOT_BACKEND_URL` in `src/theme/Layout.tsx` with your Space URL
2. Or set the `REACT_APP_CHATBOT_URL` environment variable during build

## Architecture

```
┌─────────────────┐    HTTP/HTTPS     ┌─────────────────┐
│   Docusaurus    │ ────────────────→ │  Chatbot API    │
│     Site        │                   │    Server       │
│                 │ ←───────────────  │                 │
│  ┌───────────┐  │   Responses       │ ┌─────────────┐ │
│  │ Chat UI   │  │                   │ │ AI Service  │ │
│  │ Component │  │ ←───────────────  │ │ Integration │ │
│  └───────────┘  │   AI Requests     │ └─────────────┘ │
└─────────────────┘                   └─────────────────┘
                  (Hugging Face Space)
```

## Features

- Natural language interface for book content
- User authentication with JWT tokens
- Personalized session history for each user
- Context-aware responses based on current page
- Quick-ask buttons for common book topics
- Session management with history view
- Responsive design
- Error handling and fallback responses
- User identification and logout
- Easy customization of appearance and behavior

## Customization

The chatbot can be customized:

- Visual appearance through CSS
- AI service integration (replace the mock service)
- Welcome messages and suggested questions
- Backend API endpoints and structure
- Authentication system (integrate with your user database)