# Deploying the Physical AI Book Chatbot

This guide explains how to deploy the chatbot backend on Hugging Face Spaces and integrate the frontend with your Docusaurus-based book site.

## Backend Deployment on Hugging Face Spaces

### Prerequisites
- A Hugging Face account
- Git installed on your system
- Basic knowledge of Docker and Python

### Step-by-Step Deployment

1. **Fork or clone the backend code**
   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd backend
   ```

2. **Create a new Space on Hugging Face**
   - Go to [huggingface.co/spaces](https://huggingface.co/spaces)
   - Click "Create new Space"
   - Choose:
     - **SDK**: Docker
     - **License**: Choose appropriate license
     - **Hardware**: CPU Basic (or higher if needed)

3. **Add the backend files to your Space repository**
   - Upload the following files to your Space:
     - `app.py` - Application entrypoint
     - `main.py` - FastAPI application
     - `requirements.txt` - Python dependencies
     - `Dockerfile` - Container configuration
     - `README.md` - Documentation
     - `space.yaml` - Space configuration

4. **Configure Environment Variables (Optional)**
   - In your Space settings, add environment variables if needed:
     - `JWT_SECRET_KEY`: Secret key for JWT token generation (default: "your-secret-key-change-in-production")
     - `CHATBOT_API_KEY`: For API access control (optional)

5. **Wait for Build and Deployment**
   - Hugging Face will automatically build and deploy your Space
   - Monitor the build logs in the "Logs" tab
   - Once built, your API will be accessible at `https://your-username-space-name.hf.space`

### API Endpoints
- `POST /auth` - Authenticate user and get JWT token
- `POST /chat` - Send messages to the chatbot
- `GET /health` - Health check endpoint
- `GET /chat/history` - Get user's conversation history
- `GET /chat/sessions` - Get list of user's sessions
- `DELETE /chat/session/{session_id}` - Delete a specific session

### Testing the Backend
```bash
# Authenticate user
curl -X POST "https://your-username-space-name.hf.space/auth" \
  -H "Content-Type: application/json" \
  -d '{"username": "testuser", "password": "testpass"}'

# Send a message
curl -X POST "https://your-username-space-name.hf.space/chat" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -d '{"message": "What is Physical AI?", "context": "Introduction chapter", "user_id": "user_testuser"}'

# Get user history
curl -X GET "https://your-username-space-name.hf.space/chat/history" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

## Frontend Integration

### Prerequisites
- A Docusaurus-based documentation site
- Node.js and npm installed

### Integration Steps

1. **Install the chatbot component**
   - Copy the `src/components/Chatbot` directory to your Docusaurus `src/components` directory
   - Ensure you have the following files:
     - `src/components/Chatbot/Chatbot.js`
     - `src/components/Chatbot/Chatbot.css`
     - `src/components/Chatbot/index.js`

2. **Configure the backend URL**
   - Update the `CHATBOT_BACKEND_URL` in `src/theme/Layout.tsx` with your Hugging Face Space URL

3. **Customize the appearance (optional)**
   - Modify `src/components/Chatbot/Chatbot.css` to match your site's styling
   - Adjust colors, sizes, and positioning as needed

4. **Build and deploy your Docusaurus site**
   ```bash
   npm run build
   npm run serve  # to test locally
   ```

## Architecture Overview

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

### Authentication
- JWT-based authentication system
- User sessions stored in browser localStorage
- Login/logout functionality

### Session Management
- Automatic session creation for each conversation
- Session history persisted per user
- Ability to view and switch between conversation sessions

### Book-Specific Knowledge
- Responses tailored to Physical AI & Humanoid Robotics concepts
- Context awareness of current page content
- Topic suggestions related to book chapters

### UI Features
- Floating chat interface on all pages
- Quick-ask buttons for common topics
- Responsive design for all devices
- Timestamps on messages
- User identification in header

## Configuration Options

### Frontend Configuration
The chatbot component accepts the following props:
- `backendUrl`: The URL of your deployed backend API

### Backend Configuration
Environment variables for the backend:
- `JWT_SECRET_KEY`: Secret key for JWT token generation (required for production)
- `CHATBOT_API_KEY`: API key for authentication (optional)
- `PORT`: Port number (default: 8000, usually set by Hugging Face)

## Security Considerations

- Use HTTPS for all API communications
- Implement rate limiting in production
- Use a strong secret key for JWT token generation
- Validate and sanitize all user inputs
- Consider using a CDN for additional security
- For production, implement proper user authentication with database verification

## Performance Notes

- Hugging Face Spaces may hibernate when not in use, causing initial request delays
- For production use, consider a dedicated hosting solution if you need guaranteed uptime
- Session data is stored in-memory and may reset when the Space hibernates
- For persistent conversations, implement a database backend instead of in-memory storage
- JWT tokens are stored in localStorage and persist between sessions

## Troubleshooting

### Common Issues
1. **CORS errors**: The backend already includes CORS middleware, but if needed, ensure your Space settings allow requests from your site domain.

2. **Backend not responding**: Check the Space logs in the Hugging Face UI for any errors during startup.

3. **Chat not appearing**: Ensure that the Layout wrapper is properly set up and the component is imported correctly.

4. **Authentication issues**: Check that JWT_SECRET_KEY is set in your Space environment variables for production use.

### Checking Backend Status
- Visit `https://your-username-space-name.hf.space/health` to check the API status
- Check `https://your-username-space-name.hf.space/docs` for API documentation

## Scaling Considerations

For higher traffic:
1. Upgrade your Hugging Face Space hardware
2. Implement database storage for persistent sessions
3. Add caching for common queries
4. Add a load balancer if needed
5. Consider moving to a dedicated cloud provider for higher scale

## Updating the Chatbot

### Frontend Updates
1. Modify the React component as needed
2. Rebuild and redeploy your Docusaurus site

### Backend Updates
1. Update the FastAPI application
2. Push changes to your Space repository
3. Hugging Face will automatically rebuild the Space