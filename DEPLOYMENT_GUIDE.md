# Deployment Guide for AI Book RAG Chatbot

## Overview
This guide provides instructions for deploying the RAG (Retrieval-Augmented Generation) chatbot for the Physical AI & Humanoid Robotics book. The system consists of:
- Backend API (FastAPI) with Cohere and Qdrant integration
- Frontend Docusaurus site with embedded chatbot
- Vector database (Qdrant) for document storage

## Prerequisites
- Docker and Docker Compose
- Node.js (for frontend development)
- Python 3.11+ (for backend development)
- Cohere API key
- Qdrant Cloud account (or self-hosted Qdrant instance)

## Environment Variables
Create a `.env` file in the `backend` directory with the following variables:

```bash
COHERE_API_KEY=your_cohere_api_key
QDRANT_URL=your_qdrant_cloud_url
QDRANT_API_KEY=your_qdrant_api_key
```

## Local Development

### Backend (FastAPI)
1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Start the backend server:
   ```bash
   python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

### Frontend (Docusaurus)
1. Navigate to the AI-Book directory:
   ```bash
   cd AI-Book
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm start
   ```

## Production Deployment with Docker

### Using Docker Compose
1. Ensure you have your `.env` file configured with API keys
2. Run the following command from the root directory:
   ```bash
   docker-compose up -d
   ```

This will start all services:
- Backend API on http://localhost:8000
- Frontend on http://localhost:3000
- Qdrant on http://localhost:6333

### Individual Service Deployment

#### Backend API
1. Build the backend image:
   ```bash
   cd backend
   docker build -t ai-book-backend .
   ```

2. Run the backend container:
   ```bash
   docker run -d -p 8000:8000 --env-file .env ai-book-backend
   ```

#### Frontend
1. Build the frontend:
   ```bash
   cd AI-Book
   npm run build
   ```

2. Serve the built site using a web server or CDN

## Content Ingestion
To populate the vector database with book content:

1. Make sure the backend is running
2. Run the ingestion script:
   ```bash
   cd backend
   python ingest_content.py
   ```

## Testing the Deployment
1. Check the backend health endpoint: `GET http://localhost:8000/health`
2. Test the chat endpoint: `POST http://localhost:8000/chat`
3. Verify the frontend chatbot connects to the backend

## Cloud Deployment Options

### Backend (Recommended: Railway, Render, or Vercel)
1. Create a new project
2. Connect your GitHub repository
3. Set environment variables
4. Deploy with the provided Dockerfile

### Frontend (Recommended: Vercel, Netlify, or GitHub Pages)
1. Build the site: `npm run build`
2. Deploy the `build` directory to your hosting platform
3. Configure the backend URL in environment variables

### Vector Database
- Use Qdrant Cloud for managed service
- Or self-host using the Qdrant Docker image

## Configuration
- Update the backend URL in the frontend if deploying to different domains
- Configure CORS settings in `main.py` for production domains
- Adjust rate limiting as needed for your use case

## Troubleshooting
- Check logs: `docker-compose logs`
- Verify environment variables are set correctly
- Ensure the Qdrant instance is accessible from the backend
- Check that Cohere API key has proper permissions

## Scaling
- Use a load balancer for multiple backend instances
- Consider Redis for session management in production
- Implement caching for frequently accessed responses