# Troubleshooting Guide: Backend and Chatbot Issues

## Problem Description
The backend was not running and the chatbot was not working due to missing environment variables and configuration issues.

## Root Causes Identified
1. Missing environment variables in the backend `.env` file
2. Required variables like `NEON_DB_URL` and `NEON_API_KEY` were not set
3. The Docker Compose setup was not working due to Docker not being installed

## Solution Implemented

### 1. Backend Environment Setup
Created a proper `.env` file in the backend directory with required variables:
- `NEON_DB_URL`: PostgreSQL connection string for Neon database
- `NEON_API_KEY`: API key for Neon database access
- `DB_POOL_SIZE`: Database connection pool size (default: 10)
- `DB_POOL_OVERFLOW`: Database connection pool overflow (default: 20)
- `LOG_LEVEL`: Logging level (default: INFO)
- `COHERE_API_KEY`: API key for Cohere service
- `QDRANT_URL`: URL for Qdrant vector database
- `QDRANT_API_KEY`: API key for Qdrant vector database

### 2. Backend Startup
- Installed required dependencies using `pip install -r requirements.txt`
- Started the backend server using `python -m uvicorn main:app --host 0.0.0.0 --port 8000`
- Verified the server is running by testing the `/health` endpoint

### 3. Frontend Startup
- Started the frontend using `npm start` in the AI-Book directory
- Verified the frontend is running on `http://localhost:3000`

## Current Status
✅ Backend running on `http://localhost:8000`
✅ Frontend running on `http://localhost:3000`
✅ Health check endpoint returning 200 OK
✅ Configuration health check passing

## Verification Steps
1. Backend health: `curl http://localhost:8000/health`
2. Config health: `curl http://localhost:8000/health/config`
3. Frontend accessibility: Navigate to `http://localhost:3000`

## Additional Notes
- The Docker Compose setup requires Docker Desktop to be installed
- If Docker is available, run `docker-compose up -d` to start all services
- The database service needs to be accessible for full functionality
- For production deployment, configure proper CORS settings in the backend