# Troubleshooting Guide for Docker Setup

## Common Issues and Solutions

### 1. Docker Daemon Not Running
If you get an error like:
```
failed to connect to the docker API at npipe:////./pipe/dockerDesktopLinuxEngine
```

**Solution:**
- Make sure Docker Desktop is running on your system
- On Windows, look for the Docker whale icon in your system tray
- If it's not running, start Docker Desktop from the Start menu

### 2. Insufficient Resources
If containers fail to start due to memory or CPU limitations:

**Solution:**
- Open Docker Desktop settings
- Go to Resources
- Increase memory allocation (recommended: 4GB+)
- Increase CPU cores if needed

### 3. Port Already in Use
If you get port binding errors:

**Solution:**
- Check if ports 3000, 8000, or 6333 are already in use
- Stop other applications using these ports
- Or modify the docker-compose.yml to use different ports

### 4. Environment Variables Not Loaded
If the backend fails to connect to Cohere or Qdrant:

**Solution:**
- Ensure your .env file in the backend directory has the correct API keys
- Verify that COHERE_API_KEY, QDRANT_URL, and QDRANT_API_KEY are set correctly

### 5. Build Failures
If Docker build fails:

**Solution:**
- Check the error logs for specific issues
- Ensure you have internet connection for downloading dependencies
- Try running `docker-compose build --no-cache` to rebuild from scratch

## How to Check if Everything is Working

1. After running `docker-compose up`, check if these services are running:
   - Frontend: http://localhost:3000
   - Backend: http://localhost:8000/health
   - Qdrant: http://localhost:6333/dashboard

2. You can also run `docker ps` to see running containers

## Useful Commands

- `docker-compose up --build` - Build and start all services
- `docker-compose down` - Stop all services
- `docker-compose logs` - View logs from all services
- `docker-compose logs backend` - View logs from backend service only
- `docker ps` - List running containers
- `docker images` - List downloaded images

## Restarting the Application

If you need to restart after making changes:
1. Press Ctrl+C to stop the running services
2. Run `docker-compose down` to stop and remove containers
3. Run `docker-compose up --build` to rebuild and restart

## Verifying the RAG Functionality

Once the system is running:
1. Visit http://localhost:3000
2. Use the chatbot to ask questions about Physical AI or Humanoid Robotics
3. The bot should respond with information based on the book content
4. Check the backend logs to confirm the RAG functionality is working