# Physical AI & Humanoid Robotics Book with RAG Chatbot

This repository contains the source code and content for the "Physical AI & Humanoid Robotics: A Comprehensive Guide to Embodied Intelligence" book, along with an integrated RAG (Retrieval-Augmented Generation) chatbot.

## Project Structure

- `AI-Book/` - Docusaurus-based website for the book
- `backend/` - FastAPI backend for the RAG chatbot
- `specs/` - Specification documents

## Features

- Interactive AI assistant that answers questions about Physical AI & Humanoid Robotics
- RAG (Retrieval-Augmented Generation) functionality for accurate responses based on book content
- Ability to ask questions about selected text on the page
- Vector database storage for efficient semantic search
- Floating chatbot widget accessible from any page

## Technologies Used

- **Frontend**: Docusaurus, React
- **Backend**: FastAPI, Python
- **AI/ML**: Cohere for embeddings and language model
- **Vector Database**: Qdrant
- **Deployment**: Docker, Docker Compose

## Quick Start

### Prerequisites

- Node.js (v18+)
- Python (v3.11+)
- Docker and Docker Compose
- Cohere API key
- Qdrant Cloud account or self-hosted instance

### Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. Set up environment variables:
   ```bash
   # In the backend directory
   cd backend
   cp .env .env.example  # Then edit with your actual keys
   ```

3. Using Docker Compose (recommended):
   ```bash
   docker-compose up -d
   ```

4. Access the applications:
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - Qdrant dashboard: http://localhost:6333

### Manual Setup

#### Backend
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

#### Frontend
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

## Content Ingestion

To populate the vector database with book content:

1. Ensure the backend is running
2. Run the ingestion script:
   ```bash
   cd backend
   python ingest_content.py
   ```

## API Endpoints

- `GET /health` - Health check
- `POST /chat` - Chat with the AI assistant
- `POST /ingest` - Ingest documents into the vector database

## Deployment

For detailed deployment instructions, see [DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md).

## Architecture

The system follows a RAG (Retrieval-Augmented Generation) pattern:
1. Book content is chunked and stored in a vector database (Qdrant)
2. When a user asks a question, relevant content is retrieved using semantic search
3. The retrieved context is provided to the LLM (via Cohere) to generate accurate responses
4. The frontend provides an intuitive interface for users to interact with the chatbot

## Contributing

Contributions to improve the book content or the chatbot functionality are welcome. Please follow the standard fork-and-pull request workflow.

## License

[Specify your license here]

## Support

For issues or questions, please open an issue in this repository.