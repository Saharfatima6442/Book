<<<<<<< HEAD
# RAG Chatbot for AI Book

## Objective
Create a Retrieval-Augmented chatbot embedded in a Docusaurus book that answers
questions based on book content or user-selected text only.

## LLM
- Qwen (via API key)
- Used through Claude Code execution

## Embeddings
- Provider: Cohere
- Model: embed-english-v3.0

## Vector Database
- Qdrant Cloud (Free Tier)
- Stores chunked book content

## Backend
- FastAPI
- REST endpoints:
  - /ingest
  - /chat
  - /chat/selected

## Features
- Semantic search
- Contextual answers
- Selected-text-only answers
- No hallucinations outside retrieved context
=======
# Chatbot Integration Feature Specification

## Feature Overview
This feature involves integrating an AI-powered chatbot into the Physical AI & Humanoid Robotics book. The chatbot will provide interactive assistance to readers, answering questions about the content and helping them understand complex concepts.

## Requirements
- Chatbot should understand and respond to questions about the book content
- Support for natural language queries about Physical AI, robotics, and related topics
- Responsive UI component that integrates with Docusaurus
- Separate backend API for processing chat queries
- Context-aware responses that consider the current page content
- Support for follow-up questions maintaining conversation context
- Integration with AI models for natural language processing

## Scope
- Frontend chatbot UI component for Docusaurus
- Backend API server to handle chat queries
- Natural language processing capabilities
- Contextual awareness of book content
- Conversation history management
- Integration with external AI services (e.g., OpenAI, Hugging Face)
- Authentication for usage tracking (optional)

## Out of Scope
- Voice input/output capabilities
- Video chat functionality
- Real-time human operator handoff
- Advanced machine learning model training

## Success Criteria
- Functional chatbot UI integrated into book pages
- Accurate responses to questions about book content
- Responsive and user-friendly interface
- Proper error handling and fallback responses
- Clear documentation for deployment and customization
- Secure API with appropriate rate limiting
>>>>>>> 21ec9ac21bae15c0a6c4f5fd1b014840530cb68f
