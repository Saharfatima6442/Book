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
