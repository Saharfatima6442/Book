# Test to verify the system is working

## Services Status

✅ **Qdrant** - Running on port 6333
✅ **Backend** - Running on port 8000  
✅ **Frontend** - Built successfully (needs manual verification)

## How to Test

1. **Qdrant**: Visit http://localhost:6333/dashboard to verify Qdrant is running
2. **Backend**: Visit http://localhost:8000/health to verify backend is running
3. **Frontend**: The frontend container is built but may need manual verification

## API Endpoints Available

- Health check: `GET http://localhost:8000/health`
- RAG Chat: `POST http://localhost:8000/rag-chat`
- Ingest: `POST http://localhost:8000/ingest`
- Conversations: `GET http://localhost:8000/conversations`

## Troubleshooting

If the frontend container stops immediately after starting:
1. Check the nginx configuration for syntax errors
2. Verify that the build process completed successfully
3. Make sure the Docusaurus site built without errors

## Next Steps

1. Verify that the backend can connect to Cohere and Qdrant
2. Test the RAG functionality by ingesting some content
3. Verify that the chat functionality works