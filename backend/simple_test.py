import os
import sys
sys.stdout.reconfigure(encoding='utf-8')

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Test just the Cohere connection first
try:
    import cohere
    cohere_client = cohere.Client(os.getenv("COHERE_API_KEY"))
    print("Cohere client initialized successfully")
except Exception as e:
    print(f"Failed to initialize Cohere client: {e}")

# Test just the Qdrant connection
try:
    from qdrant_client import QdrantClient
    qdrant_client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY")
    )
    print(" Qdrant client initialized successfully")
    # Try to connect and get collections
    collections = qdrant_client.get_collections()
    print(f"Connected to Qdrant, available collections: {[col.name for col in collections.collections]}")
except Exception as e:
    print(f"Failed to initialize Qdrant client: {e}")