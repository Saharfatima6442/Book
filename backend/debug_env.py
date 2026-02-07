import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("Environment variables check:")
print(f"COHERE_API_KEY exists: {'COHERE_API_KEY' in os.environ}")
print(f"QDRANT_URL exists: {'QDRANT_URL' in os.environ}")
print(f"QDRANT_API_KEY exists: {'QDRANT_API_KEY' in os.environ}")

print("\nActual values:")
print(f"COHERE_API_KEY: {os.getenv('COHERE_API_KEY', 'NOT SET')[:10]}...")
print(f"QDRANT_URL: {os.getenv('QDRANT_URL', 'NOT SET')}")
print(f"QDRANT_API_KEY: {os.getenv('QDRANT_API_KEY', 'NOT SET')[:10]}...")

# Test importing required libraries
try:
    import cohere
    print("\n✅ Cohere library imported successfully")
except ImportError as e:
    print(f"\n❌ Failed to import cohere: {e}")

try:
    from qdrant_client import QdrantClient
    print("✅ Qdrant client library imported successfully")
except ImportError as e:
    print(f"❌ Failed to import qdrant_client: {e}")

try:
    import uvicorn
    print("✅ Uvicorn library imported successfully")
except ImportError as e:
    print(f"❌ Failed to import uvicorn: {e}")