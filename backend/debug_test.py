import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Print to a file for debugging
with open('debug_output.txt', 'w') as f:
    f.write("Environment variables check:\n")
    f.write(f"COHERE_API_KEY exists: {'COHERE_API_KEY' in os.environ}\n")
    f.write(f"QDRANT_URL exists: {'QDRANT_URL' in os.environ}\n")
    f.write(f"QDRANT_API_KEY exists: {'QDRANT_API_KEY' in os.environ}\n")

    f.write(f"\nActual values:\n")
    cohere_key = os.getenv('COHERE_API_KEY', 'NOT SET')
    qdrant_url = os.getenv('QDRANT_URL', 'NOT SET')
    qdrant_key = os.getenv('QDRANT_API_KEY', 'NOT SET')
    f.write(f"COHERE_API_KEY: {cohere_key[:10] if cohere_key != 'NOT SET' else cohere_key}...\n")
    f.write(f"QDRANT_URL: {qdrant_url}\n")
    f.write(f"QDRANT_API_KEY: {qdrant_key[:10] if qdrant_key != 'NOT SET' else qdrant_key}...\n")

    # Test importing required libraries
    try:
        import cohere
        f.write("\n✅ Cohere library imported successfully\n")
    except ImportError as e:
        f.write(f"\n❌ Failed to import cohere: {e}\n")

    try:
        from qdrant_client import QdrantClient
        f.write("✅ Qdrant client library imported successfully\n")
    except ImportError as e:
        f.write(f"❌ Failed to import qdrant_client: {e}\n")

    try:
        import uvicorn
        f.write("✅ Uvicorn library imported successfully\n")
    except ImportError as e:
        f.write(f"❌ Failed to import uvicorn: {e}\n")

print("Debug output written to debug_output.txt")