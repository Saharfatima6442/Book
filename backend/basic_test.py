import os
import sys
from dotenv import load_dotenv

print("Starting basic test...")

# Load environment variables
load_dotenv()
print("Environment loaded")

# Import required libraries
try:
    import cohere
    print("Cohere imported successfully")
except ImportError as e:
    print(f"Failed to import cohere: {e}")
    sys.exit(1)

try:
    from qdrant_client import QdrantClient
    print("Qdrant client imported successfully")
except ImportError as e:
    print(f"Failed to import qdrant_client: {e}")
    sys.exit(1)

try:
    from fastapi import FastAPI
    print("FastAPI imported successfully")
except ImportError as e:
    print(f"Failed to import FastAPI: {e}")
    sys.exit(1)

print("All imports successful!")