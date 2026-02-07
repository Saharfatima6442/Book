import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("Environment loaded")

# Initialize clients with error handling
try:
    import cohere
    cohere_client = cohere.Client(os.getenv("COHERE_API_KEY"))
    print("✅ Cohere client initialized")
except Exception as e:
    print(f"❌ Failed to initialize Cohere client: {e}")
    cohere_client = None

# Store Qdrant configuration but don't connect at startup to avoid blocking
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
qdrant_client = None  # Initialize as None, connect when needed
COLLECTION_NAME = "ai_book_content"

print("Qdrant config stored")

def test_function():
    global qdrant_client
    print("Global statement works")

test_function()
print("Global statement test passed")