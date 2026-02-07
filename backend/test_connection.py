import os
from dotenv import load_dotenv
import cohere
from qdrant_client import QdrantClient

# Load environment variables
load_dotenv()

def test_connections():
    print("Testing connections...")

    # Test Cohere
    try:
        cohere_api_key = os.getenv("COHERE_API_KEY")
        if not cohere_api_key:
            print("❌ COHERE_API_KEY not found in environment")
            return False

        cohere_client = cohere.Client(cohere_api_key)
        # Test by making a simple embed call
        response = cohere_client.embed(
            texts=["test"],
            model="embed-english-v3.0",
            input_type="search_query"
        )
        print("✅ Cohere connection successful")
    except Exception as e:
        print(f"❌ Cohere connection failed: {str(e)}")
        return False

    # Test Qdrant
    try:
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")

        if not qdrant_url or not qdrant_api_key:
            print("❌ QDRANT_URL or QDRANT_API_KEY not found in environment")
            return False

        qdrant_client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key
        )

        # Test by getting collections list
        collections = qdrant_client.get_collections()
        print("✅ Qdrant connection successful")
        print(f"   Available collections: {[col.name for col in collections.collections]}")
    except Exception as e:
        print(f"❌ Qdrant connection failed: {str(e)}")
        return False

    print("\n🎉 All connections successful!")
    return True

if __name__ == "__main__":
    test_connections()