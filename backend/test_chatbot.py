import requests
import json
import time

def test_chatbot():
    """Test the chatbot functionality"""
    backend_url = "http://localhost:8000"

    print("Testing the RAG Chatbot API...")
    print(f"Backend URL: {backend_url}")
    print("-" * 50)

    # Test 1: Health check
    print("1. Testing health endpoint...")
    try:
        response = requests.get(f"{backend_url}/health")
        if response.status_code == 200:
            health_data = response.json()
            print(f"   ✓ Health check passed: {health_data}")
        else:
            print(f"   ✗ Health check failed with status: {response.status_code}")
    except Exception as e:
        print(f"   ✗ Health check failed with error: {str(e)}")

    # Test 2: Basic chat functionality
    print("\n2. Testing basic chat functionality...")
    try:
        chat_request = {
            "message": "What is Physical AI?",
            "selected_text": None
        }

        response = requests.post(f"{backend_url}/chat", json=chat_request)
        if response.status_code == 200:
            chat_data = response.json()
            print(f"   ✓ Chat response received")
            print(f"   Response: {chat_data['response'][:100]}...")
            print(f"   Sources: {chat_data['sources']}")
            print(f"   Tokens used: {chat_data['tokens_used']}")
        else:
            print(f"   ✗ Chat request failed with status: {response.status_code}")
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"   ✗ Chat request failed with error: {str(e)}")

    # Test 3: Chat with selected text (simulated)
    print("\n3. Testing chat with selected text...")
    try:
        selected_text_request = {
            "message": "Explain the key concepts mentioned in this text?",
            "selected_text": "Physical AI represents a paradigm shift from traditional artificial intelligence by integrating computational systems with the physical world. Unlike conventional AI that operates primarily on abstract data representations, Physical AI systems must perceive, reason about, and interact with tangible environments subject to the laws of physics, materials science, and mechanics."
        }

        response = requests.post(f"{backend_url}/chat", json=selected_text_request)
        if response.status_code == 200:
            chat_data = response.json()
            print(f"   ✓ Selected text response received")
            print(f"   Response: {chat_data['response'][:100]}...")
            print(f"   Sources: {chat_data['sources']}")
        else:
            print(f"   ✗ Selected text request failed with status: {response.status_code}")
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"   ✗ Selected text request failed with error: {str(e)}")

    print("\n" + "="*50)
    print("Testing complete!")
    print("If tests failed, make sure the backend server is running on http://localhost:8000")

if __name__ == "__main__":
    test_chatbot()