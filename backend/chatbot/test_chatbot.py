"""
Simple test script to verify chatbot service setup.
Run: python test_chatbot.py
"""

import os
import sys
import requests
from dotenv import load_dotenv
import psycopg2

load_dotenv('.env')

def test_database_connection():
    """Test Neon database connection."""
    print("Testing database connection...")
    try:
        conn = psycopg2.connect(os.getenv('DATABASE_URL'))
        cur = conn.cursor()
        cur.execute("SELECT 1")
        cur.close()
        conn.close()
        print("✓ Database connection successful")
        return True
    except Exception as e:
        print(f"✗ Database connection failed: {e}")
        return False

def test_health_check():
    """Test health check endpoint."""
    print("\nTesting health check endpoint...")
    try:
        response = requests.get('http://localhost:5000/health', timeout=5)
        if response.status_code == 200:
            print("✓ Health check passed")
            return True
        else:
            print(f"✗ Health check returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to server. Make sure app.py is running (python app.py)")
        return False
    except Exception as e:
        print(f"✗ Health check error: {e}")
        return False

def test_chat_endpoint():
    """Test chat endpoint with dummy token."""
    print("\nTesting chat endpoint...")
    
    # Create a dummy JWT token
    import jwt
    secret = os.getenv('JWT_SECRET', 'test-secret')
    token = jwt.encode({'sub': 'test-user'}, secret, algorithm='HS256')
    
    try:
        response = requests.post(
            'http://localhost:5000/chat',
            json={'message': 'Test message'},
            headers={'Authorization': f'Bearer {token}'},
            timeout=5
        )
        if response.status_code == 200:
            print("✓ Chat endpoint successful")
            print(f"  Response: {response.json()}")
            return True
        else:
            print(f"✗ Chat endpoint returned status {response.status_code}")
            print(f"  Response: {response.json()}")
            return False
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to server. Make sure app.py is running (python app.py)")
        return False
    except Exception as e:
        print(f"✗ Chat endpoint error: {e}")
        return False

if __name__ == '__main__':
    print("=" * 50)
    print("Neon Chatbot Service - Setup Test")
    print("=" * 50)
    
    results = []
    results.append(("Database Connection", test_database_connection()))
    results.append(("Health Check", test_health_check()))
    results.append(("Chat Endpoint", test_chat_endpoint()))
    
    print("\n" + "=" * 50)
    print("Test Summary")
    print("=" * 50)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name}: {status}")
    
    if all(result[1] for result in results):
        print("\n✓ All tests passed!")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed. Check configuration and logs.")
        sys.exit(1)
