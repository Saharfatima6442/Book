import requests

def test_backend():
    try:
        # Test the health endpoint
        response = requests.get('http://localhost:8000/health')
        print(f"Health check response: {response.status_code}")
        print(f"Response: {response.json()}")
        
        # Test the config health endpoint
        response = requests.get('http://localhost:8000/health/config')
        print(f"Config health check response: {response.status_code}")
        print(f"Response: {response.json()}")
        
    except requests.exceptions.ConnectionError:
        print("Could not connect to the backend. Make sure it's running on http://localhost:8000")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    test_backend()