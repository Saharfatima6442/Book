import subprocess
import time
import requests

def start_server_and_test():
    # Start the server in a subprocess
    server_process = subprocess.Popen([
        "python", "-c", 
        "from main import app; import uvicorn; uvicorn.run(app, host='0.0.0.0', port=8000)"
    ], cwd=r"C:\Users\Saeed\OneDrive\Desktop\Book\backend")
    
    # Wait a few seconds for the server to start
    time.sleep(10)
    
    try:
        # Test the health endpoint
        response = requests.get('http://localhost:8000/health')
        print(f"Server is running! Status code: {response.status_code}")
        print(f"Response: {response.json()}")
    except requests.exceptions.ConnectionError:
        print("Could not connect to the server")
    finally:
        # Terminate the server process
        server_process.terminate()
        server_process.wait()

if __name__ == "__main__":
    start_server_and_test()