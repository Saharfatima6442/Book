import subprocess
import sys
import os
from pathlib import Path

def install_dependencies():
    """Install required dependencies"""
    requirements_file = Path("requirements.txt")

    if requirements_file.exists():
        print("Installing dependencies from requirements.txt...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(requirements_file)])
            print("Dependencies installed successfully!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Failed to install dependencies: {e}")
            return False
    else:
        print("requirements.txt not found!")
        return False

def check_environment():
    """Check if environment variables are set"""
    print("Checking environment variables...")

    required_vars = ["COHERE_API_KEY", "QDRANT_URL", "QDRANT_API_KEY"]
    all_set = True

    for var in required_vars:
        value = os.getenv(var)
        if not value:
            print(f"❌ {var} is not set")
            all_set = False
        else:
            print(f"✅ {var} is set")

    return all_set

def start_server():
    """Start the FastAPI server"""
    try:
        from main import app
        import uvicorn

        print("Starting the FastAPI server...")
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except ImportError as e:
        print(f"Failed to import main module: {e}")
        return False
    except Exception as e:
        print(f"Failed to start server: {e}")
        return False

    return True

def main():
    print("AI Book Chatbot Backend Startup Script")
    print("="*50)

    # Change to the script's directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Check environment
    env_ok = check_environment()
    if not env_ok:
        print("\nPlease make sure all environment variables are set in the .env file")
        return

    # Try to install dependencies
    deps_ok = install_dependencies()
    if not deps_ok:
        print("\nTrying to install individual packages...")
        packages = [
            "fastapi", "uvicorn[standard]", "cohere",
            "qdrant-client", "python-dotenv", "pydantic"
        ]

        for package in packages:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"✅ Installed {package}")
            except subprocess.CalledProcessError:
                print(f"❌ Failed to install {package}")

    # Try to start the server
    print("\nAttempting to start the server...")
    start_server()

if __name__ == "__main__":
    main()