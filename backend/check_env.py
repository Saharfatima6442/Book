import os
import sys

print("Python executable:", sys.executable)
print("Python version:", sys.version)
print("Current working directory:", os.getcwd())

# Try to import the required packages
required_packages = [
    ("fastapi", "FastAPI"),
    ("uvicorn", "uvicorn"),
    ("cohere", "cohere"),
    ("qdrant_client", "qdrant_client"),
    ("dotenv", "load_dotenv"),
    ("pydantic", "BaseModel")
]

print("\nChecking required packages:")
for package_name, import_name in required_packages:
    try:
        if package_name == "dotenv":
            from dotenv import load_dotenv
        elif package_name == "pydantic":
            from pydantic import BaseModel
        else:
            __import__(package_name)
        print(f"✅ {package_name} - OK")
    except ImportError as e:
        print(f"❌ {package_name} - Error: {e}")

# Check environment variables
print("\nChecking environment variables:")
env_vars = ["COHERE_API_KEY", "QDRANT_URL", "QDRANT_API_KEY"]
for var in env_vars:
    value = os.getenv(var)
    if value:
        print(f"✅ {var} - Set (length: {len(value)})")
    else:
        print(f"❌ {var} - Not set")

print("\nEnvironment check complete!")