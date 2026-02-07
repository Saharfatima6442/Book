import os
import sys
import traceback

print("Testing imports step by step...")

try:
    from dotenv import load_dotenv
    print("✅ dotenv imported")

    load_dotenv()
    print("✅ environment loaded")

    from pydantic import BaseModel
    from typing import List, Optional
    print("✅ pydantic imported")

    import cohere
    print("✅ cohere imported")

    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    print("✅ qdrant imported")

    from fastapi import FastAPI, HTTPException, Query
    from fastapi.middleware.cors import CORSMiddleware
    print("✅ fastapi imported")

    import logging
    print("✅ logging imported")

    print("All imports successful!")

except Exception as e:
    print(f"❌ Import failed: {e}")
    traceback.print_exc()