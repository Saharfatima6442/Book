import os
from dotenv import load_dotenv
import logging
import re

def load_environment():
    """
    Load environment variables from .env file and validate required variables.
    """
    # Load environment variables from .env file
    load_dotenv()

    # Define required environment variables for RAG functionality
    required_vars = [
        'COHERE_API_KEY',
        'QDRANT_URL',
        'QDRANT_API_KEY'
    ]

    # Check if all required variables are present
    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

    # Validate COHERE_API_KEY format (typically starts with "CO-" followed by alphanumeric characters)
    cohere_api_key = os.getenv('COHERE_API_KEY')
    if not cohere_api_key or len(cohere_api_key) < 10:
        raise ValueError("COHERE_API_KEY does not match expected format")

    # Validate QDRANT_URL format (should be a valid URL)
    qdrant_url = os.getenv('QDRANT_URL')
    if not qdrant_url or not qdrant_url.startswith(('http://', 'https://')):
        raise ValueError("QDRANT_URL does not match expected format")

    # Validate QDRANT_API_KEY format (typically a JWT-like token or alphanumeric string)
    qdrant_api_key = os.getenv('QDRANT_API_KEY')
    if not qdrant_api_key or len(qdrant_api_key) < 10:
        raise ValueError("QDRANT_API_KEY does not match expected format")

    # NEON DB variables are optional for RAG functionality
    neon_db_url = os.getenv('NEON_DB_URL')
    neon_api_key = os.getenv('NEON_API_KEY')
    
    if neon_db_url and neon_api_key:
        # If both are provided, validate them
        if not neon_db_url.startswith(('postgresql://', 'postgres://')):
            raise ValueError("NEON_DB_URL must be a valid PostgreSQL connection string")
        
        if not is_valid_api_key_format(neon_api_key):
            raise ValueError("NEON_API_KEY does not match expected format")
    
    # Set default values for optional variables if not present
    if not os.getenv('DB_POOL_SIZE'):
        os.environ['DB_POOL_SIZE'] = '10'

    if not os.getenv('DB_POOL_OVERFLOW'):
        os.environ['DB_POOL_OVERFLOW'] = '20'

    if not os.getenv('LOG_LEVEL'):
        os.environ['LOG_LEVEL'] = 'INFO'

    logging.info("Environment variables loaded successfully")

    return True

def is_valid_api_key_format(api_key):
    """
    Validates the format of the API key.
    Expected format: alphanumeric characters with possible hyphens/underscores, typically 20+ chars.
    """
    if not api_key:
        return False

    # Basic pattern: alphanumeric, hyphens, underscores, dots, 20-100 chars
    pattern = r'^[a-zA-Z0-9\-_.]{20,100}$'
    return bool(re.match(pattern, api_key))

def validate_api_key_exists():
    """
    Validates that the API key exists and returns it (without logging the key for security).
    """
    api_key = os.getenv('NEON_API_KEY')
    if not api_key:
        raise ValueError("NEON_API_KEY environment variable is not set")

    if not is_valid_api_key_format(api_key):
        raise ValueError("NEON_API_KEY format is invalid")

    return True