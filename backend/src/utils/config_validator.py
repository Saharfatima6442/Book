import os
from typing import List
from src.utils.exceptions import InvalidApiKeyError

def validate_required_environment_variables(required_vars: List[str] = None) -> List[str]:
    """
    Validate that required environment variables are present and properly set.

    Args:
        required_vars: List of required environment variable names.
                      If None, uses default set for this application.

    Returns:
        List of missing environment variables

    Raises:
        InvalidApiKeyError: If API key validation fails
    """
    if required_vars is None:
        required_vars = [
            'COHERE_API_KEY',
            'QDRANT_URL',
            'QDRANT_API_KEY'
        ]

    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)

    # Validate COHERE_API_KEY format
    cohere_api_key = os.getenv('COHERE_API_KEY')
    if cohere_api_key:
        if len(cohere_api_key) < 10:
            raise InvalidApiKeyError("COHERE_API_KEY format is invalid")
    else:
        missing_vars.append('COHERE_API_KEY')

    # Validate QDRANT_URL format
    qdrant_url = os.getenv('QDRANT_URL')
    if qdrant_url:
        if not qdrant_url.startswith(('http://', 'https://')):
            raise InvalidApiKeyError("QDRANT_URL format is invalid")
    else:
        missing_vars.append('QDRANT_URL')

    # Validate QDRANT_API_KEY format
    qdrant_api_key = os.getenv('QDRANT_API_KEY')
    if qdrant_api_key:
        if len(qdrant_api_key) < 10:
            raise InvalidApiKeyError("QDRANT_API_KEY format is invalid")
    else:
        missing_vars.append('QDRANT_API_KEY')

    # NEON DB variables are optional for RAG functionality
    neon_db_url = os.getenv('NEON_DB_URL')
    neon_api_key = os.getenv('NEON_API_KEY')
    
    if neon_db_url and neon_api_key:
        # If both are provided, validate them
        import re
        if not re.match(r'^[a-zA-Z0-9\-_.]+$', neon_api_key) or len(neon_api_key) < 20:
            raise InvalidApiKeyError("NEON_API_KEY format is invalid")
    elif neon_db_url or neon_api_key:
        # If only one is provided, that's an issue
        if not neon_db_url:
            missing_vars.append('NEON_DB_URL')
        if not neon_api_key:
            missing_vars.append('NEON_API_KEY')

    return missing_vars

def validate_configuration() -> bool:
    """
    Validate the entire application configuration.

    Returns:
        True if configuration is valid, raises appropriate exceptions if not
    """
    # Validate required environment variables
    missing_vars = validate_required_environment_variables()

    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

    # Validate COHERE_API_KEY format
    cohere_api_key = os.getenv('COHERE_API_KEY')
    if cohere_api_key:
        if len(cohere_api_key) < 10:
            raise ValueError("COHERE_API_KEY format is invalid")
    else:
        raise ValueError("COHERE_API_KEY environment variable is required")

    # Validate QDRANT_URL format
    qdrant_url = os.getenv('QDRANT_URL')
    if qdrant_url:
        if not qdrant_url.startswith(('http://', 'https://')):
            raise ValueError("QDRANT_URL must be a valid HTTP/HTTPS URL")
    else:
        raise ValueError("QDRANT_URL environment variable is required")

    # Validate QDRANT_API_KEY format
    qdrant_api_key = os.getenv('QDRANT_API_KEY')
    if qdrant_api_key:
        if len(qdrant_api_key) < 10:
            raise ValueError("QDRANT_API_KEY format is invalid")
    else:
        raise ValueError("QDRANT_API_KEY environment variable is required")

    # Validate database URL format (optional for RAG functionality)
    db_url = os.getenv('NEON_DB_URL')
    if db_url:
        # Basic validation for PostgreSQL URL format
        if not db_url.startswith(('postgresql://', 'postgres://')):
            raise ValueError("NEON_DB_URL must be a valid PostgreSQL connection string")

    # Validate numeric configurations
    try:
        pool_size = int(os.getenv('DB_POOL_SIZE', '10'))
        if pool_size <= 0:
            raise ValueError("DB_POOL_SIZE must be a positive integer")
    except ValueError:
        raise ValueError("DB_POOL_SIZE must be a valid integer")

    try:
        pool_overflow = int(os.getenv('DB_POOL_OVERFLOW', '20'))
        if pool_overflow < 0:
            raise ValueError("DB_POOL_OVERFLOW must be a non-negative integer")
    except ValueError:
        raise ValueError("DB_POOL_OVERFLOW must be a valid integer")

    # Validate log level
    log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
    valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    if log_level not in valid_log_levels:
        raise ValueError(f"LOG_LEVEL must be one of {valid_log_levels}")

    return True