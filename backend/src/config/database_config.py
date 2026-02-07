import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class DatabaseConfig:
    """Configuration class for database settings"""
    
    DATABASE_URL = os.getenv("NEON_DB_URL", "")
    API_KEY = os.getenv("NEON_API_KEY", "")
    POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "10"))
    POOL_OVERFLOW = int(os.getenv("DB_POOL_OVERFLOW", "20"))
    
    @classmethod
    def validate_config(cls):
        """Validate that required configuration values are present"""
        if not cls.DATABASE_URL:
            raise ValueError("NEON_DB_URL environment variable is required")
        if not cls.API_KEY:
            raise ValueError("NEON_API_KEY environment variable is required")
        
        return True