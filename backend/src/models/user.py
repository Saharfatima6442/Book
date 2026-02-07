from .base_model import BaseModel
from datetime import datetime
from typing import Optional
import re

class User(BaseModel):
    """
    User model representing a chatbot user with preferences and historical interaction data
    """
    
    def __init__(self, username: str, email: Optional[str] = None, preferences: Optional[dict] = None):
        super().__init__()
        self.username = self._validate_username(username)
        self.email = self._validate_email(email) if email else None
        self.preferences = preferences or {}
    
    def _validate_username(self, username: str) -> str:
        """Validate username length (3-30 characters)"""
        if not username or len(username) < 3 or len(username) > 30:
            raise ValueError("Username must be between 3 and 30 characters")
        return username
    
    def _validate_email(self, email: str) -> str:
        """Validate email format if provided"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(pattern, email):
            raise ValueError(f"Invalid email format: {email}")
        return email
    
    def update_preferences(self, new_preferences: dict):
        """Update user preferences"""
        if not isinstance(new_preferences, dict):
            raise ValueError("Preferences must be a dictionary")
        self.preferences.update(new_preferences)
        self.update_timestamp()
    
    def to_dict(self):
        """Convert user model to dictionary representation"""
        result = super().to_dict()
        result['preferences'] = self.preferences
        return result