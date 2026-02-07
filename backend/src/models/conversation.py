from .base_model import BaseModel
from datetime import datetime
from typing import Optional
import re

class Conversation(BaseModel):
    """
    Conversation model representing a chat session with unique identifier, 
    timestamp, user queries, and chatbot responses
    """
    
    def __init__(self, user_id: str, title: Optional[str] = None):
        super().__init__()
        self.user_id = user_id
        self.title = title
        self.messages = []
    
    def set_title(self, title: str):
        """Set the conversation title"""
        if title and len(title) > 100:  # Reasonable limit for titles
            raise ValueError("Title must be 100 characters or less")
        self.title = title
        self.update_timestamp()
    
    def add_message(self, message):
        """Add a message to the conversation"""
        self.messages.append(message)
        self.update_timestamp()
    
    def get_messages(self):
        """Get all messages in the conversation"""
        return self.messages
    
    def to_dict(self):
        """Convert conversation model to dictionary representation"""
        result = super().to_dict()
        result['user_id'] = self.user_id
        result['title'] = self.title
        result['message_count'] = len(self.messages)
        return result