from .base_model import BaseModel
from datetime import datetime
from typing import Optional

class Message(BaseModel):
    """
    Message model representing an individual message within a conversation
    containing content, timestamp, and sender type
    """
    
    def __init__(self, conversation_id: str, sender_type: str, content: str, parent_message_id: Optional[str] = None):
        super().__init__()
        self.conversation_id = conversation_id
        self.sender_type = self._validate_sender_type(sender_type)
        self.content = self._validate_content(content)
        self.parent_message_id = parent_message_id
    
    def _validate_sender_type(self, sender_type: str) -> str:
        """Validate sender type is either 'user' or 'bot'"""
        if sender_type not in ['user', 'bot']:
            raise ValueError("Sender type must be 'user' or 'bot'")
        return sender_type
    
    def _validate_content(self, content: str) -> str:
        """Validate content length (1-10000 characters)"""
        if not content or len(content) < 1 or len(content) > 10000:
            raise ValueError("Content must be between 1 and 10000 characters")
        return content
    
    def to_dict(self):
        """Convert message model to dictionary representation"""
        result = super().to_dict()
        result['conversation_id'] = self.conversation_id
        result['sender_type'] = self.sender_type
        result['content'] = self.content
        result['parent_message_id'] = self.parent_message_id
        return result