from abc import ABC
from datetime import datetime
import uuid

class BaseModel(ABC):
    """
    Base model class that provides common functionality for all models
    """
    
    def __init__(self):
        self.id = str(uuid.uuid4())
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
    
    def update_timestamp(self):
        """Update the updated_at timestamp"""
        self.updated_at = datetime.now()
    
    def to_dict(self):
        """Convert model to dictionary representation"""
        result = {}
        for attr, value in self.__dict__.items():
            if isinstance(value, datetime):
                result[attr] = value.isoformat()
            elif hasattr(value, 'to_dict'):  # For nested objects
                result[attr] = value.to_dict()
            else:
                result[attr] = value
        return result