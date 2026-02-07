"""
Custom exception classes for the chatbot application
"""

class ChatbotBaseException(Exception):
    """Base exception class for all chatbot-related exceptions"""
    def __init__(self, message: str, error_code: str = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "UNKNOWN_ERROR"


class DatabaseConnectionError(ChatbotBaseException):
    """Raised when there's an issue connecting to the database"""
    def __init__(self, message: str = "Failed to connect to the database"):
        super().__init__(message, "DB_CONNECTION_ERROR")


class InvalidApiKeyError(ChatbotBaseException):
    """Raised when an invalid API key is provided"""
    def __init__(self, message: str = "Invalid or missing API key"):
        super().__init__(message, "INVALID_API_KEY")


class DataValidationError(ChatbotBaseException):
    """Raised when data validation fails"""
    def __init__(self, message: str = "Data validation failed"):
        super().__init__(message, "DATA_VALIDATION_ERROR")


class ResourceNotFoundError(ChatbotBaseException):
    """Raised when a requested resource is not found"""
    def __init__(self, message: str = "Requested resource not found"):
        super().__init__(message, "RESOURCE_NOT_FOUND")


class RateLimitExceededError(ChatbotBaseException):
    """Raised when rate limit is exceeded"""
    def __init__(self, message: str = "Rate limit exceeded"):
        super().__init__(message, "RATE_LIMIT_EXCEEDED")


class ConversationHistoryError(ChatbotBaseException):
    """Raised when there's an issue with conversation history"""
    def __init__(self, message: str = "Error with conversation history"):
        super().__init__(message, "CONVERSATION_HISTORY_ERROR")