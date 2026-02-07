import pytest
from unittest.mock import Mock, patch
from src.models.user import User
from src.models.conversation import Conversation
from src.models.message import Message
from src.services.user_service import UserService
from src.services.conversation_service import ConversationService
from src.services.message_service import MessageService
from src.services.database_service import DatabaseService
from src.utils.exceptions import DataValidationError, ResourceNotFoundError

class TestUserService:
    """Unit tests for the UserService"""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.mock_db_service = Mock(spec=DatabaseService)
        self.user_service = UserService(self.mock_db_service)
    
    def test_create_user_success(self):
        """Test creating a user successfully"""
        # Arrange
        username = "testuser"
        email = "test@example.com"
        preferences = {"theme": "dark"}
        
        # Act
        with patch('src.models.user.User') as mock_user_class:
            mock_user = Mock()
            mock_user.id = "test-id"
            mock_user.username = username
            mock_user.email = email
            mock_user.preferences = preferences
            mock_user_class.return_value = mock_user
            
            result = self.user_service.create_user(username, email, preferences)
        
        # Assert
        assert result == mock_user
        self.mock_db_service.execute_update.assert_called_once()
        assert "INSERT INTO users" in str(self.mock_db_service.execute_update.call_args)
    
    def test_create_user_missing_username(self):
        """Test creating a user with missing username raises an error"""
        with pytest.raises(DataValidationError, match="Username is required"):
            self.user_service.create_user("")
    
    def test_get_user_by_id_found(self):
        """Test getting a user by ID when it exists"""
        # Arrange
        user_id = "test-user-id"
        mock_result = [{
            'id': user_id,
            'username': 'testuser',
            'email': 'test@example.com',
            'preferences': {'theme': 'dark'},
            'created_at': '2023-01-01',
            'updated_at': '2023-01-01'
        }]
        self.mock_db_service.execute_query.return_value = mock_result
        
        # Act
        result = self.user_service.get_user_by_id(user_id)
        
        # Assert
        assert result is not None
        assert result.id == user_id
        assert result.username == 'testuser'
        self.mock_db_service.execute_query.assert_called_once()
    
    def test_get_user_by_id_not_found(self):
        """Test getting a user by ID when it doesn't exist"""
        # Arrange
        user_id = "nonexistent-id"
        self.mock_db_service.execute_query.return_value = []
        
        # Act
        result = self.user_service.get_user_by_id(user_id)
        
        # Assert
        assert result is None
        self.mock_db_service.execute_query.assert_called_once()
    
    def test_update_user_success(self):
        """Test updating a user successfully"""
        # Arrange
        user_id = "test-user-id"
        new_username = "updateduser"
        
        # Mock the existing user
        existing_user = User(username="olduser")
        existing_user.id = user_id
        with patch.object(self.user_service, 'get_user_by_id', return_value=existing_user):
            # Act
            result = self.user_service.update_user(user_id, username=new_username)
        
        # Assert
        assert result is not None
        assert result.username == new_username
        self.mock_db_service.execute_update.assert_called_once()
    
    def test_update_user_not_found(self):
        """Test updating a user that doesn't exist"""
        # Arrange
        user_id = "nonexistent-id"
        with patch.object(self.user_service, 'get_user_by_id', return_value=None):
            # Act & Assert
            with pytest.raises(ResourceNotFoundError):
                self.user_service.update_user(user_id, username="newuser")

class TestConversationService:
    """Unit tests for the ConversationService"""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.mock_db_service = Mock(spec=DatabaseService)
        self.conversation_service = ConversationService(self.mock_db_service)
    
    def test_create_conversation_success(self):
        """Test creating a conversation successfully"""
        # Arrange
        user_id = "test-user-id"
        title = "Test Conversation"
        
        # Act
        with patch('src.models.conversation.Conversation') as mock_conversation_class:
            mock_conversation = Mock()
            mock_conversation.id = "test-conversation-id"
            mock_conversation.user_id = user_id
            mock_conversation.title = title
            mock_conversation_class.return_value = mock_conversation
            
            result = self.conversation_service.create_conversation(user_id, title)
        
        # Assert
        assert result == mock_conversation
        self.mock_db_service.execute_update.assert_called_once()
        assert "INSERT INTO conversations" in str(self.mock_db_service.execute_update.call_args)
    
    def test_create_conversation_missing_user_id(self):
        """Test creating a conversation with missing user ID raises an error"""
        with pytest.raises(DataValidationError, match="User ID is required"):
            self.conversation_service.create_conversation("")
    
    def test_get_conversation_by_id_found(self):
        """Test getting a conversation by ID when it exists"""
        # Arrange
        conversation_id = "test-conversation-id"
        mock_result = [{
            'id': conversation_id,
            'user_id': 'test-user-id',
            'title': 'Test Conversation',
            'created_at': '2023-01-01',
            'updated_at': '2023-01-01'
        }]
        self.mock_db_service.execute_query.return_value = mock_result
        
        # Act
        result = self.conversation_service.get_conversation_by_id(conversation_id)
        
        # Assert
        assert result is not None
        assert result.id == conversation_id
        assert result.user_id == 'test-user-id'
        self.mock_db_service.execute_query.assert_called_once()

class TestMessageService:
    """Unit tests for the MessageService"""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.mock_db_service = Mock(spec=DatabaseService)
        self.message_service = MessageService(self.mock_db_service)
    
    def test_create_message_success(self):
        """Test creating a message successfully"""
        # Arrange
        conversation_id = "test-conversation-id"
        sender_type = "user"
        content = "Hello, world!"
        
        # Act
        with patch('src.models.message.Message') as mock_message_class:
            mock_message = Mock()
            mock_message.id = "test-message-id"
            mock_message.conversation_id = conversation_id
            mock_message.sender_type = sender_type
            mock_message.content = content
            mock_message_class.return_value = mock_message
            
            result = self.message_service.create_message(conversation_id, sender_type, content)
        
        # Assert
        assert result == mock_message
        self.mock_db_service.execute_update.assert_called_once()
        assert "INSERT INTO messages" in str(self.mock_db_service.execute_update.call_args)
    
    def test_create_message_invalid_sender_type(self):
        """Test creating a message with invalid sender type raises an error"""
        with pytest.raises(DataValidationError, match="Sender type must be 'user' or 'bot'"):
            self.message_service.create_message("test-id", "invalid", "content")
    
    def test_create_message_invalid_content_length(self):
        """Test creating a message with invalid content length raises an error"""
        with pytest.raises(DataValidationError, match="Content must be between 1 and 10000 characters"):
            self.message_service.create_message("test-id", "user", "")