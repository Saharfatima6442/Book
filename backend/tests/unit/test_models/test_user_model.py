import pytest
from datetime import datetime
from src.models.user import User
from src.models.conversation import Conversation
from src.models.message import Message

class TestUserModel:
    """Unit tests for the User model"""
    
    def test_create_user_with_required_fields(self):
        """Test creating a user with required fields"""
        user = User(username="testuser", email="test@example.com")
        
        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert user.id is not None
        assert isinstance(user.created_at, datetime)
        assert isinstance(user.updated_at, datetime)
        assert user.preferences == {}
    
    def test_create_user_without_optional_fields(self):
        """Test creating a user without optional fields"""
        user = User(username="testuser")
        
        assert user.username == "testuser"
        assert user.email is None
        assert user.id is not None
        assert user.preferences == {}
    
    def test_username_validation_too_short(self):
        """Test that usernames shorter than 3 characters raise an error"""
        with pytest.raises(ValueError, match="Username must be between 3 and 30 characters"):
            User(username="ab")
    
    def test_username_validation_too_long(self):
        """Test that usernames longer than 30 characters raise an error"""
        with pytest.raises(ValueError, match="Username must be between 3 and 30 characters"):
            User(username="a" * 31)
    
    def test_email_validation_invalid_format(self):
        """Test that invalid email formats raise an error"""
        with pytest.raises(ValueError, match="Invalid email format"):
            User(username="testuser", email="invalid-email")
    
    def test_update_preferences(self):
        """Test updating user preferences"""
        user = User(username="testuser")
        initial_updated_at = user.updated_at
        
        new_prefs = {"theme": "dark", "notifications": True}
        user.update_preferences(new_prefs)
        
        assert user.preferences["theme"] == "dark"
        assert user.preferences["notifications"] is True
        assert user.updated_at > initial_updated_at
    
    def test_update_preferences_invalid_type(self):
        """Test that updating preferences with non-dict raises an error"""
        user = User(username="testuser")
        
        with pytest.raises(ValueError, match="Preferences must be a dictionary"):
            user.update_preferences("invalid prefs")
    
    def test_to_dict(self):
        """Test converting user to dictionary"""
        user = User(username="testuser", email="test@example.com")
        user_dict = user.to_dict()
        
        assert "id" in user_dict
        assert user_dict["username"] == "testuser"
        assert user_dict["email"] == "test@example.com"
        assert "created_at" in user_dict
        assert "updated_at" in user_dict
        assert user_dict["preferences"] == {}

class TestConversationModel:
    """Unit tests for the Conversation model"""
    
    def test_create_conversation(self):
        """Test creating a conversation"""
        user_id = "test-user-id"
        conversation = Conversation(user_id=user_id, title="Test Conversation")
        
        assert conversation.user_id == user_id
        assert conversation.title == "Test Conversation"
        assert conversation.id is not None
        assert isinstance(conversation.created_at, datetime)
        assert isinstance(conversation.updated_at, datetime)
        assert conversation.messages == []
    
    def test_set_title(self):
        """Test setting conversation title"""
        conversation = Conversation(user_id="test-user-id")
        initial_updated_at = conversation.updated_at
        
        conversation.set_title("New Title")
        
        assert conversation.title == "New Title"
        assert conversation.updated_at > initial_updated_at
    
    def test_set_title_too_long(self):
        """Test that setting a title longer than 100 characters raises an error"""
        conversation = Conversation(user_id="test-user-id")
        
        with pytest.raises(ValueError, match="Title must be 100 characters or less"):
            conversation.set_title("a" * 101)
    
    def test_add_and_get_messages(self):
        """Test adding and retrieving messages from conversation"""
        conversation = Conversation(user_id="test-user-id")
        message = Message(conversation_id=conversation.id, sender_type="user", content="Hello")
        
        conversation.add_message(message)
        
        messages = conversation.get_messages()
        assert len(messages) == 1
        assert messages[0] == message
    
    def test_to_dict(self):
        """Test converting conversation to dictionary"""
        conversation = Conversation(user_id="test-user-id", title="Test")
        conv_dict = conversation.to_dict()
        
        assert "id" in conv_dict
        assert conv_dict["user_id"] == "test-user-id"
        assert conv_dict["title"] == "Test"
        assert "created_at" in conv_dict
        assert "updated_at" in conv_dict
        assert "message_count" in conv_dict

class TestMessageModel:
    """Unit tests for the Message model"""
    
    def test_create_message(self):
        """Test creating a message"""
        conversation_id = "test-conversation-id"
        message = Message(
            conversation_id=conversation_id,
            sender_type="user",
            content="Hello, world!"
        )
        
        assert message.conversation_id == conversation_id
        assert message.sender_type == "user"
        assert message.content == "Hello, world!"
        assert message.parent_message_id is None
        assert message.id is not None
        assert isinstance(message.created_at, datetime)
        assert isinstance(message.updated_at, datetime)
    
    def test_create_message_with_parent(self):
        """Test creating a message with a parent message"""
        parent_id = "parent-message-id"
        message = Message(
            conversation_id="test-conversation-id",
            sender_type="bot",
            content="Hi there!",
            parent_message_id=parent_id
        )
        
        assert message.parent_message_id == parent_id
    
    def test_sender_type_validation(self):
        """Test that invalid sender types raise an error"""
        with pytest.raises(ValueError, match="Sender type must be 'user' or 'bot'"):
            Message(
                conversation_id="test-conversation-id",
                sender_type="invalid",
                content="Hello"
            )
    
    def test_content_validation_empty(self):
        """Test that empty content raises an error"""
        with pytest.raises(ValueError, match="Content must be between 1 and 10000 characters"):
            Message(
                conversation_id="test-conversation-id",
                sender_type="user",
                content=""
            )
    
    def test_content_validation_too_long(self):
        """Test that content longer than 10000 characters raises an error"""
        with pytest.raises(ValueError, match="Content must be between 1 and 10000 characters"):
            Message(
                conversation_id="test-conversation-id",
                sender_type="user",
                content="a" * 10001
            )
    
    def test_to_dict(self):
        """Test converting message to dictionary"""
        message = Message(
            conversation_id="test-conversation-id",
            sender_type="user",
            content="Hello, world!",
            parent_message_id="parent-id"
        )
        message_dict = message.to_dict()
        
        assert "id" in message_dict
        assert message_dict["conversation_id"] == "test-conversation-id"
        assert message_dict["sender_type"] == "user"
        assert message_dict["content"] == "Hello, world!"
        assert message_dict["parent_message_id"] == "parent-id"
        assert "created_at" in message_dict
        assert "updated_at" in message_dict