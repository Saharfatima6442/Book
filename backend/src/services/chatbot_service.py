from typing import Dict, Any, Optional
from src.models.conversation import Conversation
from src.models.message import Message
from src.models.user import User
from src.services.user_service import UserService
from src.services.conversation_service import ConversationService
from src.services.message_service import MessageService
from src.services.database_service import DatabaseService
from src.utils.exceptions import ResourceNotFoundError, ConversationHistoryError
import logging

class ChatbotService:
    """
    Main service class for chatbot functionality
    Integrates user, conversation, and message services
    """

    def __init__(self, db_service: DatabaseService):
        self.db_service = db_service
        self.user_service = UserService(db_service)
        self.conversation_service = ConversationService(db_service)
        self.message_service = MessageService(db_service)

    def process_user_message(self, user_id: str, message_content: str,
                           conversation_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a user message and return a response
        """
        try:
            # Get or create conversation
            if conversation_id:
                conversation = self.conversation_service.get_conversation_by_id(conversation_id)
                if not conversation:
                    raise ResourceNotFoundError(f"Conversation with ID {conversation_id} not found")
            else:
                # Create a new conversation
                conversation = self.conversation_service.create_conversation(user_id)
                conversation_id = conversation.id

            # Create user message in the conversation
            user_message = self.message_service.create_message(
                conversation_id=conversation_id,
                sender_type='user',
                content=message_content
            )

            # Retrieve conversation history for context
            conversation_history = self.message_service.get_messages_by_conversation(conversation_id)

            # Generate bot response based on history and user message
            bot_response_content = self.generate_bot_response(
                user_message.content,
                conversation_history,
                user_id
            )

            # Create bot response message
            bot_message = self.message_service.create_message(
                conversation_id=conversation_id,
                sender_type='bot',
                content=bot_response_content
            )

            # Prepare response
            response = {
                "conversation_id": conversation_id,
                "user_message_id": user_message.id,
                "bot_message_id": bot_message.id,
                "response": bot_response_content,
                "timestamp": bot_message.created_at.isoformat()
            }

            logging.info(f"Processed message for user {user_id} in conversation {conversation_id}")
            return response

        except Exception as e:
            logging.error(f"Error processing user message: {str(e)}")
            raise

    def generate_bot_response(self, user_message: str, conversation_history: list, user_id: str) -> str:
        """
        Generate a bot response based on user message and conversation history
        """
        # Get user preferences to customize response
        user = self.user_service.get_user_by_id(user_id)
        user_preferences = user.preferences if user else {}

        # For now, implement a simple response mechanism
        # In a real implementation, this would connect to an AI model
        if "hello" in user_message.lower() or "hi" in user_message.lower():
            greeting = "Hello"
            if user and user.username:
                greeting = f"Hello {user.username}"
            return f"{greeting}! How can I assist you today?"
        elif "thank" in user_message.lower():
            return "You're welcome! Is there anything else I can help with?"
        elif "bye" in user_message.lower() or "goodbye" in user_message.lower():
            return "Goodbye! Feel free to come back if you have more questions."
        else:
            # Contextual response based on history
            if len(conversation_history) > 1:
                # Look for patterns in the conversation history
                context_summary = self.extract_context_from_history(conversation_history)
                return f"I understand you're asking about '{user_message}'. {context_summary} How else can I assist you?"
            else:
                return f"Thanks for your message: '{user_message}'. How can I help you further?"

    def extract_context_from_history(self, conversation_history: list) -> str:
        """
        Extract context from conversation history to provide more personalized responses
        """
        if not conversation_history:
            return ""

        # Get the last few messages to understand the context
        recent_messages = conversation_history[-5:]  # Last 5 messages

        # Count user vs bot messages
        user_messages = [msg for msg in recent_messages if msg.sender_type == 'user']
        bot_messages = [msg for msg in recent_messages if msg.sender_type == 'bot']

        if len(user_messages) > len(bot_messages):
            return "I see you've been asking several questions. "
        elif len(bot_messages) > len(user_messages):
            return "We've been having a good conversation. "
        else:
            return "Based on our conversation, "

    def get_conversation_history(self, conversation_id: str) -> Dict[str, Any]:
        """
        Retrieve full conversation history with messages
        """
        try:
            # Get conversation
            conversation = self.conversation_service.get_conversation_by_id(conversation_id)
            if not conversation:
                raise ResourceNotFoundError(f"Conversation with ID {conversation_id} not found")

            # Get all messages in the conversation
            messages = self.message_service.get_messages_by_conversation(conversation_id)

            # Prepare response
            result = {
                "id": conversation.id,
                "title": conversation.title,
                "created_at": conversation.created_at.isoformat(),
                "updated_at": conversation.updated_at.isoformat(),
                "messages": [
                    {
                        "id": msg.id,
                        "sender_type": msg.sender_type,
                        "content": msg.content,
                        "timestamp": msg.created_at.isoformat()
                    } for msg in messages
                ]
            }

            return result
        except Exception as e:
            logging.error(f"Error retrieving conversation history {conversation_id}: {str(e)}")
            raise

    def get_user_conversations(self, user_id: str, limit: int = 10, offset: int = 0) -> Dict[str, Any]:
        """
        Retrieve all conversations for a specific user
        """
        try:
            # Get user to verify it exists
            user = self.user_service.get_user_by_id(user_id)
            if not user:
                raise ResourceNotFoundError(f"User with ID {user_id} not found")

            # Get conversations for the user
            conversations = self.conversation_service.get_conversations_by_user(user_id, limit, offset)

            # Prepare response
            result = {
                "conversations": [
                    {
                        "id": conv.id,
                        "title": conv.title,
                        "created_at": conv.created_at.isoformat(),
                        "updated_at": conv.updated_at.isoformat()
                    } for conv in conversations
                ],
                "total_count": len(conversations)  # This would need to be a separate count query in a real implementation
            }

            return result
        except Exception as e:
            logging.error(f"Error retrieving conversations for user {user_id}: {str(e)}")
            raise

    def delete_conversation(self, conversation_id: str) -> bool:
        """
        Delete a conversation and all associated messages
        """
        try:
            # This will cascade delete messages due to the foreign key constraint
            success = self.conversation_service.delete_conversation(conversation_id)
            if success:
                logging.info(f"Conversation {conversation_id} deleted successfully")
            else:
                logging.warning(f"Conversation {conversation_id} not found for deletion")

            return success
        except Exception as e:
            logging.error(f"Error deleting conversation {conversation_id}: {str(e)}")
            raise

    def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """
        Retrieve user preferences to personalize the chatbot experience
        """
        try:
            user = self.user_service.get_user_by_id(user_id)
            if not user:
                raise ResourceNotFoundError(f"User with ID {user_id} not found")

            return user.preferences
        except Exception as e:
            logging.error(f"Error retrieving user preferences for user {user_id}: {str(e)}")
            raise

    def update_user_preferences(self, user_id: str, preferences: Dict[str, Any]) -> bool:
        """
        Update user preferences to personalize the chatbot experience
        """
        try:
            user = self.user_service.get_user_by_id(user_id)
            if not user:
                raise ResourceNotFoundError(f"User with ID {user_id} not found")

            # Update user preferences
            updated_user = self.user_service.update_user(user_id, preferences=preferences)
            return updated_user is not None
        except Exception as e:
            logging.error(f"Error updating user preferences for user {user_id}: {str(e)}")
            raise