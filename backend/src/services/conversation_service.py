from typing import List, Optional
from src.models.conversation import Conversation
from src.services.database_service import DatabaseService
from src.utils.exceptions import DataValidationError, ResourceNotFoundError
import logging

class ConversationService:
    """
    Service class for managing Conversation operations
    """
    
    def __init__(self, db_service: DatabaseService):
        self.db_service = db_service
        self.table_name = "conversations"
    
    def create_conversation(self, user_id: str, title: Optional[str] = None) -> Conversation:
        """
        Create a new conversation in the database
        """
        try:
            # Validate inputs
            if not user_id:
                raise DataValidationError("User ID is required")
            
            # Create conversation object
            conversation = Conversation(user_id=user_id, title=title)
            
            # Insert into database
            query = """
                INSERT INTO conversations (id, user_id, title, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s)
            """
            params = (
                conversation.id,
                conversation.user_id,
                conversation.title,
                conversation.created_at,
                conversation.updated_at
            )
            
            self.db_service.execute_update(query, params)
            logging.info(f"Conversation created successfully with ID: {conversation.id}")
            
            return conversation
        except Exception as e:
            logging.error(f"Error creating conversation: {str(e)}")
            raise
    
    def get_conversation_by_id(self, conversation_id: str) -> Optional[Conversation]:
        """
        Retrieve a conversation by its ID
        """
        try:
            query = "SELECT * FROM conversations WHERE id = %s"
            params = (conversation_id,)
            
            result = self.db_service.execute_query(query, params)
            
            if not result:
                return None
            
            row = result[0]
            conversation = Conversation(user_id=row['user_id'], title=row['title'])
            conversation.id = row['id']
            conversation.created_at = row['created_at']
            conversation.updated_at = row['updated_at']
            
            return conversation
        except Exception as e:
            logging.error(f"Error retrieving conversation {conversation_id}: {str(e)}")
            raise
    
    def get_conversations_by_user(self, user_id: str, limit: int = 10, offset: int = 0) -> List[Conversation]:
        """
        Retrieve all conversations for a specific user
        """
        try:
            query = """
                SELECT * FROM conversations 
                WHERE user_id = %s 
                ORDER BY created_at DESC 
                LIMIT %s OFFSET %s
            """
            params = (user_id, limit, offset)
            
            results = self.db_service.execute_query(query, params)
            
            conversations = []
            for row in results:
                conversation = Conversation(user_id=row['user_id'], title=row['title'])
                conversation.id = row['id']
                conversation.created_at = row['created_at']
                conversation.updated_at = row['updated_at']
                conversations.append(conversation)
            
            return conversations
        except Exception as e:
            logging.error(f"Error retrieving conversations for user {user_id}: {str(e)}")
            raise
    
    def update_conversation(self, conversation_id: str, title: Optional[str] = None) -> Optional[Conversation]:
        """
        Update conversation information
        """
        try:
            # Get existing conversation
            existing_conversation = self.get_conversation_by_id(conversation_id)
            if not existing_conversation:
                raise ResourceNotFoundError(f"Conversation with ID {conversation_id} not found")
            
            # Prepare update values
            update_fields = []
            params = []
            
            if title is not None:
                update_fields.append("title = %s")
                params.append(title)
                existing_conversation.title = title
            
            if not update_fields:
                return existing_conversation  # Nothing to update
            
            # Add updated_at timestamp
            update_fields.append("updated_at = NOW()")
            
            # Build and execute query
            query = f"UPDATE conversations SET {', '.join(update_fields)} WHERE id = %s"
            params.append(conversation_id)
            
            rows_affected = self.db_service.execute_update(query, params)
            
            if rows_affected > 0:
                logging.info(f"Conversation {conversation_id} updated successfully")
                return existing_conversation
            else:
                return None
        except Exception as e:
            logging.error(f"Error updating conversation {conversation_id}: {str(e)}")
            raise
    
    def delete_conversation(self, conversation_id: str) -> bool:
        """
        Delete a conversation by its ID
        """
        try:
            query = "DELETE FROM conversations WHERE id = %s"
            params = (conversation_id,)
            
            rows_affected = self.db_service.execute_update(query, params)
            
            if rows_affected > 0:
                logging.info(f"Conversation {conversation_id} deleted successfully")
                return True
            else:
                return False
        except Exception as e:
            logging.error(f"Error deleting conversation {conversation_id}: {str(e)}")
            raise