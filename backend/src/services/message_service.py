from typing import List, Optional
from src.models.message import Message
from src.services.database_service import DatabaseService
from src.utils.exceptions import DataValidationError, ResourceNotFoundError
import logging

class MessageService:
    """
    Service class for managing Message operations
    """
    
    def __init__(self, db_service: DatabaseService):
        self.db_service = db_service
        self.table_name = "messages"
    
    def create_message(self, conversation_id: str, sender_type: str, content: str, 
                      parent_message_id: Optional[str] = None) -> Message:
        """
        Create a new message in the database
        """
        try:
            # Validate inputs
            if not conversation_id:
                raise DataValidationError("Conversation ID is required")
            if sender_type not in ['user', 'bot']:
                raise DataValidationError("Sender type must be 'user' or 'bot'")
            if not content or len(content) < 1 or len(content) > 10000:
                raise DataValidationError("Content must be between 1 and 10000 characters")
            
            # Create message object
            message = Message(
                conversation_id=conversation_id,
                sender_type=sender_type,
                content=content,
                parent_message_id=parent_message_id
            )
            
            # Insert into database
            query = """
                INSERT INTO messages (id, conversation_id, sender_type, content, parent_message_id, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            params = (
                message.id,
                message.conversation_id,
                message.sender_type,
                message.content,
                message.parent_message_id,
                message.created_at,
                message.updated_at
            )
            
            self.db_service.execute_update(query, params)
            logging.info(f"Message created successfully with ID: {message.id}")
            
            return message
        except Exception as e:
            logging.error(f"Error creating message: {str(e)}")
            raise
    
    def get_message_by_id(self, message_id: str) -> Optional[Message]:
        """
        Retrieve a message by its ID
        """
        try:
            query = "SELECT * FROM messages WHERE id = %s"
            params = (message_id,)
            
            result = self.db_service.execute_query(query, params)
            
            if not result:
                return None
            
            row = result[0]
            message = Message(
                conversation_id=row['conversation_id'],
                sender_type=row['sender_type'],
                content=row['content'],
                parent_message_id=row['parent_message_id']
            )
            message.id = row['id']
            message.created_at = row['created_at']
            message.updated_at = row['updated_at']
            
            return message
        except Exception as e:
            logging.error(f"Error retrieving message {message_id}: {str(e)}")
            raise
    
    def get_messages_by_conversation(self, conversation_id: str, limit: int = 50, offset: int = 0) -> List[Message]:
        """
        Retrieve all messages for a specific conversation
        """
        try:
            query = """
                SELECT * FROM messages 
                WHERE conversation_id = %s 
                ORDER BY created_at ASC 
                LIMIT %s OFFSET %s
            """
            params = (conversation_id, limit, offset)
            
            results = self.db_service.execute_query(query, params)
            
            messages = []
            for row in results:
                message = Message(
                    conversation_id=row['conversation_id'],
                    sender_type=row['sender_type'],
                    content=row['content'],
                    parent_message_id=row['parent_message_id']
                )
                message.id = row['id']
                message.created_at = row['created_at']
                message.updated_at = row['updated_at']
                messages.append(message)
            
            return messages
        except Exception as e:
            logging.error(f"Error retrieving messages for conversation {conversation_id}: {str(e)}")
            raise
    
    def get_threaded_messages(self, parent_message_id: str) -> List[Message]:
        """
        Retrieve all messages that are children of a specific message (threaded conversations)
        """
        try:
            query = """
                SELECT * FROM messages 
                WHERE parent_message_id = %s 
                ORDER BY created_at ASC
            """
            params = (parent_message_id,)
            
            results = self.db_service.execute_query(query, params)
            
            messages = []
            for row in results:
                message = Message(
                    conversation_id=row['conversation_id'],
                    sender_type=row['sender_type'],
                    content=row['content'],
                    parent_message_id=row['parent_message_id']
                )
                message.id = row['id']
                message.created_at = row['created_at']
                message.updated_at = row['updated_at']
                messages.append(message)
            
            return messages
        except Exception as e:
            logging.error(f"Error retrieving threaded messages for parent {parent_message_id}: {str(e)}")
            raise
    
    def update_message(self, message_id: str, content: Optional[str] = None) -> Optional[Message]:
        """
        Update message content
        """
        try:
            # Get existing message
            existing_message = self.get_message_by_id(message_id)
            if not existing_message:
                raise ResourceNotFoundError(f"Message with ID {message_id} not found")
            
            # Prepare update values
            update_fields = []
            params = []
            
            if content is not None:
                if not content or len(content) < 1 or len(content) > 10000:
                    raise DataValidationError("Content must be between 1 and 10000 characters")
                
                update_fields.append("content = %s")
                params.append(content)
                existing_message.content = content
            
            if not update_fields:
                return existing_message  # Nothing to update
            
            # Add updated_at timestamp
            update_fields.append("updated_at = NOW()")
            
            # Build and execute query
            query = f"UPDATE messages SET {', '.join(update_fields)} WHERE id = %s"
            params.append(message_id)
            
            rows_affected = self.db_service.execute_update(query, params)
            
            if rows_affected > 0:
                logging.info(f"Message {message_id} updated successfully")
                return existing_message
            else:
                return None
        except Exception as e:
            logging.error(f"Error updating message {message_id}: {str(e)}")
            raise
    
    def delete_message(self, message_id: str) -> bool:
        """
        Delete a message by its ID
        """
        try:
            query = "DELETE FROM messages WHERE id = %s"
            params = (message_id,)
            
            rows_affected = self.db_service.execute_update(query, params)
            
            if rows_affected > 0:
                logging.info(f"Message {message_id} deleted successfully")
                return True
            else:
                return False
        except Exception as e:
            logging.error(f"Error deleting message {message_id}: {str(e)}")
            raise