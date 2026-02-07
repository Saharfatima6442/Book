from typing import Optional
from src.models.user import User
from src.services.database_service import DatabaseService
from src.utils.exceptions import DataValidationError, ResourceNotFoundError
import logging

class UserService:
    """
    Service class for managing User operations
    """
    
    def __init__(self, db_service: DatabaseService):
        self.db_service = db_service
        self.table_name = "users"
    
    def create_user(self, username: str, email: Optional[str] = None, preferences: Optional[dict] = None) -> User:
        """
        Create a new user in the database
        """
        try:
            # Validate inputs
            if not username or len(username.strip()) == 0:
                raise DataValidationError("Username is required")
            
            # Create user object
            user = User(username=username, email=email, preferences=preferences or {})
            
            # Insert into database
            query = """
                INSERT INTO users (id, username, email, preferences, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s)
            """
            params = (
                user.id,
                user.username,
                user.email,
                user.preferences,
                user.created_at,
                user.updated_at
            )
            
            self.db_service.execute_update(query, params)
            logging.info(f"User created successfully with ID: {user.id}")
            
            return user
        except Exception as e:
            logging.error(f"Error creating user: {str(e)}")
            raise
    
    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """
        Retrieve a user by their ID
        """
        try:
            query = "SELECT * FROM users WHERE id = %s"
            params = (user_id,)
            
            result = self.db_service.execute_query(query, params)
            
            if not result:
                return None
            
            row = result[0]
            user = User(username=row['username'], email=row['email'])
            user.id = row['id']
            user.created_at = row['created_at']
            user.updated_at = row['updated_at']
            user.preferences = row['preferences'] or {}
            
            return user
        except Exception as e:
            logging.error(f"Error retrieving user {user_id}: {str(e)}")
            raise
    
    def get_user_by_username(self, username: str) -> Optional[User]:
        """
        Retrieve a user by their username
        """
        try:
            query = "SELECT * FROM users WHERE username = %s"
            params = (username,)
            
            result = self.db_service.execute_query(query, params)
            
            if not result:
                return None
            
            row = result[0]
            user = User(username=row['username'], email=row['email'])
            user.id = row['id']
            user.created_at = row['created_at']
            user.updated_at = row['updated_at']
            user.preferences = row['preferences'] or {}
            
            return user
        except Exception as e:
            logging.error(f"Error retrieving user {username}: {str(e)}")
            raise
    
    def update_user(self, user_id: str, username: Optional[str] = None, email: Optional[str] = None, 
                   preferences: Optional[dict] = None) -> Optional[User]:
        """
        Update user information
        """
        try:
            # Get existing user
            existing_user = self.get_user_by_id(user_id)
            if not existing_user:
                raise ResourceNotFoundError(f"User with ID {user_id} not found")
            
            # Prepare update values
            update_fields = []
            params = []
            
            if username is not None:
                update_fields.append("username = %s")
                params.append(username)
                existing_user.username = username
            
            if email is not None:
                update_fields.append("email = %s")
                params.append(email)
                existing_user.email = email
            
            if preferences is not None:
                update_fields.append("preferences = %s")
                params.append(preferences)
                existing_user.preferences = preferences
            
            if not update_fields:
                return existing_user  # Nothing to update
            
            # Add updated_at timestamp
            update_fields.append("updated_at = NOW()")
            
            # Build and execute query
            query = f"UPDATE users SET {', '.join(update_fields)} WHERE id = %s"
            params.append(user_id)
            
            rows_affected = self.db_service.execute_update(query, params)
            
            if rows_affected > 0:
                logging.info(f"User {user_id} updated successfully")
                return existing_user
            else:
                return None
        except Exception as e:
            logging.error(f"Error updating user {user_id}: {str(e)}")
            raise
    
    def delete_user(self, user_id: str) -> bool:
        """
        Delete a user by their ID
        """
        try:
            query = "DELETE FROM users WHERE id = %s"
            params = (user_id,)
            
            rows_affected = self.db_service.execute_update(query, params)
            
            if rows_affected > 0:
                logging.info(f"User {user_id} deleted successfully")
                return True
            else:
                return False
        except Exception as e:
            logging.error(f"Error deleting user {user_id}: {str(e)}")
            raise