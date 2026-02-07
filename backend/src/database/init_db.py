"""
Database schema migration script for Neon API Chatbot Integration
Creates all required tables for the application
"""

from src.services.database_service import DatabaseService
import logging

def create_tables():
    """
    Create all required database tables
    """
    db_service = DatabaseService()
    
    # SQL statements to create tables
    create_users_table = """
    CREATE TABLE IF NOT EXISTS users (
        id VARCHAR(36) PRIMARY KEY,
        username VARCHAR(30) UNIQUE NOT NULL,
        email VARCHAR(255) UNIQUE,
        preferences JSONB,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    );
    """
    
    create_conversations_table = """
    CREATE TABLE IF NOT EXISTS conversations (
        id VARCHAR(36) PRIMARY KEY,
        user_id VARCHAR(36) REFERENCES users(id),
        title VARCHAR(255),
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    );
    """
    
    create_messages_table = """
    CREATE TABLE IF NOT EXISTS messages (
        id VARCHAR(36) PRIMARY KEY,
        conversation_id VARCHAR(36) REFERENCES conversations(id) ON DELETE CASCADE,
        sender_type VARCHAR(10) NOT NULL CHECK (sender_type IN ('user', 'bot')),
        content TEXT NOT NULL,
        parent_message_id VARCHAR(36) REFERENCES messages(id),
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    );
    """
    
    # Create indexes
    create_user_indexes = """
    CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
    CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
    """
    
    create_conversation_indexes = """
    CREATE INDEX IF NOT EXISTS idx_conversations_user_id ON conversations(user_id);
    CREATE INDEX IF NOT EXISTS idx_conversations_created_at ON conversations(created_at);
    """
    
    create_message_indexes = """
    CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id);
    CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at);
    CREATE INDEX IF NOT EXISTS idx_messages_parent_message_id ON messages(parent_message_id);
    """
    
    try:
        # Execute table creation statements
        db_service.execute_update(create_users_table)
        logging.info("Users table created or already exists")
        
        db_service.execute_update(create_conversations_table)
        logging.info("Conversations table created or already exists")
        
        db_service.execute_update(create_messages_table)
        logging.info("Messages table created or already exists")
        
        # Execute index creation statements
        db_service.execute_update(create_user_indexes)
        logging.info("User indexes created")
        
        db_service.execute_update(create_conversation_indexes)
        logging.info("Conversation indexes created")
        
        db_service.execute_update(create_message_indexes)
        logging.info("Message indexes created")
        
        print("Database schema created successfully!")
        
    except Exception as e:
        logging.error(f"Error creating database schema: {str(e)}")
        raise
    finally:
        db_service.close_all_connections()

if __name__ == "__main__":
    create_tables()