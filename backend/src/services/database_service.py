import asyncio
import logging
from typing import Optional
import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor
from src.config.database_config import DatabaseConfig

class DatabaseService:
    """
    Service class to handle database connections with connection pooling
    """
    
    def __init__(self):
        self.connection_pool: Optional[psycopg2.pool.ThreadedConnectionPool] = None
        self._initialize_pool()
    
    def _initialize_pool(self):
        """Initialize the connection pool based on configuration"""
        try:
            self.connection_pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=1,
                maxconn=DatabaseConfig.POOL_SIZE + DatabaseConfig.POOL_OVERFLOW,
                dsn=DatabaseConfig.DATABASE_URL,
                cursor_factory=RealDictCursor
            )
            logging.info(f"Database connection pool initialized with {DatabaseConfig.POOL_SIZE} base connections and {DatabaseConfig.POOL_OVERFLOW} overflow")
        except Exception as e:
            logging.error(f"Failed to initialize database connection pool: {str(e)}")
            raise
    
    def get_connection(self):
        """Get a connection from the pool"""
        if not self.connection_pool:
            raise RuntimeError("Database connection pool not initialized")
        
        return self.connection_pool.getconn()
    
    def return_connection(self, connection):
        """Return a connection to the pool"""
        if self.connection_pool:
            self.connection_pool.putconn(connection)
    
    def execute_query(self, query: str, params=None):
        """Execute a SELECT query and return results"""
        connection = None
        cursor = None
        
        try:
            connection = self.get_connection()
            cursor = connection.cursor()
            
            cursor.execute(query, params)
            results = cursor.fetchall()
            
            return results
        except Exception as e:
            logging.error(f"Error executing query: {str(e)}")
            raise
        finally:
            if cursor:
                cursor.close()
            if connection:
                self.return_connection(connection)
    
    def execute_update(self, query: str, params=None):
        """Execute an INSERT, UPDATE, or DELETE query"""
        connection = None
        cursor = None
        
        try:
            connection = self.get_connection()
            cursor = connection.cursor()
            
            cursor.execute(query, params)
            connection.commit()
            
            return cursor.rowcount
        except Exception as e:
            logging.error(f"Error executing update: {str(e)}")
            connection.rollback()
            raise
        finally:
            if cursor:
                cursor.close()
            if connection:
                self.return_connection(connection)
    
    def close_all_connections(self):
        """Close all connections in the pool"""
        if self.connection_pool:
            self.connection_pool.closeall()
            logging.info("All database connections closed")