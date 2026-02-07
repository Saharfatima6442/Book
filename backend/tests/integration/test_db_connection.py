import pytest
import os
from unittest.mock import patch, MagicMock
from src.services.database_service import DatabaseService
from src.config.database_config import DatabaseConfig

class TestDatabaseConnection:
    """Test database connection establishment with Neon API key"""
    
    def test_database_config_loading(self):
        """Test that database configuration is loaded correctly from environment"""
        # Set up environment variables for testing
        os.environ['NEON_DB_URL'] = 'postgresql://test:test@test-neon-db.internal/test_db'
        os.environ['NEON_API_KEY'] = 'test-api-key-12345'
        os.environ['DB_POOL_SIZE'] = '5'
        os.environ['DB_POOL_OVERFLOW'] = '10'
        
        # Reload the config to pick up the test environment variables
        import importlib
        importlib.reload(DatabaseConfig)
        
        # Verify the configuration values
        assert DatabaseConfig.DATABASE_URL == 'postgresql://test:test@test-neon-db.internal/test_db'
        assert DatabaseConfig.API_KEY == 'test-api-key-12345'
        assert DatabaseConfig.POOL_SIZE == 5
        assert DatabaseConfig.POOL_OVERFLOW == 10
        
        # Test validation function
        assert DatabaseConfig.validate_config() is True
        
        # Clean up environment variables
        del os.environ['NEON_DB_URL']
        del os.environ['NEON_API_KEY']
        del os.environ['DB_POOL_SIZE']
        del os.environ['DB_POOL_OVERFLOW']
    
    def test_database_config_missing_values(self):
        """Test that validation fails when required config values are missing"""
        # Temporarily remove environment variables
        original_db_url = os.environ.get('NEON_DB_URL')
        original_api_key = os.environ.get('NEON_API_KEY')
        
        if 'NEON_DB_URL' in os.environ:
            del os.environ['NEON_DB_URL']
        if 'NEON_API_KEY' in os.environ:
            del os.environ['NEON_API_KEY']
        
        # Reload config to pick up missing values
        import importlib
        importlib.reload(DatabaseConfig)
        
        # Test that validation raises an error
        with pytest.raises(ValueError, match="NEON_DB_URL environment variable is required"):
            DatabaseConfig.validate_config()
        
        # Restore original environment variables
        if original_db_url:
            os.environ['NEON_DB_URL'] = original_db_url
        if original_api_key:
            os.environ['NEON_API_KEY'] = original_api_key
    
    @patch('src.services.database_service.psycopg2.pool.ThreadedConnectionPool')
    def test_database_service_initialization(self, mock_pool_constructor):
        """Test that database service initializes connection pool correctly"""
        # Set up environment variables
        os.environ['NEON_DB_URL'] = 'postgresql://test:test@test-neon-db.internal/test_db'
        os.environ['NEON_API_KEY'] = 'test-api-key-12345'
        os.environ['DB_POOL_SIZE'] = '5'
        os.environ['DB_POOL_OVERFLOW'] = '10'
        
        # Import after setting environment variables to ensure they're picked up
        from src.config.database_config import DatabaseConfig
        importlib.reload(DatabaseConfig)
        
        # Create database service
        db_service = DatabaseService()
        
        # Verify that the connection pool was initialized with correct parameters
        mock_pool_constructor.assert_called_once()
        call_args = mock_pool_constructor.call_args
        assert call_args[1]['dsn'] == 'postgresql://test:test@test-neon-db.internal/test_db'
        
        # Verify pool size parameters
        assert call_args[1]['minconn'] == 1
        assert call_args[1]['maxconn'] == 15  # 5 + 10
        
        # Clean up
        db_service.close_all_connections()
        del os.environ['NEON_DB_URL']
        del os.environ['NEON_API_KEY']
        del os.environ['DB_POOL_SIZE']
        del os.environ['DB_POOL_OVERFLOW']
    
    @patch('src.services.database_service.psycopg2.pool.ThreadedConnectionPool')
    def test_database_service_operations(self, mock_pool_constructor):
        """Test database service operations"""
        # Mock the connection pool and connections
        mock_pool = MagicMock()
        mock_connection = MagicMock()
        mock_cursor = MagicMock()
        
        mock_pool_constructor.return_value = mock_pool
        mock_pool.getconn.return_value = mock_connection
        mock_connection.cursor.return_value = mock_cursor
        
        # Set up return values for the cursor
        mock_cursor.fetchall.return_value = [{'id': 1, 'name': 'test'}]
        
        # Set up environment variables
        os.environ['NEON_DB_URL'] = 'postgresql://test:test@test-neon-db.internal/test_db'
        os.environ['NEON_API_KEY'] = 'test-api-key-12345'
        
        # Import after setting environment variables
        from src.config.database_config import DatabaseConfig
        importlib.reload(DatabaseConfig)
        
        # Create database service
        db_service = DatabaseService()
        
        # Test execute_query
        result = db_service.execute_query("SELECT * FROM test_table")
        assert result == [{'id': 1, 'name': 'test'}]
        
        # Verify cursor was used correctly
        mock_cursor.execute.assert_called_once_with("SELECT * FROM test_table", None)
        mock_cursor.fetchall.assert_called_once()
        
        # Test execute_update
        mock_cursor.rowcount = 1
        update_result = db_service.execute_update("INSERT INTO test_table VALUES (1, 'test')")
        assert update_result == 1
        
        # Verify cursor was used correctly for update
        assert mock_cursor.execute.call_count == 2  # Previous select + this insert
        assert mock_connection.commit.call_count == 1
        
        # Clean up
        db_service.close_all_connections()
        del os.environ['NEON_DB_URL']
        del os.environ['NEON_API_KEY']