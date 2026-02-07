import pytest
import os
import logging
from io import StringIO
from unittest.mock import patch
from src.utils.logging_config import SensitiveDataFilter
from src.utils.env_loader import load_environment, is_valid_api_key_format
from src.utils.config_validator import validate_required_environment_variables
from src.utils.exceptions import InvalidApiKeyError

class TestSecureLogging:
    """Test that API keys are not exposed in logs or error messages"""
    
    def test_sensitive_data_filter_masks_api_key(self):
        """Test that the SensitiveDataFilter masks API keys in log messages"""
        # Set up environment with a fake API key
        os.environ['NEON_API_KEY'] = 'test-api-key-12345-secret'
        
        # Create a filter instance
        filter_instance = SensitiveDataFilter()
        
        # Create a mock log record with the API key
        class MockRecord:
            def __init__(self):
                self.msg = "Connecting to database with API key: test-api-key-12345-secret"
                self.args = ("Additional info with key: test-api-key-12345-secret",)
        
        record = MockRecord()
        
        # Apply the filter
        result = filter_instance.filter(record)
        
        # Verify the API key is masked in the message
        assert '[HIDDEN_API_KEY]' in record.msg
        assert 'test-api-key-12345-secret' not in record.msg
        
        # Verify the API key is masked in the args
        masked_args = ''.join(str(arg) for arg in record.args)
        assert '[HIDDEN_API_KEY]' in masked_args
        assert 'test-api-key-12345-secret' not in masked_args
        
        # Clean up
        if 'NEON_API_KEY' in os.environ:
            del os.environ['NEON_API_KEY']
    
    def test_logging_with_api_key_capture(self):
        """Test that logs don't expose API keys when using string formatting"""
        # Set up environment with a fake API key
        os.environ['NEON_API_KEY'] = 'secret-api-key-98765'
        
        # Create a string buffer to capture logs
        log_stream = StringIO()
        handler = logging.StreamHandler(log_stream)
        
        # Add our sensitive data filter to the handler
        handler.addFilter(SensitiveDataFilter())
        
        # Create a logger and add the handler
        logger = logging.getLogger('test_logger')
        logger.setLevel(logging.INFO)
        # Remove any existing handlers to avoid duplicates
        logger.handlers.clear()
        logger.addHandler(handler)
        
        # Log a message containing the API key
        api_key = os.environ.get('NEON_API_KEY')
        logger.info(f"Using API key: {api_key} for database connection")
        
        # Get the log output
        log_contents = log_stream.getvalue()
        
        # Verify the API key is not in the logs
        assert 'secret-api-key-98765' not in log_contents
        assert '[HIDDEN_API_KEY]' in log_contents
        
        # Clean up
        if 'NEON_API_KEY' in os.environ:
            del os.environ['NEON_API_KEY']
    
    def test_error_message_does_not_contain_api_key(self):
        """Test that error messages don't expose API keys"""
        # Set up environment with a fake API key
        os.environ['NEON_API_KEY'] = 'error-test-key-54321'
        
        try:
            # Simulate an error that might include the API key
            api_key = os.environ.get('NEON_API_KEY')
            error_msg = f"Connection failed with key: {api_key}"
            raise ValueError(error_msg)
        except ValueError as e:
            # Verify the API key is not in the error message
            assert 'error-test-key-54321' not in str(e)
            # Note: In a real scenario, we wouldn't put the API key in the error message
            # This test is just to show how we might detect such issues
        
        # Clean up
        if 'NEON_API_KEY' in os.environ:
            del os.environ['NEON_API_KEY']

class TestApiKeyValidation:
    """Test API key validation functions"""
    
    def test_valid_api_key_formats(self):
        """Test that valid API key formats pass validation"""
        valid_keys = [
            'valid-api-key-123',
            'VALID_API_KEY_456',
            'valid.api.key.789',
            'a' * 25,  # Long alphanumeric key
            'valid-key-with-multiple-parts-and-numbers-12345'
        ]
        
        for key in valid_keys:
            os.environ['NEON_API_KEY'] = key
            assert is_valid_api_key_format(key) == True
        
        # Clean up
        if 'NEON_API_KEY' in os.environ:
            del os.environ['NEON_API_KEY']
    
    def test_invalid_api_key_formats(self):
        """Test that invalid API key formats fail validation"""
        invalid_keys = [
            '',  # Empty
            'short',  # Too short
            'key with spaces',  # Contains spaces
            'key\nwith\nnewlines',  # Contains newlines
            'key\twith\ttabs',  # Contains tabs
            'key<with>special',  # Contains angle brackets
        ]
        
        for key in invalid_keys:
            os.environ['NEON_API_KEY'] = key
            assert is_valid_api_key_format(key) == False
        
        # Clean up
        if 'NEON_API_KEY' in os.environ:
            del os.environ['NEON_API_KEY']
    
    def test_config_validation_with_missing_api_key(self):
        """Test that config validation fails when API key is missing"""
        # Temporarily remove API key if it exists
        original_key = os.environ.get('NEON_API_KEY')
        if 'NEON_API_KEY' in os.environ:
            del os.environ['NEON_API_KEY']
        
        # Validate environment variables
        missing_vars = validate_required_environment_variables()
        
        # Check that API key is reported as missing
        assert 'NEON_API_KEY' in missing_vars
        
        # Restore original key if it existed
        if original_key:
            os.environ['NEON_API_KEY'] = original_key
    
    def test_config_validation_with_invalid_api_key(self):
        """Test that config validation fails when API key is invalid"""
        # Set an invalid API key
        os.environ['NEON_API_KEY'] = 'short'
        os.environ['NEON_DB_URL'] = 'postgresql://test:test@test-db.internal/test_db'
        
        # This should raise an InvalidApiKeyError
        with pytest.raises(InvalidApiKeyError):
            # We can't directly call validate_configuration here as it's in main.py
            # So we'll test the validation function that checks API key format
            from src.utils.env_loader import validate_api_key_exists
            validate_api_key_exists()
        
        # Clean up
        if 'NEON_API_KEY' in os.environ:
            del os.environ['NEON_API_KEY']
        if 'NEON_DB_URL' in os.environ:
            del os.environ['NEON_DB_URL']