import logging
import os
from pythonjsonlogger import jsonlogger
import re

class SensitiveDataFilter(logging.Filter):
    """
    Custom logging filter to prevent sensitive data (like API keys) from being logged
    """
    def __init__(self):
        super().__init__()
        # Get the API key from environment to filter it
        self.api_key = os.getenv('NEON_API_KEY', '')
        # Create a regex pattern to match the API key in various contexts
        self.patterns = []
        if self.api_key:
            # Escape special regex characters in the API key
            escaped_key = re.escape(self.api_key)
            self.patterns.append(re.compile(escaped_key, re.IGNORECASE))

    def filter(self, record):
        """
        Filter the log record to remove sensitive information
        """
        # Process the message to remove sensitive data
        if hasattr(record, 'msg'):
            if isinstance(record.msg, str):
                for pattern in self.patterns:
                    record.msg = pattern.sub('[HIDDEN_API_KEY]', record.msg)

        # Process the args to remove sensitive data
        if hasattr(record, 'args') and record.args:
            new_args = []
            for arg in record.args:
                if isinstance(arg, str):
                    filtered_arg = arg
                    for pattern in self.patterns:
                        filtered_arg = pattern.sub('[HIDDEN_API_KEY]', filtered_arg)
                    new_args.append(filtered_arg)
                else:
                    new_args.append(arg)
            record.args = tuple(new_args)

        return True

def setup_logging():
    """
    Set up logging configuration as per research.md requirements.
    All database operations and API interactions will be logged for debugging and monitoring.
    """
    log_level = os.getenv('LOG_LEVEL', 'INFO').upper()

    # Create logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level))

    # Prevent adding multiple handlers if called multiple times
    if logger.handlers:
        return

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level))

    # Create formatter
    if os.getenv('LOG_FORMAT', 'TEXT') == 'JSON':
        formatter = jsonlogger.JsonFormatter(
            '%(asctime)s %(name)s %(levelname)s %(message)s'
        )
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    # Add formatter to handler
    console_handler.setFormatter(formatter)

    # Add the sensitive data filter to the handler
    sensitive_data_filter = SensitiveDataFilter()
    console_handler.addFilter(sensitive_data_filter)

    # Add handler to logger
    logger.addHandler(console_handler)

    # Also set up logging for specific modules
    logging.getLogger('database').setLevel(getattr(logging, log_level))
    logging.getLogger('api').setLevel(getattr(logging, log_level))
    logging.getLogger('chatbot').setLevel(getattr(logging, log_level))

# Call setup_logging to configure logging when module is imported
setup_logging()