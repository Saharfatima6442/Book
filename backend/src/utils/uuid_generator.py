import uuid
from typing import Union

def generate_uuid() -> str:
    """
    Generate a UUID4 string
    """
    return str(uuid.uuid4())

def is_valid_uuid(uuid_string: str) -> bool:
    """
    Check if the provided string is a valid UUID
    """
    try:
        uuid.UUID(uuid_string)
        return True
    except ValueError:
        return False

def convert_to_uuid(uuid_string: Union[str, uuid.UUID]) -> uuid.UUID:
    """
    Convert a string or UUID object to a UUID object
    """
    if isinstance(uuid_string, uuid.UUID):
        return uuid_string
    return uuid.UUID(uuid_string)