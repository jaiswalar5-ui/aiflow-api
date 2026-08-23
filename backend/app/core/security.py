from passlib.context import CryptContext
import secrets

pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")

def verify_api_key(plain_api_key: str, hashed_api_key: str) -> bool:
    return pwd_context.verify(plain_api_key, hashed_api_key)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def generate_api_key() -> str:
    """Generate a new secure API key"""
    # Create something like sk-aiflow-1234567890abcdef
    raw_token = secrets.token_urlsafe(32)
    return f"sk-aiflow-{raw_token}"
