from enum import Enum
from typing import Optional, Dict, Any
import re


class ErrorType(str, Enum):
    """Normalized error type enumeration."""
    INVALID_REQUEST = "invalid_request"           # Non-retryable
    AUTHENTICATION_ERROR = "authentication_error"   # Non-retryable
    RATE_LIMIT_ERROR = "rate_limit_error"          # Retryable + backoff hint
    QUOTA_EXHAUSTED_ERROR = "quota_exhausted_error" # Retryable after cooldown
    TIMEOUT_ERROR = "timeout_error"                # Retryable
    PROVIDER_UNAVAILABLE_ERROR = "provider_unavailable_error"  # Retryable
    SERVER_ERROR = "server_error"                  # Retryable
    UNSUPPORTED_MODEL_ERROR = "unsupported_model_error"  # Non-retryable
    NETWORK_ERROR = "network_error"                # Retryable
    UNKNOWN_ERROR = "unknown_error"                # Retryable with caution


class ProviderError(Exception):
    """
    Base exception for all provider-related errors.
    Contains standardized error information for classification and retry logic.
    """
    def __init__(
        self,
        message: str,
        error_type: ErrorType,
        retryable: bool,
        status_code: Optional[int] = None,
        response_headers: Optional[Dict[str, str]] = None,
        response_body: Optional[Dict[str, Any]] = None,
        provider_name: Optional[str] = None,
        request_id: Optional[str] = None
    ):
        """
        Initialize provider error with standardized information.
        
        Args:
            message: Safe error message (no credentials/internal details)
            error_type: Normalized error type from ErrorType enum
            retryable: Whether this error is retryable
            status_code: HTTP status code if available
            response_headers: Response headers if available
            response_body: Response body if available
            provider_name: Name of the provider that caused the error
            request_id: Request identifier for tracing
        """
        # Ensure message is safe (no credentials)
        self.safe_message = self._sanitize_message(message)
        super().__init__(self.safe_message)
        
        self.error_type = error_type
        self.retryable = retryable
        self.status_code = status_code
        self.response_headers = response_headers or {}
        self.response_body = response_body or {}
        self.provider_name = provider_name
        self.request_id = request_id
        
        self.provider_metadata = {
            "provider_name": self.provider_name,
            "status_code": self.status_code,
            "request_id": self.request_id,
            "response_headers": self.response_headers,
            "response_body": self.response_body,
        }
    
    @staticmethod
    def _sanitize_message(message: str) -> str:
        """
        Sanitize error message to remove sensitive information.
        Removes API keys, credentials, internal URLs, etc.
        """
        if not message:
            return "An error occurred"
        
        # Remove common credential patterns
        # API keys
        message = re.sub(r'["\']?[A-Za-z0-9_\-]{20,}["\']?', '[REDACTED_API_KEY]', message)
        # Bearer tokens
        message = re.sub(r'Bearer\s+[A-Za-z0-9_\-\.]+', 'Bearer [REDACTED_TOKEN]', message)
        # Email addresses
        message = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[REDACTED_EMAIL]', message)
        # URLs with potential sensitive info
        message = re.sub(r'https?://[^\s<>"]+/?[^\s<>"]*', '[REDACTED_URL]', message)
        # UUIDs that might be sensitive
        message = re.sub(r'\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b', '[REDACTED_ID]', message)
        
        return message
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for API responses."""
        return {
            "error_type": self.error_type.value,
            "retryable": self.retryable,
            "message": str(self),
            "provider": self.provider_name,
            "status_code": self.status_code,
            "request_id": self.request_id
        }


class InvalidRequestError(ProviderError):
    """Non-retryable error: Invalid request parameters."""
    def __init__(
        self,
        message: str = "Invalid request",
        status_code: int = 400,
        response_headers: Optional[Dict[str, str]] = None,
        response_body: Optional[Dict[str, Any]] = None,
        provider_name: Optional[str] = None,
        request_id: Optional[str] = None
    ):
        super().__init__(
            message=message,
            error_type=ErrorType.INVALID_REQUEST,
            retryable=False,
            status_code=status_code,
            response_headers=response_headers,
            response_body=response_body,
            provider_name=provider_name,
            request_id=request_id
        )


class AuthenticationError(ProviderError):
    """Non-retryable error: Authentication failure."""
    def __init__(
        self,
        message: str = "Authentication failed",
        status_code: int = 401,
        response_headers: Optional[Dict[str, str]] = None,
        response_body: Optional[Dict[str, Any]] = None,
        provider_name: Optional[str] = None,
        request_id: Optional[str] = None
    ):
        super().__init__(
            message=message,
            error_type=ErrorType.AUTHENTICATION_ERROR,
            retryable=False,
            status_code=status_code,
            response_headers=response_headers,
            response_body=response_body,
            provider_name=provider_name,
            request_id=request_id
        )


class RateLimitError(ProviderError):
    """Retryable error: Rate limit exceeded with backoff hint."""
    def __init__(
        self,
        message: str = "Rate limit exceeded",
        status_code: int = 429,
        response_headers: Optional[Dict[str, str]] = None,
        response_body: Optional[Dict[str, Any]] = None,
        provider_name: Optional[str] = None,
        request_id: Optional[str] = None
    ):
        super().__init__(
            message=message,
            error_type=ErrorType.RATE_LIMIT_ERROR,
            retryable=True,
            status_code=status_code,
            response_headers=response_headers,
            response_body=response_body,
            provider_name=provider_name,
            request_id=request_id
        )


class QuotaExhaustedError(ProviderError):
    """Retryable error: Quota exhausted, retry after cooldown."""
    def __init__(
        self,
        message: str = "Quota exhausted",
        status_code: int = 429,
        response_headers: Optional[Dict[str, str]] = None,
        response_body: Optional[Dict[str, Any]] = None,
        provider_name: Optional[str] = None,
        request_id: Optional[str] = None
    ):
        super().__init__(
            message=message,
            error_type=ErrorType.QUOTA_EXHAUSTED_ERROR,
            retryable=True,
            status_code=status_code,
            response_headers=response_headers,
            response_body=response_body,
            provider_name=provider_name,
            request_id=request_id
        )


class TimeoutError(ProviderError):
    """Retryable error: Request timeout."""
    def __init__(
        self,
        message: str = "Request timeout",
        status_code: Optional[int] = None,
        response_headers: Optional[Dict[str, str]] = None,
        response_body: Optional[Dict[str, Any]] = None,
        provider_name: Optional[str] = None,
        request_id: Optional[str] = None
    ):
        super().__init__(
            message=message,
            error_type=ErrorType.TIMEOUT_ERROR,
            retryable=True,
            status_code=status_code,
            response_headers=response_headers,
            response_body=response_body,
            provider_name=provider_name,
            request_id=request_id
        )


class ProviderUnavailableError(ProviderError):
    """Retryable error: Provider unavailable."""
    def __init__(
        self,
        message: str = "Provider unavailable",
        status_code: int = 503,
        response_headers: Optional[Dict[str, str]] = None,
        response_body: Optional[Dict[str, Any]] = None,
        provider_name: Optional[str] = None,
        request_id: Optional[str] = None
    ):
        super().__init__(
            message=message,
            error_type=ErrorType.PROVIDER_UNAVAILABLE_ERROR,
            retryable=True,
            status_code=status_code,
            response_headers=response_headers,
            response_body=response_body,
            provider_name=provider_name,
            request_id=request_id
        )


class ServerError(ProviderError):
    """Retryable error: Provider server error."""
    def __init__(
        self,
        message: str = "Server error",
        status_code: int = 500,
        response_headers: Optional[Dict[str, str]] = None,
        response_body: Optional[Dict[str, Any]] = None,
        provider_name: Optional[str] = None,
        request_id: Optional[str] = None
    ):
        super().__init__(
            message=message,
            error_type=ErrorType.SERVER_ERROR,
            retryable=True,
            status_code=status_code,
            response_headers=response_headers,
            response_body=response_body,
            provider_name=provider_name,
            request_id=request_id
        )


class UnsupportedModelError(ProviderError):
    """Non-retryable error: Model not supported."""
    def __init__(
        self,
        message: str = "Unsupported model",
        status_code: int = 404,
        response_headers: Optional[Dict[str, str]] = None,
        response_body: Optional[Dict[str, Any]] = None,
        provider_name: Optional[str] = None,
        request_id: Optional[str] = None
    ):
        super().__init__(
            message=message,
            error_type=ErrorType.UNSUPPORTED_MODEL_ERROR,
            retryable=False,
            status_code=status_code,
            response_headers=response_headers,
            response_body=response_body,
            provider_name=provider_name,
            request_id=request_id
        )


class NetworkError(ProviderError):
    """Retryable error: Network connectivity issue."""
    def __init__(
        self,
        message: str = "Network error",
        status_code: Optional[int] = None,
        response_headers: Optional[Dict[str, str]] = None,
        response_body: Optional[Dict[str, Any]] = None,
        provider_name: Optional[str] = None,
        request_id: Optional[str] = None
    ):
        super().__init__(
            message=message,
            error_type=ErrorType.NETWORK_ERROR,
            retryable=True,
            status_code=status_code,
            response_headers=response_headers,
            response_body=response_body,
            provider_name=provider_name,
            request_id=request_id
        )


class UnknownError(ProviderError):
    """Retryable with caution: Unknown error."""
    def __init__(
        self,
        message: str = "Unknown error",
        status_code: Optional[int] = None,
        response_headers: Optional[Dict[str, str]] = None,
        response_body: Optional[Dict[str, Any]] = None,
        provider_name: Optional[str] = None,
        request_id: Optional[str] = None
    ):
        super().__init__(
            message=message,
            error_type=ErrorType.UNKNOWN_ERROR,
            retryable=True,
            status_code=status_code,
            response_headers=response_headers,
            response_body=response_body,
            provider_name=provider_name,
            request_id=request_id
        )


# Legacy aliases for backward compatibility
ProviderRateLimitError = RateLimitError
ProviderAuthenticationError = AuthenticationError
ProviderTimeoutError = TimeoutError
ProviderServerError = ServerError
ProviderInvalidRequestError = InvalidRequestError
ProviderUnsupportedModelError = UnsupportedModelError
ProviderNetworkError = NetworkError
ProviderUnknownError = UnknownError