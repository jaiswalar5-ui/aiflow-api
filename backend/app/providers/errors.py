class ProviderError(Exception):
    """Base exception for all provider-related errors."""
    def __init__(self, message: str, status_code: int = None, response_headers: dict = None, response_body: dict = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_headers = response_headers or {}
        self.response_body = response_body or {}

class ProviderRateLimitError(ProviderError):
    """Raised when a provider's rate limit is exceeded."""
    def __init__(self, message: str, status_code: int = 429, response_headers: dict = None, response_body: dict = None):
        super().__init__(message, status_code, response_headers, response_body)

class ProviderAuthenticationError(ProviderError):
    """Raised when authentication with the provider fails."""
    def __init__(self, message: str, status_code: int = 401, response_headers: dict = None, response_body: dict = None):
        super().__init__(message, status_code, response_headers, response_body)

class ProviderTimeoutError(ProviderError):
    """Raised when a provider request times out."""
    def __init__(self, message: str, status_code: int = None, response_headers: dict = None, response_body: dict = None):
        super().__init__(message, status_code, response_headers, response_body)

class ProviderServerError(ProviderError):
    """Raised when the provider experiences an internal server error (500s)."""
    def __init__(self, message: str, status_code: int = 500, response_headers: dict = None, response_body: dict = None):
        super().__init__(message, status_code, response_headers, response_body)

class ProviderInvalidRequestError(ProviderError):
    """Raised when the provider rejects the request as invalid (400s)."""
    def __init__(self, message: str, status_code: int = 400, response_headers: dict = None, response_body: dict = None):
        super().__init__(message, status_code, response_headers, response_body)

class ProviderUnsupportedModelError(ProviderError):
    """Raised when a requested model is not supported by the provider."""
    def __init__(self, message: str, status_code: int = 404, response_headers: dict = None, response_body: dict = None):
        super().__init__(message, status_code, response_headers, response_body)

class ProviderNetworkError(ProviderError):
    """Raised when a low-level network error occurs."""
    def __init__(self, message: str, status_code: int = None, response_headers: dict = None, response_body: dict = None):
        super().__init__(message, status_code, response_headers, response_body)

class ProviderUnknownError(ProviderError):
    """Raised when an unknown error occurs during a provider request."""
    def __init__(self, message: str, status_code: int = None, response_headers: dict = None, response_body: dict = None):
        super().__init__(message, status_code, response_headers, response_body)
