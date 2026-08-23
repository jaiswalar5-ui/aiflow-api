class ProviderError(Exception):
    """Base exception for all provider-related errors."""
    pass

class ProviderRateLimitError(ProviderError):
    """Raised when a provider's rate limit is exceeded."""
    pass

class ProviderAuthenticationError(ProviderError):
    """Raised when authentication with the provider fails."""
    pass

class ProviderTimeoutError(ProviderError):
    """Raised when a provider request times out."""
    pass

class ProviderServerError(ProviderError):
    """Raised when the provider experiences an internal server error (500s)."""
    pass

class ProviderInvalidRequestError(ProviderError):
    """Raised when the provider rejects the request as invalid (400s)."""
    pass

class ProviderUnsupportedModelError(ProviderError):
    """Raised when a requested model is not supported by the provider."""
    pass

class ProviderNetworkError(ProviderError):
    """Raised when a low-level network error occurs."""
    pass

class ProviderUnknownError(ProviderError):
    """Raised when an unknown error occurs during a provider request."""
    pass
