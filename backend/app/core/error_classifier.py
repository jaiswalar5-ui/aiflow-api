from typing import Optional, Dict, Any, Tuple
from enum import Enum
import json

from app.providers.errors import (
    ErrorType,
    InvalidRequestError,
    AuthenticationError,
    RateLimitError,
    QuotaExhaustedError,
    TimeoutError,
    ProviderUnavailableError,
    ServerError,
    UnsupportedModelError,
    NetworkError,
    UnknownError,
    ProviderError
)
from app.core.logging import get_logger

logger = get_logger(__name__)


class ErrorClassificationResult:
    """Result of error classification."""
    def __init__(
        self,
        error_type: ErrorType,
        retryable: bool,
        safe_message: str,
        original_error: Optional[str] = None,
        backoff_hint: Optional[int] = None,
        provider_metadata: Optional[Dict[str, Any]] = None
    ):
        self.error_type = error_type
        self.retryable = retryable
        self.safe_message = safe_message
        self.original_error = original_error
        self.backoff_hint = backoff_hint
        self.provider_metadata = provider_metadata or {}


class ErrorClassifier:
    """
    Centralized error classification engine.
    Maps vendor-specific errors to normalized error types with retry logic.
    """
    
    def __init__(self):
        self._vendor_classifiers = {
            "gemini": self._classify_gemini_error,
            "groq": self._classify_groq_error,
            "mock": self._classify_mock_error,
        }
    
    def classify_error(
        self,
        provider_name: str,
        http_status: Optional[int] = None,
        raw_error_body: Optional[str] = None,
        response_headers: Optional[Dict[str, str]] = None,
        original_error: Optional[str] = None
    ) -> ErrorClassificationResult:
        """
        Classify an error from a provider into normalized error type.
        
        Args:
            provider_name: Name of the provider
            http_status: HTTP status code if available
            raw_error_body: Raw error response body
            response_headers: Response headers
            original_error: Original error message for context
            
        Returns:
            ErrorClassificationResult with normalized error information
        """
        try:
            # Parse JSON body if available
            parsed_body = None
            if raw_error_body:
                try:
                    parsed_body = json.loads(raw_error_body)
                except json.JSONDecodeError:
                    parsed_body = {"raw": raw_error_body}
            
            # Use vendor-specific classifier if available
            classifier = self._vendor_classifiers.get(provider_name.lower())
            if classifier:
                return classifier(http_status, parsed_body, response_headers, original_error)
            
            # Fallback to generic classification
            return self._classify_generic_error(http_status, parsed_body, response_headers, original_error)
            
        except Exception as e:
            logger.error(f"Error during classification: {e}")
            # Return unknown error on classification failure
            return ErrorClassificationResult(
                error_type=ErrorType.UNKNOWN_ERROR,
                retryable=True,
                safe_message="An unknown error occurred",
                original_error=original_error
            )
    
    def _classify_generic_error(
        self,
        http_status: Optional[int],
        parsed_body: Optional[Dict[str, Any]],
        response_headers: Optional[Dict[str, str]],
        original_error: Optional[str]
    ) -> ErrorClassificationResult:
        """Generic error classification based on HTTP status codes."""
        if http_status is None:
            return ErrorClassificationResult(
                error_type=ErrorType.UNKNOWN_ERROR,
                retryable=True,
                safe_message="An unknown error occurred",
                original_error=original_error
            )
        
        status = http_status
        
        # Client errors (4xx)
        if 400 <= status < 500:
            if status == 400:
                return ErrorClassificationResult(
                    error_type=ErrorType.INVALID_REQUEST,
                    retryable=False,
                    safe_message="Invalid request",
                    original_error=original_error
                )
            elif status == 401 or status == 403:
                return ErrorClassificationResult(
                    error_type=ErrorType.AUTHENTICATION_ERROR,
                    retryable=False,
                    safe_message="Authentication failed",
                    original_error=original_error
                )
            elif status == 404:
                return ErrorClassificationResult(
                    error_type=ErrorType.UNSUPPORTED_MODEL_ERROR,
                    retryable=False,
                    safe_message="Resource not found",
                    original_error=original_error
                )
            elif status == 429:
                # Determine if it's rate limit or quota exhausted
                backoff_hint = self._extract_retry_after(response_headers)
                if self._is_quota_exhausted(parsed_body, response_headers):
                    return ErrorClassificationResult(
                        error_type=ErrorType.QUOTA_EXHAUSTED_ERROR,
                        retryable=True,
                        safe_message="Quota exhausted",
                        original_error=original_error,
                        backoff_hint=backoff_hint
                    )
                else:
                    return ErrorClassificationResult(
                        error_type=ErrorType.RATE_LIMIT_ERROR,
                        retryable=True,
                        safe_message="Rate limit exceeded",
                        original_error=original_error,
                        backoff_hint=backoff_hint
                    )
            else:
                return ErrorClassificationResult(
                    error_type=ErrorType.INVALID_REQUEST,
                    retryable=False,
                    safe_message=f"Client error: {status}",
                    original_error=original_error
                )
        
        # Server errors (5xx)
        elif 500 <= status < 600:
            if status == 503:
                return ErrorClassificationResult(
                    error_type=ErrorType.PROVIDER_UNAVAILABLE_ERROR,
                    retryable=True,
                    safe_message="Service unavailable",
                    original_error=original_error
                )
            else:
                return ErrorClassificationResult(
                    error_type=ErrorType.SERVER_ERROR,
                    retryable=True,
                    safe_message="Server error",
                    original_error=original_error
                )
        
        # Unknown status
        return ErrorClassificationResult(
            error_type=ErrorType.UNKNOWN_ERROR,
            retryable=True,
            safe_message=f"Unknown error with status {status}",
            original_error=original_error
        )
    
    def _classify_gemini_error(
        self,
        http_status: Optional[int],
        parsed_body: Optional[Dict[str, Any]],
        response_headers: Optional[Dict[str, str]],
        original_error: Optional[str]
    ) -> ErrorClassificationResult:
        """Classify Gemini-specific errors."""
        # Gemini-specific error codes and messages
        if parsed_body:
            error_code = parsed_body.get("error", {}).get("code")
            error_message = parsed_body.get("error", {}).get("message", "").lower()
            
            if error_code == "QUOTA_EXCEEDED" or "quota" in error_message:
                return ErrorClassificationResult(
                    error_type=ErrorType.QUOTA_EXHAUSTED_ERROR,
                    retryable=True,
                    safe_message="Gemini quota exceeded",
                    original_error=original_error,
                    backoff_hint=self._extract_retry_after(response_headers)
                )
            elif error_code == "RATE_LIMIT_EXCEEDED" or "rate limit" in error_message:
                return ErrorClassificationResult(
                    error_type=ErrorType.RATE_LIMIT_ERROR,
                    retryable=True,
                    safe_message="Gemini rate limit exceeded",
                    original_error=original_error,
                    backoff_hint=self._extract_retry_after(response_headers)
                )
        
        # Fallback to generic classification
        return self._classify_generic_error(http_status, parsed_body, response_headers, original_error)
    
    def _classify_groq_error(
        self,
        http_status: Optional[int],
        parsed_body: Optional[Dict[str, Any]],
        response_headers: Optional[Dict[str, str]],
        original_error: Optional[str]
    ) -> ErrorClassificationResult:
        """Classify Groq-specific errors."""
        # Groq-specific error codes
        if parsed_body:
            error_type = parsed_body.get("error", {}).get("type", "")
            error_message = parsed_body.get("error", {}).get("message", "").lower()
            
            if error_type == "rate_limit_exceeded" or "rate limit" in error_message:
                # Check if it's quota vs rate limit
                if "quota" in error_message or "usage" in error_message:
                    return ErrorClassificationResult(
                        error_type=ErrorType.QUOTA_EXHAUSTED_ERROR,
                        retryable=True,
                        safe_message="Groq quota exceeded",
                        original_error=original_error,
                        backoff_hint=self._extract_retry_after(response_headers)
                    )
                else:
                    return ErrorClassificationResult(
                        error_type=ErrorType.RATE_LIMIT_ERROR,
                        retryable=True,
                        safe_message="Groq rate limit exceeded",
                        original_error=original_error,
                        backoff_hint=self._extract_retry_after(response_headers)
                    )
            elif error_type == "invalid_request" or "invalid" in error_message:
                return ErrorClassificationResult(
                    error_type=ErrorType.INVALID_REQUEST,
                    retryable=False,
                    safe_message="Invalid request to Groq",
                    original_error=original_error
                )
        
        # Fallback to generic classification
        return self._classify_generic_error(http_status, parsed_body, response_headers, original_error)
    
    def _classify_mock_error(
        self,
        http_status: Optional[int],
        parsed_body: Optional[Dict[str, Any]],
        response_headers: Optional[Dict[str, str]],
        original_error: Optional[str]
    ) -> ErrorClassificationResult:
        """Classify Mock provider errors (for testing)."""
        # Mock provider errors are predictable based on the test setup
        if original_error and "rate limit" in original_error.lower():
            return ErrorClassificationResult(
                error_type=ErrorType.RATE_LIMIT_ERROR,
                retryable=True,
                safe_message="Mock rate limit",
                original_error=original_error
            )
        
        return self._classify_generic_error(http_status, parsed_body, response_headers, original_error)
    
    def _extract_retry_after(self, response_headers: Optional[Dict[str, str]]) -> Optional[int]:
        """Extract retry-after value from headers."""
        if not response_headers:
            return None
        
        retry_after = response_headers.get("Retry-After") or response_headers.get("retry-after")
        if retry_after:
            try:
                return int(retry_after)
            except (ValueError, TypeError):
                pass
        
        return None
    
    def _is_quota_exhausted(
        self,
        parsed_body: Optional[Dict[str, Any]],
        response_headers: Optional[Dict[str, str]]
    ) -> bool:
        """Determine if error indicates quota exhaustion vs rate limiting."""
        if parsed_body:
            # Check for quota-specific keywords
            error_data = parsed_body.get("error", {})
            error_message = error_data.get("message", "").lower()
            error_code = error_data.get("code", "").lower()
            
            quota_keywords = ["quota", "usage", "credits", "limit reached", "exceeded"]
            if any(keyword in error_message or keyword in error_code for keyword in quota_keywords):
                return True
        
        if response_headers:
            # Check for quota-related headers
            remaining = response_headers.get("X-RateLimit-Remaining") or response_headers.get("x-ratelimit-remaining")
            if remaining and int(remaining) == 0:
                return True
        
        return False
    
    def create_provider_error(
        self,
        classification: ErrorClassificationResult,
        provider_name: str,
        request_id: Optional[str] = None
    ) -> ProviderError:
        """
        Create a ProviderError instance from classification result.
        
        Args:
            classification: Error classification result
            provider_name: Name of the provider
            request_id: Request identifier
            
        Returns:
            Appropriate ProviderError subclass
        """
        error_classes = {
            ErrorType.INVALID_REQUEST: InvalidRequestError,
            ErrorType.AUTHENTICATION_ERROR: AuthenticationError,
            ErrorType.RATE_LIMIT_ERROR: RateLimitError,
            ErrorType.QUOTA_EXHAUSTED_ERROR: QuotaExhaustedError,
            ErrorType.TIMEOUT_ERROR: TimeoutError,
            ErrorType.PROVIDER_UNAVAILABLE_ERROR: ProviderUnavailableError,
            ErrorType.SERVER_ERROR: ServerError,
            ErrorType.UNSUPPORTED_MODEL_ERROR: UnsupportedModelError,
            ErrorType.NETWORK_ERROR: NetworkError,
            ErrorType.UNKNOWN_ERROR: UnknownError,
        }
        
        error_class = error_classes.get(classification.error_type, UnknownError)
        
        return error_class(
            message=classification.safe_message,
            status_code=classification.provider_metadata.get("status_code") if classification.provider_metadata else None,
            response_headers=classification.provider_metadata.get("response_headers"),
            response_body=classification.provider_metadata.get("response_body"),
            provider_name=provider_name,
            request_id=request_id
        )


# Global error classifier instance
_error_classifier: Optional[ErrorClassifier] = None


def get_error_classifier() -> ErrorClassifier:
    """Get the global error classifier instance."""
    global _error_classifier
    if _error_classifier is None:
        _error_classifier = ErrorClassifier()
    return _error_classifier


def set_error_classifier(classifier: ErrorClassifier) -> None:
    """Set the global error classifier instance (for testing)."""
    global _error_classifier
    _error_classifier = classifier


def classify_error(
    provider_name: str,
    http_status: Optional[int] = None,
    raw_error_body: Optional[str] = None,
    response_headers: Optional[Dict[str, str]] = None,
    original_error: Optional[str] = None
) -> ErrorClassificationResult:
    """
    Convenience function to classify an error using the global classifier.
    
    Args:
        provider_name: Name of the provider
        http_status: HTTP status code if available
        raw_error_body: Raw error response body
        response_headers: Response headers
        original_error: Original error message
        
    Returns:
        ErrorClassificationResult with normalized error information
    """
    classifier = get_error_classifier()
    return classifier.classify_error(
        provider_name=provider_name,
        http_status=http_status,
        raw_error_body=raw_error_body,
        response_headers=response_headers,
        original_error=original_error
    )