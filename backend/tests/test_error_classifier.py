import pytest
from app.core.error_classifier import classify_error, get_error_classifier, ErrorType
from app.providers.errors import ProviderError

def test_classify_400_invalid_request():
    result = classify_error("test_provider", http_status=400, raw_error_body='{"error": "bad request"}')
    assert result.error_type == ErrorType.INVALID_REQUEST
    assert result.retryable is False
    assert "Invalid request" in result.safe_message

def test_classify_401_authentication():
    result = classify_error("test_provider", http_status=401, raw_error_body='{"error": "unauthorized"}')
    assert result.error_type == ErrorType.AUTHENTICATION_ERROR
    assert result.retryable is False
    assert "Authentication failed" in result.safe_message

def test_classify_404_unsupported_model():
    result = classify_error("test_provider", http_status=404, raw_error_body='{"error": "not found"}')
    assert result.error_type == ErrorType.UNSUPPORTED_MODEL_ERROR
    assert result.retryable is False
    assert "Resource not found" in result.safe_message

def test_classify_429_rate_limit():
    result = classify_error("test_provider", http_status=429, raw_error_body='{"error": "too many requests"}')
    assert result.error_type == ErrorType.RATE_LIMIT_ERROR
    assert result.retryable is True
    assert "Rate limit exceeded" in result.safe_message

def test_classify_429_quota():
    result = classify_error("test_provider", http_status=429, raw_error_body='{"error": {"message": "quota exceeded"}}')
    assert result.error_type == ErrorType.QUOTA_EXHAUSTED_ERROR
    assert result.retryable is True
    assert "Quota exhausted" in result.safe_message

def test_classify_500_server_error():
    result = classify_error("test_provider", http_status=500, raw_error_body='{"error": "internal error"}')
    assert result.error_type == ErrorType.SERVER_ERROR
    assert result.retryable is True
    assert "Server error" in result.safe_message

def test_classify_timeout():
    result = classify_error("test_provider", original_error="Read timeout")
    assert result.error_type == ErrorType.TIMEOUT_ERROR
    assert result.retryable is True

def test_classify_network_failure():
    result = classify_error("test_provider", original_error="Connection refused")
    assert result.error_type == ErrorType.NETWORK_ERROR
    assert result.retryable is True

def test_classify_malformed_response():
    result = classify_error("test_provider", http_status=200, raw_error_body="<html><body>Not JSON</body></html>")
    # Should fall back to Unknown error because 200 without handled text falls to Unknown
    assert result.error_type == ErrorType.UNKNOWN_ERROR
    assert result.retryable is True

def test_safe_message_sanitization():
    raw_message = "Error: Invalid API key sk-1234567890abcdef1234567890abcdef. Failed to reach https://internal-api.com/v1"
    sanitized = ProviderError._sanitize_message(raw_message)
    assert "sk-1234567890abcdef1234567890abcdef" not in sanitized
    assert "[REDACTED_API_KEY]" in sanitized
    assert "https://internal-api.com/v1" not in sanitized
    assert "[REDACTED_URL]" in sanitized
    
def test_provider_error_properties():
    classification = classify_error("test", http_status=429, raw_error_body='{"msg":"too fast"}')
    classifier = get_error_classifier()
    err = classifier.create_provider_error(classification, "test", "req-123")
    
    assert err.safe_message == "Rate limit exceeded"
    assert err.error_type == ErrorType.RATE_LIMIT_ERROR
    assert err.retryable is True
    assert err.provider_metadata["provider_name"] == "test"
    assert err.provider_metadata["status_code"] == 429
    assert err.provider_metadata["request_id"] == "req-123"
    assert "msg" in err.provider_metadata["response_body"]
