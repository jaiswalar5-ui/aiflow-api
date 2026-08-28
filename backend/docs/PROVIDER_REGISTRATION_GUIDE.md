# Provider Registration Guide

This guide explains how to add a new AI provider to the AIFlow backend using the config-driven provider registry system.

## Overview

AIFlow uses a configuration-driven provider registry that allows you to add new providers without modifying core application code. The system supports:

- **YAML-based configuration**: Define providers in `config/providers.yaml`
- **Hot-reload**: Changes to config are applied without restart
- **Validation**: Config is validated before applying changes
- **Environment-based secrets**: API keys come from environment variables, not config files
- **Extensibility**: Support for 20+ providers without code changes

## Architecture

The provider system consists of three main components:

1. **Provider Adapter** (`app/providers/your_provider.py`): Implements the provider-specific logic
2. **Configuration** (`config/providers.yaml`): Defines provider metadata and settings
3. **Registry Service** (`app/providers/config_registry.py`): Loads config and manages provider instances

## Step-by-Step Guide

### Step 1: Create the Provider Adapter

Create a new file in `app/providers/` that implements the `ProviderBase` interface:

```python
# app/providers/your_provider.py
import time
import uuid
import httpx
from typing import List, Optional

from app.providers.base import ProviderBase
from app.providers.errors import (
    ProviderError,
    ProviderAuthenticationError,
    ProviderRateLimitError,
    # ... other error types
)
from app.models.chat import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatChoice,
    ChatMessage,
    ChatCompletionUsage,
)

class YourProvider(ProviderBase):
    """
    Provider adapter for YourProvider REST API.
    """
    def __init__(self, timeout_seconds: float = 30.0, api_key: Optional[str] = None):
        self.timeout = timeout_seconds
        self.api_key = api_key
        self.base_url = "https://api.yourprovider.com/v1"
        self.supported_models = ["your-model-1", "your-model-2"]

    def get_supported_models(self) -> List[str]:
        return self.supported_models

    async def check_health(self) -> bool:
        if not self.api_key:
            return False
        # Implement health check logic
        return True

    async def send_chat_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        if not self.api_key:
            raise ProviderAuthenticationError("YOUR_PROVIDER_API_KEY is not configured.")
        
        # Implement your provider-specific logic here
        # 1. Translate request to provider format
        # 2. Make HTTP request
        # 3. Handle errors
        # 4. Convert response to canonical format
        
        # Example implementation:
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=self._translate_request(request)
                )
                response.raise_for_status()
                return self._convert_response(response.json(), request.model)
                
        except httpx.HTTPStatusError as exc:
            self._handle_http_error(exc)
        except Exception as exc:
            raise ProviderError(f"Unexpected error: {exc}") from exc

    def _translate_request(self, request: ChatCompletionRequest) -> dict:
        """Translate canonical format to provider-specific format."""
        # Implement translation logic
        pass

    def _convert_response(self, response_data: dict, model_id: str) -> ChatCompletionResponse:
        """Convert provider response to canonical format."""
        # Implement conversion logic
        pass

    def _handle_http_error(self, exc: httpx.HTTPStatusError):
        """Map HTTP errors to standardized provider errors."""
        status = exc.response.status_code
        if status in (401, 403):
            raise ProviderAuthenticationError(f"Authentication failed: {exc.response.text}") from exc
        elif status == 429:
            raise ProviderRateLimitError(f"Rate limit exceeded: {exc.response.text}") from exc
        # ... handle other status codes
```

**Key Requirements:**

- Inherit from `ProviderBase`
- Implement `__init__` with `timeout_seconds` and optional `api_key` parameters
- Implement `get_supported_models()` to return list of supported model IDs
- Implement `send_chat_completion()` to handle chat requests
- Raise appropriate errors from `app.providers.errors`
- Convert responses to canonical `ChatCompletionResponse` format

### Step 2: Add Provider Configuration

Add your provider to `config/providers.yaml`:

```yaml
providers:
  # ... existing providers ...
  
  - name: your-provider
    adapter_type: app.providers.your_provider.YourProvider
    enabled: true
    supported_models:
      - your-model-1
      - your-model-2
      - your-model-3
    priority: 100
    timeout_seconds: 30.0
    retry_policy:
      max_attempts: 3
      backoff_factor: 2.0
      initial_delay: 1.0
      retryable_errors:
        - rate_limit_error
        - timeout_error
        - server_error
        - network_error
    health_check_settings:
      enabled: true
      interval_seconds: 60
      timeout_seconds: 5.0
      unhealthy_threshold: 3
      healthy_threshold: 1
    routing_weight: 1.0
    capabilities:
      streaming: true
      function_calling: true
      vision: false
      parallel_requests: true
    env_var_prefix: YOUR_PROVIDER
```

**Configuration Fields:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Unique provider identifier (alphanumeric, hyphens, underscores only) |
| `adapter_type` | string | Yes | Python module path to your provider class |
| `enabled` | boolean | No | Whether the provider is active (default: true) |
| `supported_models` | list | Yes | List of model identifiers this provider supports |
| `priority` | integer | No | Selection priority (1-1000, higher = preferred) |
| `timeout_seconds` | float | No | Request timeout (1-300 seconds, default: 30.0) |
| `retry_policy` | object | No | Retry configuration (see below) |
| `health_check_settings` | object | No | Health check configuration (see below) |
| `routing_weight` | float | No | Load balancing weight (0-100, default: 1.0) |
| `capabilities` | object | No | Provider capabilities (see below) |
| `env_var_prefix` | string | No | Environment variable prefix for API keys |

**Retry Policy Fields:**

| Field | Type | Default | Range | Description |
|-------|------|---------|-------|-------------|
| `max_attempts` | integer | 3 | 1-10 | Maximum retry attempts |
| `backoff_factor` | float | 2.0 | 1.0-10.0 | Exponential backoff multiplier |
| `initial_delay` | float | 1.0 | 0.1-60.0 | Initial delay in seconds |
| `retryable_errors` | list | - | - | Error types to retry |

**Health Check Settings:**

| Field | Type | Default | Range | Description |
|-------|------|---------|-------|-------------|
| `enabled` | boolean | true | - | Whether health checks are enabled |
| `interval_seconds` | integer | 60 | 10-3600 | Check interval in seconds |
| `timeout_seconds` | float | 5.0 | 1.0-30.0 | Health check timeout |
| `unhealthy_threshold` | integer | 3 | 1-10 | Failures before marking unhealthy |
| `healthy_threshold` | integer | 1 | 1-10 | Successes before marking healthy |

**Capabilities:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `streaming` | boolean | false | Supports streaming responses |
| `function_calling` | boolean | false | Supports function calling |
| `vision` | boolean | false | Supports vision/multimodal |
| `parallel_requests` | boolean | true | Supports parallel requests |

### Step 3: Set Environment Variables

Add your API key to environment variables (NOT in config file):

```bash
# In .env file or environment
YOUR_PROVIDER_API_KEY=your_api_key_here
```

The system will automatically inject this into your provider instance if `env_var_prefix` is set.

### Step 4: Test Your Provider

Create a test file to verify your provider works:

```python
# tests/test_your_provider.py
import pytest
from app.providers.your_provider import YourProvider
from app.models.chat import ChatCompletionRequest, ChatMessage

@pytest.mark.asyncio
async def test_your_provider_basic():
    provider = YourProvider(api_key="test_key")
    
    request = ChatCompletionRequest(
        messages=[ChatMessage(role="user", content="Hello")],
        model="your-model-1"
    )
    
    # Test basic functionality
    assert "your-model-1" in provider.get_supported_models()
```

### Step 5: Reload Configuration

If the application is running, the hot-reload feature will automatically detect the config change and attempt to reload:

1. Modify `config/providers.yaml`
2. The system validates the new configuration
3. If valid, it creates new provider instances
4. If invalid, it rejects the change and keeps the old configuration
5. Active requests are not dropped during reload

## Validation Rules

The system enforces these validation rules:

- **Provider names**: Must be unique, contain only alphanumeric characters, hyphens, and underscores
- **Adapter type**: Must be a valid Python module path (e.g., `app.providers.your_provider.YourProvider`)
- **Supported models**: Must have at least one model
- **Retry policy**: `max_attempts` must be 1-10, `backoff_factor` 1.0-10.0, `initial_delay` 0.1-60.0
- **Health check**: `interval_seconds` 10-3600, `timeout_seconds` 1.0-30.0
- **Priority**: Must be 1-1000
- **Timeout**: Must be 1-300 seconds
- **Routing weight**: Must be 0-100

## Error Handling

Your provider should raise appropriate errors from `app.providers.errors`:

- `ProviderAuthenticationError`: Invalid API key or authentication failure
- `ProviderRateLimitError`: Rate limit exceeded
- `ProviderInvalidRequestError`: Invalid request parameters
- `ProviderUnsupportedModelError`: Model not supported
- `ProviderServerError`: Provider server error (5xx)
- `ProviderTimeoutError`: Request timeout
- `ProviderNetworkError`: Network connectivity issues
- `ProviderUnknownError`: Other unexpected errors

## Examples

See existing providers for reference:

- **Gemini**: `app/providers/gemini.py` - Google Gemini API
- **Groq**: `app/providers/groq.py` - Groq API
- **Mock**: `app/providers/mock.py` - Testing/mock provider

## Troubleshooting

### Provider not loading

1. Check `config/providers.yaml` syntax
2. Verify `adapter_type` path is correct
3. Check logs for validation errors
4. Ensure API key environment variable is set

### Hot-reload not working

1. Verify `ENABLE_HOT_RELOAD=true` in settings
2. Check file permissions on `config/providers.yaml`
3. Review logs for reload errors

### API key not working

1. Verify environment variable name matches `env_var_prefix` + `_API_KEY`
2. Check environment variable is loaded
3. Ensure provider `__init__` accepts `api_key` parameter

## Best Practices

1. **Security**: Never commit API keys to the repository
2. **Validation**: Test your provider with invalid inputs
3. **Error handling**: Map provider-specific errors to standard errors
4. **Timeouts**: Use reasonable timeout values (30-60 seconds)
5. **Health checks**: Implement lightweight health checks
6. **Documentation**: Document model capabilities and limitations
7. **Testing**: Write comprehensive tests for your provider

## Summary

Adding a new provider requires:

1. ✅ Create provider adapter class inheriting from `ProviderBase`
2. ✅ Add configuration to `config/providers.yaml`
3. ✅ Set environment variable for API key
4. ✅ Test the provider implementation
5. ✅ Hot-reload picks up changes automatically

The system handles validation, instantiation, and routing automatically. No core code changes needed!