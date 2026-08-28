# Provider Configuration Schema

This document describes the complete schema for provider configuration in `config/providers.yaml`.

## Root Structure

```yaml
providers:
  - # Provider 1 configuration
  - # Provider 2 configuration
  - # ... more providers
```

## Provider Object Schema

### Required Fields

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| `name` | string | 1-100 chars, alphanumeric + hyphens/underscores | Unique provider identifier |
| `adapter_type` | string | Valid Python module path | Module path to provider class (e.g., `app.providers.gemini.GeminiProvider`) |
| `supported_models` | array | Min 1 item | List of supported model identifiers |

### Optional Fields

| Field | Type | Default | Constraints | Description |
|-------|------|---------|-------------|-------------|
| `enabled` | boolean | `true` | - | Whether the provider is active |
| `priority` | integer | `100` | 1-1000 | Selection priority (higher = preferred) |
| `timeout_seconds` | float | `30.0` | 1.0-300.0 | Request timeout in seconds |
| `retry_policy` | object | See below | - | Retry configuration |
| `health_check_settings` | object | See below | - | Health check configuration |
| `routing_weight` | float | `1.0` | 0.0-100.0 | Load balancing weight (0 = excluded) |
| `capabilities` | object | See below | - | Provider capabilities |
| `env_var_prefix` | string | `null` | - | Environment variable prefix for API keys |

## Retry Policy Schema

```yaml
retry_policy:
  max_attempts: 3              # 1-10
  backoff_factor: 2.0          # 1.0-10.0
  initial_delay: 1.0           # 0.1-60.0 (seconds)
  retryable_errors:
    - rate_limit_error
    - timeout_error
    - server_error
    - network_error
```

### Retry Policy Fields

| Field | Type | Default | Range | Description |
|-------|------|---------|-------|-------------|
| `max_attempts` | integer | `3` | 1-10 | Maximum number of retry attempts |
| `backoff_factor` | float | `2.0` | 1.0-10.0 | Exponential backoff multiplier |
| `initial_delay` | float | `1.0` | 0.1-60.0 | Initial delay before first retry (seconds) |
| `retryable_errors` | array | See default | - | Error types that should trigger retry |

### Valid Retryable Error Types

- `rate_limit_error` - Rate limit exceeded
- `timeout_error` - Request timeout
- `server_error` - Provider server error (5xx)
- `network_error` - Network connectivity issues

## Health Check Settings Schema

```yaml
health_check_settings:
  enabled: true                # boolean
  interval_seconds: 60         # 10-3600
  timeout_seconds: 5.0         # 1.0-30.0
  unhealthy_threshold: 3       # 1-10
  healthy_threshold: 1         # 1-10
```

### Health Check Fields

| Field | Type | Default | Range | Description |
|-------|------|---------|-------|-------------|
| `enabled` | boolean | `true` | - | Whether health checks are enabled |
| `interval_seconds` | integer | `60` | 10-3600 | Time between health checks (seconds) |
| `timeout_seconds` | float | `5.0` | 1.0-30.0 | Health check request timeout |
| `unhealthy_threshold` | integer | `3` | 1-10 | Consecutive failures before marking unhealthy |
| `healthy_threshold` | integer | `1` | 1-10 | Consecutive successes before marking healthy |

## Capabilities Schema

```yaml
capabilities:
  streaming: true              # boolean
  function_calling: true       # boolean
  vision: false                # boolean
  parallel_requests: true      # boolean
```

### Capability Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `streaming` | boolean | `false` | Provider supports streaming responses |
| `function_calling` | boolean | `false` | Provider supports function calling |
| `vision` | boolean | `false` | Provider supports vision/multimodal inputs |
| `parallel_requests` | boolean | `true` | Provider supports parallel request processing |

## Complete Example

```yaml
providers:
  - name: example-provider
    adapter_type: app.providers.example.ExampleProvider
    enabled: true
    supported_models:
      - example-model-1
      - example-model-2
      - example-model-3
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
    env_var_prefix: EXAMPLE_PROVIDER
```

## Validation Rules

### Name Validation
- Must be unique across all providers
- Only alphanumeric characters, hyphens, and underscores allowed
- Must match regex: `^[a-zA-Z0-9_-]+$`

### Adapter Type Validation
- Must be a valid Python module path
- Must contain at least one dot (e.g., `module.Class`)
- Class must exist and be importable
- Class must inherit from `ProviderBase`

### Supported Models Validation
- Must contain at least one model
- Each model should be a non-empty string

### Numeric Field Ranges
- `priority`: 1-1000
- `timeout_seconds`: 1.0-300.0
- `routing_weight`: 0.0-100.0
- `retry_policy.max_attempts`: 1-10
- `retry_policy.backoff_factor`: 1.0-10.0
- `retry_policy.initial_delay`: 0.1-60.0
- `health_check_settings.interval_seconds`: 10-3600
- `health_check_settings.timeout_seconds`: 1.0-30.0
- `health_check_settings.unhealthy_threshold`: 1-10
- `health_check_settings.healthy_threshold`: 1-10

### Environment Variable Handling
- If `env_var_prefix` is set, the system looks for `{PREFIX}_API_KEY`
- If the environment variable is not found, a warning is logged
- Provider instantiation continues but may fail at runtime if API key is required

## Error Messages

### Common Validation Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `Provider names must be unique` | Duplicate provider names | Use unique names for each provider |
| `must contain only alphanumeric characters` | Invalid characters in name | Use only letters, numbers, hyphens, underscores |
| `must be a valid module path` | Invalid adapter_type | Use format `module.path.ClassName` |
| `ensure this value has at least 1 item` | Empty supported_models | Add at least one model |
| `ensure this value is less than or equal to X` | Field exceeds maximum | Reduce value to within range |
| `ensure this value is greater than or equal to X` | Field below minimum | Increase value to within range |

## Type Definitions (Python)

```python
from typing import List, Optional
from pydantic import BaseModel, Field

class RetryPolicy(BaseModel):
    max_attempts: int = Field(default=3, ge=1, le=10)
    backoff_factor: float = Field(default=2.0, ge=1.0, le=10.0)
    initial_delay: float = Field(default=1.0, ge=0.1, le=60.0)
    retryable_errors: List[str] = Field(default_factory=lambda: [
        "rate_limit_error", "timeout_error", "server_error", "network_error"
    ])

class HealthCheckSettings(BaseModel):
    enabled: bool = Field(default=True)
    interval_seconds: int = Field(default=60, ge=10, le=3600)
    timeout_seconds: float = Field(default=5.0, ge=1.0, le=30.0)
    unhealthy_threshold: int = Field(default=3, ge=1, le=10)
    healthy_threshold: int = Field(default=1, ge=1, le=10)

class ProviderCapabilities(BaseModel):
    streaming: bool = Field(default=False)
    function_calling: bool = Field(default=False)
    vision: bool = Field(default=False)
    parallel_requests: bool = Field(default=True)

class ProviderConfig(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    adapter_type: str = Field(..., min_length=1)
    enabled: bool = Field(default=True)
    supported_models: List[str] = Field(..., min_length=1)
    priority: int = Field(default=100, ge=1, le=1000)
    timeout_seconds: float = Field(default=30.0, ge=1.0, le=300.0)
    retry_policy: RetryPolicy = Field(default_factory=RetryPolicy)
    health_check_settings: HealthCheckSettings = Field(default_factory=HealthCheckSettings)
    routing_weight: float = Field(default=1.0, ge=0.0, le=100.0)
    capabilities: ProviderCapabilities = Field(default_factory=ProviderCapabilities)
    env_var_prefix: Optional[str] = Field(default=None)
```

## Migration Notes

### From Hardcoded to Config-Driven

If migrating from hardcoded provider registration:

1. **Remove hardcoded registration** from `app/main.py`:
   ```python
   # OLD (remove this)
   from app.providers.gemini import GeminiProvider
   provider_registry.register_provider("gemini", GeminiProvider())
   ```

2. **Add config entry** to `config/providers.yaml`:
   ```yaml
   - name: gemini
     adapter_type: app.providers.gemini.GeminiProvider
     # ... other fields
   ```

3. **Update settings** to include config path (optional):
   ```python
   # In app/core/config.py
   PROVIDER_CONFIG_PATH: str | None = None  # Uses default if None
   ENABLE_HOT_RELOAD: bool = True
   ```

The system will automatically load providers from config on startup.