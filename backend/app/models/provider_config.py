from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator


class RetryPolicy(BaseModel):
    """Retry policy configuration for a provider."""
    max_attempts: int = Field(default=3, ge=1, le=10, description="Maximum number of retry attempts")
    backoff_factor: float = Field(default=2.0, ge=1.0, le=10.0, description="Exponential backoff multiplier")
    initial_delay: float = Field(default=1.0, ge=0.1, le=60.0, description="Initial delay in seconds")
    retryable_errors: List[str] = Field(
        default_factory=lambda: ["rate_limit_error", "timeout_error", "server_error", "network_error"],
        description="Error types that should trigger a retry"
    )


class HealthCheckSettings(BaseModel):
    """Health check configuration for a provider."""
    enabled: bool = Field(default=True, description="Whether health checks are enabled")
    interval_seconds: int = Field(default=60, ge=10, le=3600, description="Health check interval in seconds")
    timeout_seconds: float = Field(default=5.0, ge=1.0, le=30.0, description="Health check timeout")
    unhealthy_threshold: int = Field(default=3, ge=1, le=10, description="Consecutive failures before marking unhealthy")
    healthy_threshold: int = Field(default=1, ge=1, le=10, description="Consecutive successes before marking healthy")


class ProviderCapabilities(BaseModel):
    """Provider capabilities metadata."""
    streaming: bool = Field(default=False, description="Whether provider supports streaming responses")
    function_calling: bool = Field(default=False, description="Whether provider supports function calling")
    vision: bool = Field(default=False, description="Whether provider supports vision/multimodal")
    parallel_requests: bool = Field(default=True, description="Whether provider supports parallel requests")


class ProviderConfig(BaseModel):
    """Configuration for a single AI provider."""
    name: str = Field(..., min_length=1, max_length=100, description="Unique provider identifier")
    adapter_type: str = Field(..., min_length=1, description="Python module path to the adapter class")
    enabled: bool = Field(default=True, description="Whether the provider is active")
    supported_models: List[str] = Field(..., min_length=1, description="List of supported model identifiers")
    priority: int = Field(default=100, ge=1, le=1000, description="Priority for provider selection (higher = preferred)")
    timeout_seconds: float = Field(default=30.0, ge=1.0, le=300.0, description="Request timeout in seconds")
    retry_policy: RetryPolicy = Field(default_factory=RetryPolicy, description="Retry configuration")
    health_check_settings: HealthCheckSettings = Field(default_factory=HealthCheckSettings, description="Health check configuration")
    routing_weight: float = Field(default=1.0, ge=0.0, le=100.0, description="Weight for load balancing (0 = excluded)")
    capabilities: ProviderCapabilities = Field(default_factory=ProviderCapabilities, description="Provider capabilities")
    env_var_prefix: Optional[str] = Field(default=None, description="Environment variable prefix for API keys (e.g., GEMINI)")
    is_local_fallback: bool = Field(
        default=False,
        description=(
            "Mark this provider as a local/on-premise fallback. "
            "Failover engine will only attempt local providers after all "
            "cloud providers have been exhausted."
        ),
    )
    
    @field_validator('name')
    @classmethod
    def name_must_be_valid_identifier(cls, v: str) -> str:
        """Validate that provider name is a valid identifier."""
        if not v.replace('-', '_').replace('_', '').isalnum():
            raise ValueError('Provider name must contain only alphanumeric characters, hyphens, and underscores')
        return v
    
    @field_validator('adapter_type')
    @classmethod
    def adapter_type_must_be_valid_module(cls, v: str) -> str:
        """Validate that adapter_type looks like a valid module path."""
        if '.' not in v:
            raise ValueError('adapter_type must be a valid module path (e.g., "app.providers.gemini.GeminiProvider")')
        return v


class ProviderRegistryConfig(BaseModel):
    """Root configuration for the provider registry."""
    providers: List[ProviderConfig] = Field(..., min_length=1, description="List of provider configurations")
    
    @field_validator('providers')
    @classmethod
    def provider_names_must_be_unique(cls, v: List[ProviderConfig]) -> List[ProviderConfig]:
        """Validate that provider names are unique."""
        names = [p.name for p in v]
        if len(names) != len(set(names)):
            duplicates = [name for name in names if names.count(name) > 1]
            raise ValueError(f'Provider names must be unique. Duplicates found: {set(duplicates)}')
        return v
    
    def get_enabled_providers(self) -> List[ProviderConfig]:
        """Get list of enabled provider configurations."""
        return [p for p in self.providers if p.enabled]
    
    def get_provider_by_name(self, name: str) -> Optional[ProviderConfig]:
        """Get provider configuration by name."""
        for provider in self.providers:
            if provider.name == name:
                return provider
        return None
