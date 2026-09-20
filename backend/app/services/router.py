"""
Routing Engine — Task 13 update

Replaces the single-provider dispatch with FailoverEngine.execute_with_failover.
The failover engine handles:
  - ordered provider iteration (cloud-first, local-last)
  - per-provider retry / backoff (Task 12 RetryEngine)
  - error classification gating (Task 11)
  - structured logs with correlation_id
  - 503 AllProvidersExhaustedError when everything fails
"""
from typing import Optional
import uuid

from fastapi import HTTPException

from app.models.chat import ChatCompletionRequest, ChatCompletionResponse
from app.providers.config_registry import configurable_registry
from app.providers.errors import (
    AllProvidersExhaustedError,
    ProviderRateLimitError,
    ProviderUnsupportedModelError,
)
from app.core.quota_manager import get_quota_manager
from app.core.failover_engine import FailoverEngine, FailoverPolicy, get_failover_engine
from app.core.logging import get_logger

logger = get_logger(__name__)


class RoutingEngine:
    """
    Core routing engine — delegates multi-provider orchestration to FailoverEngine.

    Keeps eligibility selection (priority order, quota health) here and passes
    the ordered provider list to the failover engine, which owns the retry /
    failover lifecycle.
    """

    def __init__(self, registry=None, quota_manager=None, failover_engine: Optional[FailoverEngine] = None):
        self.registry = registry or configurable_registry
        self.quota_manager = quota_manager or get_quota_manager()
        self._failover_engine = failover_engine or get_failover_engine()

    async def route_chat_completion(
        self,
        request: ChatCompletionRequest,
        target_provider: Optional[str] = None,
        correlation_id: Optional[str] = None,
    ) -> ChatCompletionResponse:
        """
        Route a chat completion request through the failover engine.

        Args:
            request:          Chat completion request.
            target_provider:  Pin to a specific provider (bypasses failover ordering).
            correlation_id:   Caller-supplied request ID; auto-generated when absent.

        Returns:
            ChatCompletionResponse from the first successful provider.

        Raises:
            HTTPException 503: When all providers are exhausted.
        """
        cid = correlation_id or str(uuid.uuid4())

        await self.registry.increment_active_requests()
        try:
            if target_provider:
                # Pinned: single provider, no failover ordering
                cloud_providers = [target_provider]
                local_providers: list[str] = []
            else:
                cloud_providers, local_providers = await self._build_ordered_provider_lists(request)

            # Build provider → supported-models map for unsupported-model guard
            provider_models = {
                name: self.registry.get_provider(name).get_supported_models()
                for name in cloud_providers + local_providers
            }

            model = request.model if request.model != "default" else None

            def func_factory(provider_name: str):
                """Return a no-arg async callable for the retry engine."""
                provider = self.registry.get_provider(provider_name)

                async def _call():
                    # Pre-flight check: verify model is supported
                    if request.model != "default" and request.model not in provider.get_supported_models():
                        raise ProviderUnsupportedModelError(
                            f"Model '{request.model}' not supported by provider '{provider_name}'",
                            provider_name=provider_name
                        )

                    response = await provider.send_chat_completion(request)
                    # Record successful quota usage
                    tokens_used = (response.usage.total_tokens or 0) if response.usage else 0
                    await self.quota_manager.record_request(provider_name, tokens_used)
                    return response

                return _call

            response = await self._failover_engine.execute_with_failover(
                func_factory=func_factory,
                cloud_providers=cloud_providers,
                local_providers=local_providers,
                correlation_id=cid,
                model=model,
                provider_models=provider_models,
            )
            return response

        except AllProvidersExhaustedError as exc:
            logger.error(
                "All providers exhausted — returning 503.",
                extra={
                    "correlation_id": cid,
                    "error_code": exc.error_type.value,
                },
            )
            raise HTTPException(
                status_code=503,
                detail={
                    "error_code": exc.error_type.value,
                    "message": (
                        "The request could not be completed. All configured "
                        "providers are currently unavailable. Please try again later."
                    ),
                    "request_id": cid,
                },
            )

        except ProviderRateLimitError as exc:
            # Record rate limit event so quota manager can cool down the provider
            provider_name = exc.provider_name or "unknown"
            await self.quota_manager.record_rate_limit(
                provider_name,
                status_code=exc.status_code or 429,
                response_headers=exc.response_headers,
                response_body=exc.response_body,
            )
            logger.warning(
                "Rate limit propagated upstream.",
                extra={"correlation_id": cid, "provider": provider_name},
            )
            raise

        finally:
            await self.registry.decrement_active_requests()

    # ------------------------------------------------------------------
    # Provider ordering helpers
    # ------------------------------------------------------------------

    async def _build_ordered_provider_lists(
        self, request: ChatCompletionRequest
    ) -> tuple[list[str], list[str]]:
        """
        Return (cloud_providers, local_providers) sorted by priority and quota health.

        Cloud providers are tried first (highest-priority → lowest-priority).
        Local/fallback providers are returned separately and tried last.
        """
        all_by_priority = self.registry.get_provider_by_priority()

        # Separate local from cloud
        cloud: list[str] = []
        local: list[str] = []
        for name in all_by_priority:
            cfg = self.registry.get_provider_config(name)
            if cfg and cfg.is_local_fallback:
                local.append(name)
            else:
                cloud.append(name)

        # Prefer quota-healthy cloud providers but keep all as failover candidates
        try:
            available = await self.quota_manager.get_available_providers(cloud)
        except Exception:
            available = cloud

        if available:
            # Put healthy providers first, then the rest (still available as fallback)
            unhealthy = [p for p in cloud if p not in available]
            cloud = available + unhealthy

        return cloud, local

    # ------------------------------------------------------------------
    # Introspection helpers (unchanged interface)
    # ------------------------------------------------------------------

    def get_available_providers(self) -> list:
        """Get list of available provider names."""
        return self.registry.list_providers()

    def get_provider_config(self, provider_name: str):
        """Get configuration for a specific provider."""
        return self.registry.get_provider_config(provider_name)

    def get_providers_by_capability(self, capability: str) -> list:
        """Get providers that support a specific capability."""
        return self.registry.get_providers_by_capability(capability)

    async def get_quota_status(self, provider_name: str) -> dict:
        """Get quota status for a specific provider."""
        return await self.quota_manager.get_quota_status_for_routing(provider_name)

    async def get_all_quota_status(self) -> dict:
        """Get quota status for all providers."""
        providers = self.registry.list_providers()
        status = {}
        for provider_name in providers:
            status[provider_name] = await self.quota_manager.get_quota_status_for_routing(provider_name)
        return status


# Singleton routing engine using configurable registry and quota manager
router_engine = RoutingEngine()
