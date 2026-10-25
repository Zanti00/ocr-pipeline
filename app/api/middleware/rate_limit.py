import asyncio
import logging
import math
import time
from dataclasses import dataclass

import redis.asyncio as aioredis
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from app.config import settings
from app.dependencies import resolve_service_from_token

logger = logging.getLogger(__name__)

# Atomic Redis Lua script to increment rate counter and set expire if new key.
# Returns: [current_count, ttl_remaining_seconds]
# KEYS[1]: rate limit key
# ARGV[1]: window expiration in seconds
_REDIS_RATE_LIMIT_LUA = """
local current = redis.call('INCR', KEYS[1])
if current == 1 then
    redis.call('EXPIRE', KEYS[1], ARGV[1])
end
local ttl = redis.call('TTL', KEYS[1])
if ttl < 0 then
    redis.call('EXPIRE', KEYS[1], ARGV[1])
    ttl = tonumber(ARGV[1])
end
return {current, ttl}
"""


@dataclass(frozen=True)
class RateLimitResult:
    allowed: bool
    limit: int
    remaining: int
    reset_after: int


class InMemoryRateLimitBackend:
    """
    Thread-safe in-memory sliding fixed-window fallback when Redis is unavailable or unconfigured.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        # key -> (current_count, window_start_monotonic)
        self._entries: dict[str, tuple[int, float]] = {}

    async def check_and_increment(self, key: str, limit: int, window_seconds: int) -> RateLimitResult:
        async with self._lock:
            now = time.monotonic()
            entry = self._entries.get(key)

            if entry is None or (now - entry[1]) >= window_seconds:
                # New window
                count = 1
                self._entries[key] = (count, now)
                reset_after = window_seconds
            else:
                count = entry[0] + 1
                elapsed = now - entry[1]
                reset_after = max(1, math.ceil(window_seconds - elapsed))
                self._entries[key] = (count, entry[1])

            # Periodic cleanup of expired entries if map gets large
            if len(self._entries) > 2048:
                expired = [k for k, v in self._entries.items() if (now - v[1]) >= window_seconds]
                for k in expired:
                    self._entries.pop(k, None)

            allowed = count <= limit
            remaining = max(0, limit - count)
            return RateLimitResult(
                allowed=allowed,
                limit=limit,
                remaining=remaining,
                reset_after=reset_after,
            )

    async def reset(self) -> None:
        async with self._lock:
            self._entries.clear()


class RedisRateLimitBackend:
    """
    Redis-backed rate limiting using atomic Lua scripts with automatic in-memory fallback on failure.
    """

    def __init__(self, redis_url: str) -> None:
        self.redis_url = redis_url
        self._client: aioredis.Redis | None = None
        self._script = None
        self._fallback = InMemoryRateLimitBackend()
        self._redis_warning_logged = False

    def _get_client(self) -> aioredis.Redis:
        if self._client is None:
            self._client = aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=1.0,
                socket_timeout=1.0,
            )
            self._script = self._client.register_script(_REDIS_RATE_LIMIT_LUA)
        return self._client

    async def check_and_increment(self, key: str, limit: int, window_seconds: int) -> RateLimitResult:
        try:
            client = self._get_client()
            if self._script is None:
                self._script = client.register_script(_REDIS_RATE_LIMIT_LUA)

            result = await self._script(keys=[key], args=[window_seconds])
            current_count = int(result[0])
            ttl = int(result[1])
            reset_after = max(1, ttl)

            allowed = current_count <= limit
            remaining = max(0, limit - current_count)
            return RateLimitResult(
                allowed=allowed,
                limit=limit,
                remaining=remaining,
                reset_after=reset_after,
            )
        except Exception as exc:  # noqa: BLE001
            if not self._redis_warning_logged:
                logger.warning(
                    "Redis rate limiter failed (%s). Falling back to in-memory rate limiter.",
                    exc,
                )
                self._redis_warning_logged = True
            return await self._fallback.check_and_increment(key, limit, window_seconds)

    async def close(self) -> None:
        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception as exc:  # noqa: BLE001
                logger.debug("Error closing Redis client: %s", exc)
            self._client = None
            self._script = None

    async def reset(self) -> None:
        await self._fallback.reset()
        if self._client is not None:
            try:
                # Only clear ratelimit namespace
                keys = await self._client.keys("ocr:ratelimit:*")
                if keys:
                    await self._client.delete(*keys)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Error resetting Redis rate limit keys: %s", exc)


# Global backend singleton instance
rate_limit_backend = RedisRateLimitBackend(settings.redis_url)


def _get_client_ip(request: Request) -> str:
    """Extract client IP handling potential X-Forwarded-For header."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def _extract_bearer_token(request: Request) -> str | None:
    auth_header = request.headers.get("authorization", "")
    parts = auth_header.split()
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1]
    return None


def resolve_route_policy(request: Request) -> tuple[str, int, str] | None:
    """
    Determines (bucket_key, limit, policy_type) for the incoming request path and method.
    Returns None if route should skip rate limiting (e.g. OpenAPI docs).
    """
    path = request.url.path.rstrip("/")
    if path == "":
        path = "/"
    method = request.method.upper()

    # Skip API documentation endpoints
    if path in ("/docs", "/redoc", "/openapi.json"):
        return None

    client_ip = _get_client_ip(request)

    # 1. Health and root endpoints (public)
    if path in ("/api/health", "/"):
        limit = settings.rate_limit_health_per_minute
        key = f"ocr:ratelimit:ip:{client_ip}:health"
        return key, limit, "public"

    # Protected routes: inspect bearer token
    token = _extract_bearer_token(request)
    service = resolve_service_from_token(token)

    # 2. Failed / missing authentication on protected endpoints
    if not service:
        limit = settings.rate_limit_auth_fail_per_minute
        key = f"ocr:ratelimit:ip:{client_ip}:auth_fail"
        return key, limit, "auth_fail"

    # 3. Authenticated service requests
    if path == "/api/ocr/process" and method == "POST":
        limit = settings.rate_limit_ocr_per_minute
        key = f"ocr:ratelimit:svc:{service}:ocr"
        return key, limit, "service"

    if path == "/api/duplicate-check" and method == "POST":
        limit = settings.rate_limit_duplicate_per_minute
        key = f"ocr:ratelimit:svc:{service}:duplicate"
        return key, limit, "service"

    if path.startswith("/api/jobs") or path == "/api/metrics":
        limit = settings.rate_limit_read_per_minute
        key = f"ocr:ratelimit:svc:{service}:read"
        return key, limit, "service"

    # Fallback for any other authenticated api routes
    limit = settings.rate_limit_read_per_minute
    key = f"ocr:ratelimit:svc:{service}:general"
    return key, limit, "service"


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware enforcing Redis-backed (with in-memory fallback) rate limiting.
    Injects X-RateLimit-* headers and returns HTTP 429 on quota exhaustion.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        if not settings.rate_limit_enabled:
            return await call_next(request)

        policy = resolve_route_policy(request)
        if policy is None:
            return await call_next(request)

        key, limit, _policy_type = policy
        window_seconds = settings.rate_limit_window_seconds

        result = await rate_limit_backend.check_and_increment(
            key=key,
            limit=limit,
            window_seconds=window_seconds,
        )

        headers = {
            "X-RateLimit-Limit": str(result.limit),
            "X-RateLimit-Remaining": str(result.remaining),
            "X-RateLimit-Reset": str(result.reset_after),
        }

        if not result.allowed:
            headers["Retry-After"] = str(result.reset_after)
            logger.warning(
                "Rate limit exceeded: key=%s limit=%d reset_after=%ds client_ip=%s path=%s",
                key,
                result.limit,
                result.reset_after,
                _get_client_ip(request),
                request.url.path,
            )
            return JSONResponse(
                status_code=429,
                content={
                    "detail": f"Rate limit exceeded. Try again in {result.reset_after} seconds.",
                    "error_code": "rate_limit_exceeded",
                    "retry_after": result.reset_after,
                },
                headers=headers,
            )

        response = await call_next(request)

        # Inject rate limit headers onto response
        for k, v in headers.items():
            response.headers[k] = v

        return response
