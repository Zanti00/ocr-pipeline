from unittest.mock import MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.api.middleware.rate_limit import (
    rate_limit_backend,
)
from app.config import settings
from app.dependencies import resolve_service_from_token
from app.main import app


@pytest_asyncio.fixture(autouse=True)
async def reset_rate_limiter():
    await rate_limit_backend.reset()
    yield
    await rate_limit_backend.reset()


@pytest.mark.asyncio
async def test_resolve_service_from_token():
    assert resolve_service_from_token(settings.serms_api_key) == "serms"
    assert resolve_service_from_token(settings.prs_api_key) == "prs"
    assert resolve_service_from_token("invalid-key") is None
    assert resolve_service_from_token("") is None
    assert resolve_service_from_token(None) is None


@pytest.mark.asyncio
async def test_health_endpoint_rate_limit_and_headers():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        res = await client.get("/api/health")
        assert res.status_code == 200
        assert "X-RateLimit-Limit" in res.headers
        assert "X-RateLimit-Remaining" in res.headers
        assert "X-RateLimit-Reset" in res.headers
        assert res.headers["X-RateLimit-Limit"] == str(settings.rate_limit_health_per_minute)


@pytest.mark.asyncio
async def test_unauthenticated_protected_endpoint_returns_403_and_limits_attempts():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Invalid API key
        res = await client.get("/api/metrics", headers={"Authorization": "Bearer bad-key"})
        assert res.status_code == 403
        assert "X-RateLimit-Limit" in res.headers
        assert res.headers["X-RateLimit-Limit"] == str(settings.rate_limit_auth_fail_per_minute)

        # Missing API key
        res_no_auth = await client.get("/api/metrics")
        assert res_no_auth.status_code == 403
        assert "X-RateLimit-Limit" in res_no_auth.headers


@pytest.mark.asyncio
async def test_rate_limit_exceeded_returns_429():
    await rate_limit_backend.reset()
    with patch.object(settings, "rate_limit_health_per_minute", 2):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            res1 = await client.get("/api/health")
            assert res1.status_code == 200
            assert res1.headers["X-RateLimit-Remaining"] == "1"

            res2 = await client.get("/api/health")
            assert res2.status_code == 200
            assert res2.headers["X-RateLimit-Remaining"] == "0"

            res3 = await client.get("/api/health")
            assert res3.status_code == 429
            assert "Retry-After" in res3.headers
            data = res3.json()
            assert data["error_code"] == "rate_limit_exceeded"
            assert "detail" in data
            assert data["retry_after"] > 0


@pytest.mark.asyncio
async def test_service_isolation_serms_vs_prs():
    await rate_limit_backend.reset()
    # If SERMS reaches limit, PRS must not be blocked
    with patch.object(settings, "rate_limit_read_per_minute", 1):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            serms_headers = {"Authorization": f"Bearer {settings.serms_api_key}"}
            prs_headers = {"Authorization": f"Bearer {settings.prs_api_key}"}

            # SERMS request 1
            res_serms1 = await client.get("/api/metrics", headers=serms_headers)
            assert res_serms1.status_code == 200

            # SERMS request 2 -> blocked
            res_serms2 = await client.get("/api/metrics", headers=serms_headers)
            assert res_serms2.status_code == 429

            # PRS request 1 -> should succeed (different service bucket)
            res_prs1 = await client.get("/api/metrics", headers=prs_headers)
            assert res_prs1.status_code == 200


@pytest.mark.asyncio
async def test_ocr_process_endpoint_rate_limit():
    await rate_limit_backend.reset()
    with (
        patch.object(settings, "rate_limit_ocr_per_minute", 1),
        patch("app.api.routes.ocr.MongoDBClient.create_job") as mock_mongo,
        patch("app.api.routes.ocr.process_receipt_task.delay") as mock_celery,
    ):
        mock_mongo.return_value = None
        mock_celery.return_value = None

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            headers = {"Authorization": f"Bearer {settings.serms_api_key}"}
            payload = {
                "receipt_id": 999,
                "file_url": "https://example.com/receipt.jpg",
                "callback_url": "https://example.com/callback",
                "force_process": True,
            }

            # 1st request succeeds (202)
            res1 = await client.post("/api/ocr/process", json=payload, headers=headers)
            assert res1.status_code == 202
            assert res1.headers["X-RateLimit-Limit"] == "1"
            assert res1.headers["X-RateLimit-Remaining"] == "0"

            # 2nd request blocked (429)
            res2 = await client.post("/api/ocr/process", json=payload, headers=headers)
            assert res2.status_code == 429
            assert "Retry-After" in res2.headers


@pytest.mark.asyncio
async def test_duplicate_check_endpoint_rate_limit():
    await rate_limit_backend.reset()
    with (
        patch.object(settings, "rate_limit_duplicate_per_minute", 1),
        patch("app.api.routes.duplicate.EmbeddingGenerator.generate") as mock_emb,
        patch("app.api.routes.duplicate.find_similar_receipts") as mock_sim,
        patch("app.api.routes.duplicate.AsyncSessionLocal") as mock_session,
    ):
        mock_emb.return_value = [0.1] * 384
        mock_sim.return_value = []
        mock_session.return_value.__aenter__.return_value = MagicMock()

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            headers = {"Authorization": f"Bearer {settings.serms_api_key}"}
            payload = {
                "receipt_text": "Sample receipt text for duplication check",
                "source_service": "serms",
            }

            res1 = await client.post("/api/duplicate-check", json=payload, headers=headers)
            assert res1.status_code == 200
            assert res1.headers["X-RateLimit-Limit"] == "1"

            res2 = await client.post("/api/duplicate-check", json=payload, headers=headers)
            assert res2.status_code == 429


@pytest.mark.asyncio
async def test_redis_failure_falls_back_to_in_memory():
    # Simulate redis failure
    with patch.object(rate_limit_backend, "_get_client", side_effect=Exception("Redis connection refused")):
        res = await rate_limit_backend.check_and_increment("test:key", 2, 60)
        assert res.allowed is True
        assert res.remaining == 1

        res2 = await rate_limit_backend.check_and_increment("test:key", 2, 60)
        assert res2.allowed is True
        assert res2.remaining == 0

        res3 = await rate_limit_backend.check_and_increment("test:key", 2, 60)
        assert res3.allowed is False
        assert res3.remaining == 0


@pytest.mark.asyncio
async def test_rate_limit_disabled_flag():
    with patch.object(settings, "rate_limit_enabled", False):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            res = await client.get("/api/health")
            assert res.status_code == 200
            # Headers should not be added if rate limiter is disabled
            assert "X-RateLimit-Limit" not in res.headers
